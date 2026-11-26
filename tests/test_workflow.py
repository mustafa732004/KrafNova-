"""Workflow tests with the model calls mocked.

Runs without a GEMINI_API_KEY: every Gemini call is replaced by a fake, so
these tests check OUR logic (validation boundary, retry, graph flow, tracer,
fallbacks), not the model. The real calls stay untested until a key is set.

Run from the repo root:  python tests/test_workflow.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("IMAGE_PROVIDER", "stub")
os.environ.setdefault("GEMINI_API_KEY", "")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.agent.nodes import intent as intent_node  # noqa: E402
from app.agent.nodes.vectorize import trace_to_svg  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.design_intent import DesignIntent  # noqa: E402

VALID_JSON = (
    '{"subject": "a cedar tree", "style": "line_art", '
    '"dimensions": {"width_mm": 100, "height_mm": 100}, "notes": null}'
)
INVALID_JSON = '{"subject": "", "style": "watercolor", "dimensions": {"width_mm": -5}}'


def fake_llm(script):
    calls = iter(script)

    def _fake(message, feedback=None):
        return next(calls)

    return _fake


def test_intent_valid_first_try():
    intent_node._call_llm = fake_llm([VALID_JSON])
    out = intent_node.parse_intent({"message": "a cedar tree for a 10cm sign"})
    assert out["error"] is None and out["intent"].subject == "a cedar tree"
    print("ok  intent: valid JSON validates")


def test_intent_retry_then_success():
    intent_node._call_llm = fake_llm([INVALID_JSON, VALID_JSON])
    out = intent_node.parse_intent({"message": "a cedar tree"})
    assert out["error"] is None and out["intent"] is not None
    print("ok  intent: invalid output retried once, then accepted")


def test_intent_gives_up_after_retry():
    intent_node._call_llm = fake_llm([INVALID_JSON, INVALID_JSON])
    out = intent_node.parse_intent({"message": "a cedar tree"})
    assert out["intent"] is None and "invalid design intent" in out["error"]
    print("ok  intent: two failures produce an error (API returns 422)")


def test_full_workflow_stub_images():
    intent_node._call_llm = fake_llm([VALID_JSON])
    client = TestClient(app)
    res = client.post("/design", json={"message": "a cedar tree for a 10cm wood sign"})
    assert res.status_code == 200, res.text
    body = res.json()
    DesignIntent.model_validate(body["intent"])
    assert len(body["candidates"]) == 3
    assert all(c["svg"] and c["svg"].startswith("<svg") for c in body["candidates"])
    assert body["explanation"]  # template fallback, no key configured
    print("ok  workflow: /design 200 with stub images, SVGs, template explanation")


def test_full_workflow_422():
    intent_node._call_llm = fake_llm([INVALID_JSON, INVALID_JSON])
    client = TestClient(app)
    res = client.post("/design", json={"message": "a cedar tree"})
    assert res.status_code == 422
    print("ok  workflow: /design returns 422 when intent never validates")


GOOD_SVG = (
    '<svg viewBox="0 0 100 100"><rect width="100" height="100" fill="white"/>'
    '<path d="M50 10 L90 90 L10 90 Z" stroke="black" fill="none"/></svg>'
)


def test_full_workflow_gemini_svg_provider():
    """gemini-svg: the text model's SVG passes through vectorize untouched."""
    from app.agent.nodes import image_gen
    from app.config import settings

    intent_node._call_llm = fake_llm([VALID_JSON])
    image_gen._call_svg_llm = lambda prompt: "```svg\n" + GOOD_SVG + "\n```"
    old_provider = settings.image_provider
    settings.image_provider = "gemini-svg"
    try:
        client = TestClient(app)
        res = client.post("/design", json={"message": "a cedar tree for a 10cm sign"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert len(body["candidates"]) == 3
        for c in body["candidates"]:
            assert c["svg"].startswith("<svg"), c["svg"][:80]
            assert 'width="100mm"' in c["svg"] and "mm" in c["svg"]
            assert 'xmlns="http://www.w3.org/2000/svg"' in c["svg"]
            assert "<path" in c["svg"]  # model's artwork kept, not a placeholder
            assert c["image_url"].endswith(".svg")
    finally:
        settings.image_provider = old_provider
    print("ok  workflow: gemini-svg provider returns normalized model SVGs")


def test_extract_svg_rejects_garbage():
    from app.agent.nodes.image_gen import _extract_svg, _normalize_svg

    assert _extract_svg("no svg here") is None
    assert _extract_svg("<svg><broken") is None
    svg = _extract_svg("sure! " + GOOD_SVG + " hope that helps")
    assert svg is not None and svg.startswith("<svg")
    out = _normalize_svg(svg, 100.0, 80.0)
    assert 'width="100mm"' in out and 'height="80mm"' in out
    assert 'viewBox="0 0 100 100"' in out  # model's own viewBox preserved
    print("ok  gemini-svg: extraction rejects garbage, normalize sets mm size")


def test_tracer_is_deterministic(tmp_path="."):
    path = os.path.join(tmp_path, "_test_raster.png")
    img = Image.new("L", (32, 32), color=255)
    for x in range(8, 24):
        for y in range(8, 24):
            img.putpixel((x, y), 0)
    img.save(path)
    svg1 = trace_to_svg(path, 100.0, 100.0)
    svg2 = trace_to_svg(path, 100.0, 100.0)
    os.remove(path)
    assert svg1 == svg2, "tracer must be deterministic"
    assert 'width="100.0mm"' in svg1 and "<rect" in svg1
    # a 16x16 black square on a 32px canvas scaled to 100mm = one 50mm rect
    assert 'width="50.000" height="50.000"' in svg1, svg1
    print("ok  vectorize: deterministic SVG, rect-merge tracer correct")


if __name__ == "__main__":
    test_intent_valid_first_try()
    test_intent_retry_then_success()
    test_intent_gives_up_after_retry()
    test_full_workflow_stub_images()
    test_full_workflow_gemini_svg_provider()
    test_extract_svg_rejects_garbage()
    test_full_workflow_422()
    test_tracer_is_deterministic()
    print("\nAll tests passed.")
