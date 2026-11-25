"""Node 3: deterministic raster-to-vector step.

No LLM and no randomness here: the same input file always produces the same
SVG. The tracer is a simple rule-based bitmap -> rect-merge pass (threshold
the image to pure black/white, then merge equal horizontal runs on adjacent
rows into rectangles). It is intentionally simple and inspectable; a potrace
or vtracer backend can replace trace_to_svg without touching the graph.

Candidates without a real raster (stub image provider) keep a placeholder
SVG with the intent's dimensions, exactly like before.
"""
from PIL import Image

from app.config import settings
from app.schemas.design_intent import DesignIntent

_THRESHOLD = 128  # grayscale value below which a pixel counts as ink


def trace_to_svg(image_path: str, width_mm: float, height_mm: float) -> str:
    """Trace a raster file into an SVG of black rectangles. Deterministic."""
    img = Image.open(image_path).convert("L")
    if max(img.size) > settings.vector_max_px:
        scale = settings.vector_max_px / max(img.size)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        )
    w, h = img.size
    ink = [1 if p < _THRESHOLD else 0 for p in img.getdata()]

    # Horizontal runs per row: (row, x0, x1_exclusive).
    runs = []
    for y in range(h):
        row = ink[y * w : (y + 1) * w]
        x = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                runs.append([y, x0, x])
            else:
                x += 1

    # Merge identical runs on consecutive rows into rectangles.
    # rect is [y0, y1_exclusive, x0, x1_exclusive].
    active: dict[tuple[int, int], list] = {}
    rects = []
    for y, x0, x1 in runs:
        key = (x0, x1)
        if key in active and active[key][1] == y:
            active[key][1] = y + 1
        else:
            rect = [y, y + 1, x0, x1]
            active[key] = rect
            rects.append(rect)

    sx = width_mm / w
    sy = height_mm / h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
        f'height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}">',
        '<g fill="#000">',
    ]
    for y0, y1, x0, x1 in rects:
        parts.append(
            f'<rect x="{x0 * sx:.3f}" y="{y0 * sy:.3f}" '
            f'width="{(x1 - x0) * sx:.3f}" height="{(y1 - y0) * sy:.3f}"/>'
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _placeholder_svg(intent: DesignIntent) -> str:
    w, h = intent.dimensions.width_mm, intent.dimensions.height_mm
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
        f'viewBox="0 0 {w} {h}"><!-- placeholder path --></svg>'
    )


def vectorize(state: dict) -> dict:
    intent = state["intent"]
    out = []
    for cand in state["candidates"]:
        if cand.local_path:
            svg = trace_to_svg(
                cand.local_path,
                intent.dimensions.width_mm,
                intent.dimensions.height_mm,
            )
        else:
            svg = _placeholder_svg(intent)
        out.append(cand.model_copy(update={"svg": svg}))
    return {"candidates": out}
