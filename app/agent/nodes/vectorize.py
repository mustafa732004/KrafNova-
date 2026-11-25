"""Node 3: deterministic raster-to-vector step.

No LLM here. Same input must give the same SVG every time.
TODO: plug in a real tracer (for example potrace or vtracer) and clean up
paths to the stroke width and dimensions from the intent.
"""


def vectorize(state: dict) -> dict:
    intent = state["intent"]
    w, h = intent.dimensions.width_mm, intent.dimensions.height_mm
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
        f'viewBox="0 0 {w} {h}"><!-- placeholder path --></svg>'
    )
    return {"candidates": [c.model_copy(update={"svg": svg}) for c in state["candidates"]]}
