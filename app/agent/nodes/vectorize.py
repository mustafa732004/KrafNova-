"""Node 3: deterministic raster-to-vector step.

No LLM and no randomness here: the same input file always produces the same
SVG.

Backends (settings.vector_backend):
  - "vtracer" (default when installed): real curve tracer. The raster is
    preprocessed first (grayscale, autocontrast, Otsu binarization, small
    crop to cut the pollinations watermark strip), then traced to smooth
    bezier paths. Produces CNC-usable line art instead of blocky pixels.
  - "rects": the original rule-based bitmap -> rect-merge pass (threshold
    to pure black/white, merge equal horizontal runs on adjacent rows into
    rectangles). Kept as a dependency-free fallback.

Candidates without a real raster (stub image provider) keep a placeholder
SVG with the intent's dimensions, exactly like before.
"""
import os
import re
import tempfile

import numpy as np
from PIL import Image, ImageOps

from app.config import settings
from app.schemas.design_intent import DesignIntent

try:
    import vtracer

    _HAS_VTRACER = True
except ImportError:  # pragma: no cover - optional dependency
    vtracer = None
    _HAS_VTRACER = False

_THRESHOLD = 128  # grayscale value below which a pixel counts as ink


def _otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's binarization threshold for a uint8 grayscale array."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = float((hist * np.arange(256)).sum())
    sum_bg = 0.0
    weight_bg = 0.0
    best_var = -1.0
    best_t = 128
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0 or weight_bg == total:
            continue
        sum_bg += t * hist[t]
        weight_fg = total - weight_bg
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return best_t


def _preprocess_for_trace(image_path: str, out_path: str) -> None:
    """Clean a raster for tracing: grayscale, autocontrast, Otsu binarize.

    Also crops a small strip off the bottom edge, where the pollinations
    free tier stamps its watermark, so the mark does not end up in the SVG.
    """
    img = Image.open(image_path).convert("L")
    if max(img.size) > settings.vector_max_px:
        scale = settings.vector_max_px / max(img.size)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        )
    img = ImageOps.autocontrast(img, cutoff=1)
    w, h = img.size
    crop = round(h * settings.vector_watermark_crop)
    if crop > 0:
        img = img.crop((0, 0, w, h - crop))
    gray = np.asarray(img, dtype=np.uint8)
    threshold = _otsu_threshold(gray)
    binary = np.where(gray < threshold, 0, 255).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(out_path)


def _to_mm_svg(svg: str, width_mm: float, height_mm: float) -> str:
    """Rewrite a pixel-sized SVG root so the physical size is in mm."""
    m = re.search(r'<svg[^>]*width="([\d.]+)"[^>]*height="([\d.]+)"', svg)
    if not m:
        return svg
    px_w, px_h = float(m.group(1)), float(m.group(2))
    if "viewBox" not in svg:
        svg = svg.replace(
            "<svg",
            f'<svg viewBox="0 0 {px_w:g} {px_h:g}"',
            1,
        )
    svg = re.sub(r'width="[\d.]+"', f'width="{width_mm}mm"', svg, count=1)
    svg = re.sub(r'height="[\d.]+"', f'height="{height_mm}mm"', svg, count=1)
    return svg


def vtrace_to_svg(image_path: str, width_mm: float, height_mm: float) -> str:
    """Trace a raster into smooth bezier paths with vtracer. Deterministic."""
    with tempfile.TemporaryDirectory() as tmp:
        clean_png = os.path.join(tmp, "clean.png")
        out_svg = os.path.join(tmp, "out.svg")
        _preprocess_for_trace(image_path, clean_png)
        vtracer.convert_image_to_svg_py(
            clean_png,
            out_svg,
            colormode="binary",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=settings.vector_filter_speckle,
            corner_threshold=60,
            length_threshold=4.0,
            splice_threshold=45,
            max_iterations=10,
            path_precision=2,
        )
        with open(out_svg, encoding="utf-8") as fh:
            svg = fh.read()
    return _to_mm_svg(svg, width_mm, height_mm)


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


def _trace(image_path: str, width_mm: float, height_mm: float) -> str:
    backend = settings.vector_backend
    if backend == "vtracer" and not _HAS_VTRACER:
        backend = "rects"
    if backend == "vtracer":
        return vtrace_to_svg(image_path, width_mm, height_mm)
    return trace_to_svg(image_path, width_mm, height_mm)


def vectorize(state: dict) -> dict:
    intent = state["intent"]
    out = []
    for cand in state["candidates"]:
        if cand.local_path:
            svg = _trace(
                cand.local_path,
                intent.dimensions.width_mm,
                intent.dimensions.height_mm,
            )
        else:
            svg = _placeholder_svg(intent)
        out.append(cand.model_copy(update={"svg": svg}))
    return {"candidates": out}
