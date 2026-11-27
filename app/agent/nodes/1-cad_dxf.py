"""CAD provider: the LLM describes the design as plain geometry (JSON), and
ezdxf writes a real DXF. No model-written code is ever executed.

Geometry the model may use (all units mm, origin bottom-left, y up):
  {"type":"line","x1":..,"y1":..,"x2":..,"y2":..}
  {"type":"circle","cx":..,"cy":..,"r":..}
  {"type":"arc","cx":..,"cy":..,"r":..,"start":deg,"end":deg}   (CCW)
  {"type":"polyline","points":[[x,y],...],"closed":true|false}

Output per candidate: a .dxf (R2010, mm, layer CUT) for the CNC, and a
matching .svg preview built from the same entities, so the existing
/generated serving and vectorize pass-through keep working. image_url points
at the SVG preview; dxf_url is logged next to it.
"""
import json
import logging
import math
import os
import re
import uuid

import ezdxf

from app.config import settings
from app.schemas.design_intent import Candidate, DesignIntent

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a master ornamental designer drafting flowing, elegant, artistic "
    "2D CNC cutting art (Art Nouveau / Arabesque / botanical line work). Reply "
    'with ONLY a JSON object: {"center":[x,y],"rotations":N,"mirror":bool,'
    '"motif":[...],"fixed":[...]}. "motif" is ONE sector/petal unit that code '
    "repeats N times around center (N 4..16 for radial designs; for a subject with a left/right axis such as a scarab, lotus, winged sun disk use rotations=1 with mirror=true and draw only the RIGHT half; for an asymmetric subject such as an Eye of Horus use rotations=1, mirror=false and put everything in fixed) and, if mirror is true, also "
    "mirrors for perfect symmetry. \"fixed\" shapes are drawn once (center "
    "rosette, rings, borders). Entity types: "
    '{"type":"spline","points":[[x,y],...],"closed":true|false} - a smooth '
    "flowing curve through the points (PREFER this: use 6-14 points for "
    "organic petals, leaves, vines, teardrops, S-curves, swirls); "
    '{"type":"circle","cx","cy","r"}; {"type":"arc","cx","cy","r","start",'
    '"end"} (deg, CCW); {"type":"line",...}; {"type":"polyline","points",'
    '"closed"} (only for straight-edged parts). Units mm, origin bottom-left, '
    "y up; stay inside the canvas with 5 mm margin. Make it rich and layered: "
    "for radial subjects a large outer petal ring, a smaller inner ring of different shape "
    "offset by half a step, leaf/vine accents, curled tips, a detailed "
    "center rosette with 2-3 concentric elements. Closed splines for "
    "anything to be cut out; keep minimum feature size above 2 mm. No text, "
    "no dimensions."
)


def _num(v):
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite")
    return f


def _catmull(pts, closed, per=14):
    """Centripetal Catmull-Rom through the points, sampled densely: a smooth
    flowing curve that CNC post-processors read as a plain polyline."""
    n = len(pts)
    if n < 3:
        return list(pts)
    P = list(pts)
    segs = n if closed else n - 1
    out = []
    for i in range(segs):
        if closed:
            p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        else:
            p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, n - 1)]
        ts = [0.0]
        for a_, b_ in ((p0, p1), (p1, p2), (p2, p3)):
            ts.append(ts[-1] + max(math.dist(a_, b_), 1e-6) ** 0.5)
        t0, t1, t2, t3 = ts
        for k in range(per):
            t = t1 + (t2 - t1) * k / per

            def lerp(u, v, ua, ub):
                f = (t - ua) / (ub - ua)
                return (u[0] + (v[0] - u[0]) * f, u[1] + (v[1] - u[1]) * f)
            a1 = lerp(p0, p1, t0, t1); a2 = lerp(p1, p2, t1, t2); a3 = lerp(p2, p3, t2, t3)
            b1 = lerp(a1, a2, t0, t2); b2 = lerp(a2, a3, t1, t3)
            out.append(lerp(b1, b2, t1, t2))
    if not closed:
        out.append(P[-1])
    return out


def _one(e):
    """One raw entity -> normalized entity (or None). Splines and arcs become
    dense polylines so rotating/mirroring is exact."""
    t = e["type"]
    if t == "line":
        return {"type": "polyline", "points": [(_num(e["x1"]), _num(e["y1"])), (_num(e["x2"]), _num(e["y2"]))], "closed": False}
    if t == "circle":
        r = _num(e["r"])
        return {"type": "circle", "cx": _num(e["cx"]), "cy": _num(e["cy"]), "r": r} if r > 0 else None
    if t == "arc":
        r = _num(e["r"])
        if r <= 0:
            return None
        ee = {"cx": _num(e["cx"]), "cy": _num(e["cy"]), "r": r, "start": _num(e["start"]), "end": _num(e["end"])}
        return {"type": "polyline", "points": _arc_pts(ee), "closed": False}
    if t in ("polyline", "spline"):
        pts = [(_num(q[0]), _num(q[1])) for q in e["points"]]
        if len(pts) < 2:
            return None
        closed = bool(e.get("closed"))
        if t == "spline":
            pts = _catmull(pts, closed)
        return {"type": "polyline", "points": pts, "closed": closed}
    return None


def _norm(raw):
    out = []
    for e in raw[:600]:
        try:
            r = _one(e)
        except (KeyError, TypeError, ValueError, IndexError):
            continue  # drop malformed entity, keep the rest
        if r:
            out.append(r)
    return out


def _xf(e, fn):
    if e["type"] == "circle":
        x, y = fn(e["cx"], e["cy"])
        return {**e, "cx": x, "cy": y}
    return {**e, "points": [fn(x, y) for x, y in e["points"]]}


def parse_entities(text: str, w: float, h: float) -> list[dict]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON")
    data = json.loads(m.group(0))
    if isinstance(data.get("entities"), list):  # old flat format still accepted
        data = {"fixed": data["entities"]}
    fixed = _norm(data.get("fixed") or [])
    motif = _norm(data.get("motif") or [])
    out = list(fixed)
    if motif:
        cx, cy = data.get("center") or [w / 2, h / 2]
        cx, cy = _num(cx), _num(cy)
        n = max(1, min(int(_num(data.get("rotations", 1))), 32))
        mirror = bool(data.get("mirror"))
        for k in range(n):
            a = 2 * math.pi * k / n
            ca, sa = math.cos(a), math.sin(a)

            def rot(x, y):
                dx, dy = x - cx, y - cy
                return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)

            def mir(x, y):
                return rot(2 * cx - x, y)  # reflect across the vertical axis, then rotate
            for e in motif:
                out.append(_xf(e, rot))
                if mirror:
                    out.append(_xf(e, mir))
    if not out:
        raise ValueError("no valid entities")
    if len(out) > 800:
        raise ValueError("too many entities")
    # Reject designs that spill off the stock: bbox must sit inside the canvas.
    xs, ys = [], []
    for e in out:
        if e["type"] == "polyline":
            xs += [q[0] for q in e["points"]]; ys += [q[1] for q in e["points"]]
        else:
            xs += [e["cx"] - e["r"], e["cx"] + e["r"]]; ys += [e["cy"] - e["r"], e["cy"] + e["r"]]
    if min(xs) < -1 or min(ys) < -1 or max(xs) > w + 1 or max(ys) > h + 1:
        raise ValueError("geometry outside the canvas")
    return out


def _arc_pts(e, n=48):
    a0, a1 = math.radians(e["start"]), math.radians(e["end"])
    if a1 <= a0:
        a1 += 2 * math.pi
    return [(e["cx"] + e["r"] * math.cos(a0 + (a1 - a0) * i / n),
             e["cy"] + e["r"] * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def write_dxf(entities: list[dict], path: str) -> None:
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.layers.add("CUT", color=7)
    msp = doc.modelspace()
    at = {"layer": "CUT"}
    for e in entities:
        if e["type"] == "line":
            msp.add_line((e["x1"], e["y1"]), (e["x2"], e["y2"]), dxfattribs=at)
        elif e["type"] == "circle":
            msp.add_circle((e["cx"], e["cy"]), e["r"], dxfattribs=at)
        elif e["type"] == "arc":
            msp.add_arc((e["cx"], e["cy"]), e["r"], e["start"], e["end"], dxfattribs=at)
        else:
            msp.add_lwpolyline(e["points"], close=e["closed"], dxfattribs=at)
    doc.saveas(path)


def to_svg(entities: list[dict], w: float, h: float) -> str:
    """Preview SVG of the same geometry (y flipped: DXF is y-up)."""
    def P(x, y):
        return f"{x:.2f},{h - y:.2f}"
    els = []
    for e in entities:
        if e["type"] == "line":
            els.append(f'<path d="M{P(e["x1"], e["y1"])} L{P(e["x2"], e["y2"])}"/>')
        elif e["type"] == "circle":
            els.append(f'<circle cx="{e["cx"]:.2f}" cy="{h - e["cy"]:.2f}" r="{e["r"]:.2f}"/>')
        elif e["type"] == "arc":
            pts = _arc_pts(e)
            els.append('<path d="M' + " L".join(P(*p) for p in pts) + '"/>')
        else:
            d = "M" + " L".join(P(*p) for p in e["points"]) + (" Z" if e["closed"] else "")
            els.append(f'<path d="{d}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" height="{h:g}mm" '
        f'viewBox="0 0 {w:g} {h:g}"><rect width="{w:g}" height="{h:g}" fill="white"/>'
        '<g fill="none" stroke="black" stroke-width="0.6">' + "".join(els) + "</g></svg>"
    )


def gemini_dxf_candidates(intent: DesignIntent, call_llm, english_subject) -> list[Candidate]:
    os.makedirs(settings.generated_dir, exist_ok=True)
    batch = uuid.uuid4().hex[:8]
    w, h = intent.dimensions.width_mm, intent.dimensions.height_mm
    subject = english_subject(intent.subject)
    out: list[Candidate] = []
    for i in range(1, settings.image_candidates + 1):
        prompt = (
            f"Design: {subject}, {intent.style} style. Canvas {w:g} x {h:g} mm."
            + (f" Note: {intent.notes}" if intent.notes and intent.notes.isascii() else "")
            + f" Variation {i}."
        )
        ents = None
        for _ in range(2):  # one retry on unparseable output
            try:
                ents = parse_entities(call_llm(_SYSTEM, prompt), w, h)
                break
            except Exception as exc:
                log.warning("gemini-dxf candidate %s attempt failed: %s", i, exc)
        if ents is None:
            continue
        base = f"{batch}-cand-{i}"
        write_dxf(ents, os.path.join(settings.generated_dir, base + ".dxf"))
        svg = to_svg(ents, w, h)
        with open(os.path.join(settings.generated_dir, base + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(svg)
        log.info("gemini-dxf: /generated/%s.dxf", base)
        out.append(Candidate(id=f"cand-{i}", image_url=f"/generated/{base}.svg", svg=svg))
    return out

