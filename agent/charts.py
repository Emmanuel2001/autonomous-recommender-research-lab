"""Inline-SVG chart primitives for the run report. No libraries, no runtime.

Everything here is pure: (numbers) -> (svg string). That is deliberate -- charts are
the part of a report a reader trusts most and checks least, so they are the part most
worth being able to unit-test. tests/test_charts.py exercises the degenerate cases
that silently produce a wrong picture rather than an exception: an empty series, a
single point, and a perfectly flat series (where a naive (v-lo)/(hi-lo) is 0/0).

Colour comes from `currentColor` and CSS custom properties so one drawing works on
both the light and the dark ground; no literal hex appears in a path.
"""
from __future__ import annotations

import html
import math

__all__ = ["trajectory", "multiline", "bars", "PAD", "LABEL_ROOM"]

PAD = {"l": 58, "r": 14, "t": 14, "b": 26}
#: horizontal room bars() keeps free for the value label after the longest bar.
#: Sized for the worst realistic label ("999 / 999", ~9 chars at ~6.2u each)
#: plus the 6u gap. tests/test_charts.py asserts no text escapes the viewBox.
LABEL_ROOM = 70


def _finite(vals) -> list[float]:
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def _range(vals: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    """A drawable y-range. A flat series is the case that matters: lo == hi makes
    every (v-lo)/(hi-lo) a 0/0, so widen around the value instead of clamping the
    span to 1.0, which would squash a real series into a hairline at the bottom."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        d = max(abs(hi) * 0.01, 1e-4)
        return lo - d, hi + d
    m = (hi - lo) * pad_frac
    return lo - m, hi + m


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1000 or (a < 0.001 and a > 0):
        return f"{v:.2e}"
    return f"{v:.5f}".rstrip("0").rstrip(".") if a < 1 else f"{v:.4g}"


def _frame(w: int, h: int, lo: float, hi: float, n_x: int,
           x_label: str, y_ticks: int = 4) -> tuple[str, callable, callable]:
    """Axes plus the two mappers. Returned as functions so no caller recomputes the
    geometry -- a second copy of the arithmetic is how a chart's grid and its data
    end up on different scales."""
    x0, x1 = PAD["l"], w - PAD["r"]
    y0, y1 = h - PAD["b"], PAD["t"]

    def X(i: float) -> float:
        return x0 if n_x <= 1 else x0 + i * (x1 - x0) / (n_x - 1)

    def Y(v: float) -> float:
        return y0 - (v - lo) / (hi - lo) * (y0 - y1)

    g = [f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' class='ax-line'/>"]
    for t in range(y_ticks + 1):
        v = lo + (hi - lo) * t / y_ticks
        y = Y(v)
        g.append(f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' class='grid'/>")
        g.append(f"<text x='{x0 - 6}' y='{y + 3.5:.1f}' class='ax' "
                 f"text-anchor='end'>{_fmt(v)}</text>")
    if x_label:
        g.append(f"<text x='{(x0 + x1) / 2:.0f}' y='{h - 6}' class='ax' "
                 f"text-anchor='middle'>{html.escape(x_label)}</text>")
    return "".join(g), X, Y


def _empty(msg: str) -> str:
    return f"<p class='muted'>{html.escape(msg)}</p>"


def trajectory(values, baseline: float | None = None, band: float | None = None,
               labels=None, w: int = 880, h: int = 260,
               x_label: str = "evaluation (in ledger order)",
               aria: str = "score by evaluation") -> str:
    """Score per evaluation, with the running best as a step line.

    `baseline` draws a reference rule and `band` a +/-1 shaded strip around it, so a
    point inside the strip is visibly inside seed noise rather than merely close to
    the line. That framing is the whole reason this chart exists: without the strip
    every upward wiggle reads as progress.
    """
    vals = _finite(values)
    if not vals:
        return _empty("no evaluations to plot")
    pool = list(vals)
    if baseline is not None:
        pool += [baseline - (band or 0), baseline + (band or 0)]
    lo, hi = _range(pool)
    frame, X, Y = _frame(w, h, lo, hi, max(len(vals), 2), x_label)

    parts = [frame]
    if baseline is not None:
        if band:
            yt, yb = Y(baseline + band), Y(baseline - band)
            parts.append(f"<rect x='{PAD['l']}' y='{yt:.1f}' "
                         f"width='{w - PAD['l'] - PAD['r']}' height='{abs(yb - yt):.1f}' "
                         f"class='band'/>")
        yb_ = Y(baseline)
        parts.append(f"<line x1='{PAD['l']}' y1='{yb_:.1f}' x2='{w - PAD['r']}' "
                     f"y2='{yb_:.1f}' class='ref'/>")
        parts.append(f"<text x='{w - PAD['r'] - 2}' y='{yb_ - 5:.1f}' class='ax' "
                     f"text-anchor='end'>official baseline {_fmt(baseline)}"
                     f"{f' &plusmn;{band:g} seed sd' if band else ''}</text>")

    # running best, drawn under the samples so the samples stay readable
    run, cur = [], -math.inf
    for v in vals:
        cur = max(cur, v)
        run.append(cur)
    step = []
    for i, v in enumerate(run):
        if i:
            step.append(f"{X(i):.1f},{Y(run[i - 1]):.1f}")
        step.append(f"{X(i):.1f},{Y(v):.1f}")
    parts.append(f"<polyline points='{' '.join(step)}' class='best-step'/>")

    if len(vals) > 1:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        parts.append(f"<polyline points='{pts}' class='series'/>")
    bi = max(range(len(vals)), key=lambda i: vals[i])
    for i, v in enumerate(vals):
        cls = "pt-best" if i == bi else "pt"
        t = html.escape(str(labels[i])) if labels and i < len(labels) else f"{v:.5f}"
        parts.append(f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='{4 if i == bi else 2.6}' "
                     f"class='{cls}'><title>{t}</title></circle>")
    return (f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(aria)}'>"
            + "".join(parts) + "</svg>")


def multiline(series: list[tuple[str, list]], w: int = 880, h: int = 260,
              x_label: str = "epoch", aria: str = "curves",
              emphasise: str | None = None, invert_best: bool = False) -> str:
    """Overlaid curves that share one y-axis. `series` is [(name, values)].

    Curves of different lengths are normal here (early stopping fires at different
    epochs), so each is drawn against its own length on a shared x-scale set by the
    longest -- a curve that stopped early simply ends sooner, which is the true
    picture rather than a stretched one.
    """
    clean = [(n, _finite(v)) for n, v in series]
    clean = [(n, v) for n, v in clean if v]
    if not clean:
        return _empty("no curves recorded for these evaluations")
    pool = [v for _n, vs in clean for v in vs]
    lo, hi = _range(pool)
    n_x = max((len(v) for _n, v in clean), default=2)
    frame, X, Y = _frame(w, h, lo, hi, max(n_x, 2), x_label)

    parts = [frame]
    for n, vs in clean:
        em = (emphasise is not None and n == emphasise)
        cls = "curve-em" if em else "curve"
        if len(vs) == 1:
            parts.append(f"<circle cx='{X(0):.1f}' cy='{Y(vs[0]):.1f}' r='3' "
                         f"class='{cls.replace('curve', 'pt')}'/>")
        else:
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vs))
            parts.append(f"<polyline points='{pts}' class='{cls}'><title>"
                         f"{html.escape(n)}</title></polyline>")
        k = (min if invert_best else max)(range(len(vs)), key=lambda i: vs[i])
        parts.append(f"<circle cx='{X(k):.1f}' cy='{Y(vs[k]):.1f}' r='3' "
                     f"class='{'pt-best' if em else 'pt'}'><title>"
                     f"{html.escape(n)} @ epoch {k + 1} = {vs[k]:.6g}</title></circle>")
    return (f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(aria)}'>"
            + "".join(parts) + "</svg>")


def bars(items: list[tuple[str, float, float]], w: int = 880, bar_h: int = 22,
         aria: str = "per-action outcome") -> str:
    """Horizontal bars: [(label, total, highlighted_subset)]. Used for tried-vs-improved
    per action, where the subset is always <= the total."""
    items = [(n, float(t), float(s)) for n, t, s in items if t]
    if not items:
        return _empty("no actions recorded")
    h = PAD["t"] + PAD["b"] + len(items) * bar_h
    mx = max(t for _n, t, _s in items)
    # Reserve room for the value label that sits AFTER the widest bar. The old 40 units were
    # not enough for even one label: the max-total bar ends exactly at x1, its label starts
    # at x1+6, and "5 / 1 improved" needs ~87 units -- ending at ~919 in an 880-wide
    # viewBox, clipped in every chart that had an improvement, which is every interesting
    # one. The word "improved" is dropped too: the figure caption already explains the
    # highlight, so repeating it on every row cost width and said nothing.
    x0, x1 = 150, w - PAD["r"] - LABEL_ROOM
    parts = []
    for i, (n, tot, sub) in enumerate(items):
        y = PAD["t"] + i * bar_h
        wt = (x1 - x0) * tot / mx
        ws = (x1 - x0) * min(sub, tot) / mx
        parts.append(f"<text x='{x0 - 8}' y='{y + bar_h / 2 + 3.5:.1f}' class='ax' "
                     f"text-anchor='end'>{html.escape(n)}</text>")
        parts.append(f"<rect x='{x0}' y='{y + 3}' width='{wt:.1f}' "
                     f"height='{bar_h - 8}' class='bar'/>")
        if ws > 0:
            parts.append(f"<rect x='{x0}' y='{y + 3}' width='{ws:.1f}' "
                         f"height='{bar_h - 8}' class='bar-hi'/>")
        parts.append(f"<text x='{x0 + wt + 6:.1f}' y='{y + bar_h / 2 + 3.5:.1f}' "
                     f"class='ax'>{tot:g}{f' / {sub:g}' if sub else ''}</text>")
    return (f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(aria)}'>"
            + "".join(parts) + "</svg>")
