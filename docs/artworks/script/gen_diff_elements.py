# Prints the SVG fragments of the "differentiable" additions to the UNsim hero (1280x640).
# Usage: python unsim/script/gen_diff_elements.py {descent|descent_route}
# The fragments are in the coordinates of the earlier hero with the ring of arterials (mark at translate(130.92 90.72) scale(0.88)); the SVG places them with
# transform="translate(-120.5 -49.72) scale(1.17)", which scales them by 1.17 about (650, 316) and moves that point to the canvas centre (640, 320).
import math, sys
import numpy as np
import contourpy

W, H = 1280, 640
IX0, IX1, IY0, IY1 = 650 - 640 / 1.17, 650 + 640 / 1.17, 316 - 320 / 1.17, 316 + 320 / 1.17   # the canvas in these coordinates
def f1(v): return f"{v:.1f}".replace("-0.0", "0.0")

def simplify(pts, tol):
    # Douglas-Peucker
    pts = np.asarray(pts, float)
    if len(pts) < 3: return pts
    keep = np.zeros(len(pts), bool); keep[0] = keep[-1] = True; stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        a, b = pts[i], pts[j]; ab = b - a; n = np.hypot(*ab)
        seg = pts[i + 1:j]
        if len(seg) == 0: continue
        d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / n if n > 0 else np.hypot(*(seg - a).T)
        k = int(np.argmax(d))
        if d[k] > tol: keep[i + 1 + k] = True; stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]
def path_d(pts):
    # absolute moveto, relative lineto
    s = f"M{f1(pts[0][0])} {f1(pts[0][1])}"
    px, py = round(pts[0][0], 1), round(pts[0][1], 1)
    out = []
    for x, y in pts[1:]:
        x, y = round(x, 1), round(y, 1)
        out.append(f"{f1(x - px)} {f1(y - py)}"); px, py = x, y
    return s + ("l" + " ".join(out) if out else "")

# ---------- 1. gradient descent on a loss landscape ----------
def g2(x, y, cx, cy, sx, sy, amp, rot):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    u = (x - cx) * c + (y - cy) * s; v = -(x - cx) * s + (y - cy) * c
    return amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
LM = (955.0, 488.0)   # the minimum, below the tagline
FEAT = [(955, 488, 190, 95, -150, -6), (330, 560, 210, 110, -70, 15), (620, 150, 170, 120, 95, -25), (175, 170, 120, 130, 70, 0),
        (1150, 140, 120, 90, 75, 20), (800, 330, 300, 70, 35, -8), (560, 470, 90, 70, 30, 0)]
# valley carved along a smooth curve from beyond the right edge of the canvas, below the end of the tagline, into the basin
VALLEY = [(1250.0, 390.0), (1180.0, 420.0), (1150.0, 490.0), LM]
def _valley_pts(n=240):
    t = np.linspace(0, 1, n)[:, None]; P = [np.array(p) for p in VALLEY]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3], t[:, 0]
def loss(x, y):
    z = 0.10 * np.hypot(x - LM[0], (y - LM[1]) * 1.3)
    for p in FEAT: z = z + g2(x, y, *p)
    C, t = _valley_pts()
    x = np.asarray(x, float); y = np.asarray(y, float)
    d2 = np.full(np.shape(x), np.inf); tk = np.zeros(np.shape(x))
    for (cx, cy), ti in zip(C, t):
        e = (x - cx) ** 2 + (y - cy) ** 2; m = e < d2
        d2 = np.where(m, e, d2); tk = np.where(m, ti, tk)
    return z - (30 + 40 * tk) * np.exp(-0.5 * d2 / 32.0 ** 2)
def descent(route_only=False):
    if route_only: return descent_route()
    xs = np.arange(IX0 - 1.5, IX1 + 1.5, 1.5); ys = np.arange(IY0 - 1.5, IY1 + 1.5, 1.5)
    Xg, Yg = np.meshgrid(xs, ys); Z = loss(Xg, Yg)
    gen = contourpy.contour_generator(xs, ys, Z)
    step = 6.5; levs = np.arange(Z.min() + step * 0.5, Z.max(), step)
    reg, idx = [], []
    for i, L in enumerate(levs):
        for line in gen.lines(L):
            if len(line) < 4: continue
            (idx if i % 5 == 4 else reg).append(path_d(simplify(line, 0.35)))
    print("CONTOURS_REGULAR"); print(" ".join(reg))
    print("CONTOURS_INDEX"); print(" ".join(idx))
    descent_route()
def descent_route():
    # steepest descent from the start of the valley beyond the right edge; stops where the path starts to dither in the flat bottom
    def grad(x, y, h=0.5): return ((loss(x + h, y) - loss(x - h, y)) / (2 * h), (loss(x, y + h) - loss(x, y - h)) / (2 * h))
    x, y = VALLEY[0]; pts = [(x, y)]
    for _ in range(4000):
        gx, gy = grad(x, y); gn = math.hypot(gx, gy)
        x -= gx / gn; y -= gy / gn; pts.append((x, y))
        if math.dist((x, y), LM) < 30.0: break
    pts.append(LM)   # last stretch straight into the minimum
    P = simplify(pts, 0.25)
    enter = next(p for p in pts if p[0] <= IX1)   # where the route enters the canvas (start of the colour gradient)
    print("ROUTE"); print(path_d(P)); print(f"start {f1(pts[0][0])},{f1(pts[0][1])} enter {f1(enter[0])},{f1(enter[1])} end {f1(pts[-1][0])},{f1(pts[-1][1])}")
    # gradient-descent iterates: arc-length positions with geometrically shrinking steps whose sum is the route length
    A = np.array(pts); L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(A, axis=0).T))]); tot = L[-1]
    q = 0.68; s, d, marks = 0.0, tot * (1 - q), []
    while d > 4:
        s += d; d *= q
        j = int(np.searchsorted(L, s)); marks.append(A[min(j, len(A) - 1)])
    print("ITERATES"); print(" ".join(f"{f1(p[0])},{f1(p[1])}" for p in marks))

if __name__ == "__main__":
    {"descent": descent, "descent_route": descent_route}[sys.argv[1]]()
