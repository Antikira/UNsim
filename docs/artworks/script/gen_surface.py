# Prints the SVG fragments for the UNsim hero (1280x640): a single-peaked loss landscape drawn as a 3D wireframe with contour lines that
# fade out towards the foothills, and a steepest-ascent trajectory that climbs to the summit.
# Usage: python unsim/script/gen_surface.py [key=value ...]   (CFG holds the view, the line styles and the start of the climb; the arguments override it)
#   unsim-hero-*.svg (foothills reaching down to the height of the summit mirrored about the canvas centre): hp=240 sq=150 start=761.5,196.8
# Ground coordinates: p to the right, q away from the viewer, height h up (all in px). Hidden lines are removed by marching from each sample point towards the camera.
# Screen coordinates are those of gen_diff_elements.py; the fragments carry its transform="translate(-120.5 -49.72) scale(1.17)".
import math, os, sys
import numpy as np
import contourpy
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diff_elements import simplify, path_d, f1, IX0, IX1, IY0, IY1

TF = 'transform="translate(-120.5 -49.72) scale(1.17)"'
CFG = dict(elev=30.0, azim=-45.0, dist=1600.0, X0=640.0, Y0=316.0, k=1.0, mesh=24.0, dz=16.0, start="636,185",
           px=900.0, py=112.0, sp=140.0, sq=120.0, hp=190.0, rot=0.0,
           mc="9FB6CE", mw=0.6, mo=0.8, cw=0.9, co=0.5, t0=0.04, t1=0.5, bins=10)
# outlines of the letters (screen boxes), for the clearance of the trajectory: U, N, s, dot of i, stem of i, m, tagline
TEXT = [(604.5, 221.3, 704.3, 346.0), (711.9, 221.1, 839.5, 344.8), (838.7, 245.2, 920.1, 346.3), (930.7, 206.1, 961.2, 235.1),
        (933.4, 246.7, 958.1, 344.8), (975.5, 245.3, 1137.3, 344.8), (604.5, 378.5, 1137.3, 409.2)]

# ---------- landscape: a single peak, placed so that its summit projects to (px, py); ground sizes sp, sq; height hp; rotation rot in deg ----------
FEAT, H = None, None   # set by fragments()
def height(p, q):
    cp, cq, sp, sq, A, rot = FEAT
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    p = np.asarray(p, float) - cp; q = np.asarray(q, float) - cq
    a, b = p * c + q * s, -p * s + q * c
    return A * np.exp(-0.5 * ((a / sp) ** 2 + (b / sq) ** 2))
def grad(p, q, e=0.5): return ((height(p + e, q) - height(p - e, q)) / (2 * e), (height(p, q + e) - height(p, q - e)) / (2 * e))

# ---------- camera: orthographic (dist = 0) or perspective; the ground origin projects to (X0, Y0) ----------
class Cam:
    def __init__(s, c):
        th = math.radians(c["elev"]); s.st, s.ct = math.sin(th), math.cos(th)
        s.X0, s.Y0, s.k, s.L = c["X0"], c["Y0"], c["k"], c["dist"]
    def project(s, p, q, h):
        yc = q * s.st + h * s.ct; zc = q * s.ct - h * s.st
        f = s.L / np.maximum(s.L + zc, 1e-6) if s.L > 0 else 1.0
        return s.X0 + s.k * p * f, s.Y0 - s.k * yc * f
    def place(s, X, Y, h):
        # point (p, q) whose surface point at height h projects to (X, Y)
        m = (s.Y0 - Y) / s.k
        q = (h * s.ct * s.L + m * h * s.st - m * s.L) / (m * s.ct - s.st * s.L) if s.L > 0 else (m - h * s.ct) / s.st
        f = s.L / (s.L + q * s.ct - h * s.st) if s.L > 0 else 1.0
        return (X - s.X0) / (s.k * f), q
    def front(s, p, q, h):
        # in front of the camera (always true for the orthographic view)
        return (s.L + q * s.ct - h * s.st > 10) if s.L > 0 else np.ones(np.shape(p), bool)
    def hidden(s, p, q, h, hmax, dt=2.0, eps=0.4, t0=1.5):
        if s.L > 0:
            dx, dy, dz = -p, -s.L * s.ct - q, s.L * s.st - h; n = np.sqrt(dx * dx + dy * dy + dz * dz); dx, dy, dz = dx / n, dy / n, dz / n
        else:
            dx, dy, dz = np.zeros(p.size), np.full(p.size, -s.ct), np.full(p.size, s.st)
        hid = np.zeros(p.size, bool); idx = np.arange(p.size); t = t0
        while idx.size:
            pp, qq, hh = p[idx] + t * dx[idx], q[idx] + t * dy[idx], h[idx] + t * dz[idx]
            occ = height(pp, qq) > hh + eps
            hid[idx[occ]] = True
            idx = idx[~occ & (hh < hmax)]
            t += dt
        return hid

def surface_point(cam, X, Y):
    # visible point (p, q) on the surface that projects to (X, Y): the highest crossing of the line of sight with the surface, found by scanning
    # the height downwards from the summit and bisecting
    g = lambda h: float(height(*cam.place(X, Y, h))) - h
    hs = np.linspace(H, 0.0, 401)
    k = next(i for i in range(1, len(hs)) if g(hs[i]) >= 0)
    lo, hi = hs[k], hs[k - 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 0: lo = mid
        else: hi = mid
    return np.array(cam.place(X, Y, lo))
def alpha(h, c):
    # opacity factor of the lines: 0 at the foothills (h <= t0 H), 1 above t1 H, smoothstep in between
    x = np.clip((np.asarray(h, float) / H - c["t0"]) / (c["t1"] - c["t0"]), 0, 1)
    return x * x * (3 - 2 * x)

def polylines(curves, cam, hmax, pad=6):
    # visible parts of curves on the surface ((N, 3) arrays): screen polylines inside the canvas, with the heights of their points
    if not curves: return []
    P = np.vstack(curves); lid = np.repeat(np.arange(len(curves)), [len(cv) for cv in curves])
    X, Y = cam.project(P[:, 0], P[:, 1], P[:, 2])
    ok = cam.front(P[:, 0], P[:, 1], P[:, 2]) & (X > IX0 - pad) & (X < IX1 + pad) & (Y > IY0 - pad) & (Y < IY1 + pad)
    i = np.where(ok)[0]
    ok[i] = ~cam.hidden(P[i, 0], P[i, 1], P[i, 2], hmax)
    idx = np.where(ok)[0]
    if idx.size == 0: return []
    brk = np.where((np.diff(idx) > 1) | (np.diff(lid[idx]) != 0))[0] + 1
    return [(np.column_stack([X[g], Y[g]]), P[g, 2]) for g in np.split(idx, brk) if len(g) >= 2]

def by_opacity(runs, c):
    # cut each polyline where its opacity, quantised to 1/bins, changes; {level: [polylines]} (level 0 is dropped)
    out = {}
    for xy, h in runs:
        k = np.rint(alpha(h, c) * c["bins"]).astype(int); i = 0
        for j in range(1, len(k) + 1):
            if j == len(k) or k[j] != k[i]:
                piece = xy[i:min(j + 1, len(k))]
                if k[i] > 0 and len(piece) >= 2: out.setdefault(int(k[i]), []).append(piece)
                i = j
    return out

def ground_box(cam, hmax, pad=40):
    # part of the ground that can appear inside the canvas
    PP, QQ = np.meshgrid(np.linspace(-3000, 3000, 601), np.linspace(-3000, 6000, 901))
    ok = np.zeros(PP.shape, bool)
    for h in (0.0, hmax):
        Hh = np.full(PP.shape, h); X, Y = cam.project(PP, QQ, Hh)
        ok |= cam.front(PP, QQ, Hh) & (X > IX0 - pad) & (X < IX1 + pad) & (Y > IY0 - pad) & (Y < IY1 + pad)
    return PP[ok].min() - 20, PP[ok].max() + 20, QQ[ok].min() - 20, QQ[ok].max() + 20

def mesh(c, cam, box, hmax):
    # lines of constant u and constant v of the mesh, which is turned by azim against the screen
    P0, P1, Q0, Q1 = box; ph = math.radians(c["azim"]); cph, sph = math.cos(ph), math.sin(ph)
    cs = [(P0, Q0), (P0, Q1), (P1, Q0), (P1, Q1)]
    U = [p * cph + q * sph for p, q in cs]; V = [-p * sph + q * cph for p, q in cs]
    d = c["mesh"]; curves = []
    for fam in (0, 1):
        A, B = (U, V) if fam == 0 else (V, U)
        b = np.arange(min(B), max(B), 2.0)
        for a in np.arange(math.floor(min(A) / d) * d, max(A) + d, d):
            u, v = (np.full(b.size, a), b) if fam == 0 else (b, np.full(b.size, a))
            p, q = u * cph - v * sph, u * sph + v * cph
            k = (p >= P0) & (p <= P1) & (q >= Q0) & (q <= Q1)
            if k.sum() >= 2: curves.append(np.column_stack([p[k], q[k], height(p[k], q[k])]))
    return polylines(curves, cam, hmax)

def contours(c, cam, box, hmax):
    P0, P1, Q0, Q1 = box
    ps, qs = np.arange(P0, P1 + 1, 2.5), np.arange(Q0, Q1 + 1, 2.5)
    gen = contourpy.contour_generator(ps, qs, height(*np.meshgrid(ps, qs)))
    curves = []
    for L in np.arange(c["dz"], hmax, c["dz"]):
        curves += [np.column_stack([ln, np.full(len(ln), L)]) for ln in gen.lines(L) if len(ln) >= 4]
    return polylines(curves, cam, hmax)

def trajectory(c, cam, hmax):
    # steepest ascent from the surface point under the screen point "start", with 0.5 px steps along the normalised gradient, until the steps
    # turn back; then the summit itself
    p, q = surface_point(cam, *map(float, c["start"].split(","))); pts = [(p, q)]; prev = None
    for _ in range(40000):
        gp, gq = grad(p, q); n = math.hypot(gp, gq)
        if n < 1e-9: break
        st = (gp / n, gq / n)
        if prev is not None and st[0] * prev[0] + st[1] * prev[1] < 0: break
        p, q = p + 0.5 * st[0], q + 0.5 * st[1]; pts.append((p, q)); prev = st
    top = minimize(lambda x: -float(height(x[0], x[1])), [p, q], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-9)).x
    pts.append(tuple(top))
    G = np.array(pts); Hp = height(G[:, 0], G[:, 1]) + 0.3
    X, Y = cam.project(G[:, 0], G[:, 1], Hp)
    nhid = int(cam.hidden(G[:, 0], G[:, 1], Hp, hmax).sum())
    # iterates: positions along the 3D path with geometrically shrinking steps (as in proposal 1)
    S = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(G[:, 0]) ** 2 + np.diff(G[:, 1]) ** 2 + np.diff(Hp) ** 2))]); tot = S[-1]
    s, d, its = 0.0, tot * (1 - 0.68), []
    while d > 4:
        s += d; d *= 0.68; j = min(int(np.searchsorted(S, s)), len(S) - 1); its.append((X[j], Y[j]))
    # clearance from the letters
    gap = min(float(np.min(np.hypot(np.maximum(np.maximum(x0 - X, 0), X - x1), np.maximum(np.maximum(y0 - Y, 0), Y - y1)))) for x0, y0, x1, y1 in TEXT)
    return np.column_stack([X, Y]), its, f"hidden {nhid}, clearance {gap:.1f}, start height {Hp[0] / H:.2f} H", tot

def fragments(over=None):
    global FEAT, H
    c = dict(CFG); c.update({k: type(CFG[k])(v) for k, v in (over or {}).items() if k in CFG})
    cam = Cam(c); H = hmax = c["hp"]
    FEAT = (*cam.place(c["px"], c["py"], H), c["sp"], c["sq"], H, c["rot"])
    box = ground_box(cam, hmax)
    lines = []
    for name, runs, w, o in (("mesh", mesh(c, cam, box, hmax), c["mw"], c["mo"]), ("contours", contours(c, cam, box, hmax), c["cw"], c["co"])):
        for lev, pls in sorted(by_opacity(runs, c).items()):
            d = " ".join(path_d(simplify(pl, 0.35)) for pl in pls)
            col = f' stroke="#{c["mc"]}"' if name == "mesh" else ""
            lines.append(f'    <path{col} stroke-width="{w}" stroke-opacity="{o * lev / c["bins"]:.3f}" d="{d}"/>\n')
    T, its, tinfo, tot = trajectory(c, cam, hmax)
    Ts = simplify(T, 0.25); a, b = T[0], T[-1]
    # colours: blue at the foot, yellow at the summit
    stops = [("0", "#306998"), (".3", "#5A9FD4"), (".65", "#F5B800"), ("1", "#FFD43B")]
    defs = (f'    <linearGradient id="unsim-hero-white-route" gradientUnits="userSpaceOnUse" x1="{f1(a[0])}" y1="{f1(a[1])}" x2="{f1(b[0])}" y2="{f1(b[1])}">'
            + "".join(f'<stop offset="{o}" stop-color="{col}"/>' for o, col in stops) + "</linearGradient>\n")
    land = ('  <!-- Loss landscape with a single peak: wireframe and contour lines seen in perspective, hidden lines removed; the lines fade out towards the foothills; unsim/script/gen_surface.py -->\n'
            f'  <g {TF} fill="none" stroke="#306998" stroke-linejoin="round">\n' + "".join(lines) + '  </g>\n')
    traj = ('  <!-- Steepest ascent to the summit; the white dots are iterates with geometrically shrinking steps -->\n'
            f'  <g {TF}>\n'
            '    <g fill="none" stroke-linecap="round" stroke-linejoin="round">\n'
            f'      <path stroke="#fff" stroke-width="7.2" d="{path_d(Ts)}"/>\n'
            f'      <path stroke="url(#unsim-hero-white-route)" stroke-width="4" d="{path_d(Ts)}"/>\n    </g>\n'
            '    <g fill="#fff">\n' + "".join(f'      <circle cx="{f1(x)}" cy="{f1(y)}" r="1.4"/>\n' for x, y in its) + '    </g>\n'
            f'    <circle cx="{f1(a[0])}" cy="{f1(a[1])}" r="4.2" fill="#fff" stroke="#306998" stroke-width="2.4"/>\n'
            f'    <circle cx="{f1(b[0])}" cy="{f1(b[1])}" r="6.2" fill="#F5B800" stroke="#fff" stroke-width="2.4"/>\n  </g>\n')
    info = f"box {[round(v) for v in box]} paths {len(lines)} chars {sum(len(l) for l in lines)} traj {f1(a[0])},{f1(a[1])} -> {f1(b[0])},{f1(b[1])} len {tot:.0f} {tinfo}"
    return dict(defs=defs, land=land, traj=traj, info=info)

if __name__ == "__main__":
    fr = fragments(dict(a.split("=", 1) for a in sys.argv[1:]))
    print("INFO", fr["info"]); print("DEFS"); print(fr["defs"], end=""); print("LAND"); print(fr["land"], end=""); print("TRAJ"); print(fr["traj"], end="")
