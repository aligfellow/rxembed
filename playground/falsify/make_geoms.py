"""Generate haptic reference geometries (the MUST-KEEP half of the falsification set)."""

import numpy as np


def write(name, sy, xyz, note):
    with open(f"{name}.xyz", "w") as f:
        f.write(f"{len(sy)}\n{note}\n")
        for s, (x, y, z) in zip(sy, xyz):
            f.write(f"{s:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    print(f"  {name}.xyz  ({len(sy)} atoms)  {note}")


def ring(n, r, z, phase=0.0):
    a = np.arange(n) * 2 * np.pi / n + phase
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], axis=1)


# --- ferrocene, D5d staggered: C-C 1.43 -> r 1.216; Fe-C 2.06 -> h 1.663
r, h = 1.2159, 1.6628
top, bot = ring(5, r, h), ring(5, r, -h, np.pi / 5)
hr, hh = r + 1.08, h + 0.01  # H's radially out, in-plane
htop, hbot = ring(5, hr, hh), ring(5, hr, -hh, np.pi / 5)
sy = ["Fe"] + ["C"] * 10 + ["H"] * 10
write("ferrocene", sy, np.vstack([[[0, 0, 0]], top, bot, htop, hbot]), "ferrocene D5d | eta5+eta5 | MUST KEEP all 10 C")

# --- Zeise's anion [PtCl3(C2H4)]-: eta2 C=C perpendicular to the PtCl3 plane
cc, ptc = 1.375, 2.128
hz = np.sqrt(max(ptc**2 - (cc / 2) ** 2, 0.01))  # Pt->C=C midpoint
sy = ["Pt", "Cl", "Cl", "Cl", "C", "C", "H", "H", "H", "H"]
xyz = [[0, 0, 0], [2.30, 0, 0], [-2.30, 0, 0], [0, -2.32, 0], [0, hz, cc / 2], [0, hz, -cc / 2]]
for dz, dx in ((cc / 2, 1), (cc / 2, -1), (-cc / 2, 1), (-cc / 2, -1)):
    xyz.append([dx * 0.93, hz + 0.55, dz * 1.30])
write("zeise", sy, np.array(xyz), "Zeise anion (charge -1) | eta2 | MUST KEEP C4,C5")

# --- bis(eta3-allyl)Ni  -- STARTING GEOMETRY, needs optimisation
cc, ang = 1.40, np.deg2rad(120.0)
c2 = np.array([0.0, 0.0, 0.0])
c1 = np.array([-cc * np.sin(ang / 2), cc * np.cos(ang / 2), 0.0])
c3 = np.array([cc * np.sin(ang / 2), cc * np.cos(ang / 2), 0.0])
allyl = np.vstack([c1, c2, c3])
cen = allyl.mean(axis=0)
M = cen + np.array([0.0, 0.30, 1.93])  # leans toward the open C1-C3 edge


def hyd(a):  # 2 H on each terminal, 1 on central
    out = []
    for c, d in ((c1, np.array([-1.0, 0.35, 0.0])), (c3, np.array([1.0, 0.35, 0.0]))):
        u = d / np.linalg.norm(d)
        out += [c + 1.08 * u + np.array([0, 0, 0.50]), c + 1.08 * u + np.array([0, 0, -0.50])]
    out.append(c2 + np.array([0.0, -1.08, 0.0]))
    return np.array(out)


H1 = hyd(allyl)
flip = np.diag([1.0, -1.0, -1.0])  # second allyl below, rotated
A2, H2, M2 = allyl @ flip, H1 @ flip, M @ flip
shift = M - M2
sy = ["Ni"] + ["C"] * 3 + ["H"] * 5 + ["C"] * 3 + ["H"] * 5
write(
    "bisallyl_ni_UNOPT",
    sy,
    np.vstack([[M], allyl, H1, A2 + shift, H2 + shift]),
    "bis(eta3-allyl)Ni | STARTING GEOM - NEEDS OPT | MUST KEEP all 6 C (esp. central C2,C10)",
)
