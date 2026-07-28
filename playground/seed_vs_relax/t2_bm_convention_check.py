"""Verify the bm[lo]/bm[hi] index convention before trusting the 85% result.

If the convention were flipped, essentially EVERY pair would read as out-of-bounds. So: measure the
fraction of ALL atom pairs whose realised distance lies inside the matrix bounds. A correct
convention gives a high fraction (ETKDG largely honours the matrix); a flipped one gives ~0.
"""

import logging, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import rxembed as rx
from rxembed.constraints import metal as _metal
from rxembed.embed import bounds as _bounds
from t2_seed_vs_relax import NCONF, SEED, TMQM, crystal

logging.disable(logging.WARNING)

for nm in ["ZOPNOH", "NUKHEG", "DEYMIE", "HIBPAK"]:
    sym, cry, q = crystal(os.path.join(TMQM, f"{nm}.xyz"))
    ens = rx.embed(os.path.join(TMQM, f"{nm}.xyz"), charge=q, n=NCONF, seed=SEED)
    work = _metal.materialise_phantoms(ens.mol, ens.cons.haptic)
    bm, tol = _bounds._feasible_bounds(work, ens.cons)
    n = ens.mol.GetNumAtoms()
    cid = list(ens.ids)[1]
    p = ens.mol.GetConformer(cid).GetPositions()
    iu = np.triu_indices(n, 1)
    d = np.linalg.norm(p[iu[0]] - p[iu[1]], axis=1)
    hi = np.asarray(bm)[iu]  # bm[a][b], a<b  -> UPPER
    lo = np.asarray(bm).T[iu]  # bm[b][a]        -> LOWER
    assert (lo <= hi + 1e-9).all(), f"{nm}: convention flipped (lower > upper)"
    ok = np.mean((d >= lo - 1e-6) & (d <= hi + 1e-6))
    # the crystal, for reference
    dc = np.linalg.norm(cry[iu[0]] - cry[iu[1]], axis=1)
    okc = np.mean((dc >= lo - 1e-6) & (dc <= hi + 1e-6))
    print(
        f"{nm}: smooth_tol={tol:.3f}  lower<=upper OK  |  ETKDG seed pairs inside bm: {100 * ok:.1f}%  |  crystal pairs inside bm: {100 * okc:.1f}%"
    )
