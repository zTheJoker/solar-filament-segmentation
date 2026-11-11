"""Paste-back: the final, pixel-disjoint masks of the kept candidates (final pipeline).

In (per frame): the decoder candidates and the fused map, the nz1 outlines of all candidates with their chosen
level (``postprocess.outline_nz1``), the keep decision, the FZ verifier score u_FZ and the refined probabilities
(``refiner.Refiner.refine``). Out: a compact label image, label k = k-th kept candidate in candidate order.

1. Sub-pixel shift. The annotators' drawings sit about 0.23 px above (and 0.05 px right of) the masks of a
   flip-symmetric pipeline (measured on matched training instances). A binary mask cannot move by a fraction of
   a pixel, but the map it is cut from can: with ``shift`` = (dy, dx), ``ndimage.shift(a, s)[i] = a[i - s]``,
   bilinear, edges repeated.
   * Re-drawn candidates: the refined probability is shifted before thresholding.
   * Other kept candidates: the same nz1 level is re-cut from the shifted fused map (``shifted_outlines``), in
     the same territories, so the outlines stay disjoint. With shift (0, 0) this returns the nz1 outlines bit
     for bit.
2. Re-drawn candidates: kept, u_FZ <= ``umax`` and a refined probability exists. Mask = shifted refined
   probability >= ``thr``, only the 8-connected parts that touch the candidate's (unshifted) nz1 outline, holes
   filled. An empty result keeps the shifted nz1 outline.
3. Overlaps: every contested pixel goes to the instance with the higher refined probability there (nz1 masks
   count as 255; ties: higher u_FZ, larger area, lower index); then each instance keeps only the parts still
   connected to its nz1 outline (the largest part if none is). One owner image, so the result is disjoint.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .postprocess import territories

EIGHT = np.ones((3, 3), bool)


def shift_map(a: np.ndarray, shift) -> np.ndarray:
    return ndi.shift(np.asarray(a, np.float32), tuple(float(v) for v in shift), order=1, mode='nearest')


def shifted_outlines(labels: np.ndarray, fg: np.ndarray, choice: np.ndarray, shift, levels, base_level=.4,
                     grow_px=16) -> np.ndarray:
    """nz1 outlines (the level ``choice[j]`` of each candidate) re-cut from the fused map moved by ``shift``.

    labels: decoder candidates (1..n); fg: the fused foreground map they were decoded from. The level-``base_level``
    mask changes only on its one-pixel boundary band (a pixel changes side where the two maps disagree about
    the threshold); grow and shrink levels are cut from the shifted map exactly as ``outline_nz1`` cuts them.
    """
    fgs = fg if not any(shift) else np.clip(shift_map(fg, shift), 0., 1.)
    terr = territories(labels, grow_px)
    n = int(labels.max())
    out = np.zeros(labels.shape, np.int32)
    for j, tb in enumerate(ndi.find_objects(terr, max_label=n), 1):
        y0, y1, x0, x1 = tb[0].start, tb[0].stop, tb[1].start, tb[1].stop
        base = labels[y0:y1, x0:x1] == j
        own = terr[y0:y1, x0:x1] == j
        pc, ps = fg[y0:y1, x0:x1], fgs[y0:y1, x0:x1]
        band = ndi.binary_dilation(base, EIGHT) & ~ndi.binary_erosion(base, EIGHT)
        add = band & ~base & own & (ps >= base_level) & (pc < base_level)
        rem = band & base & (ps < base_level) & (pc >= base_level)
        bs = (base | add) & ~rem
        if not bs.any():
            bs = base
        t = levels[int(choice[j-1])]
        if abs(t-base_level) < 1e-9:
            m = bs
        elif t > base_level:
            m = ndi.binary_fill_holes(bs & (ps >= t)) & bs
        else:
            cc, _ = ndi.label((ps >= t) & own, EIGHT)
            ids = np.unique(cc[bs])
            ids = ids[ids > 0]
            m = ndi.binary_fill_holes(np.isin(cc, ids) | bs) & own
        if m.sum() == 0:
            m = bs
        view = out[y0:y1, x0:x1]
        assert not (view[m] > 0).any(), 'shifted nz1 outlines must be disjoint'
        view[m] = j
    return out


def _instance(j, seeds, seeds_shifted, objects, objects_shifted, u_fz, refined, thr, fill, umax, shift):
    if j in refined and u_fz[j] <= umax:
        y0, y1, x0, x1, P = refined[j]
        seed = seeds[y0:y1, x0:x1] == j+1
        if shift is not None:
            P = shift_map(P, shift)
        M = P >= thr*255
        lab, _ = ndi.label(M, EIGHT)
        ids = np.unique(lab[seed & M])
        ids = ids[ids > 0]
        if len(ids):
            M = np.isin(lab, ids)
            if fill:
                M = ndi.binary_fill_holes(M)
            return {'j': j, 'box': (y0, y1, x0, x1), 'mask': M, 'score': P.astype(np.float32), 'refined': True}
    b, source = objects[j], seeds
    if seeds_shifted is not None and objects_shifted[j] is not None:
        b, source = objects_shifted[j], seeds_shifted
    mask = source[b] == j+1
    return {'j': j, 'box': (b[0].start, b[0].stop, b[1].start, b[1].stop), 'mask': mask,
            'score': np.where(mask, 255., 0.).astype(np.float32), 'refined': False}


def _intersection(a, b) -> int:
    ya, yb = max(a['box'][0], b['box'][0]), min(a['box'][1], b['box'][1])
    xa, xb = max(a['box'][2], b['box'][2]), min(a['box'][3], b['box'][3])
    if yb <= ya or xb <= xa:
        return 0
    ma = a['mask'][ya-a['box'][0]:yb-a['box'][0], xa-a['box'][2]:xb-a['box'][2]]
    mb = b['mask'][ya-b['box'][0]:yb-b['box'][0], xa-b['box'][2]:xb-b['box'][2]]
    return int((ma & mb).sum())


def paste_back(seeds: np.ndarray, keep: np.ndarray, u_fz: np.ndarray, refined: dict, seeds_shifted=None,
               thr=.60, fill=True, umax=.55, shift=None) -> tuple[np.ndarray, list[int], dict]:
    """Compact label image of the kept candidates, the candidate index of each label, and which were re-drawn."""
    keep = np.asarray(keep, bool)
    u = np.asarray(u_fz, np.float64)
    n = len(keep)
    assert len(u) == n
    objects = ndi.find_objects(seeds, max_label=n)
    objects_shifted = ndi.find_objects(seeds_shifted, max_label=n) if seeds_shifted is not None else None
    inst = [_instance(int(j), seeds, seeds_shifted, objects, objects_shifted, u, refined, thr, fill, umax, shift)
            for j in np.flatnonzero(keep)]
    for a in inst:
        a['area'] = int(a['mask'].sum())
    inst = [a for a in inst if a['area'] > 0]
    priority = lambda a: (-u[a['j']], -a['area'], a['j'])  # smaller tuple = higher priority
    overlapping = any(_intersection(inst[p], inst[q]) for p in range(len(inst)) for q in range(p+1, len(inst)))
    if overlapping:
        owner = np.zeros(seeds.shape, np.int32)
        best = np.full(seeds.shape, -1., np.float32)
        for a in sorted(inst, key=priority):
            y0, y1, x0, x1 = a['box']
            bv, ov = best[y0:y1, x0:x1], owner[y0:y1, x0:x1]
            update = a['mask'] & (a['score'] > bv)
            bv[update] = a['score'][update]
            ov[update] = a['j']+1
        for a in inst:  # keep only what is still connected to the nz1 outline
            y0, y1, x0, x1 = a['box']
            ov = owner[y0:y1, x0:x1]
            region = ov == a['j']+1
            if int(region.sum()) != a['area']:
                lab, k = ndi.label(region, EIGHT)
                if k > 1:
                    ids = np.unique(lab[region & (seeds[y0:y1, x0:x1] == a['j']+1)])
                    ids = ids[ids > 0]
                    if not len(ids):
                        ids = [int(np.argmax(np.bincount(lab.ravel())[1:]))+1]
                    drop = region & ~np.isin(lab, ids)
                    ov[drop] = 0
                    region = region & ~drop
            a['mask'], a['area'] = region, int(region.sum())
        inst = [a for a in inst if a['area'] > 0]
    inst.sort(key=lambda a: a['j'])
    labels = np.zeros(seeds.shape, np.int32)
    for i, a in enumerate(inst, 1):
        y0, y1, x0, x1 = a['box']
        labels[y0:y1, x0:x1][a['mask']] = i
    return labels, [a['j'] for a in inst], {a['j']: a['refined'] for a in inst}
