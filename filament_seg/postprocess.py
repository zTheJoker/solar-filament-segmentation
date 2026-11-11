"""Post-processing: an ordered, configurable list of steps on the candidate set.

State passed between steps (``InstanceState``):
  labels  int32 label image, candidate j has id j+1 (all candidates, kept or not)
  keep    bool per candidate (the verifier decision)
  fg      native foreground map that produced the candidates
  gray    native 8-bit image

Steps of the FZ and lite pipelines (``configs/fz_nz1.yaml``, ``configs/f_lite.yaml``):

1. ``outline_nz1`` - expected-IoU outline. The ensemble map approximates the
   annotator-mean probability that a pixel belongs to the filament, so for
   each candidate we pick the nested super-level set S of the map that
   maximises the expected IoU with an annotator mask,
       E[IoU](S) ~= sum_S q / (|S| + sum_{band minus S} q),   q = p ** gamma
   (Nowozin 2014), among thresholds 0.25 ... 0.70. Lower thresholds grow the
   candidate inside its own Voronoi territory (at most ``grow_px`` away), higher
   thresholds shrink it inside itself, so instances stay disjoint whatever
   level each one chooses. The sums are evaluated on 20-bin histograms of p.
2. ``drop_rejected`` - remove candidates the verifier rejected and renumber
   the rest 1..K in candidate order.

The final pipeline (``configs/final.yaml``) runs ``outline_nz1`` on all candidates before the keep decision (the
detector votes are measured on these outlines) and then replaces ``drop_rejected`` by the refiner paste-back
(``pasteback.py``), which also drops the rejected candidates.

To add a step (e.g. a fragment merge), write ``def step(state, **params) ->
InstanceState``, register it in ``STEPS`` and list it in the config.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(2, 1)
EIGHT = np.ones((3, 3), bool)


@dataclass
class InstanceState:
    labels: np.ndarray
    keep: np.ndarray
    fg: np.ndarray
    gray: np.ndarray
    choice: np.ndarray | None = None   # nz1 level index per candidate (set by outline_nz1)


def territories(labels: np.ndarray, grow_px: int) -> np.ndarray:
    """Voronoi territory of every candidate (nearest candidate pixel), capped at ``grow_px``."""
    d, idx = ndi.distance_transform_edt(labels == 0, return_indices=True)
    t = labels[idx[0], idx[1]]
    t[d > grow_px] = 0
    return t


def _level_masks(base, own, p, levels, base_level):
    """Mask per threshold level for one candidate (None when empty); all inside its territory ``own``."""
    masks = []
    for t in levels:
        if abs(t-base_level) < 1e-9:
            m = base
        elif t > base_level:  # shrink: keep the confident part (holes filled inside the candidate)
            m = ndi.binary_fill_holes(base & (p >= t)) & base
        else:  # grow: components of p >= t in the territory that touch the candidate
            cc, _ = ndi.label((p >= t) & own, EIGHT)
            ids = np.unique(cc[base])
            ids = ids[ids > 0]
            m = ndi.binary_fill_holes(np.isin(cc, ids) | base) & own
        masks.append(m if m.sum() else None)
    return masks


def outline_nz1(state: InstanceState, levels, base_level=.4, grow_px=16, band_px=4, bins=20, gamma=1.) -> InstanceState:
    """Replace every candidate by the threshold level with the highest expected IoU (see module docstring)."""
    labels, p = state.labels, state.fg
    n = int(labels.max())
    if n == 0:
        return state
    terr = territories(labels, grow_px)
    boxes = ndi.find_objects(terr, max_label=n)
    hist_in = np.zeros((n, len(levels), bins))   # p histogram inside each level's mask
    hist_band = np.zeros((n, bins))              # p histogram of the band around the largest mask
    valid = np.zeros((n, len(levels)), bool)
    all_masks = []
    for j in range(1, n+1):
        box = boxes[j-1]
        base = labels[box] == j
        own = terr[box] == j
        pc = p[box]
        masks = _level_masks(base, own, pc, levels, base_level)
        grown = masks[0] if masks[0] is not None else base
        band = ndi.binary_dilation(grown | base, CROSS, iterations=band_px) & own
        hist_band[j-1] = np.histogram(pc[band], bins, (0, 1))[0]
        for li, m in enumerate(masks):
            hist_in[j-1, li] = np.histogram(pc[m if m is not None else base], bins, (0, 1))[0]
        valid[j-1] = [m is not None for m in masks]
        all_masks.append((box, masks))
    q = ((np.arange(bins)+.5)/bins)**gamma
    inside_q = hist_in@q
    expected_iou = inside_q/np.maximum(hist_in.sum(-1)+np.maximum((hist_band@q)[:, None]-inside_q, 0), 1e-9)
    choice = np.argmax(np.where(valid, expected_iou, -1), 1)  # ties keep the lowest level
    out = np.zeros_like(labels)
    for j, ((box, masks), c) in enumerate(zip(all_masks, choice), 1):
        view = out[box]
        assert not (view[masks[c]] > 0).any(), 'outlines must stay disjoint'
        view[masks[c]] = j
    return replace(state, labels=out, choice=choice)


def drop_rejected(state: InstanceState) -> InstanceState:
    """Remove rejected candidates and renumber the kept ones 1..K in candidate order."""
    keep = np.asarray(state.keep, bool)
    if len(keep) != int(state.labels.max()) and state.labels.max() > 0:
        raise ValueError('One keep decision is required per candidate id.')
    mapping = np.zeros(len(keep)+1, np.int32)
    mapping[1:][keep] = np.arange(1, int(keep.sum())+1)
    return replace(state, labels=mapping[state.labels], keep=np.ones(int(keep.sum()), bool))


STEPS = {'outline_nz1': outline_nz1, 'drop_rejected': drop_rejected}


def apply_steps(state: InstanceState, steps: list[dict]) -> InstanceState:
    """Apply the configured steps in order (no check of the result)."""
    for spec in steps:
        params = {k: v for k, v in spec.items() if k != 'name'}
        state = STEPS[spec['name']](state, **params)
    return state


def run_steps(state: InstanceState, steps: list[dict]) -> InstanceState:
    """Apply the configured steps in order; the result must be a compact label image of kept instances."""
    state = apply_steps(state, steps)
    if not state.keep.all() or int(state.labels.max()) != len(state.keep):
        raise ValueError('Post-processing must end with only kept, compactly numbered instances (add drop_rejected).')
    return state
