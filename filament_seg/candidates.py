"""Candidate instances: connected components of the thresholded ensemble map.

Candidates = 8-connected components of ``foreground >= fg_threshold`` with at
least ``min_area`` native pixels; enclosed holes are filled per instance.
Labels are compact (1..n) in raster order of the components. Every candidate
is later kept or rejected by the verifier and re-outlined by post-processing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)


@dataclass(frozen=True)
class CandidateSettings:
    fg_threshold: float = .4
    min_area: int = 150
    fill_holes: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> 'CandidateSettings':
        c = cfg['candidates']
        return cls(float(c['fg_threshold']), int(c['min_area']), bool(c['fill_holes']))


def fill_instance_holes(labels: np.ndarray) -> np.ndarray:
    """Fill background enclosed by an instance; never take another instance's pixels."""
    result = np.asarray(labels).copy()
    for label, box in enumerate(ndi.find_objects(result), 1):
        if box is None:
            continue
        view = result[box]
        filled = ndi.binary_fill_holes(view == label)
        view[filled & (view == 0)] = label
    return result


def candidate_labels(foreground: np.ndarray, settings: CandidateSettings) -> np.ndarray:
    """Compact int32 label image of candidates decoded from a native foreground probability map."""
    fg = np.asarray(foreground)
    if fg.ndim != 2 or not np.isfinite(fg).all() or ((fg < 0) | (fg > 1)).any():
        raise ValueError('Expected a finite 2-D probability map in [0, 1].')
    components, _ = ndi.label(fg >= settings.fg_threshold, EIGHT)
    counts = np.bincount(components.ravel())
    keep = counts >= settings.min_area
    keep[0] = False
    labels = ndi.label(keep[components], structure=EIGHT)[0]
    if settings.fill_holes:
        labels = fill_instance_holes(labels)
    return labels.astype(np.int32)
