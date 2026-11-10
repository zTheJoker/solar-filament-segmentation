"""Submission encoding: one CSV row per instance, compressed COCO RLE of a 2048x2048 mask.

``filament_id`` is ``<image id>_<k>`` with k = 1..K; images without instances
simply have no rows. Instances are disjoint by construction and this is
re-verified on the written file (pairwise RLE IoU is exactly zero).
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects

from .data import NATIVE


def encode_label_image(labels: np.ndarray, *, roundtrip: bool = True) -> list[str]:
    """Compressed COCO RLE string per instance of a compact label image (column-major runs, as pycocotools)."""
    from pycocotools import mask as cm
    labels = np.asarray(labels)
    height, width = labels.shape
    encoded = []
    for label, box in enumerate(find_objects(labels), 1):
        if box is None:
            continue
        crop = labels[box] == label
        yy, xx = np.nonzero(crop)
        positions = np.sort((xx+box[1].start)*height+yy+box[0].start)
        gaps = np.diff(positions) > 1
        starts, ends = positions[np.r_[True, gaps]], positions[np.r_[gaps, True]]+1
        runs = np.empty(len(starts)*2+1, np.int64)
        runs[0] = starts[0]
        runs[1:-1:2] = ends-starts
        runs[2:-1:2] = starts[1:]-ends[:-1]
        runs[-1] = height*width-ends[-1]
        if runs[-1] == 0:
            runs = runs[:-1]  # pycocotools omits a terminal empty run
        rle = cm.frPyObjects(dict(size=[height, width], counts=runs.tolist()), height, width)
        if roundtrip:
            decoded = cm.decode(rle)
            assert int(decoded.sum()) == len(positions) and np.array_equal(crop, decoded[box]), 'RLE mismatch'
        encoded.append(rle['counts'].decode('ascii'))
    return encoded


def write_submission(rles_by_image: dict[str, list[str]], path: str | Path) -> str:
    """Write the CSV (images in sorted order) and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        for key in sorted(rles_by_image):
            for k, rle in enumerate(rles_by_image[key], 1):
                writer.writerow({'filament_id': f'{key}_{k}', 'segmentation_rle': rle})
    return sha256(path)


def read_submission(path: str | Path) -> dict[str, list[str]]:
    """CSV -> {image id: [RLE strings in row order]}."""
    rows = defaultdict(list)
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('CSV columns must be filament_id,segmentation_rle')
        for row in reader:
            rows[row['filament_id'].split('_', 1)[0]].append(row['segmentation_rle'])
    return dict(rows)


def validate_submission(path: str | Path, image_ids) -> dict:
    """Check ids, mask size, non-empty masks, unique ids and pairwise disjointness per image."""
    from pycocotools import mask as cm
    known = set(image_ids)
    rows = read_submission(path)
    with open(path, newline='') as handle:
        ids = [r['filament_id'] for r in csv.DictReader(handle)]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate filament_id')
    for image, rles in rows.items():
        if image not in known:
            raise ValueError(f'Unknown image id {image}')
        objs = [{'size': [NATIVE, NATIVE], 'counts': r.encode('ascii')} for r in rles]
        if any(a == 0 for a in cm.area(objs)):
            raise ValueError(f'Empty mask in {image}')
        iou = np.asarray(cm.iou(objs, objs, [0]*len(objs)), float)
        np.fill_diagonal(iou, 0)
        if (iou > 0).any():
            raise ValueError(f'Overlapping or duplicate instances in {image}')
    return {'rows': len(ids), 'images_with_instances': len(rows), 'images_without_instances': len(known-set(rows)),
            'disjoint': True}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
