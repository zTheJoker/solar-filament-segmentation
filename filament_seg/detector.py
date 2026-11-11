"""Detector votes: instance-segmentation detectors as six agreement columns per candidate.

Twenty YOLO26m-seg detectors (Ultralytics; four training seeds x the five calendar-month folds, all trained on
the 707 labelled frames) run on the native frame. Their instances are never output; they only tell the keep
model how a different kind of model sees each candidate.

Detector input: the native 8-bit grayscale frame replicated to three channels. One forward pass per model at
``imgsz`` 1536 with confidence >= 0.05, at most 300 instances, masks at native resolution (``retina_masks``).
Instances are sorted by descending confidence (stable).

Columns, computed per model against the candidate's nz1 outline (the outline step's label image, all
candidates, label j+1 = candidate j), then averaged:
  d_iou     best mask IoU with any instance
  d_conf    that instance's confidence if d_iou >= 0.1, else 0
  d_conf50  highest confidence among instances with IoU > 0.5
  d_cover   intersection with the best instance / candidate area
  d_incl    intersection with the best instance / that instance's area
  d_n       number of instances with confidence >= 0.1 that overlap by >= 20% of the smaller area
Averaging follows the submitted run: the mean over the five fold models of each seed (float64, stored as
float32), then the float32 mean over the seeds.

The Ultralytics library is AGPL-3.0. Telemetry is switched off before it is imported (``YOLO_OFFLINE``).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

COLUMNS = ['d_iou', 'd_conf', 'd_conf50', 'd_cover', 'd_incl', 'd_n']


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_detector(name: str, weights_dir: str | Path, manifest: dict):
    """An Ultralytics YOLO model from ``weights/<file>``, after checking the file's SHA-256 against the manifest."""
    os.environ.setdefault('YOLO_OFFLINE', '1')
    os.environ.setdefault('YOLO_VERBOSE', 'False')
    from ultralytics import YOLO
    entry = manifest[name]
    path = Path(weights_dir)/entry['files'][0]
    if file_sha256(path) != entry['file_sha256']:
        raise ValueError(f'{name}: {path.name} does not match the manifest hash')
    return YOLO(str(path), task='segment')


def predict_instances(model, gray: np.ndarray, imgsz=1536, conf=.05, iou=.7, max_det=300, device=None) -> dict:
    """One model on one frame -> {'conf': float32 [J], 'rles': [COCO RLE], 'area': int64 [J]}, by descending confidence."""
    from pycocotools import mask as cm
    image = np.ascontiguousarray(np.repeat(np.asarray(gray, np.uint8)[:, :, None], 3, 2))
    r = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, retina_masks=True, verbose=False,
                      half=False, device=device)[0]
    if r.masks is None or len(r.boxes) == 0:
        return {'conf': np.zeros(0, np.float32), 'rles': [], 'area': np.zeros(0, np.int64)}
    scores = r.boxes.conf.float().cpu().numpy()
    masks = r.masks.data.bool()
    assert tuple(masks.shape[1:]) == gray.shape, masks.shape
    rles = cm.encode(np.asfortranarray(masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)))
    order = np.argsort(-scores, kind='stable')
    return {'conf': scores[order].astype(np.float32), 'rles': [rles[j] for j in order],
            'area': np.asarray([int(cm.area(rles[j])) for j in order], np.int64)}


def intersections(seeds: np.ndarray, n: int, inst: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel intersections [n candidates, J instances], candidate areas [n], instance areas [J]."""
    from pycocotools import mask as cm
    area_c = np.bincount(seeds.ravel(), minlength=n+1)[1:n+1].astype(np.float64)
    area_j = np.asarray(inst['area'], np.float64)
    inter = np.zeros((n, len(inst['conf'])))
    for j, r in enumerate(inst['rles']):
        if area_j[j] == 0 or n == 0:
            continue
        x, y, w, h = [int(v) for v in cm.toBbox(r)]
        m = cm.decode(r)[y:y+h+1, x:x+w+1].astype(bool)
        inter[:, j] = np.bincount(seeds[y:y+h+1, x:x+w+1][m], minlength=n+1)[1:n+1]
    return inter, area_c, area_j


def columns(inter: np.ndarray, area_c: np.ndarray, area_j: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """The six columns [n, 6] of one model (instances sorted by descending confidence; ties go to the first)."""
    n, J = inter.shape
    out = np.zeros((n, 6))
    if n == 0 or J == 0:
        return out
    union = area_c[:, None]+area_j[None, :]-inter
    iou = np.where(union > 0, inter/np.maximum(union, 1), 0.)
    jb = np.argmax(iou, 1)
    bi = iou[np.arange(n), jb]
    out[:, 0] = bi
    out[:, 1] = np.where(bi >= .1, conf[jb], 0.)
    out[:, 2] = np.where(iou > .5, conf[None, :], 0.).max(1)
    ib = inter[np.arange(n), jb]
    out[:, 3] = np.where(bi > 0, ib/np.maximum(area_c, 1), 0.)
    out[:, 4] = np.where(bi > 0, ib/np.maximum(area_j[jb], 1), 0.)
    small = np.minimum(area_c[:, None], area_j[None, :])
    out[:, 5] = ((inter >= .2*small) & (inter > 0) & (conf[None, :] >= .1)).sum(1)
    return out


def model_columns(seeds: np.ndarray, n: int, inst: dict) -> np.ndarray:
    return columns(*intersections(seeds, n, inst), np.asarray(inst['conf'], np.float64))


def average_columns(per_model: list[np.ndarray], groups: list[int]) -> np.ndarray:
    """float32 [n, 6]: mean over the models of each group (seed), then the float32 mean over the groups."""
    per_group, start = [], 0
    for size in groups:
        per_group.append(np.mean(per_model[start:start+size], 0).astype(np.float32))
        start += size
    assert start == len(per_model)
    return per_group[0] if len(per_group) == 1 else np.mean(per_group, 0).astype(np.float32)


class DetectorVoter:
    """The detector ensemble of a config; ``votes(gray, seeds, n)`` returns the averaged columns [n, 6]."""

    def __init__(self, cfg: dict, weights_dir: str | Path, manifest: dict, names=None, device=None):
        d = cfg['inference']['detector']
        self.names = list(names if names is not None else cfg['weights']['detectors'])
        self.groups = [int(g) for g in d.get('groups', [len(self.names)])]
        assert sum(self.groups) == len(self.names), 'detector groups must cover the detector list'
        self.settings = {k: d[k] for k in ('imgsz', 'conf', 'iou', 'max_det')}
        self.device = device
        self.models = [load_detector(n, weights_dir, manifest) for n in self.names]

    def instances(self, gray: np.ndarray) -> list[dict]:
        dev = None if self.device is None else (0 if self.device.type == 'cuda' else 'cpu')
        return [predict_instances(m, gray, device=dev, **self.settings) for m in self.models]

    def votes(self, gray: np.ndarray, seeds: np.ndarray, n: int, instances: list[dict] | None = None) -> np.ndarray:
        if n == 0:
            return np.zeros((0, 6), np.float32)
        instances = self.instances(gray) if instances is None else instances
        return average_columns([model_columns(seeds, n, inst) for inst in instances], self.groups)
