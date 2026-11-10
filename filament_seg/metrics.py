"""Evaluation exactly as in the host's Self_Evaluation_Notebook (v6).

For every annotator-image record (one annotator's labels of one frame) the
GT x prediction IoU and Dice matrices are computed; the same predictions are
reused for every annotator of a frame. Then, pooled over all records:

* TP = every (GT, prediction) pair with IoU strictly > 0.5,
  FP = prediction columns with no such pair, FN = GT rows with no such pair,
  PQ = sum(IoU of TP pairs) / (TP + FP/2 + FN/2)   (no averaging per image)
* IoU / Dice distributions over all pairs with a positive score
* one-to-many / many-to-one relations: for every GT, the number of
  predictions overlapping it (IoU > 0), and vice versa.
Also returned (not part of the host notebook): every GT's area and whether it
was hit (IoU > 0.5), for recall by filament size.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .data import NATIVE


def overlap_matrices(gt_rles: list[dict], pred_rles: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(n_gt, n_pred) IoU and Dice matrices of COCO RLE masks."""
    from pycocotools import mask as cm
    if not gt_rles or not pred_rles:
        z = np.zeros((len(gt_rles), len(pred_rles)))
        return z, z.copy()
    iou = np.asarray(cm.iou(list(pred_rles), list(gt_rles), [0]*len(gt_rles)), np.float64).T
    return iou, 2*iou/(1+iou)


def pq_counts(iou: np.ndarray, threshold: float = .5) -> dict:
    """Host matching for one record: sum of TP IoUs, TP, FP, FN."""
    hit = iou > threshold
    return {'sum_iou': float(iou[hit].sum()), 'tp': int(hit.sum()),
            'fp': int((~hit.any(axis=0)).sum()) if iou.shape[0] else iou.shape[1],
            'fn': int((~hit.any(axis=1)).sum()) if iou.shape[1] else iou.shape[0]}


def pq_from_counts(c: dict) -> dict:
    den = c['tp']+.5*c['fp']+.5*c['fn']
    return {**c, 'pq': c['sum_iou']/den if den else 0., 'sq': c['sum_iou']/c['tp'] if c['tp'] else 0.,
            'rq': c['tp']/den if den else 0.}


def evaluate(ground_truth: dict[str, list], predictions: dict[str, list[str]]) -> dict:
    """Score predictions against every annotator record.

    ground_truth: {frame id: [(annotator_image_id, [COCO RLE dict, ...]), ...]} (data.load_ground_truth)
    predictions:  {frame id: [compressed RLE string, ...]}
    """
    from pycocotools import mask as cm
    total = Counter()
    records, ious, dices, gt_degree, pred_degree, gt_area, gt_hit = [], [], [], [], [], [], []
    for key in sorted(ground_truth):
        pred = [{'size': [NATIVE, NATIVE], 'counts': r.encode('ascii')} for r in predictions.get(key, [])]
        for annotator_image, gt in ground_truth[key]:
            iou, dice = overlap_matrices(gt, pred)
            c = pq_counts(iou)
            total.update(c)
            records.append({'annotator_image': annotator_image, 'image': key, 'n_gt': len(gt), 'n_pred': len(pred),
                            **pq_from_counts(c)})
            ious.extend(iou[iou > 0].tolist())
            dices.extend(dice[dice > 0].tolist())
            gt_degree.extend((iou > 0).sum(1).tolist())
            pred_degree.extend((iou > 0).sum(0).tolist())
            gt_area.extend(cm.area(gt).tolist() if gt else [])
            gt_hit.extend((iou > .5).any(1).tolist())
    return {'overall': pq_from_counts({k: total[k] for k in ['sum_iou', 'tp', 'fp', 'fn']}),
            'records': records, 'positive_ious': np.asarray(ious), 'positive_dices': np.asarray(dices),
            'gt_degree': Counter(gt_degree), 'pred_degree': Counter(pred_degree),
            'gt_area': np.asarray(gt_area), 'gt_hit': np.asarray(gt_hit, bool)}
