"""Fast checks of the data-free parts (run: python -m pytest tests  or  python tests/test_pipeline_parts.py)."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg.candidates import CandidateSettings, candidate_labels  # noqa: E402
from filament_seg.metrics import evaluate, pq_counts  # noqa: E402
from filament_seg.postprocess import InstanceState, run_steps  # noqa: E402
from filament_seg.submission import encode_label_image  # noqa: E402

STEPS = [{'name': 'outline_nz1', 'levels': [.25, .3, .35, .4, .45, .5, .55, .6, .65, .7], 'base_level': .4,
          'grow_px': 16, 'band_px': 4, 'bins': 20, 'gamma': 1.}, {'name': 'drop_rejected'}]


def synthetic_map():
    """Three blurred bars on a 2048^2 grid: two close together, one far away."""
    from scipy import ndimage as ndi
    p = np.zeros((2048, 2048), np.float32)
    p[500:520, 400:700] = 1
    p[530:545, 400:700] = 1
    p[1500:1520, 1200:1400] = .6
    return np.clip(ndi.gaussian_filter(p, 4)*1.6, 0, 1)


def test_rle_roundtrip_and_pq():
    from pycocotools import mask as cm
    labels = candidate_labels(synthetic_map(), CandidateSettings())
    rles = encode_label_image(labels)
    for k, r in enumerate(rles, 1):
        decoded = cm.decode({'size': [2048, 2048], 'counts': r.encode()}).astype(bool)
        assert np.array_equal(decoded, labels == k)
    gt = {'f': [('a', [{'size': [2048, 2048], 'counts': r.encode()} for r in rles])]}
    assert abs(evaluate(gt, {'f': rles})['overall']['pq']-1) < 1e-12
    c = pq_counts(np.array([[.6, .0], [.51, .2]]))  # host rule: every pair > .5 counts
    assert (c['tp'], c['fp'], c['fn']) == (2, 1, 0)


def test_postprocess_disjoint_and_compact():
    p = synthetic_map()
    labels = candidate_labels(p, CandidateSettings())
    keep = np.ones(int(labels.max()), bool)
    keep[-1] = False
    state = run_steps(InstanceState(labels, keep, p, np.zeros_like(labels, np.uint8)), STEPS)
    assert int(state.labels.max()) == keep.sum() and state.keep.all()
    rles = encode_label_image(state.labels)
    from pycocotools import mask as cm
    objs = [{'size': [2048, 2048], 'counts': r.encode()} for r in rles]
    iou = np.asarray(cm.iou(objs, objs, [0]*len(objs)))
    assert np.allclose(iou, np.eye(len(objs)))


def test_folds_file_is_reproducible():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'train'))
    from common import make_folds
    stored = json.loads((Path(__file__).resolve().parents[1]/'configs'/'folds_full.json').read_text())
    remade = make_folds(stored['train_sources'])
    assert [f['heldout_sources'] for f in remade['folds']] == [f['heldout_sources'] for f in stored['folds']]
    assert [f['train_sources'] for f in remade['folds']] == [f['train_sources'] for f in stored['folds']]
    assert [f['inner_validation_sources'] for f in remade['folds']] == [f['inner_validation_sources'] for f in stored['folds']]


def test_fusion_matches_stored_arithmetic():
    from filament_seg.inference import fuse, specialist_mean
    rng = np.random.default_rng(0)
    base = rng.random((2, 64, 64)).astype(np.float16).astype(np.float32)
    q8 = [rng.integers(0, 256, (64, 64), dtype=np.uint8) for _ in range(5)]
    q = specialist_mean(q8)
    assert np.allclose(q, np.mean(q8, 0)/255, atol=1e-6)
    fused = fuse(base, q, .5)
    assert np.array_equal(fused[1], base[1])
    assert np.array_equal(fused[0], ((base[0]+q)/2).astype(np.float16).astype(np.float32))


def test_shifted_outlines_identity_at_zero_shift():
    from filament_seg.pasteback import shifted_outlines
    p = synthetic_map()
    labels = candidate_labels(p, CandidateSettings())
    keep = np.ones(int(labels.max()), bool)
    state = run_steps(InstanceState(labels, keep, p, np.zeros_like(labels, np.uint8)), STEPS[:1]+[{'name': 'drop_rejected'}])
    from filament_seg.postprocess import apply_steps
    seeds = apply_steps(InstanceState(labels, keep, p, np.zeros_like(labels, np.uint8)), STEPS[:1])
    assert np.array_equal(state.labels, seeds.labels)  # all kept: drop_rejected changes nothing
    same = shifted_outlines(labels, p, seeds.choice, (0., 0.), STEPS[0]['levels'])
    assert np.array_equal(same, seeds.labels)
    moved = shifted_outlines(labels, p, seeds.choice, (-.23, .05), STEPS[0]['levels'])
    assert int(moved.max()) == int(labels.max()) and 0 < int((moved != seeds.labels).sum()) < .2*int((seeds.labels > 0).sum())


def test_pasteback_disjoint_compact_and_connected():
    from filament_seg.pasteback import paste_back
    p = synthetic_map()
    labels = candidate_labels(p, CandidateSettings())
    n = int(labels.max())
    keep = np.ones(n, bool)
    u = np.full(n, .3)
    # a refined probability for candidate 0 that grows into candidate 1 and has a detached blob
    ys, xs = np.nonzero(labels == 1)
    y0, y1, x0, x1 = ys.min()-20, ys.max()+40, xs.min()-20, xs.max()+20
    P = np.zeros((y1-y0, x1-x0), np.uint8)
    P[labels[y0:y1, x0:x1] == 1] = 220
    P[labels[y0:y1, x0:x1] == 2] = 200
    P[-5:, :5] = 255
    out, ids, redrawn = paste_back(labels, keep, u, {0: (y0, y1, x0, x1, P)}, None, thr=.6, fill=True, umax=.55, shift=None)
    assert ids == list(range(n)) and redrawn[0] and not redrawn[1]
    assert int(out.max()) == n and not out[y1-5:y1, x0:x0+5].any()
    # candidate 1 keeps the pixels where its own score (255 for an unrefined mask) beats the refined 200
    assert np.array_equal(out == 2, labels == 2)
    drop = keep.copy(); drop[0] = False
    out2, ids2, _ = paste_back(labels, drop, u, {0: (y0, y1, x0, x1, P)}, None)
    assert ids2 == list(range(1, n)) and int(out2.max()) == n-1


def test_detector_columns_and_averaging():
    from filament_seg.detector import average_columns, columns
    inter = np.array([[60., 0.], [0., 10.]])
    area_c, area_j, conf = np.array([100., 50.]), np.array([80., 40.]), np.array([.9, .3])
    c = columns(inter, area_c, area_j, conf)
    assert np.allclose(c[0], [60/120, .9, 0, .6, .75, 1])  # IoU exactly .5 is not > .5
    assert np.allclose(c[1], [10/80, .3, 0, .2, .25, 1])
    a, b, d = np.ones((2, 6)), np.zeros((2, 6)), np.full((2, 6), .5)
    assert np.allclose(average_columns([a, b, d, d], [2, 2]), .5)
    assert average_columns([a, b], [2]).dtype == np.float32


def test_refiner_crop_geometry():
    from filament_seg.refiner import BUCKETS, build_crops, crop_roi, unwarp, warp
    p = synthetic_map()
    labels = candidate_labels(p, CandidateSettings())
    n = int(labels.max())
    recs = build_crops(np.zeros(labels.shape, np.float32), labels, np.array([.5]*n), [p]*5, .10)
    assert [r['cand'] for r in recs] == list(range(n))
    for r in recs:
        (h, w), (my, mx) = BUCKETS[r['bucket']]['inner'], BUCKETS[r['bucket']]['margin']
        assert r['x'].shape == (3, h+2*my, w+2*mx) and r['pf'].shape == (5, h+2*my, w+2*mx)
        # the seed channel back-projected onto the frame overlaps the candidate
        (y0, y1, x0, x1), Mr = crop_roi(r['M'], (h+2*my, w+2*mx))
        back = unwarp(r['x'][1].astype(np.float32)/255, Mr, (y1-y0, x1-x0)) > .5
        own = labels[y0:y1, x0:x1] == r['cand']+1
        assert (back & own).sum() > .9*own.sum()


def test_new_networks_forward():
    import torch
    from filament_seg.models import ConvNeXtUNet, RefineUNet
    with torch.inference_mode():
        assert ConvNeXtUNet(in_chans=3).eval()(torch.zeros(1, 3, 64, 64)).shape == (1, 2, 64, 64)
        assert RefineUNet(in_chans=4).eval()(torch.zeros(2, 4, 64, 192)).shape == (2, 1, 64, 192)


class Probe:
    """Stands in for a GBM: returns column 53, the first detector column if the groups are stacked in feature_order."""

    def predict(self, x):
        return x[:, 53]


def test_keep_model_column_order():
    import gzip, pickle, tempfile
    from filament_seg.keep import KeepModel
    path = Path(tempfile.mkdtemp())/'k.pkl.gz'
    path.write_bytes(gzip.compress(pickle.dumps({'models': [Probe()], 'feature_order': ['x46', 'cnn', 'l2', 'dv']})))
    km = KeepModel(path, .25)
    cols = {'x46': np.zeros((3, 46)), 'cnn': np.zeros(3), 'l2': np.zeros((3, 6)), 'dv': np.array([[.1]*6, [.3]*6, [2.]*6])}
    u = km.scores(cols)
    assert u.dtype == np.float32 and np.allclose(u, [.1, .3, 1.]) and list(km.keep(u)) == [False, True, True]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
