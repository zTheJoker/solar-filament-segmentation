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


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
