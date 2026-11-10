"""Convert training checkpoints into the repository's weights/ folder.

Every network state dict is stored as fp32 safetensors, split into shards of
at most --max-mb (GitHub rejects files over 100 MB; no Git LFS needed). The
manifest records, per model, its shards, the SHA-256 of the reassembled state
dict (verified at load time) and the SHA-256 of the source checkpoint.

    python train/export_weights.py \
        --cnx runs/cnx_f{0..4}/best.pt --r34 runs/r34_f{0..4}/best.pt \
        --crop-cnn runs/crop_cnn/cnn_fold{0..4}.pt --gbm runs/verifier_gbm.pkl \
        --extra-gbm verifier_gbm_without_fold0.pkl=runs/verifier_gbm_without_fold0.pkl --out weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_seg.models import state_dict_sha256  # noqa: E402


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def shard(state: dict, max_bytes: int) -> list[dict]:
    """Split a state dict, in key order, into consecutive groups of at most max_bytes."""
    parts, current, size = [], {}, 0
    for key, t in state.items():
        n = t.numel()*t.element_size()
        if current and size+n > max_bytes:
            parts.append(current)
            current, size = {}, 0
        current[key] = t.contiguous()
        size += n
    parts.append(current)
    return parts


def export_net(name, checkpoint, architecture, out, max_bytes, extra):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ck['model'] if 'model' in ck else ck
    files = []
    parts = shard(state, max_bytes)
    for i, part in enumerate(parts, 1):
        fname = f'{name}.safetensors' if len(parts) == 1 else f'{name}.part{i}of{len(parts)}.safetensors'
        save_file(part, str(out/fname))
        files.append(fname)
    entry = {'architecture': architecture, 'files': files, 'state_sha256': state_dict_sha256(state),
             'source_checkpoint_sha256': file_sha256(checkpoint), 'dtype': 'float32', **extra}
    if isinstance(ck, dict) and 'config' in ck:
        cfg = ck['config']
        entry['encoder'] = cfg.get('encoder')
        entry['training'] = {k: cfg[k] for k in ['fold', 'seed', 'epochs', 'epochs_a', 'epochs_b', 'train_sources',
                                                 'inner_sources', 'precision', 'folds_sha256'] if k in cfg}
        entry['training']['selected_epoch'] = ck.get('epoch')
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cnx', nargs=5, required=True)
    ap.add_argument('--r34', nargs=5, required=True)
    ap.add_argument('--crop-cnn', nargs=5, required=True)
    ap.add_argument('--gbm', required=True)
    ap.add_argument('--extra-gbm', nargs='*', default=[], metavar='NAME=PATH',
                    help='additional verifier pickles, e.g. the fold-0-excluded GBM used for held-out evaluation')
    ap.add_argument('--out', type=Path, default=Path('weights'))
    ap.add_argument('--max-mb', type=float, default=90.)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(a.max_mb*1e6)
    manifest = {}
    for k, path in enumerate(a.cnx):
        manifest[f'cnx_fold{k}'] = export_net(f'cnx_fold{k}', path, 'ConvNeXtUNet', a.out, max_bytes, {'fold': k})
    for k, path in enumerate(a.r34):
        manifest[f'r34_fold{k}'] = export_net(f'r34_fold{k}', path, 'ResNet34UNet', a.out, max_bytes, {'fold': k})
    for k, path in enumerate(a.crop_cnn):
        manifest[f'crop_cnn_fold{k}'] = export_net(f'crop_cnn_fold{k}', path, 'CropVerifierCNN', a.out, max_bytes, {'fold': k})
    for name, path in [('verifier_gbm.pkl', a.gbm)]+[tuple(e.split('=', 1)) for e in a.extra_gbm]:
        shutil.copyfile(path, a.out/name)
        manifest[name] = {'architecture': 'bagged sklearn HistGradientBoostingRegressor (pickle)',
                          'files': [name], 'sha256': file_sha256(a.out/name), 'source_sha256': file_sha256(path)}
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=1))
    for f in sorted(a.out.iterdir()):
        assert f.stat().st_size < 95e6, f'{f.name} is too large for GitHub'
    print(json.dumps({k: v['files'] for k, v in manifest.items()}, indent=1))


if __name__ == '__main__':
    main()
