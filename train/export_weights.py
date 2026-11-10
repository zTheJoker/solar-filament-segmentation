"""Convert training checkpoints into the repository's weights/ folder.

* U-Nets (ConvNeXt, ResNet34, small-filament specialists): fp32 safetensors,
  split into shards of at most --max-mb (GitHub rejects files over 100 MB).
* Crop CNNs: they only run under bf16 autocast, which casts conv/linear
  weights to bf16 on every call; those tensors are therefore stored in bf16
  (identical computation, half the size). BatchNorm tensors stay fp32.
* GBM verifiers: gzip-compressed pickles.
The manifest records, per model, its files, the SHA-256 of the state dict as
the loader reconstructs it (verified at load time) and the SHA-256 of the
source checkpoint.

    python train/export_weights.py \
        --cnx runs/cnx_fold{0..4}/best.pt --r34 runs/r34_fold{0..4}/best.pt \
        --crop-cnn runs/crop_cnn/cnn_fold{0..4}.pt --gbm runs/verifier_gbm.pkl \
        --spec runs/spec_fold{0..4}/best.pt --crop-cnn-fz runs/crop_cnn_fz/cnn_fold{0..4}.pt \
        --gbm-fz runs/verifier_gbm_fz.pkl \
        --extra-gbm verifier_gbm_fz_without_fold0.pkl.gz=runs/verifier_gbm_fz_without_fold0.pkl --out weights
"""
from __future__ import annotations

import argparse
import gzip
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


def autocast_tensors(model_cls, state):
    """Names of the conv/linear weights and biases (the tensors bf16 autocast casts)."""
    net = model_cls(pretrained=False)
    names = set()
    for mod_name, mod in net.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            names |= {f'{mod_name}.{p}' for p, _ in mod.named_parameters(recurse=False)}
    assert names <= set(state)
    return names


def export_net(name, checkpoint, architecture, out, max_bytes, extra, bf16_autocast_tensors=False):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ck['model'] if 'model' in ck else ck
    stored = dict(state)
    if bf16_autocast_tensors:
        from filament_seg.models import ARCHITECTURES
        for k in autocast_tensors(ARCHITECTURES[architecture], state):
            stored[k] = state[k].to(torch.bfloat16)
    files = []
    parts = shard(stored, max_bytes)
    for i, part in enumerate(parts, 1):
        fname = f'{name}.safetensors' if len(parts) == 1 else f'{name}.part{i}of{len(parts)}.safetensors'
        save_file(part, str(out/fname))
        files.append(fname)
    loaded = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in stored.items()}
    entry = {'architecture': architecture, 'files': files, 'state_sha256': state_dict_sha256(loaded),
             'source_checkpoint_sha256': file_sha256(checkpoint),
             'dtype': 'conv/linear tensors bf16 (run under bf16 autocast), rest fp32' if bf16_autocast_tensors else 'float32',
             **extra}
    if isinstance(ck, dict) and 'config' in ck:
        cfg = ck['config']
        entry['encoder'] = cfg.get('encoder')
        entry['training'] = {k: cfg[k] for k in ['fold', 'seed', 'epochs', 'epochs_a', 'epochs_b', 'train_sources',
                                                 'inner_sources', 'precision', 'folds_sha256', 'side', 'crop',
                                                 'small_w', 'init_sha256'] if k in cfg}
        entry['training']['selected_epoch'] = ck.get('epoch')
    return entry


def export_gbm(name, path, out):
    with open(path, 'rb') as src, open(out/name, 'wb') as raw, \
            gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=raw, mtime=0) as dst:
        shutil.copyfileobj(src, dst)  # mtime 0 and no file name: byte-identical output on every export
    return {'architecture': 'bagged sklearn HistGradientBoostingRegressor (gzip-compressed pickle)', 'files': [name],
            'sha256': file_sha256(out/name), 'source_sha256': file_sha256(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cnx', nargs=5, required=True)
    ap.add_argument('--r34', nargs=5, required=True)
    ap.add_argument('--crop-cnn', nargs=5, required=True, help='crop CNNs of the lite (F) verifier')
    ap.add_argument('--gbm', required=True, help='GBM of the lite (F) verifier')
    ap.add_argument('--spec', nargs=5, help='small-filament specialists (FZ)')
    ap.add_argument('--crop-cnn-fz', nargs=5, help='crop CNNs of the FZ verifier')
    ap.add_argument('--gbm-fz', help='GBM of the FZ verifier')
    ap.add_argument('--extra-gbm', nargs='*', default=[], metavar='NAME=PATH',
                    help='additional verifier pickles, e.g. fold-0-excluded GBMs used for held-out evaluation')
    ap.add_argument('--out', type=Path, default=Path('weights'))
    ap.add_argument('--max-mb', type=float, default=90.)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mb = int(a.max_mb*1e6)
    manifest = {}
    for prefix, paths, arch, bf16 in [('cnx', a.cnx, 'ConvNeXtUNet', False), ('r34', a.r34, 'ResNet34UNet', False),
                                      ('crop_cnn', a.crop_cnn, 'CropVerifierCNN', True),
                                      ('spec', a.spec or [], 'ConvNeXtUNet', False),
                                      ('crop_cnn_fz', a.crop_cnn_fz or [], 'CropVerifierCNN', True)]:
        for k, path in enumerate(paths):
            manifest[f'{prefix}_fold{k}'] = export_net(f'{prefix}_fold{k}', path, arch, a.out, mb, {'fold': k}, bf16)
    gbms = [('verifier_gbm.pkl.gz', a.gbm)]+([('verifier_gbm_fz.pkl.gz', a.gbm_fz)] if a.gbm_fz else [])
    for name, path in gbms+[tuple(e.split('=', 1)) for e in a.extra_gbm]:
        manifest[name] = export_gbm(name, path, a.out)
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=1))
    for f in sorted(a.out.iterdir()):
        assert f.stat().st_size < 95e6, f'{f.name} is too large for GitHub'
    print(json.dumps({k: v['files'] for k, v in manifest.items()}, indent=1))


if __name__ == '__main__':
    main()
