"""Fine-tune one fold of the small-filament specialist (weights/spec_fold*.safetensors, FZ pipeline).

The base ConvNeXt U-Net sees frames downsampled to 1024, which loses faint,
thin, small filaments. The specialist is the same network, warm-started from
the fold's ConvNeXt checkpoint, fine-tuned on native-resolution crops:

  prep   per labelled frame, native 2048 uint8 arrays: image, soft target
         (mean over annotators of the filament union), boundary (mean of 2-px
         instance bands), "small" map (instances < 1000 px dilated by 3 px).
  train  12 epochs, 768-px native crops, 2 per frame per epoch, half of them
         centred (with jitter) on a random small filament; dihedral, scale
         .85-1.15, gamma/gain jitter. Loss = .5 BCE weighted 1 + 3*small +
         .5 soft Dice + .15 boundary BCE. AdamW lr 1e-4, 150-step warm-up +
         cosine, batch 12, bf16 autocast, EMA .998; the EMA epoch with the
         lowest loss on fixed crops of the fold's inner month slice is kept.
         ~15 min per fold on an A100.

    python train/train_specialist.py prep --out work/native
    python train/train_specialist.py train --cache work/native --fold 0 \
        --init work/runs/cnx_fold0/best.pt --out work/runs/spec_fold0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import atomic_save, device_and_autocast, load_fold, seed_everything  # noqa: E402
from filament_seg.config import data_path, load_config  # noqa: E402
from filament_seg.data import list_images, load_ground_truth, read_gray  # noqa: E402
from filament_seg.models import ConvNeXtUNet  # noqa: E402

cv2.setNumThreads(1)
SMALL_AREA = 1000
CROSS = ndi.generate_binary_structure(2, 1)


def prep_frame(job):
    """uint8 (4, 2048, 2048): image, soft target, boundary, small map; plus percentile lo/hi and small centres."""
    from pycocotools import mask as cm
    key, path, records, out = job
    g = read_gray(path)
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    disk = ((yy-(h-1)/2)/(h*.45))**2+((xx-(w-1)/2)/(w*.45))**2 < 1
    lo, hi = np.percentile(g.astype(np.float32)[disk], [1, 99])
    soft, edge, small = (np.zeros((h, w), np.float32) for _ in range(3))
    centres = []
    for rles in records:
        uni, ed, sm = (np.zeros((h, w), bool) for _ in range(3))
        for r in rles:
            m = cm.decode(r).astype(bool)
            area = int(m.sum())
            if not area:
                continue
            uni |= m
            box = ndi.find_objects(m.astype(np.uint8))[0]
            y0, y1 = max(0, box[0].start-4), min(h, box[0].stop+4)
            x0, x1 = max(0, box[1].start-4), min(w, box[1].stop+4)
            mc = m[y0:y1, x0:x1]
            ed[y0:y1, x0:x1] |= ndi.binary_dilation(mc, CROSS, 2) ^ ndi.binary_erosion(mc, CROSS, 2)
            if area < SMALL_AREA:
                sm[y0:y1, x0:x1] |= ndi.binary_dilation(mc, CROSS, 3)
                centres.append(((box[0].start+box[0].stop)//2, (box[1].start+box[1].stop)//2))
        soft += uni
        edge += ed
        small += sm
    q = lambda a: np.round(a/max(1, len(records))*255).astype(np.uint8)
    np.save(Path(out)/f'{key}.npy', np.stack([g, q(soft), q(edge), q(small)]))
    (Path(out)/f'{key}.json').write_text(json.dumps({'lo': float(lo), 'hi': float(hi),
                                                     'centers': [list(map(int, c)) for c in centres]}))


def normalise(g, lo, hi):
    v = np.clip((g.astype(np.float32)-lo)/max(float(hi-lo), 5.), 0., 1.)
    return ((v-.5)/.25).astype(np.float32)


class NativeCrops(Dataset):
    """Native crops; training crops are random (half centred on small filaments), validation crops fixed."""

    def __init__(self, cache, keys, crop=768, per_frame=2, small_frac=.5, augment=True):
        self.cache, self.crop, self.small_frac, self.augment = Path(cache), crop, small_frac, augment
        self.items = [(k, j) for k in sorted(keys) for j in range(per_frame)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, j = self.items[i]
        arr = np.load(self.cache/f'{key}.npy', mmap_mode='r')
        meta = json.loads((self.cache/f'{key}.json').read_text())
        rng = random if self.augment else random.Random(f'{key}/{j}')
        side = int(round(self.crop/rng.uniform(.85, 1.15))) if self.augment else self.crop
        if not self.augment:
            cy, cx = [(512, 512), (512, 1536), (1536, 512), (1536, 1536)][j % 4]
        elif meta['centers'] and rng.random() < self.small_frac:
            cy, cx = meta['centers'][rng.randrange(len(meta['centers']))]
            cy, cx = cy+rng.randint(-side//3, side//3), cx+rng.randint(-side//3, side//3)
        else:
            cy, cx = rng.randint(side//2, 2048-side//2), rng.randint(side//2, 2048-side//2)
        y0, x0 = int(np.clip(cy-side//2, 0, 2048-side)), int(np.clip(cx-side//2, 0, 2048-side))
        x = normalise(np.asarray(arr[0, y0:y0+side, x0:x0+side]), meta['lo'], meta['hi'])
        y = np.asarray(arr[1:, y0:y0+side, x0:x0+side]).astype(np.float32)/255
        if side != self.crop:
            x = cv2.resize(x, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
            y = np.stack([cv2.resize(c, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR) for c in y])
        if self.augment:
            k = random.randrange(4)
            x, y = np.rot90(x, k), np.rot90(y, k, axes=(-2, -1))
            if random.random() < .5:
                x, y = x[:, ::-1], y[:, :, ::-1]
            if random.random() < .7:
                x = (np.clip(x*.25+.5, 0, 1)**random.uniform(.75, 1.35)-.5)/.25
            x = x*random.uniform(.9, 1.1)+random.uniform(-.1, .1)
        return torch.from_numpy(np.ascontiguousarray(x[None])).float(), torch.from_numpy(np.ascontiguousarray(y)).float()


def small_weighted_loss(logits, target, small_w=3.):
    fg, edge_logit = logits[:, 0:1].float(), logits[:, 1:2].float()
    t, e, s = target[:, 0:1], target[:, 1:2], target[:, 2:3]
    w = 1+small_w*s
    bce = (F.binary_cross_entropy_with_logits(fg, t, reduction='none')*w).mean()/w.mean()
    p = fg.sigmoid()
    dice = 1-((2*(p*t).sum((1, 2, 3))+1)/((p+t).sum((1, 2, 3))+1)).mean()
    edge = F.binary_cross_entropy_with_logits(edge_logit, e, pos_weight=fg.new_tensor(5.))
    return .5*bce+.5*dice+.15*edge


def prep(a):
    cfg = load_config(a.config)
    paths = list_images(data_path(cfg, 'train_images'))
    keys = sorted(paths)[::len(paths)//8][:8] if a.smoke else sorted(paths)
    truth = load_ground_truth(data_path(cfg, 'train_annotations'), set(keys))
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(prep_frame, [(k, paths[k], [r for _, r in truth[k]], str(a.out)) for k in keys]))
    print(f'prepared {len(keys)} native frames', flush=True)


def train(a):
    fold = load_fold(a.folds, a.fold)
    train_ids = [k for k in fold['train_sources'] if (a.cache/f'{k}.npy').exists()]
    inner_ids = [k for k in fold['inner_validation_sources'] if (a.cache/f'{k}.npy').exists()]
    if a.smoke:
        a.epochs, a.batch, a.workers, a.crop = 1, 2, 0, 256
    seed_everything(700+a.fold)
    device, autocast = device_and_autocast()
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    model = ConvNeXtUNet(ck['config']['encoder'])
    model.load_state_dict(ck['model'])
    model = model.to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    config = {'architecture': 'ConvNeXtUNet', 'encoder': ck['config']['encoder'], 'fold': a.fold, 'seed': 700+a.fold,
              'side': 2048, 'crop': a.crop, 'epochs': a.epochs, 'small_w': a.small_w, 'lr': a.lr, 'batch': a.batch,
              'ema': a.ema, 'train_sources': len(train_ids), 'inner_sources': len(inner_ids),
              'init_sha256': hashlib.sha256(a.init.read_bytes()).hexdigest(), 'precision': 'bf16 autocast'}
    a.out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(NativeCrops(a.cache, train_ids, a.crop), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, drop_last=True, pin_memory=device.type == 'cuda')
    inner = DataLoader(NativeCrops(a.cache, inner_ids, a.crop, per_frame=4, small_frac=0., augment=False),
                       batch_size=a.batch, num_workers=a.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    total = max(1, a.epochs*len(loader))
    warm = max(1, min(150, total//10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else
                                              .5*(1+math.cos(math.pi*(s-warm)/max(1, total-warm))))

    @torch.no_grad()
    def inner_loss():
        tot, n = 0., 0
        for x, y in inner:
            x, y = x.to(device), y.to(device)
            with autocast():
                z = ema(x)
            tot += float(small_weighted_loss(z.float(), y, a.small_w))*len(x)
            n += len(x)
        return tot/max(1, n)
    best, steps, history, start = float('inf'), 0, [], time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                z = model(x)
            loss = small_weighted_loss(z, y, a.small_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            opt.step()
            sched.step()
            steps += 1
            losses.append(float(loss.detach()))
            decay = min(a.ema, (1+steps)/(10+steps))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1-decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        val = inner_loss()
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)) if losses else None, 'inner_loss': val,
                        'seconds': round(time.monotonic()-start, 1)})
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best = val
            atomic_save({'model': ema.state_dict(), 'config': config, 'epoch': epoch}, a.out/'best.pt')
    (a.out/'history.json').write_text(json.dumps({'config': config, 'history': history}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('--config', default='configs/fz_nz1.yaml')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--smoke', action='store_true', help='8 frames')
    t = sub.add_parser('train')
    t.add_argument('--cache', type=Path, required=True)
    t.add_argument('--folds', type=Path, default=Path('configs/folds_full.json'))
    t.add_argument('--fold', type=int, required=True)
    t.add_argument('--init', type=Path, required=True, help="this fold's ConvNeXt checkpoint (train_convnext_unet.py)")
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=12)
    t.add_argument('--crop', type=int, default=768)
    t.add_argument('--batch', type=int, default=12)
    t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--small-w', type=float, default=3.)
    t.add_argument('--ema', type=float, default=.998)
    t.add_argument('--workers', type=int, default=6)
    t.add_argument('--smoke', action='store_true', help='1 epoch, 256-px crops, batch 2, on the prepared frames')
    a = ap.parse_args()
    prep(a) if a.cmd == 'prep' else train(a)


if __name__ == '__main__':
    main()
