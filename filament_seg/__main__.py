"""Command line: ``python -m filament_seg [--config configs/<name>.yaml | --fz | --lite] [--images <dir>] [--output submission.csv]``."""
import argparse

from .config import DEFAULT_CONFIG, FZ_CONFIG, LITE_CONFIG, data_path, load_config, resolve
from .data import list_images
from .pipeline import Pipeline, predict_frames
from .submission import validate_submission, write_submission


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(DEFAULT_CONFIG), help='default: the final pipeline (configs/final.yaml)')
    ap.add_argument('--fz', action='store_true', help=f'use the FZ_nz1 pipeline ({FZ_CONFIG.name})')
    ap.add_argument('--lite', action='store_true', help=f'use the lite pipeline ({LITE_CONFIG.name})')
    ap.add_argument('--images', help='folder of test frames (default: data.root/test_images from the config)')
    ap.add_argument('--output', default='outputs/submission.csv')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    a = ap.parse_args()
    cfg = load_config(LITE_CONFIG if a.lite else FZ_CONFIG if a.fz else a.config, {'inference.device': a.device} if a.device else None)
    paths = list_images(a.images or data_path(cfg, 'test_images'))
    frames = predict_frames(Pipeline(cfg), paths, resolve(cfg['output']['dir'])/f'frames_{cfg["name"].split()[0]}')
    digest = write_submission({k: v['rles'] for k, v in frames.items()}, a.output)
    print(validate_submission(a.output, paths), 'sha256', digest)


if __name__ == '__main__':
    main()
