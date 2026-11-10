"""Command line: ``python -m filament_seg --images <dir> --output submission.csv [--config ...]``."""
import argparse

from .config import DEFAULT_CONFIG, data_path, load_config, resolve
from .data import list_images
from .pipeline import Pipeline, predict_frames
from .submission import validate_submission, write_submission


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--images', help='folder of test frames (default: data.root/test_images from the config)')
    ap.add_argument('--output', default='outputs/submission.csv')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    a = ap.parse_args()
    cfg = load_config(a.config, {'inference.device': a.device} if a.device else None)
    paths = list_images(a.images or data_path(cfg, 'test_images'))
    frames = predict_frames(Pipeline(cfg), paths, resolve(cfg['output']['dir'])/'frames')
    digest = write_submission({k: v['rles'] for k, v in frames.items()}, a.output)
    print(validate_submission(a.output, paths), 'sha256', digest)


if __name__ == '__main__':
    main()
