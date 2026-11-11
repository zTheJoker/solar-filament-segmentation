"""Load the single YAML configuration and resolve its paths.

All paths in the YAML file are relative to the repository root, so the
repository can be cloned anywhere. ``FILAMENT_DATA_ROOT`` (environment) may
point the pipeline at a data folder outside the repository.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT/'configs'/'final.yaml'
FZ_CONFIG = REPO_ROOT/'configs'/'fz_nz1.yaml'
LITE_CONFIG = REPO_ROOT/'configs'/'f_lite.yaml'


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    """Read the YAML config; ``overrides`` maps dotted keys to values (e.g. {'inference.device': 'cpu'})."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT/path
    cfg = yaml.safe_load(path.read_text())
    if os.environ.get('FILAMENT_DATA_ROOT'):
        cfg['data']['root'] = os.environ['FILAMENT_DATA_ROOT']
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split('.')
        for key in parents:
            node = node[key]
        node[leaf] = copy.deepcopy(value)
    if cfg['postprocess']['steps'] and cfg['postprocess']['steps'][0]['name'] == 'outline_nz1':
        # The nz1 outline treats the decoder threshold as its "unchanged" level.
        assert abs(cfg['postprocess']['steps'][0]['base_level']-cfg['candidates']['fg_threshold']) < 1e-9
    return cfg


def resolve(path: str | Path) -> Path:
    """Absolute path for a config entry (relative entries are relative to the repository root)."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT/path


def data_path(cfg: dict, key: str) -> Path:
    """Path of a data entry (``test_images``, ``train_images``, ``train_annotations``) under data.root."""
    return resolve(cfg['data']['root'])/cfg['data'][key]
