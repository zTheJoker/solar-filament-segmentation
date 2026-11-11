"""Solar filament instance segmentation (Kaggle filament-segmentation-2026): the final pipeline and two earlier ones."""
from .config import load_config
from .pipeline import Pipeline, predict_frames

__all__ = ['load_config', 'Pipeline', 'predict_frames']
