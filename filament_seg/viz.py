"""Plots used by the demo notebook: host-style metric plots and overlays."""
from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy import ndimage as ndi


def plot_distribution(scores: np.ndarray, title: str, x_label: str, ax=None):
    """Histogram of positive pairwise scores with the mean marked (host notebook style)."""
    ax = ax or plt.gca()
    bins = np.linspace(0, 1, 51)
    counts, edges = np.histogram(scores, bins=bins)
    prob = counts/counts.sum() if counts.sum() else counts
    ax.bar((edges[:-1]+edges[1:])/2, prob, width=.02, edgecolor='black', alpha=.7, color='navy')
    if len(scores):
        ax.axvline(scores.mean(), color='crimson', linestyle='--', linewidth=1.5, label=f'Mean = {scores.mean():.3f}')
        ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--')
    ax.set(xlabel=x_label, ylabel='Probability', title=title)


def plot_relations(gt_degree: Counter, pred_degree: Counter, ax=None, title='Many to One and One to Many Distribution'):
    """Split bar chart of overlap degrees: GT -> #predictions (1:n, right), prediction -> #GT (n:1, left)."""
    ax = ax or plt.gca()
    for degree, count in gt_degree.items():
        ax.bar(degree+.5, count, width=.8, color='crimson' if degree == 0 else 'forestgreen', edgecolor='black', alpha=.7)
    for degree, count in pred_degree.items():
        ax.bar(-(degree+.5), count, width=.8, color='darkorange' if degree == 0 else 'steelblue', edgecolor='black', alpha=.7)
    ticks = sorted([d+.5 for d in gt_degree]+[-(d+.5) for d in pred_degree])
    ax.set_xticks(ticks, [f'1:{int(t-.5)}' if t > 0 else f'{int(-t-.5)}:1' for t in ticks], rotation=45, fontsize=9)
    ax.axvline(0, color='black', linewidth=.8)
    ax.grid(axis='y', linestyle='--')
    hits = lambda c: sum(v for d, v in c.items() if d > 0)
    ax.legend(handles=[Patch(facecolor='forestgreen', label=f'GT hits (1:n, n>0): {hits(gt_degree)}'),
                       Patch(facecolor='crimson', label=f'GT misses (1:0): {gt_degree.get(0, 0)}'),
                       Patch(facecolor='darkorange', label=f'Prediction misses (0:1): {pred_degree.get(0, 0)}'),
                       Patch(facecolor='steelblue', label=f'Prediction hits (n:1, n>0): {hits(pred_degree)}')], fontsize=8)
    ax.set(xlabel='Matching case', ylabel='Frequency', title=title)


def outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, iterations=2)


def overlay(gray: np.ndarray, layers: list[tuple[np.ndarray, tuple]], ax=None, title='', crop=None):
    """Grayscale frame with coloured instance outlines; layers = [(label image or mask, rgb), ...]."""
    ax = ax or plt.gca()
    rgb = np.repeat(gray[..., None], 3, -1).astype(np.float32)/255
    for labels, colour in layers:
        edge = np.zeros(gray.shape, bool)
        for k, box in enumerate(ndi.find_objects(np.asarray(labels).astype(np.int32)), 1):
            if box is not None:
                edge[box] |= outline(labels[box] == k)
        rgb[edge] = colour
    if crop is not None:
        rgb = rgb[crop]
    ax.imshow(rgb)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
