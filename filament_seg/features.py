"""The 46 per-candidate features of the verifier (image-only; no annotation input).

* 26 shape / probability / photometry features (``SHAPE_PHOTOMETRY``)
* 14 context features: agreement between the D4 map and the single-view map,
  high-confidence core fractions, position on the solar disk, neighbours
  (``CONTEXT``)
* 6 vote features from the independent ResNet34 ensemble (``VOTE``)

The column order is fixed: the GBM verifier was trained on exactly this order.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

SHAPE_PHOTOMETRY = [
    'log_area', 'log_height', 'log_width', 'bbox_axis_ratio', 'bbox_fill',
    'perimeter_over_sqrt_area', 'moment_axis_ratio',
    'fg_mean', 'fg_std', 'fg_p10', 'fg_p50', 'fg_p90',
    'boundary_mean', 'boundary_p90', 'edge_boundary_mean',
    'gray_mean', 'gray_std', 'gray_p10', 'gray_median',
    'ring_gray_mean', 'ring_gray_std', 'ring_minus_inside_contrast',
    'centroid_radius_over_geometric_sun_radius', 'min_radius', 'max_radius',
    'fraction_beyond_point9_radius',
]
# Names kept from the training run: for the D4 decode, "d4_*" and
# "plain_d4_absdiff" are computed on the *single-view* map (the "other" view).
CONTEXT = ['d4_mean', 'd4_p10', 'd4_frac_ge5', 'plain_d4_absdiff', 'fg_max', 'fg_frac_ge8', 'fg_frac_ge95',
           'disk_rho', 'disk_rho_max', 'n_neighbors_100', 'neighbor_median_log_area', 'log_area_rank_in_image',
           'n_candidates_image', 'image_fg_fraction']
VOTE = ['vote_mean', 'vote_p10', 'vote_frac_ge5', 'vote_max', 'vote_best_iou', 'vote_n_overlap']
FEATURE_NAMES = SHAPE_PHOTOMETRY+CONTEXT+VOTE

# Solar disk geometry of the GONG frames used by the context features (native pixels).
DISK_CENTER = (1024.1, 1023.1)
DISK_RADIUS = 899.7


def shape_photometry_features(gray: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """26 features per candidate from its mask, the decode map (fg, boundary) and the 8-bit image."""
    gray = np.asarray(gray)
    height, width = gray.shape
    sun_radius = min(height, width)*.45
    rows = []
    for label, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            raise ValueError('Candidate labels must be compact.')
        mask = labels[box] == label
        yy, xx = np.nonzero(mask)
        y = yy+box[0].start
        x = xx+box[1].start
        area = len(y)
        object_height, object_width = mask.shape
        fg = probabilities[0, y, x].astype(np.float64)
        edge = probabilities[1, y, x].astype(np.float64)
        pixels = gray[y, x].astype(np.float64)/255.
        perimeter_mask = mask & ~ndi.binary_erosion(mask)
        edge_values = probabilities[1][box][perimeter_mask]
        covariance = np.cov(np.vstack((y, x))) if area > 1 else np.zeros((2, 2))
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        moment_ratio = np.sqrt((eigenvalues[-1]+1)/(eigenvalues[0]+1))
        # Local 8-px ring around the candidate, excluding every candidate pixel.
        pad = 8
        ring_box = (slice(max(0, box[0].start-pad), min(height, box[0].stop+pad)),
                    slice(max(0, box[1].start-pad), min(width, box[1].stop+pad)))
        local = labels[ring_box]
        ring = ndi.binary_dilation(local == label, iterations=pad) & (local == 0)
        ring_pixels = gray[ring_box][ring].astype(np.float64)/255.
        ring_mean = float(ring_pixels.mean()) if len(ring_pixels) else float(pixels.mean())
        ring_std = float(ring_pixels.std()) if len(ring_pixels) else 0.
        radii = np.hypot(y-(height-1)/2, x-(width-1)/2)/sun_radius
        centroid_radius = np.hypot(y.mean()-(height-1)/2, x.mean()-(width-1)/2)/sun_radius
        rows.append([
            np.log1p(area), np.log1p(object_height), np.log1p(object_width),
            max(object_height, object_width)/min(object_height, object_width),
            area/(object_height*object_width), perimeter_mask.sum()/np.sqrt(area), moment_ratio,
            fg.mean(), fg.std(), *np.percentile(fg, [10, 50, 90]), edge.mean(),
            np.percentile(edge, 90), edge_values.mean(),
            pixels.mean(), pixels.std(), np.percentile(pixels, 10), np.median(pixels),
            ring_mean, ring_std, (ring_mean-np.median(pixels))/max(ring_std, .02),
            centroid_radius, radii.min(), radii.max(), (radii > .9).mean(),
        ])
    result = np.asarray(rows, np.float64).reshape(-1, len(SHAPE_PHOTOMETRY))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite candidate feature.')
    return result


def context_features(labels: np.ndarray, decode: np.ndarray, other: np.ndarray) -> np.ndarray:
    """14 features: agreement with the other view, cores, disk position and neighbourhood."""
    n = int(labels.max())
    objs = ndi.find_objects(labels)
    cents = np.asarray([[(b[0].start+b[0].stop)/2, (b[1].start+b[1].stop)/2] for b in objs]) if n else np.zeros((0, 2))
    areas = np.bincount(labels.ravel(), minlength=n+1)[1:]
    order = np.argsort(np.argsort(-areas)) if n else np.zeros(0)
    image_fg = float((decode[0] >= .5).mean())
    rows = []
    for i, box in enumerate(objs):
        m = labels[box] == i+1
        p = decode[0][box][m]
        q = other[0][box][m]
        yy, xx = np.nonzero(m)
        yy = yy+box[0].start
        xx = xx+box[1].start
        rho = np.hypot(yy-DISK_CENTER[0], xx-DISK_CENTER[1])/DISK_RADIUS
        dist = np.hypot(*(cents-cents[i]).T)
        near = (dist < 100) & (np.arange(n) != i)
        rows.append([q.mean(), np.percentile(q, 10), (q >= .5).mean(), np.abs(p-q).mean(), p.max(), (p >= .8).mean(),
                     (p >= .95).mean(), np.hypot(yy.mean()-DISK_CENTER[0], xx.mean()-DISK_CENTER[1])/DISK_RADIUS,
                     rho.max(), near.sum(), np.median(np.log1p(areas[near])) if near.any() else 0.,
                     order[i]/max(1, n), n, image_fg])
    return np.asarray(rows, np.float64).reshape(-1, len(CONTEXT))


def vote_features(labels: np.ndarray, vote: np.ndarray) -> np.ndarray:
    """6 features: how the independent ResNet34 ensemble sees each candidate."""
    comp, _ = ndi.label(vote[0] >= .5, np.ones((3, 3), bool))
    rows = []
    for i, box in enumerate(ndi.find_objects(labels)):
        m = labels[box] == i+1
        q = vote[0][box][m]
        oc = comp[box][m]
        ids, inter = np.unique(oc[oc > 0], return_counts=True)
        best = 0.
        if len(ids):
            sizes = np.bincount(comp.ravel())[ids]
            best = float(np.max(inter/(m.sum()+sizes-inter)))
        rows.append([q.mean(), np.percentile(q, 10), (q >= .5).mean(), q.max(), best, len(ids)])
    return np.asarray(rows, np.float64).reshape(-1, len(VOTE))


def candidate_features(gray: np.ndarray, labels: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    """(n_candidates, 46) feature matrix in the verifier's column order."""
    return np.hstack([shape_photometry_features(gray, maps['decode'], labels),
                      context_features(labels, maps['decode'], maps['single']),
                      vote_features(labels, maps['vote'])])
