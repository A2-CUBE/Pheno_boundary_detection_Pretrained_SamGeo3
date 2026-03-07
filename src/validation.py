"""
Validation utilities for comparing FTW predictions against cadastral ground truth.

Provides pixel-level, boundary-level, object-level, and temporal statistics
for evaluating raw and post-processed field boundary masks against
Provincial Cadastre parcel polygons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# AOI and grid constants (South Tyrol study area, 10m Sentinel-2)
# ---------------------------------------------------------------------------
AOI_BBOX_WGS84 = [11.290770, 46.356466, 11.315060, 46.389037]
TARGET_CRS = "EPSG:32632"
PIXEL_SIZE = 10.0  # metres
GRID_SHAPE = (368, 177)  # (rows=y, cols=x)


def _get_utm_bbox():
    """Compute UTM bounding box from WGS84 AOI."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    xmin, ymin = t.transform(AOI_BBOX_WGS84[0], AOI_BBOX_WGS84[1])
    xmax, ymax = t.transform(AOI_BBOX_WGS84[2], AOI_BBOX_WGS84[3])
    return xmin, ymin, xmax, ymax


def _get_transform():
    """Build rasterio Affine transform for the FTW output grid."""
    from rasterio.transform import from_bounds
    xmin, _, _, ymax = _get_utm_bbox()
    # Grid covers exactly GRID_SHAPE pixels at PIXEL_SIZE
    xmax_grid = xmin + GRID_SHAPE[1] * PIXEL_SIZE
    ymin_grid = ymax - GRID_SHAPE[0] * PIXEL_SIZE
    return from_bounds(xmin, ymin_grid, xmax_grid, ymax, GRID_SHAPE[1], GRID_SHAPE[0])


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_cadastre(
    shp_path: str,
    aoi_bbox: List[float] = None,
    target_crs: str = TARGET_CRS,
) -> "gpd.GeoDataFrame":
    """
    Load cadastral parcels, clip to AOI, and reproject.

    Parameters
    ----------
    shp_path : str
        Path to shapefile or GeoJSON.
    aoi_bbox : list, optional
        [xmin, ymin, xmax, ymax] in WGS84. Defaults to study AOI.
    target_crs : str
        Target CRS for reprojection.

    Returns
    -------
    GeoDataFrame
        Clipped, reprojected cadastral parcels with area in m².
    """
    import geopandas as gpd
    from shapely.geometry import box

    if aoi_bbox is None:
        aoi_bbox = AOI_BBOX_WGS84

    gdf = gpd.read_file(shp_path)

    # Fix invalid geometries (common with cadastral data)
    from shapely.validation import make_valid
    gdf['geometry'] = gdf['geometry'].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )

    # Clip to AOI
    aoi_polygon = box(*aoi_bbox)
    gdf = gdf[gdf.intersects(aoi_polygon)].copy()
    gdf = gpd.clip(gdf, aoi_polygon)

    # Reproject
    if gdf.crs and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Compute projected area
    gdf['area_m2'] = gdf.geometry.area

    return gdf


def rasterize_parcels(
    gdf: "gpd.GeoDataFrame",
    parcel_type: Optional[str] = None,
    shape: Tuple[int, int] = GRID_SHAPE,
) -> np.ndarray:
    """
    Rasterize vector parcels to a binary mask matching FTW output grid.

    Parameters
    ----------
    gdf : GeoDataFrame
        Cadastral parcels in UTM CRS.
    parcel_type : str, optional
        Filter by PPOL_TIPO before rasterizing. 'T' = land, 'E' = building.
        None = all parcels.
    shape : tuple
        Output raster shape (rows, cols).

    Returns
    -------
    np.ndarray
        Binary mask (bool), same shape as FTW output.
    """
    from rasterio import features

    transform = _get_transform()

    subset = gdf
    if parcel_type is not None and 'PPOL_TIPO' in gdf.columns:
        subset = gdf[gdf['PPOL_TIPO'] == parcel_type]

    if len(subset) == 0:
        return np.zeros(shape, dtype=bool)

    geometries = subset.geometry.values
    mask = features.rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def rasterize_parcels_labeled(
    gdf: "gpd.GeoDataFrame",
    parcel_type: Optional[str] = None,
    shape: Tuple[int, int] = GRID_SHAPE,
) -> np.ndarray:
    """
    Rasterize parcels with unique labels per parcel (for per-parcel stats).

    Returns
    -------
    np.ndarray
        Integer array where each parcel has a unique ID (0 = no parcel).
    """
    from rasterio import features

    transform = _get_transform()

    subset = gdf
    if parcel_type is not None and 'PPOL_TIPO' in gdf.columns:
        subset = gdf[gdf['PPOL_TIPO'] == parcel_type].copy()

    if len(subset) == 0:
        return np.zeros(shape, dtype=np.int32)

    subset = subset.reset_index(drop=True)
    geometries = [(geom, idx + 1) for idx, geom in enumerate(subset.geometry.values)]

    labeled = features.rasterize(
        geometries,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    return labeled


# ---------------------------------------------------------------------------
# A. Pixel-level metrics
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict:
    """
    Compute pixel-level accuracy metrics.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask.
    gt_mask : np.ndarray
        Ground truth binary mask.

    Returns
    -------
    dict
        Precision, Recall, F1, IoU, Accuracy, Kappa, and pixel counts.
    """
    pred = pred_mask.astype(bool).ravel()
    gt = gt_mask.astype(bool).ravel()

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    total = len(pred)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / total

    # Cohen's Kappa
    p_obs = accuracy
    p_exp = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    kappa = (p_obs - p_exp) / max(1 - p_exp, 1e-8)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


# ---------------------------------------------------------------------------
# B. Boundary metrics
# ---------------------------------------------------------------------------

def _extract_boundaries(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels using morphological gradient."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(2, 1)
    dilated = binary_dilation(mask.astype(bool), struct)
    return np.logical_xor(dilated, mask.astype(bool))


def compute_boundary_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerances: List[int] = None,
) -> Dict:
    """
    Compute boundary F1 at multiple distance tolerances.

    A predicted boundary pixel is considered correct if it falls within
    `tolerance` pixels of a ground truth boundary pixel.

    Parameters
    ----------
    pred_mask, gt_mask : np.ndarray
        Binary masks.
    tolerances : list of int
        Distance tolerances in pixels (1px = 10m).

    Returns
    -------
    dict
        Boundary precision, recall, F1 at each tolerance.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure, distance_transform_edt

    if tolerances is None:
        tolerances = [1, 2, 3]

    pred_boundary = _extract_boundaries(pred_mask)
    gt_boundary = _extract_boundaries(gt_mask)

    results = {}
    for tol in tolerances:
        # Distance transform from GT boundary
        gt_dist = distance_transform_edt(~gt_boundary)
        pred_dist = distance_transform_edt(~pred_boundary)

        # Boundary precision: fraction of pred boundary within tolerance of GT
        pred_pts = pred_boundary.sum()
        if pred_pts > 0:
            bp = (gt_dist[pred_boundary] <= tol).sum() / pred_pts
        else:
            bp = 0.0

        # Boundary recall: fraction of GT boundary within tolerance of pred
        gt_pts = gt_boundary.sum()
        if gt_pts > 0:
            br = (pred_dist[gt_boundary] <= tol).sum() / gt_pts
        else:
            br = 0.0

        bf1 = 2 * bp * br / max(bp + br, 1e-8)

        label = f"{tol * int(PIXEL_SIZE)}m"
        results[f'boundary_precision_{label}'] = float(bp)
        results[f'boundary_recall_{label}'] = float(br)
        results[f'boundary_f1_{label}'] = float(bf1)

    return results


# ---------------------------------------------------------------------------
# C. Per-parcel object-level stats
# ---------------------------------------------------------------------------

def compute_per_parcel_stats(
    pred_mask: np.ndarray,
    gdf: "gpd.GeoDataFrame",
    parcel_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per-parcel detection statistics.

    For each cadastral parcel, measures overlap with the prediction mask.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary field mask (368, 177).
    gdf : GeoDataFrame
        Cadastral parcels (UTM CRS).
    parcel_type : str, optional
        Filter parcels by PPOL_TIPO.

    Returns
    -------
    DataFrame
        Per-parcel stats: parcel_id, area_m2, type, iou, coverage, detected.
    """
    labeled = rasterize_parcels_labeled(gdf, parcel_type=parcel_type)
    pred = pred_mask.astype(bool)

    subset = gdf
    if parcel_type is not None and 'PPOL_TIPO' in gdf.columns:
        subset = gdf[gdf['PPOL_TIPO'] == parcel_type].copy()
    subset = subset.reset_index(drop=True)

    records = []
    for idx in range(len(subset)):
        parcel_id = idx + 1
        parcel_pixels = labeled == parcel_id
        n_parcel = parcel_pixels.sum()

        if n_parcel == 0:
            continue

        # Local IoU: use a bounding box around the parcel with buffer
        ys, xs = np.where(parcel_pixels)
        buf = 5  # 5 pixels = 50m buffer
        y0 = max(ys.min() - buf, 0)
        y1 = min(ys.max() + buf + 1, parcel_pixels.shape[0])
        x0 = max(xs.min() - buf, 0)
        x1 = min(xs.max() + buf + 1, parcel_pixels.shape[1])

        local_parcel = parcel_pixels[y0:y1, x0:x1]
        local_pred = pred[y0:y1, x0:x1]

        intersection = np.logical_and(local_pred, local_parcel).sum()
        union = np.logical_or(local_pred, local_parcel).sum()
        local_iou = float(intersection / max(union, 1))

        # Coverage: fraction of parcel covered by prediction
        n_pred_in_parcel = intersection
        coverage = float(n_pred_in_parcel / max(n_parcel, 1))

        row = subset.iloc[idx]
        records.append({
            'parcel_idx': parcel_id,
            'ppol_id': row.get('PPOL_ID', None),
            'ppol_tipo': row.get('PPOL_TIPO', None),
            'ppol_tip1': row.get('PPOL_TIP_1', None),
            'area_m2': row.get('PPOL_AREA', row.get('area_m2', n_parcel * PIXEL_SIZE**2)),
            'n_pixels': int(n_parcel),
            'n_detected': int(n_pred_in_parcel),
            'coverage': float(coverage),
            'iou': local_iou,
            'detected': bool(coverage > 0.1),
        })

    return pd.DataFrame(records)


def compute_size_accuracy(per_parcel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detection accuracy binned by parcel area.

    Bins: <500 m² (tiny), 500-5000 m² (small), 5000-20000 m² (medium), >20000 m² (large).

    Returns
    -------
    DataFrame
        Stats per size class: count, detection_rate, mean_iou, mean_coverage.
    """
    bins = [0, 500, 5000, 20000, float('inf')]
    labels = ['<500m²', '500-5000m²', '5000-20000m²', '>20000m²']

    df = per_parcel_df.copy()
    df['size_class'] = pd.cut(df['area_m2'], bins=bins, labels=labels, right=False)

    summary = df.groupby('size_class', observed=True).agg(
        count=('iou', 'size'),
        detection_rate=('detected', 'mean'),
        mean_iou=('iou', 'mean'),
        median_iou=('iou', 'median'),
        mean_coverage=('coverage', 'mean'),
    ).reset_index()

    return summary


# ---------------------------------------------------------------------------
# D. Stratified analysis
# ---------------------------------------------------------------------------

def compute_stratified_stats(
    pred_mask: np.ndarray,
    gdf: "gpd.GeoDataFrame",
    stratify_col: str = 'PPOL_TIPO',
) -> pd.DataFrame:
    """
    Compute pixel-level metrics stratified by a categorical column.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask.
    gdf : GeoDataFrame
        Cadastral parcels.
    stratify_col : str
        Column to stratify by (e.g. 'PPOL_TIPO' for T/E).

    Returns
    -------
    DataFrame
        Metrics per stratum.
    """
    if stratify_col not in gdf.columns:
        raise ValueError(f"Column '{stratify_col}' not found in GeoDataFrame")

    strata = gdf[stratify_col].unique()
    records = []

    for stratum in strata:
        gt_mask = rasterize_parcels(gdf[gdf[stratify_col] == stratum])
        metrics = compute_pixel_metrics(pred_mask, gt_mask)
        metrics['stratum'] = stratum
        metrics['n_parcels'] = int((gdf[stratify_col] == stratum).sum())
        records.append(metrics)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# E. Temporal + ground truth
# ---------------------------------------------------------------------------

def compute_temporal_gt_stats(
    masks_dict: Dict[int, np.ndarray],
    gt_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Compute year-pair IoU restricted to within cadastral boundaries.

    Parameters
    ----------
    masks_dict : dict
        {year: binary_field_mask}.
    gt_mask : np.ndarray
        Ground truth binary mask (union of cadastral parcels).

    Returns
    -------
    DataFrame
        Pairwise IoU matrix (within-cadastre only).
    """
    from .stability import compute_iou

    years = sorted(masks_dict.keys())
    n = len(years)
    matrix = np.zeros((n, n))

    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            m1 = masks_dict[y1]
            m2 = masks_dict[y2]
            if m1 is None or m2 is None:
                matrix[i, j] = np.nan
                continue
            # Restrict to cadastral area
            m1_gt = np.logical_and(m1, gt_mask)
            m2_gt = np.logical_and(m2, gt_mask)
            matrix[i, j] = compute_iou(m1_gt, m2_gt)

    return pd.DataFrame(matrix, index=years, columns=years)


# ---------------------------------------------------------------------------
# F. Post-processing quality
# ---------------------------------------------------------------------------

def compare_raw_vs_filtered(
    raw_mask: np.ndarray,
    filtered_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Triple comparison: raw prediction vs VITO-filtered vs ground truth.

    Returns
    -------
    DataFrame
        Metrics for raw-vs-GT and filtered-vs-GT side by side.
    """
    raw_metrics = compute_pixel_metrics(raw_mask, gt_mask)
    filtered_metrics = compute_pixel_metrics(filtered_mask, gt_mask)

    df = pd.DataFrame({
        'metric': list(raw_metrics.keys()),
        'raw_vs_gt': list(raw_metrics.values()),
        'filtered_vs_gt': list(filtered_metrics.values()),
    })

    # Add improvement column for ratio metrics
    for col in ['precision', 'recall', 'f1', 'iou', 'accuracy', 'kappa']:
        idx = df[df['metric'] == col].index
        if len(idx) > 0:
            i = idx[0]
            df.loc[i, 'improvement'] = df.loc[i, 'filtered_vs_gt'] - df.loc[i, 'raw_vs_gt']

    return df


def compute_segment_purity(
    segments: np.ndarray,
    gt_labeled: np.ndarray,
) -> Dict:
    """
    Measure how well VITO segments align with individual cadastral parcels.

    A segment is "pure" if >80% of its pixels belong to a single parcel.

    Parameters
    ----------
    segments : np.ndarray
        VITO segment labels.
    gt_labeled : np.ndarray
        Rasterized parcel labels (per-parcel unique ID).

    Returns
    -------
    dict
        n_segments, n_pure, purity_rate, mean_max_overlap.
    """
    unique_segs = np.unique(segments)
    n_pure = 0
    max_overlaps = []

    for seg_id in unique_segs:
        seg_mask = segments == seg_id
        gt_in_seg = gt_labeled[seg_mask]

        if gt_in_seg.sum() == 0:
            # Segment is entirely outside any parcel
            continue

        # Find dominant parcel
        parcel_ids, counts = np.unique(gt_in_seg[gt_in_seg > 0], return_counts=True)
        if len(parcel_ids) == 0:
            continue

        max_count = counts.max()
        total_parcel_pixels = (gt_in_seg > 0).sum()
        max_overlap = max_count / max(total_parcel_pixels, 1)
        max_overlaps.append(max_overlap)

        if max_overlap >= 0.8:
            n_pure += 1

    n_evaluated = len(max_overlaps)
    return {
        'n_segments_evaluated': n_evaluated,
        'n_pure': n_pure,
        'purity_rate': n_pure / max(n_evaluated, 1),
        'mean_max_overlap': float(np.mean(max_overlaps)) if max_overlaps else 0.0,
    }


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------

def generate_full_report(
    raw_results: Dict[int, Dict],
    filtered_results: Dict[int, Dict],
    gdf: "gpd.GeoDataFrame",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Generate comprehensive validation report.

    Computes all metric categories (A-F) for every year, for both
    raw and VITO-filtered masks, against cadastral ground truth.

    Parameters
    ----------
    raw_results : dict
        {year: FTW results dict} with 'field_mask'.
    filtered_results : dict
        {year: post-processed dict} with 'field_mask_smooth', 'segments'.
    gdf : GeoDataFrame
        Cadastral parcels (already clipped + reprojected to UTM).
    output_dir : str, optional
        Directory to save CSV outputs.

    Returns
    -------
    dict
        Full report with all metrics organized by category.
    """
    import os

    years = sorted(raw_results.keys())

    # Rasterize ground truth once
    gt_all = rasterize_parcels(gdf)
    gt_land = rasterize_parcels(gdf, parcel_type='T')
    gt_building = rasterize_parcels(gdf, parcel_type='E')
    gt_labeled = rasterize_parcels_labeled(gdf, parcel_type='T')

    report = {
        'pixel_metrics': [],
        'boundary_metrics': [],
        'per_parcel': {},
        'size_accuracy': {},
        'stratified': {},
        'triple_comparison': {},
        'segment_purity': {},
        'temporal_raw': None,
        'temporal_filtered': None,
        'temporal_raw_within_gt': None,
        'temporal_filtered_within_gt': None,
    }

    for year in years:
        raw = raw_results.get(year)
        filt = filtered_results.get(year)
        if raw is None or filt is None:
            continue

        raw_mask = raw['field_mask']
        filt_mask = filt['field_mask_smooth']

        # --- A. Pixel metrics ---
        for mask, label in [(raw_mask, 'raw'), (filt_mask, 'filtered')]:
            pm = compute_pixel_metrics(mask, gt_land)
            pm['year'] = year
            pm['source'] = label
            report['pixel_metrics'].append(pm)

        # --- B. Boundary metrics ---
        for mask, label in [(raw_mask, 'raw'), (filt_mask, 'filtered')]:
            bm = compute_boundary_metrics(mask, gt_land)
            bm['year'] = year
            bm['source'] = label
            report['boundary_metrics'].append(bm)

        # --- C. Per-parcel stats ---
        for mask, label in [(raw_mask, 'raw'), (filt_mask, 'filtered')]:
            pp = compute_per_parcel_stats(mask, gdf, parcel_type='T')
            report['per_parcel'][(year, label)] = pp
            report['size_accuracy'][(year, label)] = compute_size_accuracy(pp)

        # --- D. Stratified ---
        for mask, label in [(raw_mask, 'raw'), (filt_mask, 'filtered')]:
            report['stratified'][(year, label)] = compute_stratified_stats(mask, gdf)

        # --- F. Triple comparison ---
        report['triple_comparison'][year] = compare_raw_vs_filtered(
            raw_mask, filt_mask, gt_land
        )

        # --- Segment purity ---
        if 'segments' in filt:
            report['segment_purity'][year] = compute_segment_purity(
                filt['segments'], gt_labeled
            )

    # --- E. Temporal stability ---
    raw_masks = {y: raw_results[y]['field_mask'] for y in years if raw_results[y]}
    filt_masks = {y: filtered_results[y]['field_mask_smooth'] for y in years if filtered_results[y]}

    from .stability import compute_stability_matrix
    report['temporal_raw'] = compute_stability_matrix(raw_masks)
    report['temporal_filtered'] = compute_stability_matrix(filt_masks)
    report['temporal_raw_within_gt'] = compute_temporal_gt_stats(raw_masks, gt_land)
    report['temporal_filtered_within_gt'] = compute_temporal_gt_stats(filt_masks, gt_land)

    # Convert lists to DataFrames
    report['pixel_metrics'] = pd.DataFrame(report['pixel_metrics'])
    report['boundary_metrics'] = pd.DataFrame(report['boundary_metrics'])

    # Save CSVs if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report['pixel_metrics'].to_csv(
            os.path.join(output_dir, 'pixel_metrics.csv'), index=False
        )
        report['boundary_metrics'].to_csv(
            os.path.join(output_dir, 'boundary_metrics.csv'), index=False
        )
        report['temporal_raw'].to_csv(
            os.path.join(output_dir, 'temporal_iou_raw.csv')
        )
        report['temporal_filtered'].to_csv(
            os.path.join(output_dir, 'temporal_iou_filtered.csv')
        )
        report['temporal_raw_within_gt'].to_csv(
            os.path.join(output_dir, 'temporal_iou_raw_within_gt.csv')
        )
        report['temporal_filtered_within_gt'].to_csv(
            os.path.join(output_dir, 'temporal_iou_filtered_within_gt.csv')
        )
        for year in years:
            if year in report['triple_comparison']:
                report['triple_comparison'][year].to_csv(
                    os.path.join(output_dir, f'triple_comparison_{year}.csv'), index=False
                )

    return report
