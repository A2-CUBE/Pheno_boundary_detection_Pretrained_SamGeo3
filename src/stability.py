"""
Stability analysis utilities.

This module handles:
- IoU computation between masks
- Pairwise stability matrix generation
- Change detection between years
- Temporal stability classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary masks (boolean or 0/1)
    
    Returns
    -------
    float
        IoU score in [0, 1]
    """
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        # Both masks are empty - consider as perfect match
        return 1.0
    
    return float(intersection / union)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.
    
    Dice = 2|A ∩ B| / (|A| + |B|)
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary masks
    
    Returns
    -------
    float
        Dice score in [0, 1]
    """
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 1.0
    
    return float(2 * intersection / total)


def compute_stability_matrix(
    masks_dict: Dict[int, np.ndarray],
    metric: str = 'iou'
) -> pd.DataFrame:
    """
    Compute pairwise similarity matrix for all years.
    
    Parameters
    ----------
    masks_dict : dict
        {year: binary_mask} for each year
    metric : str
        Similarity metric: 'iou' or 'dice'
    
    Returns
    -------
    pd.DataFrame
        Symmetric similarity matrix with years as index/columns
    """
    metric_fn = compute_iou if metric == 'iou' else compute_dice
    
    years = sorted(masks_dict.keys())
    n = len(years)
    matrix = np.zeros((n, n))
    
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            if masks_dict[y1] is None or masks_dict[y2] is None:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = metric_fn(masks_dict[y1], masks_dict[y2])
    
    return pd.DataFrame(matrix, index=years, columns=years)


def compute_boundary_stability(
    boundaries_dict: Dict[int, np.ndarray],
    dilation_radius: int = 0
) -> pd.DataFrame:
    """
    Compute stability specifically for boundary predictions.
    
    Boundaries are thin structures, so optional dilation can be applied
    to make matching more robust.
    
    Parameters
    ----------
    boundaries_dict : dict
        {year: boundary_probability_map}
    dilation_radius : int
        Radius for binary dilation (0 = no dilation)
    
    Returns
    -------
    pd.DataFrame
        Boundary stability matrix
    """
    from scipy import ndimage
    
    # Threshold and optionally dilate boundaries
    processed = {}
    for year, boundary in boundaries_dict.items():
        if boundary is None:
            processed[year] = None
            continue
        
        binary = boundary > 0.5
        
        if dilation_radius > 0:
            struct = ndimage.generate_binary_structure(2, 1)
            binary = ndimage.binary_dilation(binary, struct, iterations=dilation_radius)
        
        processed[year] = binary
    
    return compute_stability_matrix(processed)


def detect_boundary_changes(
    mask_prev: np.ndarray,
    mask_curr: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Identify pixels where field coverage changed between two years.
    
    Parameters
    ----------
    mask_prev : np.ndarray
        Previous year's field probability or binary mask
    mask_curr : np.ndarray
        Current year's field probability or binary mask
    threshold : float
        Probability threshold for binarization
    
    Returns
    -------
    dict
        'added': New field areas (was non-field, now field)
        'removed': Lost field areas (was field, now non-field)
        'stable_field': Consistently field
        'stable_nonfield': Consistently non-field
        'changed': Any change (union of added and removed)
    """
    prev = mask_prev > threshold if mask_prev.dtype != bool else mask_prev
    curr = mask_curr > threshold if mask_curr.dtype != bool else mask_curr
    
    return {
        'added': np.logical_and(~prev, curr),           # Field expansion
        'removed': np.logical_and(prev, ~curr),         # Field contraction
        'stable_field': np.logical_and(prev, curr),     # Persistent field
        'stable_nonfield': np.logical_and(~prev, ~curr),# Persistent non-field
        'changed': prev != curr,                        # Any change
    }


def compute_change_statistics(changes_dict: Dict[str, np.ndarray]) -> Dict:
    """
    Summarize change detection results as statistics.
    
    Parameters
    ----------
    changes_dict : dict
        Output from detect_boundary_changes()
    
    Returns
    -------
    dict
        Change statistics including pixel counts and percentages
    """
    total_pixels = changes_dict['changed'].size
    
    stats = {
        'total_pixels': int(total_pixels),
        'added_pixels': int(changes_dict['added'].sum()),
        'removed_pixels': int(changes_dict['removed'].sum()),
        'stable_field_pixels': int(changes_dict['stable_field'].sum()),
        'stable_nonfield_pixels': int(changes_dict['stable_nonfield'].sum()),
        'changed_pixels': int(changes_dict['changed'].sum()),
        
        'added_pct': 100 * changes_dict['added'].sum() / total_pixels,
        'removed_pct': 100 * changes_dict['removed'].sum() / total_pixels,
        'stable_field_pct': 100 * changes_dict['stable_field'].sum() / total_pixels,
        'stable_nonfield_pct': 100 * changes_dict['stable_nonfield'].sum() / total_pixels,
        'changed_pct': 100 * changes_dict['changed'].sum() / total_pixels,
        'stability_pct': 100 * (1 - changes_dict['changed'].sum() / total_pixels),
    }
    
    # Net change
    stats['net_change_pixels'] = stats['added_pixels'] - stats['removed_pixels']
    stats['net_change_pct'] = stats['added_pct'] - stats['removed_pct']
    
    return stats


def multi_year_change_analysis(
    masks_dict: Dict[int, np.ndarray],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Analyze changes between all consecutive year pairs.
    
    Parameters
    ----------
    masks_dict : dict
        {year: field_mask_or_probability}
    threshold : float
        Probability threshold for binarization
    
    Returns
    -------
    pd.DataFrame
        Change statistics for each year-to-year transition
    """
    years = sorted(masks_dict.keys())
    transitions = []
    
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        
        if masks_dict[y1] is None or masks_dict[y2] is None:
            continue
        
        changes = detect_boundary_changes(
            masks_dict[y1], 
            masks_dict[y2], 
            threshold
        )
        stats = compute_change_statistics(changes)
        stats['from_year'] = y1
        stats['to_year'] = y2
        stats['transition'] = f"{y1}-{y2}"
        
        transitions.append(stats)
    
    if not transitions:
        return pd.DataFrame()
    
    return pd.DataFrame(transitions)


def classify_stability_zones(
    masks_dict: Dict[int, np.ndarray],
    stable_threshold: float = 0.8,
    field_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each pixel by temporal stability across all years.
    
    Parameters
    ----------
    masks_dict : dict
        {year: field_mask_or_probability}
    stable_threshold : float
        Fraction of years required for 'stable' classification
    field_threshold : float
        Probability threshold for field classification
    
    Returns
    -------
    tuple
        (classification_map, frequency_map)
        
        classification_map values:
            0 = Never field (stable non-field)
            1 = Always field (stable field)
            2 = Sometimes field (unstable/changing)
        
        frequency_map:
            Fraction of years each pixel was classified as field
    """
    years = sorted(masks_dict.keys())
    valid_masks = [masks_dict[y] for y in years if masks_dict[y] is not None]
    
    if not valid_masks:
        raise ValueError("No valid masks found")
    
    n_years = len(valid_masks)
    
    # Stack and binarize
    stack = np.stack([
        (m > field_threshold if m.dtype != bool else m).astype(float) 
        for m in valid_masks
    ], axis=0)
    
    # Compute frequency (fraction of years as field)
    frequency = stack.mean(axis=0)
    
    # Classify
    classification = np.zeros_like(frequency, dtype=np.uint8)
    
    # Stable field: field in >= stable_threshold of years
    classification[frequency >= stable_threshold] = 1
    
    # Unstable: sometimes field but not stable
    unstable = (frequency > (1 - stable_threshold)) & (frequency < stable_threshold)
    classification[unstable] = 2
    
    # Class 0 (never/rarely field) remains as default
    
    return classification, frequency


def compute_persistence_index(
    masks_dict: Dict[int, np.ndarray],
    field_threshold: float = 0.5
) -> np.ndarray:
    """
    Compute a continuous persistence index for each pixel.
    
    Higher values indicate more consistent field presence.
    
    Parameters
    ----------
    masks_dict : dict
        {year: field_mask_or_probability}
    field_threshold : float
        Probability threshold
    
    Returns
    -------
    np.ndarray
        Persistence index in [0, 1]
    """
    years = sorted(masks_dict.keys())
    valid_masks = [masks_dict[y] for y in years if masks_dict[y] is not None]
    
    if not valid_masks:
        raise ValueError("No valid masks found")
    
    # Stack and binarize
    stack = np.stack([
        (m > field_threshold if m.dtype != bool else m).astype(float) 
        for m in valid_masks
    ], axis=0)
    
    # Simple persistence: fraction of years as field
    frequency = stack.mean(axis=0)
    
    # Transform to persistence (max at 0.5 indicates instability)
    # Use distance from 0.5 to indicate stability
    persistence = 2 * np.abs(frequency - 0.5)
    
    return persistence


def summarize_stability(
    stability_matrix: pd.DataFrame,
    change_df: pd.DataFrame = None
) -> Dict:
    """
    Generate summary statistics for stability analysis.
    
    Parameters
    ----------
    stability_matrix : pd.DataFrame
        Pairwise IoU matrix
    change_df : pd.DataFrame, optional
        Year-to-year change statistics
    
    Returns
    -------
    dict
        Summary statistics
    """
    # Extract off-diagonal elements (year-to-year comparisons)
    values = stability_matrix.values
    n = len(values)
    off_diag = values[~np.eye(n, dtype=bool)]
    off_diag = off_diag[~np.isnan(off_diag)]
    
    summary = {
        'n_years': n,
        'mean_iou': float(np.mean(off_diag)) if len(off_diag) > 0 else np.nan,
        'min_iou': float(np.min(off_diag)) if len(off_diag) > 0 else np.nan,
        'max_iou': float(np.max(off_diag)) if len(off_diag) > 0 else np.nan,
        'std_iou': float(np.std(off_diag)) if len(off_diag) > 0 else np.nan,
    }
    
    # Add interpretation
    mean_iou = summary['mean_iou']
    if mean_iou >= 0.9:
        summary['interpretation'] = "Very stable boundaries"
    elif mean_iou >= 0.75:
        summary['interpretation'] = "Stable boundaries with minor changes"
    elif mean_iou >= 0.5:
        summary['interpretation'] = "Moderate boundary changes"
    else:
        summary['interpretation'] = "Significant boundary instability"
    
    if change_df is not None and len(change_df) > 0:
        summary['mean_change_pct'] = float(change_df['changed_pct'].mean())
        summary['max_change_pct'] = float(change_df['changed_pct'].max())
        summary['total_added_pct'] = float(change_df['added_pct'].sum())
        summary['total_removed_pct'] = float(change_df['removed_pct'].sum())
    
    return summary
