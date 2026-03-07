"""
Post-processing utilities for FTW inference outputs.

Implements VITO-style graph-based region segmentation to clean up
raw neural network predictions into smooth, region-consistent field masks.

Pipeline: sobel edges -> felzenszwalb superpixels -> RAG boundary ->
          graph cut merge -> confidence mask -> per-segment field voting
"""

import numpy as np
from typing import Dict, Optional, Tuple


def apply_vito_filter(
    field_prob: np.ndarray,
    scale: float = 120,
    sigma: float = 0.0,
    min_size: int = 30,
    merge_threshold: float = 0.15,
    confidence_threshold: float = 0.3,
) -> Dict:
    """
    Apply VITO-style graph-based region segmentation on field probability surface.

    Adapted from VITO's parcel_delineation_utils.apply_filter(). Instead of
    morphological operations, this uses superpixel segmentation and region
    adjacency graph merging to produce clean, region-consistent parcels.

    Parameters
    ----------
    field_prob : np.ndarray
        2D array (H, W) of field probability values in [0, 1].
    scale : float
        Felzenszwalb scale parameter. Higher = fewer, larger segments.
    sigma : float
        Gaussian smoothing width before segmentation. 0 = no smoothing.
    min_size : int
        Minimum segment size in pixels. At 10m resolution, 30px = 0.3 ha.
    merge_threshold : float
        RAG graph cut threshold. Adjacent segments with boundary weight
        below this value are merged.
    confidence_threshold : float
        Pixels with field_prob below this are masked out.

    Returns
    -------
    dict
        'segments': Labeled segment array (int, H x W)
        'field_mask_smooth': Clean binary field mask (bool, H x W)
        'n_segments': Number of unique segments after merging
        'masked_segments': Segments with low-confidence areas set to NaN
    """
    from skimage.filters import sobel
    from skimage import segmentation, graph

    # Step 1: Edge detection on probability surface
    edges = sobel(field_prob)

    # Step 2: Felzenszwalb superpixel segmentation
    segments = segmentation.felzenszwalb(
        field_prob, scale=scale, sigma=sigma, min_size=min_size, channel_axis=None
    )

    # Step 3: Build Region Adjacency Graph weighted by boundary strength
    bgraph = graph.rag_boundary(segments, edges)

    # Step 4: Merge segments with weak boundaries
    merged_segments = graph.cut_threshold(segments, bgraph, merge_threshold, in_place=False)

    # Step 5: Confidence masking — set low-probability regions to background
    masked_segments = merged_segments.astype(float)
    masked_segments[field_prob < confidence_threshold] = np.nan

    # Step 6: Convert segments to clean binary field mask
    field_mask_smooth = segments_to_field_mask(merged_segments, field_prob)

    # Apply confidence mask to the binary output too
    field_mask_smooth[field_prob < confidence_threshold] = False

    n_segments = len(np.unique(merged_segments))

    return {
        'segments': merged_segments,
        'field_mask_smooth': field_mask_smooth,
        'n_segments': n_segments,
        'masked_segments': masked_segments,
    }


def segments_to_field_mask(
    segments: np.ndarray,
    field_prob: np.ndarray,
    segment_threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert labeled segments to a binary field mask using per-segment voting.

    For each segment, computes the mean field probability. If the mean
    exceeds the threshold, the entire segment is classified as field.
    This produces region-consistent masks without pixel-level noise.

    Parameters
    ----------
    segments : np.ndarray
        Labeled segment array (int, H x W) from felzenszwalb + graph cut.
    field_prob : np.ndarray
        Field probability surface (float, H x W) in [0, 1].
    segment_threshold : float
        Mean field_prob threshold for classifying a segment as field.

    Returns
    -------
    np.ndarray
        Binary field mask (bool, H x W).
    """
    field_mask = np.zeros_like(field_prob, dtype=bool)
    unique_segments = np.unique(segments)

    for seg_id in unique_segments:
        seg_pixels = segments == seg_id
        mean_prob = field_prob[seg_pixels].mean()
        if mean_prob > segment_threshold:
            field_mask[seg_pixels] = True

    return field_mask


def postprocess_results(
    results_dict: Dict,
    scale: float = 120,
    sigma: float = 0.0,
    min_size: int = 30,
    merge_threshold: float = 0.15,
    confidence_threshold: float = 0.3,
    segment_threshold: float = 0.5,
) -> Dict:
    """
    Post-process a single year's FTW inference results through VITO filter.

    Parameters
    ----------
    results_dict : dict
        Single year's FTW output with keys: 'field_prob', 'boundary_prob',
        'class_map', 'field_mask', 'raw_probs'.
    scale, sigma, min_size, merge_threshold, confidence_threshold :
        VITO filter parameters (see apply_vito_filter).
    segment_threshold : float
        Per-segment field classification threshold.

    Returns
    -------
    dict
        Original results plus:
        'field_mask_smooth': Cleaned binary field mask
        'segments': Labeled segment array
        'n_segments': Segment count
    """
    field_prob = results_dict['field_prob']

    filter_result = apply_vito_filter(
        field_prob,
        scale=scale,
        sigma=sigma,
        min_size=min_size,
        merge_threshold=merge_threshold,
        confidence_threshold=confidence_threshold,
    )

    # Override the segment-level mask with custom threshold
    filter_result['field_mask_smooth'] = segments_to_field_mask(
        filter_result['segments'], field_prob, segment_threshold
    )
    # Re-apply confidence mask
    filter_result['field_mask_smooth'][field_prob < confidence_threshold] = False

    # Merge with original results
    output = dict(results_dict)
    output['field_mask_smooth'] = filter_result['field_mask_smooth']
    output['segments'] = filter_result['segments']
    output['n_segments'] = filter_result['n_segments']
    output['masked_segments'] = filter_result['masked_segments']

    return output


def postprocess_all_years(
    all_results: Dict[int, Dict],
    **filter_params,
) -> Dict[int, Dict]:
    """
    Apply VITO post-processing to all years' FTW results.

    Parameters
    ----------
    all_results : dict
        {year: results_dict} from FTW inference.
    **filter_params :
        Parameters forwarded to postprocess_results().

    Returns
    -------
    dict
        {year: postprocessed_results_dict} with added smooth masks.
    """
    processed = {}

    for year in sorted(all_results.keys()):
        results = all_results[year]
        if results is None:
            print(f"  {year}: Skipped (no results)")
            processed[year] = None
            continue

        pp = postprocess_results(results, **filter_params)
        raw_pct = 100 * results['field_mask'].sum() / results['field_mask'].size
        smooth_pct = 100 * pp['field_mask_smooth'].sum() / pp['field_mask_smooth'].size
        print(
            f"  {year}: {pp['n_segments']} segments | "
            f"raw {raw_pct:.1f}% -> smooth {smooth_pct:.1f}% field"
        )
        processed[year] = pp

    return processed
