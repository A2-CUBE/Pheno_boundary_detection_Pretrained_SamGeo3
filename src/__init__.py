"""
Pheno-Boundary: Dynamic Agricultural Parcel Delineation
========================================================

Modules:
    data_loader: EOPF STAC data access and datacube formation
    preprocessing: Cloud masking, seasonal compositing, FTW input prep
    inference: FTW model wrapper and batch inference
    stability: IoU metrics and change detection
    visualization: Plotting utilities
"""

from .data_loader import (
    reproject_bbox,
    load_single_scene,
    build_datacube,
)

from .preprocessing import (
    apply_cloud_mask,
    create_seasonal_composite,
    prepare_ftw_input,
    compute_ndvi,
    compute_temporal_statistics,
)

from .inference import FTWInference, run_multi_year_inference

from .stability import (
    compute_iou,
    compute_stability_matrix,
    detect_boundary_changes,
    compute_change_statistics,
    multi_year_change_analysis,
    classify_stability_zones,
)

from .visualization import (
    plot_stability_matrix,
    plot_multi_year_masks,
    plot_change_map,
    plot_stability_zones,
)

from .vectorization import (
    find_most_stable_year,
    vectorize_segments,
    create_interactive_map,
    run_vectorization_pipeline,
)

__version__ = "0.1.0"
__author__ = "Ahmad Ahmad Abubakar, Ousama Bin Zamir"
