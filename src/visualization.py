"""
Visualization utilities for Pheno-Boundary project.

This module handles:
- Stability matrix heatmaps
- Multi-year mask grids
- Change detection maps
- Temporal stability classification visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


# Default color schemes
STABILITY_CMAP = ListedColormap(['#808080', '#228B22', '#FFD700'])  # Gray, Green, Gold
CHANGE_COLORS = {
    'stable_field': '#228B22',    # Forest green
    'added': '#1E90FF',           # Dodger blue
    'removed': '#DC143C',         # Crimson
    'stable_nonfield': '#2F2F2F', # Dark gray
}


def plot_stability_matrix(
    stability_df: pd.DataFrame,
    title: str = "Field Boundary Stability (IoU) Across Years",
    cmap: str = 'RdYlGn',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot IoU stability matrix as a heatmap.
    
    Parameters
    ----------
    stability_df : pd.DataFrame
        Pairwise IoU matrix with years as index/columns
    title : str
        Plot title
    cmap : str
        Matplotlib colormap name
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        stability_df,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'IoU Score'},
        square=True,
        linewidths=0.5
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Year', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_multi_year_masks(
    masks_dict: Dict[int, np.ndarray],
    title: str = "Field Masks by Year",
    cmap: str = 'Greens',
    output_path: Optional[str] = None,
    max_cols: int = 4
) -> plt.Figure:
    """
    Plot field masks for all years in a grid.
    
    Parameters
    ----------
    masks_dict : dict
        {year: field_mask_array}
    title : str
        Overall figure title
    cmap : str
        Colormap for masks
    output_path : str, optional
        Path to save figure
    max_cols : int
        Maximum columns in grid
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    years = sorted([y for y in masks_dict.keys() if masks_dict[y] is not None])
    n = len(years)
    
    if n == 0:
        raise ValueError("No valid masks to plot")
    
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single subplot case
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, year in enumerate(years):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        mask = masks_dict[year]
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'{year}', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Add field coverage percentage
        if mask.dtype == bool or mask.max() <= 1:
            coverage = 100 * (mask > 0.5).sum() / mask.size
            ax.text(0.02, 0.98, f'{coverage:.1f}%', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_change_map(
    changes_dict: Dict[str, np.ndarray],
    year1: int,
    year2: int,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Visualize boundary changes between two years.
    
    Parameters
    ----------
    changes_dict : dict
        Output from detect_boundary_changes()
    year1, year2 : int
        Years being compared
    title : str, optional
        Custom title
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create RGB change map
    h, w = changes_dict['changed'].shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color assignments (RGB values)
    colors = {
        'stable_field': (34, 139, 34),      # Forest green
        'added': (30, 144, 255),            # Dodger blue
        'removed': (220, 20, 60),           # Crimson
        'stable_nonfield': (47, 47, 47),    # Dark gray
    }
    
    # Apply colors
    change_map[changes_dict['stable_field']] = colors['stable_field']
    change_map[changes_dict['added']] = colors['added']
    change_map[changes_dict['removed']] = colors['removed']
    change_map[changes_dict['stable_nonfield']] = colors['stable_nonfield']
    
    ax.imshow(change_map)
    
    if title is None:
        title = f'Field Boundary Changes: {year1} → {year2}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#228B22', label='Stable Field'),
        mpatches.Patch(color='#1E90FF', label='Field Added'),
        mpatches.Patch(color='#DC143C', label='Field Removed'),
        mpatches.Patch(color='#2F2F2F', label='Stable Non-field'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Add statistics annotation
    total = changes_dict['changed'].size
    stats_text = (
        f"Changed: {100*changes_dict['changed'].sum()/total:.1f}%\n"
        f"Added: {100*changes_dict['added'].sum()/total:.1f}%\n"
        f"Removed: {100*changes_dict['removed'].sum()/total:.1f}%"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_stability_zones(
    classification: np.ndarray,
    frequency: np.ndarray,
    title: str = "Temporal Stability Analysis",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualize temporal stability classification.
    
    Parameters
    ----------
    classification : np.ndarray
        Stability classification map (0=never, 1=stable, 2=unstable)
    frequency : np.ndarray
        Field presence frequency map [0, 1]
    title : str
        Figure title
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Stability classification
    ax1 = axes[0]
    cmap_class = ListedColormap(['#808080', '#228B22', '#FFD700'])
    im1 = ax1.imshow(classification, cmap=cmap_class, vmin=0, vmax=2)
    ax1.set_title('Stability Classification', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Class legend
    legend_elements = [
        mpatches.Patch(color='#808080', label='Never Field'),
        mpatches.Patch(color='#228B22', label='Stable Field'),
        mpatches.Patch(color='#FFD700', label='Unstable/Changing'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Add class percentages
    total = classification.size
    pct_text = (
        f"Never: {100*(classification==0).sum()/total:.1f}%\n"
        f"Stable: {100*(classification==1).sum()/total:.1f}%\n"
        f"Unstable: {100*(classification==2).sum()/total:.1f}%"
    )
    ax1.text(0.02, 0.02, pct_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: Field frequency
    ax2 = axes[1]
    im2 = ax2.imshow(frequency, cmap='YlGn', vmin=0, vmax=1)
    ax2.set_title('Field Presence Frequency', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Fraction of Years', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_change_timeline(
    change_df: pd.DataFrame,
    title: str = "Year-to-Year Change Statistics",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot change statistics over time as a bar chart.
    
    Parameters
    ----------
    change_df : pd.DataFrame
        Output from multi_year_change_analysis()
    title : str
        Figure title
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    if len(change_df) == 0:
        raise ValueError("Empty change dataframe")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(change_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, change_df['added_pct'], width, 
                   label='Field Added', color='#1E90FF')
    bars2 = ax.bar(x + width/2, change_df['removed_pct'], width,
                   label='Field Removed', color='#DC143C')
    
    ax.set_xlabel('Transition', fontsize=10)
    ax.set_ylabel('Percentage of Total Area', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(change_df['transition'], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add stability line
    ax2 = ax.twinx()
    ax2.plot(x, change_df['stability_pct'], 'go-', linewidth=2, 
             markersize=8, label='Stability %')
    ax2.set_ylabel('Stability (%)', fontsize=10, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_rgb_composite(
    datacube,
    time_index: int = 0,
    title: str = "RGB Composite",
    vmin: float = 0,
    vmax: float = 0.25,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot RGB composite from datacube.
    
    Parameters
    ----------
    datacube : xr.Dataset
        Dataset with b04, b03, b02 bands
    time_index : int
        Time index to visualize
    title : str
        Figure title
    vmin, vmax : float
        Value range for normalization
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    import xarray as xr
    
    # Select time slice
    if 'time' in datacube.dims:
        rgb = datacube.isel(time=time_index)[['b04', 'b03', 'b02']]
    else:
        rgb = datacube[['b04', 'b03', 'b02']]
    
    # Normalize
    rgb_norm = ((rgb - vmin) / (vmax - vmin)).clip(0, 1)
    
    # Convert to image array
    img = rgb_norm.to_array().transpose('y', 'x', 'variable').values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(
        img,
        extent=[
            float(rgb.x.values.min()),
            float(rgb.x.values.max()),
            float(rgb.y.values.min()),
            float(rgb.y.values.max()),
        ]
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_summary_figure(
    masks_dict: Dict[int, np.ndarray],
    stability_df: pd.DataFrame,
    classification: np.ndarray,
    frequency: np.ndarray,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary figure with all key visualizations.
    
    Parameters
    ----------
    masks_dict : dict
        {year: field_mask}
    stability_df : pd.DataFrame
        IoU stability matrix
    classification : np.ndarray
        Stability classification map
    frequency : np.ndarray
        Field frequency map
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        The generated figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2 rows
    # Row 1: Stability matrix (left), Stability zones (right)
    # Row 2: All year masks
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    # Panel 1: Stability matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(stability_df, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'IoU'})
    ax1.set_title('Boundary Stability Matrix', fontweight='bold')
    
    # Panel 2: Stability zones
    ax2 = fig.add_subplot(gs[0, 1])
    cmap_class = ListedColormap(['#808080', '#228B22', '#FFD700'])
    ax2.imshow(classification, cmap=cmap_class)
    ax2.set_title('Stability Classification', fontweight='bold')
    ax2.axis('off')
    legend_elements = [
        mpatches.Patch(color='#808080', label='Never'),
        mpatches.Patch(color='#228B22', label='Stable'),
        mpatches.Patch(color='#FFD700', label='Changing'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Panel 3: Year masks (spanning bottom row)
    years = sorted([y for y in masks_dict.keys() if masks_dict[y] is not None])
    n_years = len(years)
    
    gs_bottom = gs[1, :].subgridspec(1, n_years, wspace=0.05)
    
    for idx, year in enumerate(years):
        ax = fig.add_subplot(gs_bottom[idx])
        ax.imshow(masks_dict[year], cmap='Greens', vmin=0, vmax=1)
        ax.set_title(str(year), fontsize=10)
        ax.axis('off')
    
    fig.suptitle('Pheno-Boundary: Field Stability Analysis Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Post-processing and ground truth comparison plots
# ---------------------------------------------------------------------------

def plot_raw_vs_filtered(
    raw_mask: np.ndarray,
    filtered_mask: np.ndarray,
    year: int,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Side-by-side comparison of raw field mask vs VITO-filtered mask.

    Parameters
    ----------
    raw_mask : np.ndarray
        Raw binary field mask from FTW inference.
    filtered_mask : np.ndarray
        Post-processed mask from VITO filter.
    year : int
        Year label.
    output_path : str, optional
        Path to save figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Raw mask
    axes[0].imshow(raw_mask, cmap='Greens', vmin=0, vmax=1)
    axes[0].set_title(f'{year} Raw Mask', fontweight='bold')
    raw_pct = 100 * raw_mask.astype(bool).sum() / raw_mask.size
    axes[0].text(0.02, 0.98, f'{raw_pct:.1f}%', transform=axes[0].transAxes,
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].axis('off')

    # Filtered mask
    axes[1].imshow(filtered_mask, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title(f'{year} VITO Filtered', fontweight='bold')
    filt_pct = 100 * filtered_mask.astype(bool).sum() / filtered_mask.size
    axes[1].text(0.02, 0.98, f'{filt_pct:.1f}%', transform=axes[1].transAxes,
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].axis('off')

    # Difference map
    diff = np.zeros((*raw_mask.shape, 3), dtype=np.uint8)
    raw_b = raw_mask.astype(bool)
    filt_b = filtered_mask.astype(bool)
    both = np.logical_and(raw_b, filt_b)
    raw_only = np.logical_and(raw_b, ~filt_b)
    filt_only = np.logical_and(~raw_b, filt_b)

    diff[both] = [34, 139, 34]       # Green: in both
    diff[raw_only] = [220, 20, 60]   # Red: raw only (removed by filter)
    diff[filt_only] = [30, 144, 255] # Blue: filter only (added by filter)

    axes[2].imshow(diff)
    axes[2].set_title(f'{year} Difference', fontweight='bold')
    axes[2].axis('off')

    legend_elements = [
        mpatches.Patch(color='#228B22', label='Both'),
        mpatches.Patch(color='#DC143C', label='Raw only'),
        mpatches.Patch(color='#1E90FF', label='Filtered only'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_prediction_vs_cadastre(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    year: int,
    label: str = "Prediction",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Overlay prediction on ground truth cadastral raster.

    Shows TP (green), FP (red), FN (blue), TN (dark gray).

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask.
    gt_mask : np.ndarray
        Ground truth binary mask (rasterized cadastre).
    year : int
        Year label.
    label : str
        Prediction label (e.g. "Raw" or "Filtered").
    output_path : str, optional
        Path to save figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Prediction
    axes[0].imshow(pred_mask, cmap='Greens', vmin=0, vmax=1)
    axes[0].set_title(f'{year} {label}', fontweight='bold')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(gt_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(f'{year} Cadastre', fontweight='bold')
    axes[1].axis('off')

    # Confusion overlay
    confusion = np.zeros((*pred.shape, 3), dtype=np.uint8)
    confusion[tp] = [34, 139, 34]     # Green: TP
    confusion[fp] = [220, 20, 60]     # Red: FP
    confusion[fn] = [30, 144, 255]    # Blue: FN
    confusion[~pred & ~gt] = [40, 40, 40]  # Dark gray: TN

    axes[2].imshow(confusion)
    axes[2].set_title(f'{year} Confusion', fontweight='bold')
    axes[2].axis('off')

    legend_elements = [
        mpatches.Patch(color='#228B22', label='TP'),
        mpatches.Patch(color='#DC143C', label='FP'),
        mpatches.Patch(color='#1E90FF', label='FN'),
        mpatches.Patch(color='#282828', label='TN'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_accuracy_by_size(
    per_parcel_df: "pd.DataFrame",
    title: str = "Detection Accuracy by Parcel Size",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Scatter + box plot of per-parcel IoU vs parcel area.

    Parameters
    ----------
    per_parcel_df : DataFrame
        Output from compute_per_parcel_stats().
    title : str
        Figure title.
    output_path : str, optional
        Path to save figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    df = per_parcel_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter: area vs IoU
    ax1 = axes[0]
    ax1.scatter(df['area_m2'], df['iou'], alpha=0.4, s=15, c='#228B22')
    ax1.set_xscale('log')
    ax1.set_xlabel('Parcel Area (m²)', fontsize=10)
    ax1.set_ylabel('IoU', fontsize=10)
    ax1.set_title('Per-Parcel IoU vs Area', fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    # Box plot by size class
    bins = [0, 500, 5000, 20000, float('inf')]
    labels = ['<500', '500-5k', '5k-20k', '>20k']
    df['size_class'] = pd.cut(df['area_m2'], bins=bins, labels=labels, right=False)

    ax2 = axes[1]
    df.boxplot(column='iou', by='size_class', ax=ax2)
    ax2.set_xlabel('Parcel Size Class (m²)', fontsize=10)
    ax2.set_ylabel('IoU', fontsize=10)
    ax2.set_title('IoU Distribution by Size', fontweight='bold')
    fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_triple_comparison(
    raw_mask: np.ndarray,
    filtered_mask: np.ndarray,
    gt_mask: np.ndarray,
    year: int,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    3-panel comparison: raw prediction / VITO-filtered / ground truth.

    Parameters
    ----------
    raw_mask : np.ndarray
        Raw FTW field mask.
    filtered_mask : np.ndarray
        VITO-filtered field mask.
    gt_mask : np.ndarray
        Rasterized cadastral ground truth.
    year : int
        Year label.
    output_path : str, optional
        Path to save figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, mask, label, cmap in [
        (axes[0], raw_mask, 'Raw FTW', 'Greens'),
        (axes[1], filtered_mask, 'VITO Filtered', 'Greens'),
        (axes[2], gt_mask, 'Cadastre GT', 'Blues'),
    ]:
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'{year} {label}', fontweight='bold')
        ax.axis('off')

        pct = 100 * mask.astype(bool).sum() / mask.size
        ax.text(0.02, 0.98, f'{pct:.1f}%', transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'{year}: Raw vs Filtered vs Ground Truth', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig
