"""
Preprocessing utilities for Sentinel-2 data.

This module handles:
- Cloud masking using Scene Classification Layer
- Seasonal composite generation
- FTW model input preparation
- NDVI and temporal statistics computation
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Optional, Tuple
import pandas as pd


# SCL classification codes for invalid pixels
SCL_INVALID = [0, 1, 3, 7, 8, 9, 10]
# 0: NO_DATA
# 1: SATURATED_DEFECTIVE
# 3: CLOUD_SHADOW
# 7: CLOUD_LOW_PROBABILITY
# 8: CLOUD_MEDIUM_PROBABILITY
# 9: CLOUD_HIGH_PROBABILITY
# 10: THIN_CIRRUS


def apply_cloud_mask(
    datacube: xr.Dataset,
    scl_var: str = 'scl',
    invalid_codes: List[int] = None
) -> xr.Dataset:
    """
    Mask invalid pixels using Scene Classification Layer.
    
    Parameters
    ----------
    datacube : xr.Dataset
        Input datacube with SCL variable
    scl_var : str
        Name of SCL variable in dataset
    invalid_codes : list
        SCL codes to mask. Default uses standard cloud/shadow codes.
    
    Returns
    -------
    xr.Dataset
        Cloud-masked datacube (invalid pixels = NaN)
    """
    if invalid_codes is None:
        invalid_codes = SCL_INVALID
    
    if scl_var not in datacube:
        print(f"Warning: {scl_var} not found in datacube. Returning unmasked data.")
        return datacube
    
    valid_mask = ~datacube[scl_var].isin(invalid_codes)
    masked = datacube.where(valid_mask)
    
    return masked


def assign_season(month: int) -> str:
    """Map month number to meteorological season name."""
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'


def get_season_months(season: str) -> List[int]:
    """Get month numbers for a season."""
    season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }
    return season_months.get(season, [])


def create_seasonal_composite(
    datacube: xr.Dataset,
    year: int,
    season: str,
    method: str = 'median',
    min_observations: int = 1
) -> xr.Dataset:
    """
    Create cloud-free composite for a specific year and season.
    
    Parameters
    ----------
    datacube : xr.Dataset
        Multi-temporal datacube (should be cloud-masked)
    year : int
        Target year
    season : str
        One of 'spring', 'summer', 'autumn', 'winter'
    method : str
        Aggregation method ('median', 'mean', 'max')
    min_observations : int
        Minimum number of valid observations required
    
    Returns
    -------
    xr.Dataset
        Seasonal composite with metadata
    
    Raises
    ------
    ValueError
        If insufficient data for the specified year/season
    """
    months = get_season_months(season)
    
    # Handle winter crossing year boundary
    if season == 'winter':
        time_mask = (
            ((datacube.time.dt.year == year) & (datacube.time.dt.month == 12)) |
            ((datacube.time.dt.year == year + 1) & (datacube.time.dt.month.isin([1, 2])))
        )
    else:
        time_mask = (
            (datacube.time.dt.year == year) & 
            (datacube.time.dt.month.isin(months))
        )
    
    subset = datacube.where(time_mask, drop=True)
    
    n_obs = len(subset.time) if hasattr(subset, 'time') else 0
    
    if n_obs < min_observations:
        raise ValueError(
            f"Insufficient data for {year} {season}: "
            f"found {n_obs} observations, need {min_observations}"
        )
    
    # Compute composite
    if method == 'median':
        composite = subset.median(dim='time', skipna=True)
    elif method == 'mean':
        composite = subset.mean(dim='time', skipna=True)
    elif method == 'max':
        composite = subset.max(dim='time', skipna=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add metadata
    composite.attrs['year'] = year
    composite.attrs['season'] = season
    composite.attrs['method'] = method
    composite.attrs['n_observations'] = n_obs
    
    return composite


def prepare_ftw_input(
    datacube: xr.Dataset,
    year: int,
    spring_months: List[int] = [3, 4, 5],
    summer_months: List[int] = [7, 8, 9],
    normalize_max: float = 0.5
) -> np.ndarray:
    """
    Prepare 8-channel input tensor for FTW model.
    
    FTW expects bi-temporal input:
    - Channels 0-3: B02, B03, B04, B08 from planting window (spring)
    - Channels 4-7: B02, B03, B04, B08 from harvest window (summer)
    
    Parameters
    ----------
    datacube : xr.Dataset
        Cloud-masked multi-temporal datacube
    year : int
        Target year
    spring_months : list
        Months for spring/planting composite
    summer_months : list
        Months for summer/harvest composite
    normalize_max : float
        Maximum reflectance value for normalization
    
    Returns
    -------
    np.ndarray
        Shape (8, H, W) normalized to [0, 1]
    """
    # Create spring composite
    spring_mask = (
        (datacube.time.dt.year == year) & 
        (datacube.time.dt.month.isin(spring_months))
    )
    spring_data = datacube.where(spring_mask, drop=True)
    spring = spring_data.median(dim='time', skipna=True)
    
    # Create summer composite
    summer_mask = (
        (datacube.time.dt.year == year) & 
        (datacube.time.dt.month.isin(summer_months))
    )
    summer_data = datacube.where(summer_mask, drop=True)
    summer = summer_data.median(dim='time', skipna=True)
    
    # Extract bands in FTW order
    bands = ['b02', 'b03', 'b04', 'b08']
    
    spring_stack = np.stack([spring[b].values for b in bands], axis=0)
    summer_stack = np.stack([summer[b].values for b in bands], axis=0)
    
    # Concatenate: [B02_spring, B03_spring, B04_spring, B08_spring,
    #               B02_summer, B03_summer, B04_summer, B08_summer]
    ftw_input = np.concatenate([spring_stack, summer_stack], axis=0)
    
    # Normalize to [0, 1]
    ftw_input = np.clip(ftw_input / normalize_max, 0, 1).astype(np.float32)
    
    # Handle NaN values (set to 0)
    ftw_input = np.nan_to_num(ftw_input, nan=0.0)
    
    return ftw_input


def compute_ndvi(
    datacube: xr.Dataset,
    red: str = 'b04',
    nir: str = 'b08'
) -> xr.DataArray:
    """
    Compute Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Parameters
    ----------
    datacube : xr.Dataset
        Dataset containing red and NIR bands
    red : str
        Name of red band variable
    nir : str
        Name of NIR band variable
    
    Returns
    -------
    xr.DataArray
        NDVI values clipped to [-1, 1]
    """
    ndvi = (datacube[nir] - datacube[red]) / (datacube[nir] + datacube[red])
    return ndvi.clip(-1, 1)


def compute_temporal_statistics(
    datacube: xr.Dataset,
    year: int,
    growing_season_months: List[int] = [4, 5, 6, 7, 8, 9]
) -> xr.Dataset:
    """
    Compute NDVI temporal statistics for boundary detection.
    
    High temporal variance indicates field boundaries due to:
    - Mixed pixel effects at edges
    - Different phenology between adjacent fields
    
    Parameters
    ----------
    datacube : xr.Dataset
        Cloud-masked multi-temporal datacube
    year : int
        Target year
    growing_season_months : list
        Months to include in analysis
    
    Returns
    -------
    xr.Dataset
        Contains ndvi_mean, ndvi_std, ndvi_cv, ndvi_range
    """
    # Filter to growing season
    time_mask = (
        (datacube.time.dt.year == year) & 
        (datacube.time.dt.month.isin(growing_season_months))
    )
    growing_season = datacube.where(time_mask, drop=True)
    
    # Compute NDVI time series
    ndvi = compute_ndvi(growing_season)
    
    # Compute statistics
    ndvi_mean = ndvi.mean(dim='time', skipna=True)
    ndvi_std = ndvi.std(dim='time', skipna=True)
    
    stats = xr.Dataset({
        'ndvi_mean': ndvi_mean,
        'ndvi_std': ndvi_std,
        'ndvi_cv': ndvi_std / ndvi_mean.where(ndvi_mean != 0),  # Coefficient of variation
        'ndvi_range': ndvi.max(dim='time') - ndvi.min(dim='time'),
        'ndvi_min': ndvi.min(dim='time'),
        'ndvi_max': ndvi.max(dim='time'),
    })
    
    stats.attrs['year'] = year
    stats.attrs['n_observations'] = len(growing_season.time)
    
    return stats


def validate_ftw_input(input_array: np.ndarray) -> Dict:
    """
    Validate FTW input array format and values.
    
    Returns
    -------
    dict
        Validation results with any issues found
    """
    issues = []
    
    # Check shape
    if input_array.ndim != 3:
        issues.append(f"Expected 3D array, got {input_array.ndim}D")
    elif input_array.shape[0] != 8:
        issues.append(f"Expected 8 channels, got {input_array.shape[0]}")
    
    # Check data type
    if input_array.dtype != np.float32:
        issues.append(f"Expected float32, got {input_array.dtype}")
    
    # Check value range
    if np.any(input_array < 0) or np.any(input_array > 1):
        issues.append("Values outside [0, 1] range")
    
    # Check for NaN
    nan_count = np.isnan(input_array).sum()
    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN values")
    
    # Compute statistics
    stats = {
        'shape': input_array.shape,
        'dtype': str(input_array.dtype),
        'min': float(np.nanmin(input_array)),
        'max': float(np.nanmax(input_array)),
        'mean': float(np.nanmean(input_array)),
        'nan_fraction': float(nan_count / input_array.size),
        'valid': len(issues) == 0,
        'issues': issues,
    }
    
    return stats
