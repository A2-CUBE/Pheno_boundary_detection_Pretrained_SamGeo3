"""
Data loading utilities for EOPF Sentinel-2 Zarr data.

This module handles:
- STAC catalog connection
- Bounding box reprojection
- Single scene loading with band selection
- Multi-temporal datacube formation
"""

import xarray as xr
import dask
from pyproj import Transformer
from typing import List, Dict, Optional, Tuple
import pystac_client


def connect_stac_catalog(catalog_url: str = "https://stac.core.eopf.eodc.eu"):
    """
    Connect to EOPF STAC catalog.
    
    Parameters
    ----------
    catalog_url : str
        STAC catalog endpoint URL
    
    Returns
    -------
    pystac_client.Client
        Connected STAC client
    """
    return pystac_client.Client.open(catalog_url)


def search_sentinel2(
    catalog,
    bbox: List[float],
    start_date: str,
    end_date: str,
    collection: str = "sentinel-2-l2a"
) -> List[Dict]:
    """
    Search for Sentinel-2 scenes in the catalog.
    
    Parameters
    ----------
    catalog : pystac_client.Client
        Connected STAC client
    bbox : list
        Bounding box [west, south, east, north] in EPSG:4326
    start_date : str
        Start date in ISO format
    end_date : str
        End date in ISO format
    collection : str
        STAC collection name
    
    Returns
    -------
    list
        List of STAC item dictionaries
    """
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=[start_date, end_date],
    )
    return list(search.items_as_dicts())


def reproject_bbox(
    bbox: List[float],
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:32632"
) -> List[float]:
    """
    Transform bounding box between coordinate reference systems.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
    src_crs : str
        Source CRS
    dst_crs : str
        Destination CRS
    
    Returns
    -------
    list
        Transformed bounding box [xmin, ymin, xmax, ymax]
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])
    return [xmin, ymin, xmax, ymax]


def load_single_scene(
    item_dict: Dict,
    bbox_ll: List[float],
    bands: List[str] = ['b02', 'b03', 'b04', 'b08'],
    include_scl: bool = True
) -> xr.Dataset:
    """
    Load and crop a single Sentinel-2 scene from Zarr.
    
    Parameters
    ----------
    item_dict : dict
        STAC item dictionary
    bbox_ll : list
        Bounding box in EPSG:4326 [west, south, east, north]
    bands : list
        List of 10m band names to load
    include_scl : bool
        Whether to include Scene Classification Layer
    
    Returns
    -------
    xr.Dataset
        Cropped dataset with selected bands
    """
    # Extract base path from asset href
    href = item_dict['assets']['SR_10m']['href']
    base_path = href.split('/measurements')[0]
    
    # Open datatree
    ds = xr.open_datatree(base_path, engine="zarr", chunks={}, mask_and_scale=True)
    
    # Get CRS and reproject bbox
    dst_crs = item_dict['properties'].get("proj:code", "EPSG:32632")
    utm_bbox = reproject_bbox(bbox_ll, dst_crs=dst_crs)
    
    # Load 10m bands
    ds_ref = (
        ds["measurements"]["reflectance"]["r10m"]
        .to_dataset()[bands]
        .sel(x=slice(utm_bbox[0], utm_bbox[2]), y=slice(utm_bbox[3], utm_bbox[1]))
    )
    
    datasets_to_merge = [ds_ref]
    
    if include_scl:
        # Load SCL mask (20m resolution)
        ds_scl = (
            ds["conditions"]["mask"]["l2a_classification"]["r20m"]
            .to_dataset()
            .sel(x=slice(utm_bbox[0], utm_bbox[2]), y=slice(utm_bbox[3], utm_bbox[1]))
        )
        
        # Interpolate SCL to 10m grid
        ds_scl_interp = ds_scl.interp(x=ds_ref.x, y=ds_ref.y, method="nearest")
        datasets_to_merge.append(ds_scl_interp)
    
    # Merge and add time dimension
    merged = xr.merge(datasets_to_merge)
    merged = merged.expand_dims(time=[item_dict['properties']['datetime']])
    
    return merged


def build_datacube(
    items: List[Dict],
    bbox: List[float],
    bands: List[str] = ['b02', 'b03', 'b04', 'b08'],
    include_scl: bool = True,
    parallel: bool = True
) -> xr.Dataset:
    """
    Build multi-temporal datacube from STAC items.
    
    Parameters
    ----------
    items : list
        List of STAC item dictionaries
    bbox : list
        Bounding box in EPSG:4326
    bands : list
        List of band names to load
    include_scl : bool
        Whether to include Scene Classification Layer
    parallel : bool
        Whether to use Dask parallelism
    
    Returns
    -------
    xr.Dataset
        Multi-temporal datacube with time dimension
    """
    if parallel:
        delayed_results = [
            dask.delayed(load_single_scene)(item, bbox, bands, include_scl)
            for item in items
        ]
        results = dask.compute(*delayed_results)
    else:
        results = [
            load_single_scene(item, bbox, bands, include_scl)
            for item in items
        ]
    
    # Filter out failed loads
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        raise ValueError("No scenes loaded successfully")
    
    # Get CRS from first item
    crs = items[0]['properties'].get("proj:code", "EPSG:32632")
    
    # Concatenate along time
    datacube = xr.concat(results, dim="time").sortby("time")
    datacube = datacube.rio.write_crs(crs)
    
    return datacube


def get_temporal_info(datacube: xr.Dataset) -> Dict:
    """
    Extract temporal information from datacube.
    
    Returns
    -------
    dict
        Temporal statistics and coverage info
    """
    import pandas as pd
    
    times = pd.DatetimeIndex(datacube.time.values)
    
    return {
        'n_scenes': len(times),
        'first_date': times.min(),
        'last_date': times.max(),
        'years': sorted(times.year.unique().tolist()),
        'scenes_per_year': times.year.value_counts().to_dict(),
        'scenes_per_month': times.month.value_counts().to_dict(),
    }
