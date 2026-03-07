"""
Vectorization and interactive mapping utilities.

Converts raster segment labels from VITO post-processing into vector
polygon GeoDataFrames, identifies the most temporally stable year via
IoU analysis, and creates interactive leafmap overlays combining FTW
predicted parcels with WFS cadastral ground truth.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .validation import (
    AOI_BBOX_WGS84,
    TARGET_CRS,
    PIXEL_SIZE,
    GRID_SHAPE,
    _get_transform,
    load_cadastre,
)


def find_most_stable_year(
    results_dict: Dict[int, Dict],
) -> Tuple[int, Dict[int, float], "pd.DataFrame"]:
    """
    Identify the most temporally stable year using IoU analysis.

    Computes pairwise IoU between all years' field masks, then selects
    the year with the highest mean IoU against all other years.

    Parameters
    ----------
    results_dict : dict
        {year: results_dict} with 'field_mask_smooth' per year.

    Returns
    -------
    best_year : int
        Year with highest mean IoU across all other years.
    mean_ious : dict
        {year: mean_iou} for each year.
    iou_matrix : pd.DataFrame
        Full pairwise IoU matrix.
    """
    from .stability import compute_stability_matrix

    masks = {
        yr: res['field_mask_smooth']
        for yr, res in results_dict.items()
        if res is not None
    }

    iou_matrix = compute_stability_matrix(masks, metric='iou')

    years = iou_matrix.index.tolist()
    mean_ious = {}
    for yr in years:
        off_diag = iou_matrix.loc[yr].drop(yr)
        mean_ious[yr] = float(off_diag.mean())

    best_year = max(mean_ious, key=mean_ious.get)
    return best_year, mean_ious, iou_matrix


def vectorize_segments(
    segments: np.ndarray,
    field_mask: np.ndarray,
    simplify_tolerance: float = 5.0,
    output_path: Optional[str] = None,
) -> "gpd.GeoDataFrame":
    """
    Convert labeled segment raster to vector polygons.

    Uses rasterio.features.shapes() with the project's affine transform
    to produce georeferenced polygons in EPSG:32632.

    Parameters
    ----------
    segments : np.ndarray
        Labeled segment array (int, H x W) from VITO post-processing.
    field_mask : np.ndarray
        Binary field mask (bool, H x W). Non-field pixels are excluded.
    simplify_tolerance : float
        Douglas-Peucker simplification tolerance in meters.
        5.0 = slight smoothing at 10m resolution. 0 = no simplification.
    output_path : str, optional
        Path to save GeoJSON. None = don't save.

    Returns
    -------
    gpd.GeoDataFrame
        Polygons with columns: segment_id, area_m2, geometry.
        CRS is EPSG:32632.
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    transform = _get_transform()

    # Shift segment IDs by +1 so original segment 0 becomes 1,
    # and use 0 exclusively for masked-out (non-field) background
    masked = (segments + 1).astype(np.int32)
    masked[~field_mask] = 0

    polygons = []
    seg_ids = []

    for geom_dict, value in shapes(masked, transform=transform):
        if value == 0:
            continue
        polygons.append(shape(geom_dict))
        seg_ids.append(int(value))

    gdf = gpd.GeoDataFrame(
        {'segment_id': seg_ids, 'geometry': polygons},
        crs=TARGET_CRS,
    )

    gdf['area_m2'] = gdf.geometry.area

    if simplify_tolerance > 0:
        gdf['geometry'] = gdf.geometry.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )

    if output_path:
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"  Saved vectorized parcels: {output_path}")

    return gdf


def create_interactive_map(
    gdf_ftw: "gpd.GeoDataFrame",
    gdf_cadastre: "gpd.GeoDataFrame",
    best_year: int,
    output_html: str = "data/outputs/interactive_map.html",
    zoom: int = 15,
) -> object:
    """
    Create a leafmap interactive map with FTW parcels and cadastral overlay.

    Both GeoDataFrames are reprojected to WGS84 for web display.
    Layers can be toggled via the built-in layer control.

    Parameters
    ----------
    gdf_ftw : gpd.GeoDataFrame
        Vectorized FTW parcel polygons (any CRS, will be reprojected).
    gdf_cadastre : gpd.GeoDataFrame
        WFS cadastral parcels clipped to AOI (any CRS, will be reprojected).
    best_year : int
        Year label for the FTW layer name.
    output_html : str
        Output path for the standalone HTML map file.
    zoom : int
        Initial map zoom level.

    Returns
    -------
    leafmap.Map
        The interactive map object.
    """
    import folium
    import pandas as pd

    # Reproject to WGS84 for web display
    ftw_wgs84 = gdf_ftw.to_crs("EPSG:4326")
    cad_wgs84 = gdf_cadastre.to_crs("EPSG:4326").copy()

    # Convert any Timestamp/datetime columns to strings for JSON serialization
    for col in cad_wgs84.columns:
        if col == 'geometry':
            continue
        if pd.api.types.is_datetime64_any_dtype(cad_wgs84[col]):
            cad_wgs84[col] = cad_wgs84[col].astype(str)

    center_lat = (AOI_BBOX_WGS84[1] + AOI_BBOX_WGS84[3]) / 2
    center_lon = (AOI_BBOX_WGS84[0] + AOI_BBOX_WGS84[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        control_scale=True,
    )

    # --- Basemap layers (toggle between them) ---
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite (Esri)",
        overlay=False,
    ).add_to(m)

    # --- FTW predicted parcels — orange outlines ---
    ftw_layer = folium.FeatureGroup(name=f"FTW Parcels ({best_year})", show=True)
    folium.GeoJson(
        ftw_wgs84.to_json(),
        name=f"FTW Parcels ({best_year})",
        style_function=lambda feature: {
            "color": "#ff7800",
            "weight": 2,
            "fillColor": "#ffcc66",
            "fillOpacity": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["segment_id", "area_m2"],
            aliases=["Segment ID", "Area (m²)"],
            localize=True,
        ),
    ).add_to(ftw_layer)
    ftw_layer.add_to(m)

    # --- WFS cadastral parcels — blue outlines ---
    cad_layer = folium.FeatureGroup(name="WFS Cadastral Parcels", show=True)
    # Pick useful tooltip fields from available columns
    cad_cols = [c for c in cad_wgs84.columns if c != 'geometry']
    tooltip_fields = [c for c in ['PPOL_ID', 'PPOL_CODIC', 'area_m2'] if c in cad_cols]
    tooltip_aliases = {
        'PPOL_ID': 'Parcel ID',
        'PPOL_CODIC': 'Parcel Code',
        'area_m2': 'Area (m²)',
    }
    folium.GeoJson(
        cad_wgs84.to_json(),
        name="WFS Cadastral Parcels",
        style_function=lambda feature: {
            "color": "#0000ff",
            "weight": 1.5,
            "fillColor": "#6699ff",
            "fillOpacity": 0.15,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[tooltip_aliases.get(f, f) for f in tooltip_fields],
            localize=True,
        ) if tooltip_fields else None,
    ).add_to(cad_layer)
    cad_layer.add_to(m)

    # --- Layer control panel (top-right corner) ---
    folium.LayerControl(
        position="topright",
        collapsed=False,
    ).add_to(m)

    m.save(output_html)
    print(f"  Saved interactive map: {output_html}")

    return m


def run_vectorization_pipeline(
    results_pkl_path: str = "data/outputs/ftw_results_VITO_filtered.pkl",
    cadastre_path: str = "data/South_Tyrol_parcels.shp",
    output_dir: str = "data/outputs",
) -> Dict:
    """
    End-to-end pipeline: find stable year, vectorize, build leafmap.

    Parameters
    ----------
    results_pkl_path : str
        Path to the VITO-filtered results pickle.
    cadastre_path : str
        Path to WFS cadastral shapefile or GeoJSON.
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        'best_year', 'mean_ious', 'iou_matrix', 'gdf_ftw', 'gdf_cadastre'.
    """
    import os
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print("Loading results...")
    with open(results_pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    # Step 1: Find most stable year
    print("\nComputing stability matrix...")
    best_year, mean_ious, iou_matrix = find_most_stable_year(all_results)
    print(f"\nMean IoU per year:")
    for yr, val in sorted(mean_ious.items()):
        marker = " <-- MOST STABLE" if yr == best_year else ""
        print(f"  {yr}: {val:.4f}{marker}")

    # Step 2: Vectorize the most stable year's segments
    print(f"\nVectorizing {best_year} segments...")
    r = all_results[best_year]
    geojson_path = os.path.join(output_dir, f"ftw_parcels_{best_year}.geojson")
    gdf_ftw = vectorize_segments(
        r['segments'], r['field_mask_smooth'],
        simplify_tolerance=5.0,
        output_path=geojson_path,
    )
    print(f"  {len(gdf_ftw)} polygons, total area {gdf_ftw['area_m2'].sum()/1e4:.1f} ha")

    # Step 3: Load and clip cadastral parcels
    print(f"\nLoading cadastral parcels from {cadastre_path}...")
    gdf_cadastre = load_cadastre(cadastre_path)
    cad_geojson = os.path.join(output_dir, "cadastre_clipped.geojson")
    gdf_cadastre.to_crs("EPSG:4326").to_file(cad_geojson, driver='GeoJSON')
    print(f"  {len(gdf_cadastre)} parcels clipped to AOI")
    print(f"  Saved: {cad_geojson}")

    # Step 4: Create interactive map
    print("\nBuilding interactive map...")
    html_path = os.path.join(output_dir, "interactive_map.html")
    create_interactive_map(gdf_ftw, gdf_cadastre, best_year, output_html=html_path)

    return {
        'best_year': best_year,
        'mean_ious': mean_ious,
        'iou_matrix': iou_matrix,
        'gdf_ftw': gdf_ftw,
        'gdf_cadastre': gdf_cadastre,
    }
