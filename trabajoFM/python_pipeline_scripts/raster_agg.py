from __future__ import annotations

"""
Lightweight raster utilities extracted from TEMP_change_chm_insparation.py

Optional dependencies (install as needed):
- rasterio, rasterstats, geopandas, shapely, matplotlib
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .provenance import RealizationProvenance
from .utils import get_logger


def reproject_raster_to_epsg(input_path, output_path, epsg: int = 25830, overwrite_cache: bool = False) -> bool:
    """Reproject with simple on-disk caching.

    Returns True if work was performed; False if served from cache.
    """
    import os, json
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    input_path = str(input_path)
    output_path = str(output_path)
    meta_path = f"{output_path}.meta.json"
    if os.path.exists(output_path) and not overwrite_cache and os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            src_mtime = os.path.getmtime(input_path)
            if int(meta.get("epsg")) == int(epsg) and meta.get("src") == input_path and abs(meta.get("src_mtime", -1) - src_mtime) < 1e-6:
                return False
        except Exception:
            pass
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(src.crs, f"EPSG:{epsg}", src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": f"EPSG:{epsg}", "transform": transform, "width": width, "height": height})
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=f"EPSG:{epsg}",
                    resampling=Resampling.nearest,
                )
    try:
        Path(meta_path).write_text(
            json.dumps({"epsg": int(epsg), "src": input_path, "src_mtime": os.path.getmtime(input_path)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    return True


def clip_raster_with_bounds(input_path, output_path, bounds, overwrite_cache: bool = False) -> bool:
    """Clip with simple on-disk caching.

    Returns True if work was performed; False if served from cache.
    """
    import os, json
    import rasterio
    import rasterio.mask
    from shapely.geometry import box

    input_path = str(input_path)
    output_path = str(output_path)
    meta_path = f"{output_path}.meta.json"
    if os.path.exists(output_path) and not overwrite_cache and os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            src_mtime = os.path.getmtime(input_path)
            if meta.get("src") == input_path and abs(meta.get("src_mtime", -1) - src_mtime) < 1e-6 and meta.get("bounds") == list(bounds):
                return False
        except Exception:
            pass
    geom = [box(*bounds)]
    with rasterio.open(input_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    try:
        Path(meta_path).write_text(
            json.dumps({"src": input_path, "src_mtime": os.path.getmtime(input_path), "bounds": list(bounds)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    return True


def plot_raster_and_zones(raster_path, gdf, raster_alias: str, title: str, label_field: str) -> None:
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LinearSegmentedColormap

    with rasterio.open(raster_path) as src:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
        raster_data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            raster_data[(raster_data == nodata) | (raster_data == 0)] = np.nan
        else:
            raster_data[raster_data == 0] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        min_val = np.nanmin(raster_data)
        max_val = np.nanmax(raster_data)
        colors = [(0.9, 0.9, 0.9), (0.0, 0.0, 0.0)]
        cmap = LinearSegmentedColormap.from_list("lightgrey_to_black", colors)
        norm = Normalize(vmin=min_val, vmax=max_val)
        im = ax.imshow(raster_data, extent=extent, origin="upper", cmap=cmap, norm=norm, alpha=1.0)
        gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(centroid.x, centroid.y, str(row[label_field]), fontsize=9, fontweight="bold", color="red", ha="center", va="center")
        ax.set_facecolor("white")
        ax.set_title(title, color="black")
        ax.set_xlabel("Easting", color="black")
        ax.set_ylabel("Northing", color="black")
        ax.tick_params(colors="black")
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(raster_alias, color="black")
        plt.grid(False)
        plt.tight_layout()
        plt.show()


def raster_zonal_aggregation_to_gpkg(
    *,
    raster_folder: str | Path,
    zones_fp: str | Path,
    zone_field: str,
    label_field: str,
    output_gpkg: str | Path,
    files_end_with: str = ".tif",
    stat_operation: str = "sum",
    raster_alias: str = "year",
    zone_meaning: str = "Sub-basin",
    overwrite_cache: bool = False,
    plot: bool = False,
    config: Optional[Dict[str, Any]] = None,
    rp: Optional[RealizationProvenance] = None,
    write_manifest: bool = False,
    manifest_path: Optional[Path] = None,
) -> str:
    """Compute zonal aggregation for each raster in folder and write to GPKG layer.

    Returns the layer name path-like string 'output.gpkg:layer'.
    """
    import os
    import numpy as np
    import geopandas as gpd
    from rasterstats import zonal_stats

    log = get_logger(__name__, config)

    raster_folder = str(raster_folder)
    output_gpkg = str(output_gpkg)
    log.info("Zonal aggregation start | zones=%s | rasters_dir=%s", zones_fp, raster_folder)
    gdf = gpd.read_file(zones_fp).to_crs("EPSG:25830")
    valid_mask = gdf[zone_field].notna() & (gdf[zone_field] != "") & (gdf[zone_field] != "NA")
    gdf = gdf[valid_mask].copy()
    bounds = gdf.total_bounds
    log.info("Valid %s features: %s | bounds=%s", zone_meaning, len(gdf), bounds)

    tif_files = [os.path.join(raster_folder, f) for f in os.listdir(raster_folder) if f.lower().endswith(files_end_with)]
    log.info("Found %s raster(s) matching '%s'", len(tif_files), files_end_with)
    cache_dir = os.path.join(raster_folder, "temp_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Provenance step spans the whole operation
    ctx = rp.step(
        "raster_zonal_aggregation_to_gpkg",
        module=__name__,
        args={
            "zone_field": zone_field,
            "stat_operation": stat_operation,
            "files_end_with": files_end_with,
            "raster_alias": raster_alias,
            "zone_meaning": zone_meaning,
        },
        inputs=[zones_fp] + tif_files,
    ) if rp else None
    enter = ctx.__enter__ if ctx else (lambda: None)
    exit_ = ctx.__exit__ if ctx else (lambda exc_type, exc, tb: None)
    enter()
    try:
        for raster_path in tif_files:
            raster_filename = os.path.splitext(os.path.basename(raster_path))[0]
            alias = raster_filename.split("_")[-1] if raster_alias == "year" else raster_filename
            field_name = f"{stat_operation}_{alias}"
            proj_raster_path = os.path.join(cache_dir, f"{raster_filename}_reproj.tif")
            clip_raster_path = os.path.join(cache_dir, f"{raster_filename}_clip.tif")

            did_reproj = reproject_raster_to_epsg(raster_path, proj_raster_path, epsg=25830, overwrite_cache=overwrite_cache)
            log.info("Reprojected%s -> %s", " (cached)" if not did_reproj else "", proj_raster_path)
            did_clip = clip_raster_with_bounds(proj_raster_path, clip_raster_path, bounds, overwrite_cache=overwrite_cache)
            log.info("Clipped%s -> %s", " (cached)" if not did_clip else "", clip_raster_path)

            stats = zonal_stats(
                gdf,
                clip_raster_path,
                stats=[stat_operation, "count"],
                geojson_out=False,
                nodata=0,
                all_touched=False,
            )
            aggregations = [round(s[stat_operation]) if s[stat_operation] is not None else 0 for s in stats]
            counts = [s["count"] if s["count"] is not None else 0 for s in stats]
            gdf[field_name] = np.array(aggregations, dtype=np.float64)
            gdf[f"pixels_included_{alias}"] = np.array(counts, dtype=np.int32)
    finally:
        exit_(None, None, None)

    layer_name = f"values_by_{zone_meaning.lower().replace(' ', '_')}"
    gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
    log.info("Wrote: %s (layer=%s)", output_gpkg, layer_name)
    if rp:
        rp.record_outputs([Path(output_gpkg)], kind="gpkg")
        log.info("Provenance updated: outputs recorded in ledger.")
    # Optional manifest next to the GPKG
    if write_manifest:
        try:
            man_path = Path(manifest_path) if manifest_path else Path(str(output_gpkg) + ".manifest.json")
            import json
            manifest = {
                "rasters": [str(Path(p).resolve()) for p in tif_files],
                "zones": str(Path(zones_fp).resolve()),
                "bounds": list(bounds),
                "epsg": 25830,
                "files_end_with": files_end_with,
                "stat_operation": stat_operation,
                "raster_alias": raster_alias,
                "zone_meaning": zone_meaning,
                "gpkg": str(Path(output_gpkg).resolve()),
                "layer": layer_name,
            }
            Path(man_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Wrote manifest: %s", man_path)
        except Exception as e:
            log.warning("Could not write manifest: %s", e)
    return f"{output_gpkg}:{layer_name}"
