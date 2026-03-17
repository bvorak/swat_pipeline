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



# this was my original that worked
def rasterZonalAggregationToGPKG(
    raster_folder,
    zones_fp,
    zone_field,
    label_field,
    output_gpkg,
    files_end_with = "*.tif",  # Default to all .tif files
    stat_operation="sum",  # Options: "mean", "sum", "min", "max", "median"
    raster_alias="year",  # or "full_name"
    zone_meaning="Sub-basin",
    overwrite_cache=False
):

    import os
    import numpy as np
    import geopandas as geopandas
    from rasterstats import zonal_stats
    import rasterio

    print(f"→ Reading {zone_meaning.lower()} shapefile: {zones_fp}")
    gdf = geopandas.read_file(zones_fp).to_crs("EPSG:25830")

    # Filter out invalid zone_field values
    valid_mask = gdf[zone_field].notna() & (gdf[zone_field] != "") & (gdf[zone_field] != "NA")
    gdf = gdf[valid_mask].copy()
    print(f"✔ Filtered valid {zone_meaning.lower()}s: {len(gdf)} features retained where `{zone_field}` is not null, empty, or 'NA'.")


    bounds = gdf.total_bounds
    print(f"✔ Reprojected {zone_meaning.lower()}s to EPSG:25830")
    print(f"→ Using bounding box for clipping: {bounds}")


    # Case-insensitive matching of raster files with postfix
    tif_files = [
        os.path.join(raster_folder, f)
        for f in os.listdir(raster_folder)
        if f.lower().endswith(files_end_with)
    ]

    print(f"→ Found {len(tif_files)} raster file(s) matching pattern (case-insensitive)")


    cache_dir = os.path.join(raster_folder, "temp_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for raster_path in tif_files:
        raster_filename = os.path.splitext(os.path.basename(raster_path))[0]
        if raster_alias == "year":
            alias = raster_filename.split("_")[-1]
        else:
            alias = raster_filename

        field_name = f"{stat_operation}_{alias}"
        print(f"\n🔹 Processing raster: {raster_path} → alias: {alias}")

        proj_raster_path = os.path.join(cache_dir, f"{raster_filename}_reproj.tif")
        clip_raster_path = os.path.join(cache_dir, f"{raster_filename}_clip.tif")

        reproject_raster_to_epsg(raster_path, proj_raster_path, epsg=25830, overwrite_cache=overwrite_cache)

        with rasterio.open(proj_raster_path) as src:
            data = src.read(1).astype(np.float32)
            data[data == 0] = np.nan
            print(f"→ Raster stats BEFORE clip:")
            print(f"   MIN: {np.nanmin(data)}, MAX: {np.nanmax(data)}, MEAN: {np.nanmean(data)}, SUM: {np.nansum(data)}")

        clip_raster_with_bounds(proj_raster_path, clip_raster_path, bounds, overwrite_cache=overwrite_cache)

        with rasterio.open(clip_raster_path) as src:
            clipped = src.read(1).astype(np.float32)
            clipped[clipped == 0] = np.nan
            print(f"→ Raster stats AFTER clip:")
            print(f"   MIN: {np.nanmin(clipped)}, MAX: {np.nanmax(clipped)}, MEAN: {np.nanmean(clipped)}, SUM: {np.nansum(clipped)}")

        print(f"→ Plotting raster with {zone_meaning.lower()} overlay:")
        plot_raster_and_zones(clip_raster_path, gdf, title=f"Raster {alias} and {zone_meaning} Alignment", raster_alias=alias, label_field=label_field)

        with rasterio.open(clip_raster_path) as src:
            nodata_val = 0
        stats = zonal_stats(
            gdf,
            clip_raster_path,
            stats=[stat_operation, "count"],
            geojson_out=False,
            nodata=nodata_val,
            all_touched=False
        )

        aggregations = [s[stat_operation] if s[stat_operation] is not None else 0 for s in stats]
        counts = [s["count"] if s["count"] is not None else 0 for s in stats]

        gdf[field_name] = np.array(aggregations, dtype=np.float64)
        gdf[f"pixels_included_{alias}"] = np.array(counts, dtype=np.int32)

        print(f"✔ Zonal statistics for {field_name}:")
        for idx, stat in enumerate(stats):
            zone_id = gdf.iloc[idx][zone_field]
            s = stat[stat_operation] or 0
            c = stat["count"] or 0
            avg = s / c if c else "NA"
            print(f"   {zone_meaning} {zone_id}: {stat_operation}={s}, COUNT={c}, AVG={avg}")

        if overwrite_cache:
            print("→ Cleaning up intermediate files")
            for f in [proj_raster_path, clip_raster_path]:
                if os.path.exists(f):
                    os.remove(f)
        else:
            print("⚠ Keeping intermediate files for future runs (cached)")

    output_layer_name = f"values_by_{zone_meaning.lower().replace(' ', '_')}"
    print(f"\n→ Writing results to GeoPackage: {output_gpkg} (layer: {output_layer_name})")

    pixel_cols = sorted([col for col in gdf.columns if col.startswith("pixels_included_")])
    value_cols = sorted([col for col in gdf.columns if col.startswith("{stat_operation}_")])
    other_cols = [col for col in gdf.columns if col not in pixel_cols + value_cols]

    gdf = gdf[other_cols + pixel_cols + value_cols]
    gdf.to_file(output_gpkg, layer=output_layer_name, driver="GPKG")

    # Write CSV next to GPKG with same base name
    csv_path = str(Path(output_gpkg).with_suffix(".csv"))
    try:
        gdf.drop(columns="geometry").to_csv(csv_path, sep=";", float_format="%.8f", index=False)
        #gdf.drop(columns="geometry").to_csv(csv_path, index=False)
        #log.info("Wrote CSV: %s", csv_path)
    except Exception as e:
        #log.warning("Could not write CSV: %s", e)
        print(f"⚠ Could not write CSV: {e}")

    print("✔ Done")
    return os.path.join(output_gpkg, output_layer_name)

