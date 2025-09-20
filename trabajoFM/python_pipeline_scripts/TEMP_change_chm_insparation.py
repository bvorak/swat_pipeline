# %%
import os
import pandas as pd

# %% [markdown]
# ## Create Zonal stats (without arcpy)

# %%
import os
import glob
import rasterio
import rasterio.mask
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap





def reproject_raster_to_epsg(input_path, output_path, epsg=25830, overwrite_cache=False):
    if os.path.exists(output_path) and not overwrite_cache:
        print(f"⚠ Skipping reprojection: {output_path} already exists (cached)")
        return

    print(f"→ Reprojecting raster {input_path} to EPSG:{epsg}")
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{epsg}", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": f"EPSG:{epsg}",
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=f"EPSG:{epsg}",
                    resampling=Resampling.nearest
                )
    print(f"✔ Reprojected raster saved to {output_path}")


def clip_raster_with_bounds(input_path, output_path, bounds, overwrite_cache=False):
    if os.path.exists(output_path) and not overwrite_cache:
        print(f"⚠ Skipping clipping: {output_path} already exists (cached)")
        return

    print(f"→ Clipping raster {input_path} to bounds {bounds}")
    geom = [box(*bounds)]
    with rasterio.open(input_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    print(f"✔ Clipped raster saved to {output_path}")



def plot_raster_and_zones(raster_path, gdf, raster_alias, title, label_field):
    with rasterio.open(raster_path) as src:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        raster_data = src.read(1).astype(float)

        nodata = src.nodata
        # Mask nodata and zero values as NaN (transparent)
        if nodata is not None:
            raster_data[(raster_data == nodata) | (raster_data == 0)] = np.nan
        else:
            raster_data[raster_data == 0] = np.nan

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        # Find min positive value to avoid black on black for smallest values
        min_val = np.nanmin(raster_data)
        max_val = np.nanmax(raster_data)

        # Define a simple grey colormap from light grey to black
        colors = [(0.9, 0.9, 0.9), (0.0, 0.0, 0.0)]  # light grey to black
        cmap = LinearSegmentedColormap.from_list('lightgrey_to_black', colors)

        # Normalize with min and max
        norm = Normalize(vmin=min_val, vmax=max_val)

        im = ax.imshow(
            raster_data,
            extent=extent,
            origin='upper',
            cmap=cmap,
            norm=norm,
            alpha=1.0
        )

        # Plot zone boundaries with black edges
        gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1)

        # Add HRU labels
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x, centroid.y,
                str(row[label_field]),
                fontsize=9,
                fontweight='bold',
                color='red',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7)
            )

        ax.set_facecolor('white')
        ax.set_title(title, color='black')
        ax.set_xlabel("Easting", color='black')
        ax.set_ylabel("Northing", color='black')
        ax.tick_params(colors='black')

        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(raster_alias, color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        plt.grid(False)
        plt.tight_layout()
        plt.show()


# %%

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
    print(f"→ Reading {zone_meaning.lower()} shapefile: {zones_fp}")
    gdf = gpd.read_file(zones_fp).to_crs("EPSG:25830")

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

        aggregations = [round(s[stat_operation]) if s[stat_operation] is not None else 0 for s in stats]
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
    gdf.to_csv("output.csv", float_format="%.8f", index=False)

    print("✔ Done")
    return os.path.join(output_gpkg, output_layer_name)


# %%
import os
import pandas as pd
from pathlib import Path

output_gpkg = r"..\..\Genil GEO_INFO_POOL\Input Data\Diffuse loads\Soil chemical composition\python calculated hru stats\hru_chem_stats.gpkg"

result_layer_path = rasterZonalAggregationToGPKG(
    raster_folder=r"..\..\Genil_ArcGIS_Ruben\ArcGIS _ base de suelo quimico _ ruben _ 30-05-25\raster",
    zones_fp=r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Archivos de Cesar Ruben Fernandez De Villaran San Juan - swat_cubillas\cubillas_hru\Watershed\Shapes\hru1.shp",  
    zone_field="HRU_GIS", # "HRUGIS",
    label_field ="OBJECTID", # "GRIDCODE",
    output_gpkg= output_gpkg,
    files_end_with="_rediam.tif",
    stat_operation = "mean",
    raster_alias ="full_name",
    zone_meaning="HRU",
    overwrite_cache=False
)




# Read the result layer from the GeoPackage
layer_name = "values_by_hru"
gdf = gpd.read_file(output_gpkg, layer=layer_name)

# Export to CSV (excluding geometry)
csv_output = output_gpkg.replace(".gpkg", ".csv")
gdf.drop(columns="geometry").to_csv(csv_output, sep=';', index=False)

print(f"Exported to CSV: {csv_output}")


# %% [markdown]
# ## Read soil zonal statistics (HRUGIS)

# %%
#### Load the CSV file
# The CSV file should have the last columns representing chemical measurements with their respective units and the first column being an Unique ID
# optimally this CSV comes out of the ArcGIS Pro Model in "base de suelo_quimica.aprx"

# data from ArcGIS Pro Model
#file_path = r'HRU_cubillas_HRUGIS_join_zonal_quimica.csv'
file_path = r"..\..\Genil GEO_INFO_POOL\Input Data\Diffuse loads\Soil chemical composition\python calculated hru stats\hru_chem_stats.csv"

identifier_column = "HRU_GIS" # 'HRUGIS' for the old model
df = pd.read_csv(file_path, sep=';')
df

# %%
# Convert the last 3 columns to integers, being aware that comma is used as a decimal separator
for col in df.columns[-3:]:
    # Replace comma with dot for decimal, then convert to float
    df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

# Convert the first column to integers 
df[df.columns[0]] = df[df.columns[0]].apply(lambda x: int(str(x).split(',')[0]))

# %% [markdown]
# ## Transform from total values to org/anorg

# %%
def convert_soil_nutrients(N_total_percent, P_mg_100g_P2O5,
                           nitrate_frac=None, organicN_frac=None,
                           labileP_frac=None, organicP_frac=None):
    # --- Set default fractions with sources ---
    
    # Nitrate-N typically 1–2% of total N in agricultural soils
    if nitrate_frac is None:
        nitrate_frac = 0.02

    # Organic N is the major pool (~95–98%) of total N
    if organicN_frac is None:
        organicN_frac = 0.98

##### Phosphorus fractions
    # Olsen-P reflects available (labile) P
    if labileP_frac is None:
        labileP_frac = 1

    # Organic P  - until we have a better heuristic we set organicP_frac to 0 and hope swat initialzes 0.0 values in a senseful way
    if organicP_frac is None:
        organicP_frac = 0

    # --- Start conversions ---

    # Convert % to mg/kg
    N_total_mg_kg = N_total_percent * 10_000
    print(f"N_total ({N_total_percent}%): {N_total_mg_kg} mg/kg")

    # Convert P2O5 from mg/100g to mg/kg
    P2O5_mg_kg = P_mg_100g_P2O5 * 10
    print(f"P2O5: {P_mg_100g_P2O5} mg/100g = {P2O5_mg_kg} mg/kg")

    # Convert P2O5 to elemental P: 1 mg P2O5 = 0.4364 mg P
    available_P_mg_kg = P2O5_mg_kg * 0.4364
    print(f"Available P (as elemental P): {available_P_mg_kg:.1f} mg/kg")

    # Compute nutrient pools
    soil_NO3   = N_total_mg_kg * nitrate_frac
    soil_orgN  = N_total_mg_kg * organicN_frac
    soil_labP  = available_P_mg_kg * labileP_frac
    soil_orgP  = available_P_mg_kg * organicP_frac

    print(f"Soil NO3: {soil_NO3:.1f} mg/kg")
    print(f"Soil organic N: {soil_orgN:.1f} mg/kg")
    print(f"Soil labile P: {soil_labP:.1f} mg/kg")
    print(f"Soil organic P: {soil_orgP:.1f} mg/kg")

    return {
        "Soil NO3 [mg/kg]": round(soil_NO3, 2),
        "Soil organic N [mg/kg]": round(soil_orgN, 2),
        "Soil labile P [mg/kg]": round(soil_labP, 2),
        "Soil organic P [mg/kg]": round(soil_orgP, 2)
    }

# %% [markdown]
# ## modify .chm files

# %%
def read_modify_and_save_chm_preserve_all_spacing(file_path, replacements_dict, output_folder, print_before=True, pperco_val=None):
    """
    Reads a .chm file, prints Layer 1 values for specific variables,
    replaces them with values from a dictionary while:
    - Keeping ':' and rightmost digit of Layer 1 aligned
    - Preserving the exact original spacing between all columns
    - Optionally modifies the 'Phosphorus perc coef' line if pperco_val is given

    Parameters:
    - file_path: str, path to the .chm file
    - replacements_dict: dict of replacements for Layer 1 values
    - output_folder: str, folder to save the modified file
    - print_before: bool, whether to print original Layer 1 values
    - pperco_val: float or None, optional new value for 'Phosphorus perc coef'
    """
    import os
    import re

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(file_path))

    with open(file_path, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        modified = False
        # Handle replacements from dictionary
        for key, new_val in replacements_dict.items():
            if line.strip().startswith(key):
                colon_index = line.index(":")
                label = line[:colon_index]
                after_colon = line[colon_index + 1:].rstrip("\n")

                matches = list(re.finditer(r"\s*\S+", after_colon))
                if not matches:
                    break

                original_val = matches[0].group().strip()
                if print_before:
                    print(f"{key} (original Layer 1): {original_val}")

                new_val_str = f"{new_val:.2f}"
                original_field = matches[0].group()
                field_end_pos = matches[0].end()
                field_start_pos = matches[0].start()

                start_pos = field_end_pos - len(new_val_str)
                padding = " " * (start_pos - field_start_pos)
                replaced_field = padding + new_val_str

                new_line = (
                    label + ":" +
                    replaced_field +
                    after_colon[matches[0].end():] +
                    "\n"
                )
                modified_lines.append(new_line)
                modified = True
                break

        # Handle PPERCO if given
        if not modified and pperco_val is not None and line.strip().startswith("Phosphorus perc coef"):
            colon_index = line.index(":")
            label = line[:colon_index]
            after_colon = line[colon_index + 1:].rstrip("\n")

            matches = list(re.finditer(r"\s*\S+", after_colon))
            if matches:
                original_val = matches[0].group().strip()
                if print_before:
                    print(f"Phosphorus perc coef (original): {original_val}")

                new_val_str = f"{pperco_val:.2f}"
                field_end_pos = matches[0].end()
                field_start_pos = matches[0].start()

                start_pos = field_end_pos - len(new_val_str)
                padding = " " * (start_pos - field_start_pos)
                replaced_field = padding + new_val_str

                new_line = (
                    label + ":" +
                    replaced_field +
                    after_colon[matches[0].end():] +
                    "\n"
                )
                modified_lines.append(new_line)
                modified = True

        if not modified:
            modified_lines.append(line)

    with open(output_path, "w") as out_file:
        out_file.writelines(modified_lines)

    return output_path


# %%
input_folder = r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Archivos de Cesar Ruben Fernandez De Villaran San Juan - swat_cubillas\cubillas_hru\Scenarios\Default\TxtInOut"
output_folder = r"C:\SWAT\RSWAT\cubillas\cubillas_set_219_ruben\cubillas_BASE_DIFFUSE_set-219"

not_found_identifiers = []
manipulated_count = 0
count = 0

for file_name in os.listdir(input_folder):
    if file_name.endswith(".chm"):
        count += 1
        print(f"Processing file: {file_name}")
        identifier = int(file_name.split('.')[0])
        print(f"Trying identifier: {identifier}")
        if identifier in df[identifier_column].values:
            dict = convert_soil_nutrients(
                df.loc[df[identifier_column] == identifier, 'mean_Nitrogeno_total_porcent_resample_Rediam'].values[0],
                df.loc[df[identifier_column] == identifier, 'mean_Fosforo_mg_100g_P205_rediam'].values[0]
            )
            file_path = os.path.join(input_folder, file_name)
            modified_file_path = read_modify_and_save_chm_preserve_all_spacing(file_path, dict, output_folder, pperco_val=15)
            print(f"Modified file saved to: {modified_file_path}")
            manipulated_count += 1
        else:
            print(f"❌ Error: Identifier {identifier} not found in dataframe.")
            not_found_identifiers.append(identifier)

print(f"Total files processed: {count}")
print(f"Total files manipulated: {manipulated_count}")
print(f"Identifiers not found in df: {not_found_identifiers}")



