# %%
import os
os.getcwd()

# %% [markdown]
# ## GIS part

# %% [markdown]
# ### Non ArcPy version

# %%
#!pip install geopandas rasterio rasterstats shapely fiona numpy matplotlib


# %%
import os
import glob
import rasterio
import rasterio.mask
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
from matplotlib.colors import Normalize, LinearSegmentedColormap

def plot_raster_and_zones(raster_path, gdf, title):
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

        # Add GRIDCODE labels
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x, centroid.y,
                str(row["GRIDCODE"]),
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
        cbar.set_label('Population', color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        plt.grid(False)
        plt.tight_layout()
        plt.show()



def subbasinPopulationAggregationToGPKG(raster_folder, sub_basin_fp, zone_field, output_gpkg, overwrite_cache=False):
    print(f"→ Reading sub-basin shapefile: {sub_basin_fp}")
    gdf = gpd.read_file(sub_basin_fp).to_crs("EPSG:25830")

    bounds = gdf.total_bounds
    print(f"✔ Reprojected sub-basins to EPSG:25830")
    print(f"→ Using bounding box for clipping: {bounds}")

    tif_files = glob.glob(os.path.join(raster_folder, "hipgdac_es_100_*.tif"))

    # Create cache directory
    cache_dir = os.path.join(raster_folder, "temp_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for raster_path in tif_files:
        raster_filename = os.path.splitext(os.path.basename(raster_path))[0]
        year = raster_filename.split("_")[-1]
        field_name = f"popul_sum_{year}"
        print(f"\n🔹 Processing raster: {raster_path} → year: {year}")

        proj_raster_path = os.path.join(cache_dir, f"{raster_filename}_reproj.tif")
        clip_raster_path = os.path.join(cache_dir, f"{raster_filename}_clip.tif")

        # Step 1: Reproject
        reproject_raster_to_epsg(raster_path, proj_raster_path, epsg=25830, overwrite_cache=overwrite_cache)

        # Step 2: Stats before clipping
        with rasterio.open(proj_raster_path) as src:
            data = src.read(1).astype(np.float32)
            data[data == 0] = np.nan
            print(f"→ Raster stats BEFORE clip:")
            print(f"   MIN: {np.nanmin(data)}, MAX: {np.nanmax(data)}, MEAN: {np.nanmean(data)}, SUM: {np.nansum(data)}")

        # Step 3: Clip
        clip_raster_with_bounds(proj_raster_path, clip_raster_path, bounds, overwrite_cache=overwrite_cache)

        # Step 4: Stats after clipping
        with rasterio.open(clip_raster_path) as src:
            clipped = src.read(1).astype(np.float32)
            clipped[clipped == 0] = np.nan
            print(f"→ Raster stats AFTER clip:")
            print(f"   MIN: {np.nanmin(clipped)}, MAX: {np.nanmax(clipped)}, MEAN: {np.nanmean(clipped)}, SUM: {np.nansum(clipped)}")

        # Step 5: Plot
        print("→ Plotting raster with sub-basin overlay:")
        plot_raster_and_zones(clip_raster_path, gdf, title=f"Raster {year} and Sub-basin Alignment")

        # Step 6: Zonal stats
        with rasterio.open(clip_raster_path) as src:
            nodata_val = 0
        stats = zonal_stats(
            gdf,
            clip_raster_path,
            stats=["sum", "count"],
            geojson_out=False,
            nodata=nodata_val,
            all_touched=False
        )

        sums = [round(s["sum"]) if s["sum"] is not None else 0 for s in stats]
        counts = [s["count"] if s["count"] is not None else 0 for s in stats]

        gdf[field_name] = np.array(sums, dtype=np.float64)
        gdf[f"pixels_included_{year}"] = np.array(counts, dtype=np.int32)

        print(f"✔ Zonal statistics for {field_name}:")
        for idx, stat in enumerate(stats):
            zone_id = gdf.iloc[idx][zone_field]
            s = stat["sum"] or 0
            c = stat["count"] or 0
            avg = s / c if c else "NA"
            print(f"   Zone {zone_id}: SUM={s}, COUNT={c}, AVG={avg}")

        # Step 7: Debug Zone 10
        debug_zone_index = 9
        zone_geom = [gdf.iloc[debug_zone_index].geometry]
        with rasterio.open(clip_raster_path) as src:
            out_image, _ = rasterio.mask.mask(src, zone_geom, crop=True)
            values = out_image[0].flatten()
            valid_values = values[values > 0]
            print(f"🔍 Debug: Zone 10 pixel values (non-zero): {valid_values[:10]}...")
            print(f"→ Zone 10 pixel count (non-zero): {len(valid_values)}, sum: {np.sum(valid_values)}")

        # Step 8: Cleanup
        if overwrite_cache:
            print("→ Cleaning up intermediate files")
            for f in [proj_raster_path, clip_raster_path]:
                if os.path.exists(f):
                    os.remove(f)
        else:
            print("⚠ Keeping intermediate files for future runs (cached)")

    output_layer_name = "population_by_subbasin"
    print(f"\n→ Writing results to GeoPackage: {output_gpkg} (layer: {output_layer_name})")



    # Separate column lists
    pixel_cols = sorted([col for col in gdf.columns if col.startswith("pixels_included_")])
    popul_cols = sorted([col for col in gdf.columns if col.startswith("popul_sum_")])

    # Keep all other columns (non these two types)
    other_cols = [col for col in gdf.columns if col not in pixel_cols + popul_cols]

    # New column order: other columns + pixels_included fields + popul_sum fields
    new_col_order = other_cols + pixel_cols + popul_cols

    # Reorder GeoDataFrame columns
    gdf = gdf[new_col_order]

    # Then save

    
    gdf.to_file(output_gpkg, layer=output_layer_name, driver="GPKG")
    gdf.to_csv("output.csv", float_format="%.8f", index=False)

    print("✔ Done")

    return os.path.join(output_gpkg, output_layer_name)


# %%
import os
import pandas as pd
from pathlib import Path

# Define paths
raster_folder = r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Granada\TrabajoFM\Genil GEO_INFO_POOL\Input Data\Population data\HIPGDAC-ES-v1.0.0\fgoerlich-HIPGDAC-ES-cd11f21\HIPGDAC-ES\1970-2021 copy"
basin_fc = r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Granada\TrabajoFM\Genil GEO_INFO_POOL\SWaT outputs\Cubillas\shapes cubillas\Sub_basin.shp"
output_folder = r"C:\Users\Usuario\OneDrive - UNIVERSIDAD DE HUELVA\Granada\TrabajoFM\Genil GEO_INFO_POOL\Input Data\Population data\HIPGDAC-ES-v1.0.0\custom_aggregations\open_source"
output_gpkg = os.path.join(output_folder, "cubillas_population.gpkg")

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Run processing function (assumes you already defined the open-source version above)
result_layer_path = subbasinPopulationAggregationToGPKG(
    raster_folder=raster_folder,
    sub_basin_fp=basin_fc,
    zone_field="GRIDCODE",
    output_gpkg=output_gpkg,
    overwrite_cache=False  # Set True to force reprocessing
)

# Read the result layer from the GeoPackage
layer_name = "population_by_subbasin"
gdf = gpd.read_file(output_gpkg, layer=layer_name)

# Export to CSV (excluding geometry)
csv_output = output_gpkg.replace(".gpkg", ".csv")
gdf.drop(columns="geometry").to_csv(csv_output, sep=';', index=False)

print(f"Exported to CSV: {csv_output}")


# %% [markdown]
# # Non-GIS part

# %% [markdown]
# ## Define non GIS script's inputs & outputs

# %%
import pandas as pd
import os
os.getcwd()

#### Load the CSV file (relative path to this scripts folder)
# The CSV file should have the last columns representing years and their population counts and the first column being an Unique ID
# optimally this CSV comes out of the ArcGIS Pro Model in T"rabajoFM\Genil_ArcGIS_Pascal"

basin_name = "Cubillas"
# Population aggregated according to (sub)basins' subbasins:
file_path = r'..\..\Genil GEO_INFO_POOL\Input Data\Population data\Basin Aggregations\Cubillas population loads\cuenca_cubillas_habitantes_decadas_1970_2021_arcgis_output.csv'
file_path_2 = r"..\..\Genil GEO_INFO_POOL\Input Data\Population data\HIPGDAC-ES-v1.0.0\custom_aggregations\open_source\cubillas_population.csv"
identifier_column = "GRIDCODE"

# to extract years that have been and shall be simulated
cio_file = r"C:\SWAT\ArcSWAT\Databases\cubillas_hru_playground\Scenarios\Default\TxtInOut\file.cio"

output_dir = r".\swat_ready_recyear_files"


df = pd.read_csv(file_path_2, sep=';')
df.head()

# %%
# Convert the last 6 columns to integers, extracting the first part of the string if necessary
for col in df.columns[-6:]:
    df[col] = df[col].apply(lambda x: int(str(x).split('.')[0]))

# Convert the first column to integers 
df[df.columns[0]] = df[df.columns[0]].apply(lambda x: int(str(x).split(',')[0]))

# Define correct year labels
year_labels = [1970, 1981, 1991, 2001, 2011, 2021]

# Replace the current column names (last 6) with correct years
df.rename(columns=dict(zip(df.columns[-6:], year_labels)), inplace=True)
df.head(20)

# %% [markdown]
# ## Inter- & extra-polate decade data

# %%

def interpolate_dataframe(df, id_column, year_start, year_end, num_year_cols=6):
    """
    Interpolates numeric year data for each unique ID in a DataFrame so that all years that are between existent colums get a column with interpolated values

    Parameters:
    - df: input DataFrame
    - id_column: name of the ID column (string)
    - year_start: start year (inclusive, int)
    - year_end: end year (inclusive, int)
    - num_year_cols: how many columns from the end to treat as year columns (default 6)

    Returns:
    - result_wide: DataFrame with interpolated values, one row per ID, with 'identifier' column first
    """
    # Select year columns (last N columns)
    year_cols = df.columns[-num_year_cols:]
    
    # Clean year columns: drop after comma, convert to int
    for col in year_cols:
        df[col] = df[col].apply(lambda x: int(str(x).split(',')[0]))

    # Create full year range
    full_years = pd.DataFrame({'Year': range(year_start, year_end + 1)})
    
    # Prepare result table
    result = pd.DataFrame({'Year': full_years['Year']})
    
    # Interpolate for each unique ID
    for _, group in df.groupby(id_column):
        subset = group.melt(id_vars=[id_column], value_vars=year_cols, var_name='Year', value_name='Value')
        subset['Year'] = subset['Year'].astype(int)
        merged = full_years.merge(subset, on='Year', how='left').sort_values('Year')
        merged['Value'] = merged['Value'].interpolate(method='linear').ffill().bfill()
        label = f"{group[id_column].values[0]}"
        result[label] = merged['Value'].values

    # Reshape to wide format
    result_wide = result.set_index('Year').T.reset_index()

    # Assign 'identifier' column
    if pd.api.types.is_numeric_dtype(df[id_column]):
        result_wide['identifier'] = pd.to_numeric(result_wide['index'], errors='raise')
    else:
        result_wide['identifier'] = result_wide['index']

    # Drop helper column and reorder
    result_wide.drop(columns=['index'], inplace=True)
    cols = ['identifier'] + [col for col in result_wide.columns if col != 'identifier']
    result_wide = result_wide[cols]

    # Ensure all numeric columns are integers (drop any decimals)
    numeric_cols = result_wide.columns[1:]  # exclude 'identifier'
    result_wide[numeric_cols] = result_wide[numeric_cols].applymap(lambda x: int(float(x)))

    # Rename columns: just the year numbers (no label 'Year')
    result_wide.columns = ['identifier'] + [str(year) for year in range(year_start, year_end + 1)]
    
    return result_wide



# %%
def extrapolate_to_2025_with_fill(df):
    """
    Extrapolates numeric trends from the last two decade columns 
    and fills all years up to 2025 with interpolated/extrapolated values.

    Parameters:
    - df: input DataFrame
    - id_column: name of the ID column (string)

    Returns:
    - df_filled: DataFrame with new year columns up to 2025
    """
    # Identify year columns (numeric names only)
    year_cols = [col for col in df.columns if str(col).isdigit()]
    year_cols_sorted = sorted(year_cols, key=int)

    last_year = int(year_cols_sorted[-1])
    decade_earlier = int(year_cols_sorted[-2])

    # Calculate slope per year
    year_diff = last_year - decade_earlier
    new_years = list(range(last_year + 1, 2026))

    # Make a copy
    df_copy = df.copy()

    # Clean numeric values (strip commas, cast to int)
    for col in [str(decade_earlier), str(last_year)]:
        df_copy[col] = df_copy[col].apply(lambda x: int(str(x).split(',')[0]))

    # For each row, compute and fill values for each new year
    for year in new_years:
        df_copy[str(year)] = df_copy.apply(
            lambda row: row[str(last_year)] + ((year - last_year) / year_diff) * (row[str(last_year)] - row[str(decade_earlier)]),
            axis=1
        ).round().astype(int)

    return df_copy


# %% [markdown]
# ## Building yearly population time series, up till present

# %%
df_interpol_1970_to_2021 = interpolate_dataframe(df, id_column=identifier_column, year_start=1970, year_end=2021, num_year_cols=6)

# %%
df_1970_to_2025 = extrapolate_to_2025_with_fill(df_interpol_1970_to_2021)
df_1970_to_2025 

# %%
# Visualize trends

try:
    import matplotlib.pyplot as plt
    ENABLE_PLOTTING = True
except ImportError:
    ENABLE_PLOTTING = False

if ENABLE_PLOTTING:

    plt.figure(figsize=(14, 8))

    series_list = []
    years = [str(y) for y in range(1970, 2025)]

    # Get the name of the first column (used for labels)
    label_column = df_1970_to_2025.columns[0]

    for idx, row in df_1970_to_2025.iterrows():
        values = row[years].values.flatten()
        label = row[label_column]
        first_value = values[0]
        series_list.append((first_value, label, years, values))

    # Sort by first value (descending) so the legend matches the line starting order
    series_list.sort(reverse=True, key=lambda x: x[0])

    # Plot in sorted order
    for _, label, years, values in series_list:
        plt.plot([int(y) for y in years], values, alpha=0.5, linewidth=1, label=label)

    plt.title('Interpolated Population Over Time for All Polygons')
    plt.xlabel('Year')
    plt.ylabel('Population (integer)')
    plt.grid(True)

    # Show legend if manageable
    if len(series_list) <= 15:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()



# %% [markdown]
# ## Calculating chemical loads from Population data

# %% [markdown]
# #### CONSTANTs: assumed wastewater production from per person per day (liters/day) wastwater production & mg/liter concentration values from literature
# 
# 

# %%
WASTEWATER_L_PER_PERSON_PER_DAY = 150

150 * 15 * 4600 /1000000

# Concentraciones esperadas (mg/L) en aguas residuales - ORDEN COMO SWAT LO REQUIERE SEGUN ch. 31 del swat 2012 io handbook

# Valores tomado desde Metcalf (2000) - "Ingeniería de aguas residuales: tratamiento, vertido y reutilización"
expected_mgL_values = {
    "ORGNYR": 15,       # Nitrógeno orgánico — proteínas, urea, etc.
    "ORGPYR": 3,        # Fósforo orgánico — asociado a materia particulada
    "NO3YR": 0,         # Nitrato — suele ser 0 en aguas residuales crudas (antes de nitrificación)
    "NH3YR": 25,        # Amoníaco libre — forma principal de N inorgánico en agua residual
    "NO2YR": 0,         # Nitrito — normalmente inestable y cercano a cero
    "MINPYR": 5,        # Fósforo inorgánico soluble (PO₄³⁻) — disponible biológicamente
    "SEDYR": 720,       # Sólidos totales en suspensión — proxy para carga de sedimentos
    "CBODYR": 220,      # Demanda Bioquímica de Oxígeno (CBOD / DBO₅) — carga de materia orgánica biodegradable
    "DISOXYR": 2.5,     # Oxígeno disuelto — suele estar en valores bajos en aguas residuales
    "CHLAYR": 0.001     # Clorofila-a — muy baja en aguas residuales (agua turbia impide crecimiento de algas)
}

###### Comentarios explicativos (referencia para revisión técnica)
# ORGNYR: Organic nitrogen concentration (mg/L) — from proteins, urea, etc.
# ORGPYR: Organic phosphorus concentration (mg/L) — associated with organic matter and detritus
# NO3YR: Nitrate concentration (mg/L) — highly soluble, product of nitrification (usually near zero in raw wastewater)
# NH3YR: Ammonia concentration (mg/L) — reduced nitrogen form, main N species in domestic wastewater
# NO2YR: Nitrite concentration (mg/L) — intermediate in nitrification, usually unstable and near zero
# MINPYR: Mineral (soluble) phosphorus concentration (mg/L) — orthophosphate readily bioavailable
# SEDYR: Suspended solids concentration (mg/L) — total suspended solids proxy, major sediment load
# CBODYR: Carbonaceous BOD (mg/L) — biological oxygen demand (BOD5), high in untreated wastewater
# DISOXYR: Dissolved oxygen (mg/L) — low due to high oxygen consumption
# CHLAYR: Chlorophyll-a (mg/L) — proxy for algae biomass, very low in wastewater due to turbidity and low light

# %%
print((150 * 15 * 4600 )/1000000)

# %% [markdown]
# #### Math functions

# %%
def mgL_to_kg_day(mg_per_l, persons):
    """
    Convert concentration (mg/L) to total mass per day (kg/day),
    based on wastewater produced per person.
    Formula: mg/L × liters/day × persons ÷ 1,000,000 → kg/day
    """
    return mg_per_l * WASTEWATER_L_PER_PERSON_PER_DAY * persons / 1_000_000

def build_point_load_timeseries_dataframes(row, expected_mgL_values, years, final_columns=None):
    """
    For a given row (representing one unit, e.g., subbasin), 
    build a DataFrame with yearly SWAT point source values from expected mg/L urban wastewater values. 
    OUTPUT: kg/day for each pollutant variable and total wastewater flow (FLOYR) in m³/day.
    Allows specifying the final columns and their order; fills missing columns with 0s.
    """
    df_out = pd.DataFrame({'YEAR': years})
    # Extract population series from the row
    df_out['POPULATION'] = row[[str(y) for y in years]].values.flatten()
    # Calculate total wastewater flow (FLOYR) in m³/day
    df_out['FLOYR'] = df_out['POPULATION'] * WASTEWATER_L_PER_PERSON_PER_DAY / 1000

    # Prepare columns to fill
    if final_columns is None:
        # Default: all expected_mgL_values keys
        final_columns = ['YEAR', 'FLOYR'] + list(expected_mgL_values.keys())
    else:
        # Ensure 'YEAR' and 'FLOYR' are present
        if 'YEAR' not in final_columns:
            final_columns = ['YEAR'] + final_columns
        if 'FLOYR' not in final_columns:
            final_columns = ['YEAR', 'FLOYR'] + [col for col in final_columns if col not in ('YEAR', 'FLOYR')]

    # Print union/intersection for debug
    expected_vars = set(expected_mgL_values.keys())
    requested_vars = set(final_columns)
    print("Columns in expected_mgL_values:", expected_vars)
    print("Requested final columns:", requested_vars)
    print("Intersection (will be filled):", expected_vars & requested_vars)
    print("Missing in expected_mgL_values (will be filled with 0):", requested_vars - expected_vars - {'YEAR', 'FLOYR', 'POPULATION'})
    print("Extra in expected_mgL_values (not requested):", expected_vars - requested_vars)

    # Fill columns
    for col in final_columns:
        if col in ('YEAR', 'FLOYR', 'POPULATION'):
            continue
        elif col in expected_mgL_values:
            mgL = expected_mgL_values[col]
            df_out[col] = df_out['POPULATION'].apply(lambda p: round(mgL_to_kg_day(mgL, p), 6))
        else:
            df_out[col] = 0

    # Reorder columns
    df_out = df_out[[c for c in final_columns if c in df_out.columns] + [c for c in df_out.columns if c not in final_columns]]

    return df_out

# %% [markdown]
# ## Get number of years our swat model simulates (from file.cio)

# %%
def getModelParameter(prameter:str,parameterfile:str)->int|str|float|None:
        with open(parameterfile,"r") as f:
            for line in f.readlines():
                if(line.find(prameter)!=-1):
                   return line.partition("|")[0].strip()

def getSimulatedPeriod(swatiofile: str) -> tuple[int, int]:
    skip_year = int(getModelParameter("NYSKIP", swatiofile))
    sim_year = int(getModelParameter("NBYR", swatiofile))
    start_year = int(getModelParameter("IYR", swatiofile))
    start_sim_year = start_year + skip_year
    end_sim_year = start_sim_year + sim_year - 1
    return start_sim_year, end_sim_year

start_year, end_year = getSimulatedPeriod(cio_file)
print(f"start_year = {start_year} \nend_year   = {end_year}")

# %% [markdown]
# ## Constructing SWAT ready Tables

# %%
def build_swat_ready_tables(input_df, expected_mgL_values, start_year: int, end_year: int, id_column='GRIDCODE', swat_columns_order=None):
    """
    For an input DataFrame (wide format: ID + year columns),
    generate a dictionary of SWAT-ready DataFrames per ID, limited to a specific simulation period.
    
    Parameters:
    - input_df: DataFrame with one row per unit (e.g., subbasin) and columns: ID + year cols
    - expected_mgL_values: dictionary of variable: mg/L values
    - start_year: first year to include (inclusive)
    - end_year: last year to include (inclusive)
    - id_column: the column name identifying each unit (default: 'GRIDCODE')
    
    Returns:
    - dict { id_value: DataFrame with yearly SWAT variables }
    """
    # Filter only year columns within the simulation period
    years = [int(col) for col in input_df.columns if col.isdigit() and start_year <= int(col) <= end_year]
    
    swat_ready_dataframes = {}

    for idx, row in input_df.iterrows():
        id_value = row[id_column]
        swat_ready_dataframes[id_value] = build_point_load_timeseries_dataframes(row, expected_mgL_values, years, swat_columns_order)
    
    return swat_ready_dataframes


# %%
swat_columns_order = [
    "YEAR", "FLOYR", "SEDYR", "ORGNYR", "ORGPYR", "NO3YR", "NH3YR", "NO2YR",
    "MINPYR", "CBODYR", "DISOXYR", "CHLAYR", "SOLPSTYR", "SRBPSTYR",
    "BACTPYR", "BACTLPYR", "CMTL1YR", "CMTL2YR", "CMTL3YR"
]



swat_ready_dataframes = build_swat_ready_tables(df_1970_to_2025, expected_mgL_values=expected_mgL_values, start_year=start_year, end_year=end_year, id_column='identifier', swat_columns_order=swat_columns_order)
swat_ready_dataframes[17]

# %% [markdown]
# ### Save to swat ready .dat files

# %%


# %%
import os

def write_recyear_files(
    swat_ready_dataframes: dict,
    output_folder: str,
    start_year: int,
    end_year: int,
    swat_columns_order: list,
    df_for_metadata=None
):
    """
    Write each SWAT-ready dataframe to a .dat file following a strict column format:
    - 1 space + 4 right-aligned chars for YEAR column.
    - 1 space + 16 chars for other columns (right-aligned).
    - Floats are adjusted dynamically to fit exactly 16 chars.
    """

    os.makedirs(output_folder, exist_ok=True)

    def format_float_16(value: float) -> str:
        """
        Format a float to exactly 16 characters
        Adjust precision dynamically so the total string length is always 16.
        """
        # Start with scientific notation and trim/expand
        s = f"{value:.10E}"  # start with high precision
        if len(s) > 16:
            # Reduce precision if too long
            for p in range(9, -1, -1):
                s = f"{value:.{p}E}"
                if len(s) <= 16:
                    break
        else:
            # If shorter, pad left
            s = s.rjust(16)
        return s

    for id_value, df in swat_ready_dataframes.items():
        present_columns = [col for col in swat_columns_order if col in df.columns]
        df_filtered = df[present_columns]

        filename = f"rcyr_{id_value}.dat"
        filepath = os.path.join(output_folder, filename)

        # Drainage area
        drainage_area = None
        if df_for_metadata is not None and "GRIDCODE" in df_for_metadata.columns and "Area" in df_for_metadata.columns:
            match = df_for_metadata[df_for_metadata["GRIDCODE"] == id_value]
            if not match.empty:
                area_ha = match.iloc[0]["Area"]
                drainage_area = area_ha / 100.0  # ha to km²

        with open(filepath, 'w') as f:
            # Metadata
            area_str = f"{drainage_area:.3f}" if drainage_area else "0.000"
            f.write(f" TITLE LINE 1 - Subbasin ID {id_value} | Simulation Years: {start_year}-{end_year} | DRAINAGE_AREA (km²): {area_str}\n")
            f.write(" TITLE LINE 2 - Source: TrabajoFM model\n")
            f.write(" TITLE LINE 3 - Units: kg/day\n")
            f.write(f" TITLE LINE 4 - Period: {start_year}-{end_year}\n")
            f.write(" TITLE LINE 5 - \n")

            # Header line
            header_line = f"{'YEAR':>5}"  # 1 space + 4 chars for YEAR
            for col in present_columns:
                if col != 'YEAR':
                    header_line += f"{col:>17}"  # 1 space + 16 chars
            f.write(header_line + "\n")

            # Data lines
            for _, row in df_filtered.iterrows():
                line = f"{int(row.iloc[0]):>5}"  # YEAR
                for v in row.iloc[1:]:
                    line += " " + format_float_16(float(v))
                f.write(line + "\n")

        print(f"✅ File saved: {filepath}")


# %%
write_recyear_files(swat_ready_dataframes, output_dir, start_year, end_year, swat_columns_order, df_for_metadata=df)
