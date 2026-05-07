# %%
import os
os.getcwd()

# %% [markdown]
# ## Define script's inputs & outputs

# %%
import pandas as pd

#### Load the CSV file (relative path to this scripts folder)
# The CSV file should have the last columns representing years and their population counts and the first column being an Unique ID
# optimally this CSV comes out of the ArcGIS Pro Model in T"rabajoFM\Genil_ArcGIS_Pascal"

basin_name = "Cubillas"
# Population aggregated according to (sub)basins' subbasins:
file_path = r'..\..\Genil GEO_INFO_POOL\Input Data\Population data\Basin Aggregations\Cubillas population loads\cuenca_cubillas_habitantes_decadas_1970_2021_arcgis_output.csv'
identifier_column = "GRIDCODE"

# to extract years that have been and shall be simulated
cio_file = r"..\..\Genil GEO_INFO_POOL\SWaT outputs\file.cio"

output_dir = r".\swat_ready_recyear_files"


df = pd.read_csv(file_path, sep=';')
df.head()

# %%
# Convert the last 6 columns to integers, extracting the first part of the string if necessary
for col in df.columns[-6:]:
    df[col] = df[col].apply(lambda x: int(str(x).split(',')[0]))

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
    "CHLAYR": 0.001,     # Clorofila-a — muy baja en aguas residuales (agua turbia impide crecimiento de algas)

    "SOLPSTYR": 0,      # JUST to satisfy a swat
    "SRBPSTYR": 0,
    "BACTPYR": 0,
    "BACTLPYR": 0,
    "CMTL1YR": 0,
    "CMTL2YR": 0,
    "CMTL3YR": 0,
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

def build_point_load_timeseries_dataframes(row, expected_mgL_values, years):
    """
    For a given row (representing one unit, e.g., subbasin), 
    build a DataFrame with yearly SWAT point source values from expeted mg/L urban wastewater values. 
    OUTPUT: kg/day for each pollutant variable and total wastewater flow (FLOYR) in m³/day.
    """
    df_out = pd.DataFrame({'YEAR': years})
    
    # Extract population series from the row
    df_out['POPULATION'] = row[[str(y) for y in years]].values.flatten()
    
    # Calculate total wastewater flow (FLOYR) in m³/day
    df_out['FLOYR'] = df_out['POPULATION'] * WASTEWATER_L_PER_PERSON_PER_DAY / 1000
    
    # For each pollutant variable, compute kg/day using the defined mg/L value
    for var, mgL in expected_mgL_values.items():
        df_out[var] = df_out['POPULATION'].apply(lambda p: round(mgL_to_kg_day(mgL, p), 6))

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
# #### Constructing SWAT ready Tables

# %%
def build_swat_ready_tables(input_df, expected_mgL_values, start_year: int, end_year: int, id_column='GRIDCODE'):
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
        swat_ready_dataframes[id_value] = build_point_load_timeseries_dataframes(row, expected_mgL_values, years)
    
    return swat_ready_dataframes


# %%
swat_ready_dataframes = build_swat_ready_tables(df_1970_to_2025, expected_mgL_values=expected_mgL_values, start_year=start_year, end_year=end_year, id_column='identifier')
swat_ready_dataframes[17]

# %% [markdown]
# ## Save to swat ready .dat files

# %%
def write_recyear_files(swat_ready_dataframes: dict, output_folder: str, start_year: int, end_year: int):
    """
    Write each SWAT-ready dataframe to a .dat file following the RECYEAR.DAT format,
    with structured metadata headers and console reporting.

    Parameters:
    - swat_ready_dataframes: dict { id_value: DataFrame with yearly values }
    - output_folder: path to save the .dat files
    - start_year, end_year: simulation period for metadata reporting
    """
    os.makedirs(output_folder, exist_ok=True)

    valid_columns = [
        "YEAR", "FLOYR", "SEDYR", "ORGNYR", "ORGPYR", "NO3YR", "NH3YR", "NO2YR",
        "MINPYR", "CBODYR", "DISOXYR", "CHLAYR", "SOLPSTYR", "SRBPSTYR",
        "BACTPYR", "BACTLPYR", "CMTL1YR", "CMTL2YR", "CMTL3YR"
    ]

    for id_value, df in swat_ready_dataframes.items():
        present_columns = [col for col in valid_columns if col in df.columns]
        df_filtered = df[present_columns]

        filename = f"recyear_{id_value}.dat"
        filepath = os.path.join(output_folder, filename)

        # --- Console metadata ---
        print(f"\n📝 Writing RECYEAR file for {basin_name}'s Subbasin with ID: {id_value}")
        print(f"   ▸ Years covered: {start_year}–{end_year}")
        print(f"   ▸ Variables included ({len(present_columns)}): {', '.join(present_columns)}")
        print(f"   ▸ Method: Calculated from interpolated population * wastewater generation")
        print(f"   ▸ Source: Aggregated ArcGIS population maps (TrabajoFM model output)")
        print(f"   ▸ Literature: Metcalf (2000) for mg/L assumptions\n")

        with open(filepath, 'w') as f:
            # Metadata title lines (SWAT ignores, but humans will love it)
            f.write(f"TITLE LINE 1 - Subbasin ID {id_value} | Simulation Years: {start_year}-{end_year}\n")
            f.write(f"TITLE LINE 2 - Source: Interpolated population data from ArcGIS + expected mg/L loads (Metcalf, 2000)\n")
            f.write(f"TITLE LINE 3 - Method: kg/day = (pop * conc (mg/L) * 150 L/person/day) / 1,000,000\n")
            f.write(f"TITLE LINE 4 - Generated by TrabajoFM model | Date range: {start_year}-{end_year}\n")
            f.write(f"TITLE LINE 5 - \n")
            f.write(f"TITLE LINE 6 - Variables: {', '.join(present_columns)}\n")

            # Write data lines
            for _, row in df_filtered.iterrows():
                line = " ".join(f"{v:.6g}" if isinstance(v, float) else str(int(v)) for v in row.values)
                f.write(line + "\n")

        print(f"✅ File saved: {filepath}")


# %%
write_recyear_files(swat_ready_dataframes, output_dir, start_year, end_year)

# %% [markdown]
# ## Archived:

# %% [markdown]
# ## save swat ready dataframes to .csv

# %%
""" 

import os

# save the DataFrames to Excel files

output_folder = 'C:\\Users\\Usuario\\OneDrive - UNIVERSIDAD DE HUELVA\\Granada\\TrabajoFM\\Genil GEO_INFO_POOL\\Input Data\\Population data\\Basin Aggregations\\Cubillas population loads\\test'

os.makedirs(output_folder, exist_ok=True)

for name, df in swat_ready_dataframes.items():
    safe_name = str(name).replace(' ', '_').replace('/', '_')  # clean filename
    file_path = os.path.join(output_folder, f'{safe_name}.xlsx')  # or .csv
    df.to_excel(file_path, index=False)  # or df.to_csv()


 """

# %% [markdown]
# ## master dataframe of timeseries of all point loads for all subbasins together:

# %%
""" 

# master dataframe of timeseries of all point loads for all subbasins together:

# Get all DataFrames into a list
dfs = list(swat_ready_dataframes.values())

# Get the name of the first (non-numeric) column
id_column_name = dfs[0].columns[0]
id_column_values = dfs[0][id_column_name]

# Initialize summed DataFrame (excluding identifier column)
summed_df = dfs[0].drop(columns=[id_column_name]).copy()

# Add the rest
for df in dfs[1:]:
    summed_df += df.drop(columns=[id_column_name])

# Optionally reattach the identifier if you want (but it may not make sense when summing)
summed_df.insert(0, id_column_name, id_column_values)



# Manual test to ensure the summed DataFrame is correct
population_sum_1970 = 0
for df in dfs:
    population_sum_1970 += df[df['YEAR'] == 1970].iloc[0, 1]
print(f"Total population in 1970 across all subbasins: {population_sum_1970}")

# Result: summed_df contains the matrix sum of all numeric columns
summed_df

#summed_df.to_csv(output_folder+r"\\summbed_cubillas_point_loads.csv", index=False)

 """


