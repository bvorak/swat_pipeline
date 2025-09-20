import os
import pandas as pd
from datetime import datetime, date, timedelta

def parse_swat_sub_to_df(sub_file_path: str) -> pd.DataFrame:
    """
    Parses a SWAT .sub file and its corresponding .cio file in the same folder,
    returning a DataFrame with decoded dates and additional columns.

    Parameters:
    - sub_file_path: Path to the .sub file.

    Returns:
    - pd.DataFrame with parsed SWAT subbasin results.
    """
    # Find .cio file in the same folder
    folder = os.path.dirname(sub_file_path)
    cio_files = [f for f in os.listdir(folder) if f.lower().endswith('.cio')]
    if not cio_files:
        raise FileNotFoundError(f"No .cio file found in folder: {folder}")
    cio_file_path = os.path.join(folder, cio_files[0])

    # Column specs for SWAT .sub file
    column_specs = [
        {"start": 7, "end": 11, "name": "SUB", "type": "int", "desc": "Subbasin number"},
        {"start": 11, "end": 20, "name": "GIS", "type": "int", "desc": "GIS code from .fig"},
        {"start": 20, "end": 25, "name": "MON", "type": "int", "desc": "Julian date or month"},
        {"start": 26, "end": 35, "name": "AREA", "type": "float", "desc": "Area of subbasin (km²)"},
        {"start": 35, "end": 45, "name": "PRECIP", "type": "float", "desc": "Precipitation (mm/day)"},
        {"start": 45, "end": 55, "name": "SNOMELT", "type": "float", "desc": "Snowmelt (mm/day)"},
        {"start": 55, "end": 65, "name": "PET", "type": "float", "desc": "Potential ET (mm/day)"},
        {"start": 65, "end": 75, "name": "ET", "type": "float", "desc": "Actual ET (mm/day)"},
        {"start": 75, "end": 85, "name": "SW", "type": "float", "desc": "Soil water content (mm)"},
        {"start": 85, "end": 95, "name": "PERC", "type": "float", "desc": "Percolation past root zone (mm)"},
        {"start": 95, "end": 105, "name": "SURQ", "type": "float", "desc": "Surface runoff (mm)"},
        {"start": 105, "end": 115, "name": "GW_Q", "type": "float", "desc": "Groundwater contribution to streamflow (mm)"},
        {"start": 115, "end": 125, "name": "WYLD", "type": "float", "desc": "Water yield (mm)"},
        {"start": 125, "end": 135, "name": "SYLD", "type": "float", "desc": "Sediment yield (t/ha)"},
        {"start": 135, "end": 145, "name": "ORGN", "type": "float", "desc": "Organic nitrogen yield (kg N/ha)"},
        {"start": 145, "end": 155, "name": "ORGP", "type": "float", "desc": "Organic phosphorus yield (kg P/ha)"},
        {"start": 155, "end": 165, "name": "NSURQ", "type": "float", "desc": "NO3 in surface runoff (kg N/ha)"},
        {"start": 165, "end": 175, "name": "SOLP", "type": "float", "desc": "Soluble phosphorus yield (kg P/ha)"},
        {"start": 175, "end": 185, "name": "SEDP", "type": "float", "desc": "Mineral phosphorus yield (kg P/ha)"}
    ]

    def safe_cast(value_str, value_type, col_name):
        value_str = value_str.strip()
        if not value_str:
            return 0
        try:
            if value_type == 'int':
                return int(float(value_str))
            elif value_type == 'float':
                return float(value_str)
            else:
                return value_str
        except ValueError:
            return 0

    def parse_sub_file_to_df(inputfile: str, column_specs: list) -> pd.DataFrame:
        with open(inputfile, 'r') as f:
            lines = f.readlines()[9:]
        data = []
        for line in lines:
            row = []
            for spec in column_specs:
                value_str = line[spec['start']: spec['end']]
                value = safe_cast(value_str, spec['type'], spec['name'])
                row.append(value)
            data.append(row)
        col_names = [spec['name'] for spec in column_specs]
        return pd.DataFrame(data, columns=col_names)

    def getModelParameter(prameter:str,parameterfile:str)->int|str|float|None:
        with open(parameterfile,"r") as f:
            for line in f.readlines():
                if(line.find(prameter)!=-1):
                   return line.partition("|")[0].strip()

    def getStartDate(swatiofile: str) -> date:
        skip_year = int(getModelParameter("NYSKIP", swatiofile))
        sim_year = int(getModelParameter("NBYR", swatiofile))
        start_year = int(getModelParameter("IYR", swatiofile))
        start_day = int(getModelParameter("IDAF", swatiofile))
        day = date(start_year+skip_year, 1, 1)+timedelta(days=start_day-1)
        return day

    # Parse .sub file
    df_sub = parse_sub_file_to_df(sub_file_path, column_specs)

    # Datetime decoding
    df_sub.index.name = 'total_days'
    df_sub.reset_index(drop=False, inplace=True)
    df_sub["total_days"] = df_sub["total_days"] // 17

    startDate = getStartDate(swatiofile=cio_file_path)
    df_sub["date"] = df_sub["total_days"].apply(lambda x: timedelta(days=x) + startDate)
    df_sub["date"] = pd.to_datetime(df_sub["date"])
    df_sub["YEAR"] = df_sub["date"].dt.year

    df_sub.insert(3, "date", df_sub.pop("date"))
    df_sub.insert(4, "YEAR", df_sub.pop("YEAR"))

    df_sub["area_ha"] = df_sub["AREA"] * 100  # 100 hectare = 1 km²
    df_sub.insert(6, "area_ha", df_sub.pop("area_ha"))

    return df_sub
