"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Data loading functions for ODSL analysis.
"""

### --- OBSERVATIONAL DATA --- ###
#----------------------------------------------------------------------------------------#
# --- Altimetry data ---
# DOI: 10.48670/moi-00148
# Link to data: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
#----------------------------------------------------------------------------------------#
# --- Budget data ---
# Frederikse, T., Landerer, F., Caron, L., Adhikari, S., Parkes, D., Humphrey, V. W., ... & Wu, Y. H. (2020). The causes of sea-level rise since 1900. Nature, 584(7821), 393-397.
# Link to data: https://zenodo.org/records/3862995
#----------------------------------------------------------------------------------------#
# --- GIA data ---
# Peltier, W. R., Argus, D. F., & Drummond, R. (2015). Space geodesy constrains ice age terminal deglaciation: The global ICE‐6G_C (VM5a) model. Journal of Geophysical Research: Solid Earth, 120(1), 450-487.
# Link to data: https://www.atmosp.physics.utoronto.ca/~peltier/data.php
#----------------------------------------------------------------------------------------#
# --- Wind data ---
# Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Hor´anyi, A., Mu˜noz Sabater, J., . . . Th´epaut, J.-N. (2023).
# ERA5 monthly averaged data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 
# Retrieved from https://doi.org/10.24381/cds.f17050d7 (Accessed: 2026-01-23) doi: 10.24381/cds.f17050d7
#----------------------------------------------------------------------------------------#


### --- CMIP DATA --- ###
#----------------------------------------------------------------------------------------#
# Preprocessing done by Dewi Le Bars, KNMI
# https://github.com/dlebars/CMIP_SeaLevel/blob/master/code/PreparePlaneVariables.py
#----------------------------------------------------------------------------------------#

from utils import (cache_result, rotate_longitude, create_region_mask)
from config import (CMIP_VERSION, TARGET_CMIP5_MODELS, TARGET_CMIP6_MODELS, CMIP_SCENARIOS, CMIP5_FUTURE_SCENARIO, CMIP6_FUTURE_SCENARIO, START_YEAR, END_YEAR, EXTENT)

import xarray as xr
import os
import glob
import pandas as pd
import numpy as np
import re

#correct configuration
if CMIP_VERSION == 'CMIP5':
    TARGET_MODELS = TARGET_CMIP5_MODELS
    FUTURE_SCENARIO = CMIP5_FUTURE_SCENARIO
elif CMIP_VERSION == 'CMIP6':
    TARGET_MODELS = TARGET_CMIP6_MODELS
    FUTURE_SCENARIO = CMIP6_FUTURE_SCENARIO
else:
    raise ValueError(f"Unsupported CMIP_VERSION: {CMIP_VERSION}. Must be 'CMIP5' or 'CMIP6'.")

def find_folder_by_name(folder_name, start_path=None, max_depth=5):
    """Search for a folder by name, starting from current directory and going up."""

    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
    
    #search upward from current location
    current = start_path
    for _ in range(max_depth):
        for root, dirs, files in os.walk(current):
            if folder_name in dirs:
                return os.path.join(root, folder_name)
        #up one directory
        current = os.path.dirname(current)
        if current == os.path.dirname(current): 
            break
    
    raise FileNotFoundError(f"Folder '{folder_name}' not found within {max_depth} levels")

@cache_result('altimetry_data')
def load_altimetry_data():
    """Load and process altimetry data."""

    print("Loading altimetry data...")

    try:
        duacs_dir = find_folder_by_name("Altimetry")
    except FileNotFoundError:
        print("Warning: Altimetry folder not found, trying alternative search...")
        data_path = find_folder_by_name("Data")
        duacs_dir = os.path.join(data_path, "Altimetry")

    duacs_pattern = os.path.join(duacs_dir, 'cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_*.nc')
    duacs_ds      = xr.open_mfdataset(duacs_pattern, combine='by_coords', chunks={'time': 12})
    duacs_ds      = rotate_longitude(duacs_ds, 'longitude')
    duacs_ds['sla'] *= 100  #m to cm

    print(f"Altimetry data loaded.")

    return duacs_ds

@cache_result('budget_data')
def load_budget_data(extend_to_year=None):
    """Load Frederikse budget data."""

    try:
        budget_parent = find_folder_by_name("Budget")
        budget_dir = os.path.join(budget_parent, "Frederikse")
        if not os.path.exists(budget_dir):
            budget_dir = find_folder_by_name("Frederikse")
    except FileNotFoundError:
        print("Warning: Budget/Frederikse folder not found")
        raise

    frederikse_file = os.path.join(budget_dir, 'total.nc')
    ds_frederikse   = xr.open_dataset(frederikse_file)

    #geocentric sea level = RSL + RAD
    rsl_component  = ds_frederikse['total_rsl_mean']  #Relative Sea Level (RSL)          [mm]
    vlm_component  = ds_frederikse['total_rad_mean']  #RAD or Vertical Land Motion (VLM) [mm]
    asl_frederikse = rsl_component + vlm_component   #geocentric sea level

    #rotate and standardize coordinates
    asl_frederikse = rotate_longitude(asl_frederikse, 'lon')
    asl_frederikse = asl_frederikse.rename({'lon': 'longitude', 'lat': 'latitude', 'time': 'year'})

    #extend budget data beyond 2018 if needed based on the last 10 years linear trend
    if extend_to_year and asl_frederikse.year.max() < extend_to_year:
        print(f"Extending budget data to year {extend_to_year}...")
        last_year = asl_frederikse.year.max().item()
        
        #last 10 years for trend calculation
        trend_period = asl_frederikse.sel(year=slice(last_year-9, last_year))
        
        #trend for each grid cell
        slope = trend_period.polyfit(dim='year', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)
        
        last_data = asl_frederikse.sel(year=last_year)
        
        years_to_add       = range(last_year + 1, extend_to_year + 1)
        extended_data_list = []
        for year_to_add in years_to_add:
            years_diff         = year_to_add - last_year
            extrapolated_slice = last_data + (slope * years_diff)
            extrapolated_slice = extrapolated_slice.assign_coords(year=year_to_add)
            extended_data_list.append(extrapolated_slice)
            
        extended_data  = xr.concat(extended_data_list, dim='year')
        asl_frederikse = xr.concat([asl_frederikse, extended_data], dim='year') # type: ignore

    print(f"Budget data sea level range: {asl_frederikse.min().item():.2f} to {asl_frederikse.max().item():.2f} mm")

    return asl_frederikse

@cache_result('gia_data')
def load_gia_data():
    """Load GIA data."""

    try:
        gia_dir = find_folder_by_name("GIA")
    except FileNotFoundError:
        budget_parent = find_folder_by_name("Budget")
        gia_dir       = os.path.join(budget_parent, "GIA")
    fig_dir = './figures/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print("Loading GIA data...")
    
    gia_rad_file = os.path.join(gia_dir, 'drad.1grid_O512.nc')
    gia_sea_file = os.path.join(gia_dir, 'dsea.1grid_O512.nc')
    gia_rad      = xr.open_dataset(gia_rad_file)
    gia_sea      = xr.open_dataset(gia_sea_file)

    #process
    gia_rad_da = gia_rad['Drad_250']  #[mm/yr]
    gia_sea_da = gia_sea['Dsea_250']  #[mm/yr]

    #standardize coordinate names and rotate longitude
    gia_rad_da = gia_rad_da.rename({'Lon': 'longitude', 'Lat': 'latitude'})
    gia_sea_da = gia_sea_da.rename({'Lon': 'longitude', 'Lat': 'latitude'})
    gia_rad_da = rotate_longitude(gia_rad_da, 'longitude')
    gia_sea_da = rotate_longitude(gia_sea_da, 'longitude')

    print(f"GIA rad range: {gia_rad_da.min().item():.2f} to {gia_rad_da.max().item():.2f} mm/yr")
    print(f"GIA sea range: {gia_sea_da.min().item():.2f} to {gia_sea_da.max().item():.2f} mm/yr")
    
    #dataset
    gia_dataset = xr.Dataset({'gia_rad': gia_rad_da, 'gia_sea': gia_sea_da})
    
    #metadata
    gia_dataset.attrs['description']          = 'GIA data with rad and sea components'
    gia_dataset['gia_rad'].attrs['long_name'] = 'GIA radial component'
    gia_dataset['gia_rad'].attrs['units']     = 'mm/yr'
    gia_dataset['gia_sea'].attrs['long_name'] = 'GIA sea component'  
    gia_dataset['gia_sea'].attrs['units']     = 'mm/yr'
    
    return gia_dataset

def get_scenario_files(cmip_version, scenario, base_path, return_models=False):
    """Get all NetCDF files for a specific CMIP scenario."""

    folder = CMIP_SCENARIOS.get(cmip_version, {}).get(scenario)
    if not folder:
        raise ValueError(f"Unknown scenario {scenario} for {cmip_version}")
    
    pattern = os.path.join(base_path, folder, "*.nc")
    files   = glob.glob(pattern)
    
    if return_models:
        models = []
        for f in files:
            basename = os.path.basename(f).replace('.nc', '')
            #remove prefix and suffix to get model name
            #e.g., "cmip6_zos_historical_ACCESS-CM2_1850_2014.nc" -> "ACCESS-CM2"
            try:
                core_name = basename.rsplit('_', 2)[0]
                prefix    = f"{cmip_version.lower()}_zos_{scenario}_"

                if core_name.startswith(prefix):
                    model_name = core_name.removeprefix(prefix)
                    models.append(model_name)
                else:
                    print(f"Warning: Filename '{basename}' did not match expected prefix '{prefix}'. Skipping.")

            except IndexError:
                print(f"Warning: Could not parse model name from filename '{basename}'. Skipping.")

        return files, models
    
    return files

@cache_result('cmip_files_inventory')
def get_cmip_files_inventory(cmip_version):
    """Get inventory of all available CMIP files."""

    print(f"Finding all available {cmip_version} models and files...")
    
    #paths
    CMIP_BASE_PATH = find_folder_by_name("CMIP")
    base_path      = os.path.join(CMIP_BASE_PATH, cmip_version)
    
    #define scenarios
    scenarios = CMIP_SCENARIOS.get(cmip_version, {})
    
    #find all available files and unique models
    all_files  = {}
    all_models = set()
    
    for scenario in scenarios:
        print(f"Searching for {scenario} files...")
        try:
            files, models       = get_scenario_files(cmip_version, scenario, base_path, return_models=True)
            #dictionary for easy lookup: {model_name: filepath}
            all_files[scenario] = {model: file for model, file in zip(models, files)}
            all_models.update(models)
            print(f"Found {len(files)} files for {scenario}")
        except Exception as e:
            print(f"Warning: Could not load {scenario}: {e}")
            all_files[scenario] = {}
    
    return {'all_files': all_files, 'all_models': list(all_models), 'base_path': base_path}

def load_cmip_model_data(model_name, hist_scenario='historical', future_scenario=None, cmip_version=None, start_year=None, end_year=None):
    """Load and process a single CMIP model, optionally combining historical and future scenarios."""
    
    if cmip_version is None:
        raise ValueError("CMIP_VERSION must be provided ('CMIP5' or 'CMIP6').")

    #files inventory
    inventory = get_cmip_files_inventory(cmip_version)
    all_files = inventory['all_files']
    
    #model check
    hist_file = all_files.get(hist_scenario, {}).get(model_name)
    
    future_file = None
    if future_scenario:
        future_file = all_files.get(future_scenario, {}).get(model_name)
    
    if not hist_file:
        raise FileNotFoundError(f"Model {model_name} not found for {hist_scenario}")
    
    if future_scenario and not future_file:
         raise FileNotFoundError(f"Model {model_name} not found for future scenario {future_scenario}")
    
    print(f"Loading {model_name} ({cmip_version} {hist_scenario}" + (f" + {future_scenario}" if future_file else "").strip() + ")...")
    
    #load datasets
    with xr.open_dataset(hist_file) as ds_hist:
        zos_hist = ds_hist['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
        
        if future_file:
            with xr.open_dataset(future_file) as ds_future:
                zos_future = ds_future['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
                #combine historical and future
                combined_zos = xr.concat([zos_hist, zos_future], dim='time')
        else:
            combined_zos = zos_hist
        
        #subset time period
        if start_year or end_year:
            time_slice   = slice(start_year, end_year)
            combined_zos = combined_zos.sel(time=time_slice)
        
        return combined_zos

@cache_result('amo_index_yearly')
def load_amo_index():
    """Loads the ERSST AMO index from the provided text file, skips the header, and computes annual averages."""
    
    print("Loading AMO index data...")

    try:
        modes_dir     = find_folder_by_name("Modes")
        amo_file_path = os.path.join(modes_dir, "AMO", "AMO_monthly.txt")

        #load
        df = pd.read_csv(amo_file_path, skiprows=2, header=None, sep=r'\s+', names=['year', 'month', 'ssta'])

        #annual mean SSTA
        yearly_amo = df.groupby('year')['ssta'].mean().reset_index()

        #xarray DataArray
        amo_da         = xr.DataArray(data=yearly_amo['ssta'].values, coords={'year': yearly_amo['year'].values}, dims=['year'],name='amo_index').rename({'year': 'time'})
        amo_da['time'] = amo_da['time'].astype(int)
        
        print(f"Loaded AMO index from {yearly_amo['year'].min()} to {yearly_amo['year'].max()}")

        return amo_da.to_dataset(name='amo_index')

    except FileNotFoundError:
        print("Warning: AMO_monthly.txt not found in the expected Modes/AMO directory.")
        return None
    
    except Exception as e:
        print(f"An error occurred while loading the AMO index: {e}")
        return None
      
@cache_result('ao_index_yearly')
def load_ao_index():
    """Loads the monthly AO index from the provided text file, handles its specific multi-column format, and computes annual averages."""

    print("Loading AO index data...")

    try:
        modes_dir    = find_folder_by_name("Modes")
        ao_file_path = os.path.join(modes_dir, "AO", "AO_monthly.txt")

        #load
        df = pd.read_csv(ao_file_path, header=0, sep=r'\s+', index_col=0)

        yearly_ao      = df.mean(axis=1)
        yearly_ao.name = 'ao_index'
        ao_da          = yearly_ao.to_xarray()
        ao_da          = ao_da.rename({'index': 'time'})
        ao_da['time']  = ao_da['time'].astype(int)

        print(f"Loaded AO index from {int(ao_da.time.min().item())} to {int(ao_da.time.max().item())}")

        return ao_da.to_dataset(name='ao_index')

    except FileNotFoundError:
        print("Warning: AO_monthly.txt not found in the expected Modes/AO directory.")
        return None
    
    except Exception as e:
        print(f"An error occurred while loading the AO index: {e}")
        return None

@cache_result('nh_indices_yearly')
def load_nh_teleconnection_indices():
    """Northern Hemisphere teleconnection indices."""
    
    print("Loading NH teleconnection indices from combined file...")

    try:
        modes_dir = find_folder_by_name("Modes")
        file_path = os.path.join(modes_dir, "tele_index.txt")

        #column names
        col_names = ['year', 'month', 'nao', 'ea', 'wp', 'epnp', 'pna', 'eawr', 'sca', 'tnh', 'pol', 'pt', 'expl_var']

        #read
        data_rows = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        #skip header lines
        for line in lines[13:]:
            line = line.strip()
            if not line:
                continue
            
            #split negative numbers (regex)
            values = re.findall(r'-?\d+\.?\d*', line)
            
            if len(values) >= 13:
                data_rows.append([float(v) for v in values[:13]])
        
        df = pd.DataFrame(data_rows, columns=col_names)

        #exclude tnh and pt because they are seasonal instead of yearly
        index_cols = ['nao', 'ea', 'wp', 'epnp', 'pna', 'eawr', 'sca', 'pol', 'expl_var']
        
        #-99.9 or -99.90 = nan
        for col in index_cols:
            df[col] = df[col].replace([-99.9, -99.90], np.nan)

        #yearly avg
        yearly_df = df.groupby('year')[index_cols].mean()

        #dictionary of xarray DataArrays for caching
        indices_dict = {}
        for col in index_cols:
            da                = xr.DataArray(data=yearly_df[col].values, coords={'time': yearly_df.index.values.astype(int)}, dims=['time'], name=f'{col}_index')
            da['time']        = da['time'].astype(int)
            indices_dict[col] = da

        print(f"Loaded indices from {int(yearly_df.index.min())} to {int(yearly_df.index.max())}")
        print(f"Available indices: {', '.join(index_cols)}")

        return indices_dict

    except FileNotFoundError:
        print("Warning: NH teleconnection indices file not found.")
        return None
    
    except Exception as e:
        print(f"An error occurred while loading teleconnection indices: {e}")
        return None

def load_climate_indices_dict():
    """Load all climate indices into a dictionary."""

    ao_ds      = load_ao_index()
    amo_ds     = load_amo_index()
    nh_indices = load_nh_teleconnection_indices()
    
    if nh_indices is None:
        nh_indices = {}

    def ensure_dataarray(data):
        """Ensure data is a dataarray for caching."""

        if data is None:
            return None
        
        if isinstance(data, xr.Dataset):
            var_name = list(data.data_vars)[0]
            return data[var_name]
        return data
    
    return {'ao':       ensure_dataarray(ao_ds['ao_index'] if ao_ds is not None else None),
            'amo':      ensure_dataarray(amo_ds['amo_index'] if amo_ds is not None else None),
            'nao':      ensure_dataarray(nh_indices.get('nao')),
            'ea':       ensure_dataarray(nh_indices.get('ea')), 
            'wp':       ensure_dataarray(nh_indices.get('wp')),
            'epnp':     ensure_dataarray(nh_indices.get('epnp')),
            'pna':      ensure_dataarray(nh_indices.get('pna')),
            'eawr':     ensure_dataarray(nh_indices.get('eawr')),
            'sca':      ensure_dataarray(nh_indices.get('sca')),
            'pol':      ensure_dataarray(nh_indices.get('pol')),
            'expl_var': ensure_dataarray(nh_indices.get('expl_var'))}

@cache_result('wind_stress_data')
def load_wind_stress_data():
    """Load ERA5 wind stress data."""

    print("Loading wind stress data...")

    try:
        wind_dir = find_folder_by_name("Wind")
    except FileNotFoundError:
        data_path = find_folder_by_name("Data")
        wind_dir = os.path.join(data_path, "Wind")

    nc_files = glob.glob(os.path.join(wind_dir, '*.nc'))

    if nc_files:
        ds = xr.open_dataset(nc_files[0])
    else:
        raise FileNotFoundError(f"No wind stress files found in {wind_dir}")

    #variables
    tau_x_name = 'avg_iews'
    tau_y_name = 'avg_inss'

    if tau_x_name not in ds.data_vars or tau_y_name not in ds.data_vars:
        raise KeyError(f"Could not identify wind stress variables. Available: {list(ds.data_vars)}")

    ds = ds.rename({tau_x_name: 'tau_x', tau_y_name: 'tau_y'})

    #time
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    ds = ds.drop_vars(['number', 'expver', 'step', 'surface'], errors='ignore')

    #yearly avg
    ds_yearly = ds.groupby('time.year').mean('time').compute()
    ds_yearly = ds_yearly.sel(year=slice(START_YEAR, END_YEAR))
    ds_yearly = rotate_longitude(ds_yearly, 'longitude')
    ds_yearly = ds_yearly.sortby('latitude')

    tau_x_mean = ds_yearly['tau_x'].mean(dim='year')
    tau_y_mean = ds_yearly['tau_y'].mean(dim='year')

    region_mask = create_region_mask(tau_x_mean, EXTENT)

    return xr.Dataset({'tau_x_mean': tau_x_mean.where(region_mask), 'tau_y_mean': tau_y_mean.where(region_mask)})

