import xarray as xr
import os
import glob
from utils import cache_result, rotate_longitude
from config import TARGET_CMIP5_MODELS, TARGET_CMIP6_MODELS

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
    duacs_ds = xr.open_mfdataset(duacs_pattern, combine='by_coords').load()
    duacs_ds = rotate_longitude(duacs_ds, 'longitude')
    duacs_ds['sla'] *= 100  #m to cm
    duacs_yearly = duacs_ds.groupby('time.year').mean()
    print(f"Altimetry range: {duacs_yearly.sla.min().item():.2f} to {duacs_yearly.sla.max().item():.2f} cm/yr")

    return duacs_yearly

@cache_result('budget_data')
def load_budget_data():
    """Load Frederikse budget data."""
    try:
        budget_parent = find_folder_by_name("Budget")
        budget_dir = os.path.join(budget_parent, "Frederikse")
        if not os.path.exists(budget_dir):
            budget_dir = find_folder_by_name("Frederikse")
    except FileNotFoundError:
        print("Warning: Budget/Frederikse folder not found")
    frederikse_file = os.path.join(budget_dir, 'total.nc')
    ds_frederikse = xr.open_dataset(frederikse_file)

    #geocentric sea level = RSL + RAD
    rsl_component = ds_frederikse['total_rsl_mean']  #Relative Sea Level (RSL)          [mm]
    vlm_component = ds_frederikse['total_rad_mean']  #RAD or Vertical Land Motion (VLM) [mm]
    asl_frederikse = rsl_component + vlm_component   #geocentric sea level

    #rotate and standardize coordinates
    asl_frederikse = rotate_longitude(asl_frederikse, 'lon')
    asl_frederikse = asl_frederikse.rename({'lon': 'longitude', 'lat': 'latitude', 'time': 'year'})

    print(f"Geocentric sea level range: {asl_frederikse.min().item():.2f} to {asl_frederikse.max().item():.2f} cm/yr")

    return asl_frederikse

@cache_result('gia_data')
def load_gia_data():
    """Load GIA data."""
    try:
        gia_dir = find_folder_by_name("GIA")
    except FileNotFoundError:
        budget_parent = find_folder_by_name("Budget")
        gia_dir = os.path.join(budget_parent, "GIA")
    fig_dir = './figures/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print("Loading GIA data...")
    gia_rad_file = os.path.join(gia_dir, 'drad.1grid_O512.nc')
    gia_sea_file = os.path.join(gia_dir, 'dsea.1grid_O512.nc')
    gia_rad = xr.open_dataset(gia_rad_file)
    gia_sea = xr.open_dataset(gia_sea_file)

    #process
    gia_rad_da = gia_rad['Drad_250']  # [mm/yr]
    gia_sea_da = gia_sea['Dsea_250']  # [mm/yr]

    #standardize coordinate names and rotate longitude
    gia_rad_da = gia_rad_da.rename({'Lon': 'longitude', 'Lat': 'latitude'})
    gia_sea_da = gia_sea_da.rename({'Lon': 'longitude', 'Lat': 'latitude'})
    gia_rad_da = rotate_longitude(gia_rad_da, 'longitude')
    gia_sea_da = rotate_longitude(gia_sea_da, 'longitude')

    print(f"GIA rad range: {gia_rad_da.min().item():.2f} to {gia_rad_da.max().item():.2f} mm/yr")
    print(f"GIA sea range: {gia_sea_da.min().item():.2f} to {gia_sea_da.max().item():.2f} mm/yr")
    
    #dataset
    gia_dataset = xr.Dataset({
        'gia_rad': gia_rad_da,
        'gia_sea': gia_sea_da
    })
    
    #metadata
    gia_dataset.attrs['description'] = 'GIA data with rad and sea components'
    gia_dataset['gia_rad'].attrs['long_name'] = 'GIA radial component'
    gia_dataset['gia_rad'].attrs['units'] = 'mm/yr'
    gia_dataset['gia_sea'].attrs['long_name'] = 'GIA sea component'  
    gia_dataset['gia_sea'].attrs['units'] = 'mm/yr'
    
    return gia_dataset

def get_scenario_files(cmip_version, scenario, base_path, return_models=False):
    """Get all NetCDF files for a specific CMIP scenario."""
    #scenario names to folder patterns
    scenario_folders = {
        "CMIP5": {
            "historical": "cmip5_zos_historical",
            "rcp26": "cmip5_zos_rcp26",
            "rcp45": "cmip5_zos_rcp45",
            "rcp85": "cmip5_zos_rcp85"
        },
        "CMIP6": {
            "historical": "cmip6_zos_historical",
            "ssp126": "cmip6_zos_ssp126",
            "ssp245": "cmip6_zos_ssp245",
            "ssp585": "cmip6_zos_ssp585"
        }
    }
    
    folder = scenario_folders[cmip_version].get(scenario)
    if not folder:
        raise ValueError(f"Unknown scenario {scenario} for {cmip_version}")
    
    #find all .nc files in the folder
    pattern = os.path.join(base_path, folder, "*.nc")
    files = glob.glob(pattern)
    
    if return_models:
        #model names from filenames
        models = []
        for f in files:
            basename = os.path.basename(f)
            #remove prefix and suffix to get model name
            #e.g., "cmip6_zos_historical_ACCESS-CM2_1850_2014.nc" -> "ACCESS-CM2"
            parts = basename.replace('.nc', '').split('_')
            model_parts = parts[3:-2] if len(parts) > 5 else parts[3:-1]
            model_name = '_'.join(model_parts)
            models.append(model_name)
        return files, models
    
    return files

@cache_result('cmip_files_inventory')
def get_cmip_files_inventory(cmip_version="CMIP5"):
    """Get inventory of all available CMIP files."""
    print(f"Finding all available {cmip_version} models and files...")
    
    #paths
    CMIP_BASE_PATH = find_folder_by_name("CMIP")
    base_path = os.path.join(CMIP_BASE_PATH, cmip_version)
    
    #define scenarios
    scenarios = {
        "CMIP5": ["historical", "rcp26", "rcp45", "rcp85"],
        "CMIP6": ["historical", "ssp126", "ssp245", "ssp585"]
    }
    
    #find all available files and unique models
    all_files = {}
    all_models = set()
    
    for scenario in scenarios[cmip_version]:
        try:
            files, models = get_scenario_files(cmip_version, scenario, base_path, return_models=True)
            #dictionary for easy lookup: {model_name: filepath}
            all_files[scenario] = {model: file for model, file in zip(models, files)}
            all_models.update(models)
            print(f"  Found {len(files)} files for {scenario}")
        except Exception as e:
            print(f"  Warning: Could not load {scenario}: {e}")
            all_files[scenario] = {}
    
    return {
        'all_files': all_files,
        'all_models': list(all_models),
        'base_path': base_path
    }

#@cache_result('cmip_model_data')
def load_cmip_model_data(model_name, hist_scenario='historical', future_scenario='rcp45', 
                        cmip_version="CMIP5", start_year=None, end_year=None):
    """Load and process a single CMIP model combining historical and future scenarios."""
    
    #files inventory
    inventory = get_cmip_files_inventory(cmip_version)
    all_files = inventory['all_files']
    
    #model check
    hist_file = all_files.get(hist_scenario, {}).get(model_name)
    future_file = all_files.get(future_scenario, {}).get(model_name)
    
    if not hist_file or not future_file:
        raise FileNotFoundError(f"Model {model_name} not found for {hist_scenario} and/or {future_scenario}")
    
    print(f"Loading {model_name} ({cmip_version} {hist_scenario} + {future_scenario})...")
    
    #load datasets
    with xr.open_dataset(hist_file) as ds_hist, xr.open_dataset(future_file) as ds_future:
        #squeeze and rename
        zos_hist = ds_hist['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
        zos_future = ds_future['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
        
        #combine historical and future
        combined_zos = xr.concat([zos_hist, zos_future], dim='time')
        
        #subset time period
        if start_year and end_year:
            combined_zos = combined_zos.sel(time=slice(start_year, end_year))
        
        return combined_zos
    
def load_multiple_cmip_models(model_list=None, hist_scenario='historical', future_scenario='rcp45', cmip_version="CMIP5"):
    """Load multiple CMIP models."""
    
    #target models if no list provided
    if model_list is None:
        if cmip_version == "CMIP5":
            model_list = TARGET_CMIP5_MODELS
        else:
            model_list = TARGET_CMIP6_MODELS
    
    #inventory to check available models
    inventory = get_cmip_files_inventory(cmip_version)
    available_models = inventory['all_models']
    
    #load model
    loaded_models = {}
    failed_models = []
    
    for model_name in model_list:
        if model_name not in available_models:
            print(f"Warning: {model_name} not available in {cmip_version}")
            failed_models.append(model_name)
            continue
            
        try:
            model_data = load_cmip_model_data(
                model_name, 
                hist_scenario=hist_scenario,
                future_scenario=future_scenario,
                cmip_version=cmip_version
            )
            loaded_models[model_name] = model_data
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            failed_models.append(model_name)
    
    print(f"\nSuccessfully loaded {len(loaded_models)} models")
    if failed_models:
        print(f"Failed to load: {', '.join(failed_models)}")
    
    return loaded_models
