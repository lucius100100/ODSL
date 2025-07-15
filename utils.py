import xarray as xr
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from functools import wraps
import config

#cache directory
CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(exist_ok=True)

def cache_result(cache_name):
    """Cache function, prioritize netcdf and csv, pickle only if necessary."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            #cache paths
            nc_path = CACHE_DIR / f"{cache_name}.nc"
            csv_path = CACHE_DIR / f"{cache_name}.csv"
            json_path = CACHE_DIR / f"{cache_name}.json"
            pkl_path = CACHE_DIR / f"{cache_name}.pkl"

            #load from cache
            if config.USE_CACHE and not config.FORCE_RECOMPUTE:
                if nc_path.exists():
                    print(f"Loading cached xarray data from: {nc_path}")
                    return xr.open_dataset(nc_path)
                
                if csv_path.exists():
                    print(f"Loading cached DataFrame from: {csv_path}")
                    return pd.read_csv(csv_path, index_col=0)
                
                if json_path.exists():
                    print(f"Loading cached data from JSON: {json_path}")
                    with open(json_path, 'r') as f:
                        return json.load(f)
                
                if pkl_path.exists():
                    print(f"Loading cached data from pickle: {pkl_path}")
                    with open(pkl_path, 'rb') as f:
                        return pickle.load(f)

            #compute and save
            print(f"Computing {cache_name}...")
            result = func(*args, **kwargs)

            if config.USE_CACHE:
                try:
                    #netcdf
                    if isinstance(result, (xr.Dataset, xr.DataArray)):
                        print(f"Caching result as NetCDF to: {nc_path}")
                        result.to_netcdf(nc_path)
                    
                    #csv
                    elif isinstance(result, pd.DataFrame):
                        print(f"Caching result as CSV to: {csv_path}")
                        result.to_csv(csv_path)
                    
                    #json
                    elif isinstance(result, (dict, list)):
                        print(f"Caching result as JSON to: {json_path}")
                        json_result = _convert_for_json(result)
                        with open(json_path, 'w') as f:
                            json.dump(json_result, f, indent=2)
                    
                    #pickle
                    else:
                        print(f"Caching result as pickle to: {pkl_path}")
                        with open(pkl_path, 'wb') as f:
                            pickle.dump(result, f)
                            
                except Exception as e:
                    print(f"Warning: Could not cache {cache_name}: {e}")

            return result
        return wrapper
    return decorator

def _convert_for_json(obj):
    """Convert numpy types and other non-JSON-serializable objects to JSON-compatible types"""
    if isinstance(obj, dict):
        return {key: _convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (xr.Dataset, xr.DataArray)):
        return f"<xarray.{type(obj).__name__}>"
    else:
        return obj

def rotate_longitude(ds, name_lon):
    """Convert longitude from 0-360 to -180-180 coordinate system."""
    ds_copy = ds.copy()
    new_lon_values = (((ds_copy[name_lon] + 180) % 360) - 180)
    ds_copy = ds_copy.assign_coords({name_lon: new_lon_values})
    return ds_copy.sortby(ds_copy[name_lon])

def calculate_weighted_stats(data_x, mask, data_y=None):
    """Calculates area-weighted statistics based on the supplementary material 'Computation of metrics used in the analysis' from Richter et al. 2017."""
    #grid of weights based on the cosine of the latitude
    weights = np.cos(np.deg2rad(data_x.latitude))
    weights.name = "weights"

    #single xarray dataset
    ds = xr.Dataset({'data_x': data_x, 'weights': weights})
    if data_y is not None:
        ds['data_y'] = data_y

    #regional mask
    stacked = ds.where(mask).stack(z=("latitude", "longitude")).dropna(dim="z")
    
    #NaNs if no valid data
    if stacked.z.size == 0:
        keys = ['mean_x', 'std_x']
        if data_y is not None:
            keys.extend(['mean_y', 'std_y', 'rmse', 'pcc'])
        return {k: np.nan for k in keys}

    x = stacked.data_x
    #normalize weights
    w = stacked.weights / stacked.weights.sum()

    #unbiased correction factor for variance eq. (1)
    unbiased_factor = 1.0 / (1.0 - np.sum(w**2))
    
    #area weighted mean eq. (2)
    mu_x = np.sum(w * x)

    #area weighted variance eq. (4)
    var_x = unbiased_factor * np.sum(w * (x - mu_x)**2)

    #standard deviation
    std_x = np.sqrt(var_x)
    
    results = {'mean_x': mu_x.item(), 'std_x': std_x.item()}

    #MSE, RMSE, and PCC only if data_y is provided
    if data_y is not None:
        y = stacked.data_y
        
        #same stats for data_y
        mu_y = np.sum(w * y)
        var_y = unbiased_factor * np.sum(w * (y - mu_y)**2)
        std_y = np.sqrt(var_y)
        
        #area weighted MSE eq. (5)
        mse = unbiased_factor * np.sum(w * ((x - mu_x) - (y - mu_y))**2)

        #area weighted RMSE
        rmse = np.sqrt(mse)
        
        #area weighted PCC
        pcc = unbiased_factor * np.sum(w * ((x - mu_x) * (y - mu_y))) / (std_x * std_y)
        
        results.update({
            'mean_y': mu_y.item(), 'std_y': std_y.item(),
            'rmse': rmse.item(), 'pcc': pcc.item()
        })
        
    return results

#mask for regionwide statistics
def create_region_mask(data_array, extent):
    """Create a mask for the North Atlantic region."""
    lon_min, lon_max, lat_min, lat_max = extent
    mask = ((data_array.longitude >= lon_min) & 
            (data_array.longitude <= lon_max) & 
            (data_array.latitude >= lat_min) & 
            (data_array.latitude <= lat_max))
    return mask