"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Utility functions.
"""

import xarray as xr
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from functools import wraps
import hashlib
import config
import os
import sys
from eofs.xarray import Eof
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def setup_esmf_environment():
    """Check for and set the ESMFMKFILE environment variable if it's not present. This is a known issue for esmpy versions >= 8.4.0 in some environments (like VS Code terminals) where the conda activation doesn't set this required variable. This function dynamically finds the 'esmf.mk' file and sets the variable before xesmf is imported. See: https://github.com/conda-forge/esmf-feedstock/issues/91"""
    
    if 'ESMFMKFILE' not in os.environ:
        #active environment root path
        env_path = Path(sys.prefix)

        #potential locations for esmf.mk
        possible_paths = [
            env_path / "lib" / "esmf.mk",            #Linux/macOS
            env_path / "Lib" / "esmf.mk",            #Windows
            env_path / "Library" / "lib" / "esmf.mk" #Windows
        ]

        #search for file
        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break
        
        #set environment variable
        if found_path:
            os.environ['ESMFMKFILE'] = str(found_path)
            print(f"ESMFMKFILE set to: {found_path}")
        else:
            raise ImportError(
                "Could not find 'esmf.mk' in the environment.\n" "The xesmf package installation may be incomplete or corrupted.\n" "Please try reinstalling it with: `conda install -c conda-forge xesmf`."
            )

#cache directory
CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(exist_ok=True)

def cache_result(cache_key_prefix):
    """Cache function, prioritize netcdf, csv, and json, pickle only if necessary."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            #dynamic cache name
            arg_hash = generate_arg_hash(*args, **kwargs)
            cache_name = f"{cache_key_prefix}_{arg_hash}"

            #cache paths
            dir_path = CACHE_DIR / cache_name
            nc_path = CACHE_DIR / f"{cache_name}.nc"
            csv_path = CACHE_DIR / f"{cache_name}.csv"
            json_path = CACHE_DIR / f"{cache_name}.json"

            #load from cache
            if config.USE_CACHE and not config.FORCE_RECOMPUTE:

                #check for directory cache
                if dir_path.is_dir():
                    print(f"Loading cached dictionary of xarray objects from: {dir_path}")
                    reconstructed_dict = {}
                    for file_path in dir_path.glob('*.nc'):
                        key = file_path.stem 
                        reconstructed_dict[key] = xr.open_dataset(file_path)
                    return reconstructed_dict

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

            #compute and save
            print(f"Computing {cache_name}...")
            result = func(*args, **kwargs)

            if config.USE_CACHE:
                try:

                    #xarray object
                    if isinstance(result, (xr.Dataset, xr.DataArray)):
                        print(f"Caching result as NetCDF to: {nc_path}")
                        result.to_netcdf(nc_path)
                    
                    #pandas dataframe
                    elif isinstance(result, pd.DataFrame):
                        print(f"Caching result as CSV to: {csv_path}")
                        result.to_csv(csv_path)
                    
                    #dictionaries and lists
                    elif isinstance(result, (dict, list)):
                        try:

                            #serialize as JSON
                            print(f"Attempting to cache result as JSON to: {json_path}")
                            json_result = convert_for_json(result)
                            with open(json_path, 'w') as f:
                                json.dump(json_result, f, indent=2)
                            print("Successfully cached as JSON.")

                        except TypeError:

                            #if JSON fails check for dict of xarray objects
                            is_dict_of_datasets = (
                                isinstance(result, dict) and result and
                                all(isinstance(v, (xr.Dataset, xr.DataArray)) for v in result.values())
                            )
                            
                            if is_dict_of_datasets:
                                
                                #directory-based netcdf
                                print(f"JSON serialization failed. Caching dictionary of xarray objects to directory: {dir_path}")
                                dir_path.mkdir(parents=True, exist_ok=True)
                                for key, ds in result.items():
                                    ds.to_netcdf(dir_path / f"{key}.nc")
                            
                            else:
                                print(f"Warning: Could not cache {cache_name}. Object is not serializable to JSON and is not a recognized complex type.")
                    
                    else:
                        print(f"Warning: Could not cache {cache_name}. Data type '{type(result).__name__}' is not supported by the caching implementation.")
                            
                except Exception as e:
                    print(f"Warning: Could not cache {cache_name}: {e}")

            return result
        return wrapper
    return decorator

def convert_for_json(obj):
    """Convert numpy types and other non-JSON-serializable objects to JSON-compatible types"""

    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (xr.Dataset, xr.DataArray)):
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    else:
        return obj

def generate_arg_hash(*args, **kwargs):
    """Generates a short, deterministic SHA256 hash from function arguments, including complex objects like xarray and pandas."""
    
    hasher = hashlib.sha256()

    def update_hash(obj):
        
        if isinstance(obj, xr.Dataset):

            #xarray dataset
            for var_name in sorted(obj.variables):
                hasher.update(obj[var_name].values.tobytes())
            sorted_attrs = str(sorted(obj.attrs.items()))
            hasher.update(sorted_attrs.encode('utf-8'))

        elif isinstance(obj, xr.DataArray):
    
            #xarray dataarray
            hasher.update(obj.values.tobytes())
            for coord_name in sorted(obj.coords):
                hasher.update(obj[coord_name].values.tobytes())
            sorted_attrs = str(sorted(obj.attrs.items()))
            hasher.update(sorted_attrs.encode('utf-8'))

        elif isinstance(obj, pd.DataFrame):

            #hash data, index, and columns
            hasher.update(obj.values.tobytes())
            hasher.update(obj.index.values.tobytes())
            hasher.update(obj.columns.values.tobytes())

        elif isinstance(obj, (tuple, list)):

            #hash elements
            for item in obj:
                update_hash(item)

        elif isinstance(obj, dict):

            #hash key-value pairs
            for key, value in sorted(obj.items()):
                update_hash(key)
                update_hash(value)

        else:

            #hash string
            hasher.update(str(obj).encode('utf-8'))

    #positional and keyword arguments
    update_hash(args)
    update_hash(kwargs)
    
    #first 16 characters of hex digest
    return hasher.hexdigest()[:16]

def rotate_longitude(ds, name_lon):
    """Convert longitude from 0-360 to -180-180 coordinate system."""

    ds_copy = ds.copy()
    new_lon_values = (((ds_copy[name_lon] + 180) % 360) - 180)
    ds_copy = ds_copy.assign_coords({name_lon: new_lon_values})

    return ds_copy.sortby(ds_copy[name_lon])

def add_map_features(ax, extent, is_left=False, is_bottom=False):
    """Add standard map features to axis."""

    lon_min, lon_max, lat_min, lat_max = extent
    boundary_path = mpath.Path([[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]).interpolated(50)
    
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    boundary_in_proj_coords = proj_to_data.transform_path(boundary_path)
    ax.set_boundary(boundary_in_proj_coords)
    
    verts = boundary_in_proj_coords.vertices
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = is_left
    gl.bottom_labels = is_bottom

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

def create_region_mask(data_array, extent):
    """Create a mask for regionwide statistics for the North Atlantic region."""

    lon_min, lon_max, lat_min, lat_max = extent
    mask = ((data_array.longitude >= lon_min) & 
            (data_array.longitude <= lon_max) & 
            (data_array.latitude >= lat_min) & 
            (data_array.latitude <= lat_max))
    
    return mask

def detrend_timeseries(data_array, degree=1, dim='time'):
    """Detrending for variability calculation."""

    #check
    if not isinstance(degree, int) or degree < 0:
        raise ValueError(f"Degree must be a non-negative integer, but got {degree}.")

    #fit polynomial
    p = data_array.polyfit(dim=dim, deg=degree)
    fit = xr.polyval(data_array[dim], p.polyfit_coefficients)
    
    #return residuals
    return data_array - fit

def calculate_pc_index_correlations(pcs, indices_dict):
    """Helper function to correlate a single set of PCs with indices."""

    correlations = {}
    
    for index_name, index_ts in indices_dict.items():
        if index_ts is None: continue
        
        corr_values = []
        for mode in pcs.mode.values:
            pc_mode = pcs.sel(mode=mode)
            pc_aligned, index_aligned = xr.align(pc_mode, index_ts, join='inner')
            
            if pc_aligned.size > 1:
                corr_values.append(xr.corr(pc_aligned, index_aligned, dim='time').item())
            else:
                corr_values.append(np.nan)
        
        correlations[index_name] = xr.DataArray(corr_values, coords={'mode': pcs.mode.values}, dims=['mode'])

    return correlations