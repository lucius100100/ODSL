"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Execute the full ODSL calculations / analysis and plotting.
"""

from data_loader import (load_altimetry_data, load_budget_data, load_gia_data, load_cmip_model_data, get_cmip_files_inventory, load_climate_indices_dict)
from utils import (setup_esmf_environment, cache_result, calculate_weighted_stats, create_region_mask, detrend_timeseries, calculate_pc_index_correlations)
from plotting import create_all_figures 
from config import (CMIP_VERSION, START_YEAR, END_YEAR, EXTENT, TARGET_CMIP5_MODELS, TARGET_CMIP6_MODELS, VARIABILITY_DETREND_DEGREE, CMIP_SCENARIOS, CMIP5_FUTURE_SCENARIO, CMIP6_FUTURE_SCENARIO, EOF_N_MODES, APPLY_SPATIAL_SMOOTHING, SPATIAL_SMOOTHING_SIGMA)

#setup_esmf_environment()

import os
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
#import traceback
#from scipy import stats
#import statsmodels.api as sm
from eofs.xarray import Eof
from astropy.convolution import Gaussian2DKernel, convolve
#from itertools import zip_longest
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

#correct configuration
if CMIP_VERSION == 'CMIP5':
    TARGET_MODELS = TARGET_CMIP5_MODELS
    FUTURE_SCENARIO = CMIP5_FUTURE_SCENARIO
elif CMIP_VERSION == 'CMIP6':
    TARGET_MODELS = TARGET_CMIP6_MODELS
    FUTURE_SCENARIO = CMIP6_FUTURE_SCENARIO
else:
    raise ValueError(f"Unsupported CMIP_VERSION: {CMIP_VERSION}. Must be 'CMIP5' or 'CMIP6'.")

#figures directory
fig_dir = './figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

@cache_result('observed_odsl')
def calculate_observed_odsl():
    """Calculate observed ODSL from components."""

    print("\nCalculating observed ODSL...")
    
    #load data
    #altimetry
    duacs_ds_original = load_altimetry_data()
    duacs_ds = duacs_ds_original.copy(deep=True)
    #budget
    asl_frederikse = load_budget_data(extend_to_year=END_YEAR if END_YEAR > 2018 else None)
    #GIA
    gia_data = load_gia_data()
    gia_rad_da = gia_data['gia_rad']
    gia_sea_da = gia_data['gia_sea']

    #potential smoothing altimetry dataset
    if APPLY_SPATIAL_SMOOTHING:
        print(f"Applying Gaussian spatial smoothing with sigma={SPATIAL_SMOOTHING_SIGMA}...")
        
        #when only defining x astropy assumes isotropic
        kernel = Gaussian2DKernel(x_stddev=SPATIAL_SMOOTHING_SIGMA)
        
        smoothed_slices = []
        
        #loop over each time step
        for time_slice in duacs_ds['sla'].time:

            data_slice = duacs_ds['sla'].sel(time=time_slice)
            
            #apply convolution
            smoothed_values = convolve(data_slice.values, kernel, boundary='extend', preserve_nan=True)
            
            #new DataArray from smoothed values
            smoothed_da = xr.DataArray(smoothed_values, coords=data_slice.coords, dims=data_slice.dims)
            
            smoothed_slices.append(smoothed_da)
            
        smoothed_sla = xr.concat(smoothed_slices, dim='time')
        duacs_ds['sla'] = smoothed_sla
        
        print("Smoothing complete.")

    #yearly mean after optional smoothing
    duacs_yearly = duacs_ds.groupby('time.year').mean()
    
    #common years
    year_slice = slice(START_YEAR, END_YEAR)
    alt_years = duacs_yearly.sel(year=year_slice).year.values
    fr_years = asl_frederikse.sel(year=year_slice).year.values
    common_years = np.intersect1d(alt_years, fr_years)
    
    print(f"Analysis period: {common_years.min()}-{common_years.max()} ({len(common_years)} years)")
    
    #trends over common period
    #altimetry
    polyfit_ds_sla = duacs_yearly.sla.sel(year=common_years).polyfit(dim='year', deg=1)
    trend_sla_alt = polyfit_ds_sla['polyfit_coefficients'].sel(degree=1)
    
    #budget
    polyfit_ds_asl = asl_frederikse.sel(year=common_years).polyfit(dim='year', deg=1)
    trend_asl_fr = polyfit_ds_asl['polyfit_coefficients'].sel(degree=1)
    
    #regridding
    print("Creating regridders...")
    regridder_frederikse = xe.Regridder(trend_asl_fr, trend_sla_alt, 'bilinear', periodic=True)
    regridder_gia = xe.Regridder(gia_rad_da, trend_sla_alt, 'bilinear', periodic=True)
    
    #regrid to altimetry grid
    print("Regridding data...")
    trend_asl_fr_regridded = regridder_frederikse(trend_asl_fr)
    gia_rad_regridded = regridder_gia(gia_rad_da)
    gia_sea_regridded = regridder_gia(gia_sea_da)
    gia_geoid_correction_mm_yr = gia_sea_regridded + gia_rad_regridded
    
    #convert to mm/yr
    trend_sla_alt_mm_yr = trend_sla_alt * 10  #cm/yr -> mm/yr
    trend_asl_fr_regridded_mm_yr = trend_asl_fr_regridded
    gia_regridded_mm_yr = gia_geoid_correction_mm_yr
    
    #calculate ODSL
    odsl_mm_yr = trend_sla_alt_mm_yr - trend_asl_fr_regridded_mm_yr - gia_regridded_mm_yr

    #weighted global mean removal
    print("Removing global mean...")
    def remove_global_mean(data_array):
        weights = np.cos(np.deg2rad(data_array.latitude))
        weights.name = "weights"
        global_mean = data_array.weighted(weights).mean(dim=("latitude", "longitude")).item()
        return data_array - global_mean

    #apply
    trend_sla_alt_mm_yr.name = 'MSL'
    trend_asl_fr_regridded_mm_yr.name = 'Geoid'
    gia_regridded_mm_yr.name = 'GIA'
    odsl_mm_yr.name = 'ODSL'

    trend_sla_alt_mm_yr = remove_global_mean(trend_sla_alt_mm_yr)
    trend_asl_fr_regridded_mm_yr = remove_global_mean(trend_asl_fr_regridded_mm_yr)
    gia_regridded_mm_yr = remove_global_mean(gia_regridded_mm_yr)
    odsl_mm_yr = remove_global_mean(odsl_mm_yr)

    print(f"ODSL range: {odsl_mm_yr.min().item():.2f} to {odsl_mm_yr.max().item():.2f} mm/yr")

    print("Calculating yearly ODSL fields...")
    
    #yearly ODSL fields
    sla_yearly_mm = duacs_yearly.sla.sel(year=common_years) * 10  #cm -> mm
    asl_yearly = asl_frederikse.sel(year=common_years)
    
    #regrid
    asl_yearly_regridded = regridder_frederikse(asl_yearly)
    time_delta = xr.DataArray(common_years - common_years[0], dims='year', coords={'year': common_years})
    gia_cumulative_signal = gia_geoid_correction_mm_yr * time_delta
    
    #subtract yearly budget and cumulative GIA
    odsl_yearly_fields = sla_yearly_mm - asl_yearly_regridded - gia_cumulative_signal
    
    #variability
    detrended_odsl = detrend_timeseries(odsl_yearly_fields, degree=VARIABILITY_DETREND_DEGREE, dim='year')
    observed_variability = detrended_odsl.std(dim='year') / 10    #mm -> cm

    #anomaly from global mean for overview figure
    temporal_mean = odsl_yearly_fields.mean(dim='year')
    odsl_yearly_anomaly = odsl_yearly_fields - temporal_mean
    
    print(f"Yearly ODSL Anomaly range: {odsl_yearly_anomaly.min().item():.2f} to {odsl_yearly_anomaly.max().item():.2f} mm")

    #clean up regridders
    try:
        regridder_frederikse.clean_weight_file()
        regridder_gia.clean_weight_file()
    except AttributeError:
        pass

    odsl_yearly_fields.name = 'odsl_yearly'
    odsl_yearly_anomaly.name = 'odsl_yearly_anomaly'

    output_ds = xr.Dataset(
        {
            'odsl': odsl_mm_yr,
            'odsl_yearly': odsl_yearly_fields,
            'odsl_yearly_anomaly': odsl_yearly_anomaly,
            'msl': trend_sla_alt_mm_yr,
            'geoid': trend_asl_fr_regridded_mm_yr,
            'gia': gia_regridded_mm_yr,
            'variability': observed_variability,
        }
    )

    #global attributes
    output_ds.attrs['description'] = "Observed ODSL and its components."
    output_ds.attrs['analysis_period'] = f"{common_years.min()}-{common_years.max()}"
    output_ds.attrs['common_years_list'] = common_years.tolist() 

    return output_ds

@cache_result('smoothing_sensitivity_results')
def calculate_smoothing_sensitivity():
    """Calculates altimetry trends for different levels of Gaussian smoothing."""

    print("\nPerforming smoothing sensitivity analysis...")

    duacs_ds = load_altimetry_data()
    
    #think about values, number of operations scales with O(sigma^2)
    sigmas = [0, 3, 6, 9]
    trend_maps = []

    print(f"Calculating trends for sigma levels: {sigmas}")

    for sigma in sigmas:

        #copy
        ds_to_process = duacs_ds.copy(deep=True)

        if sigma > 0:
            print(f"  Applying smoothing for sigma={sigma}...")
            kernel = Gaussian2DKernel(x_stddev=sigma)
            
            #loop over time slices and apply convolution
            smoothed_slices = []
            for time_slice in ds_to_process['sla'].time:
                data_slice = ds_to_process['sla'].sel(time=time_slice)
                
                smoothed_values = convolve(
                    data_slice.values, 
                    kernel, 
                    boundary='extend', 
                    preserve_nan=True
                )
                
                smoothed_da = xr.DataArray(
                    smoothed_values,
                    coords=data_slice.coords,
                    dims=data_slice.dims
                )
                smoothed_slices.append(smoothed_da)
            
            #combine
            smoothed_sla = xr.concat(smoothed_slices, dim='time')
            ds_to_process['sla'] = smoothed_sla
        
        #annual mean and trend
        duacs_yearly = ds_to_process.groupby('time.year').mean()
        year_slice = slice(START_YEAR, END_YEAR)
        sla_yearly_da = duacs_yearly['sla']
        
        #polyfit
        trend = sla_yearly_da.sel(year=year_slice).polyfit(
            dim='year', deg=1
        )['polyfit_coefficients'].sel(degree=1)
        
        trend_maps.append(trend)

    #combine with sigma coordinate
    combined_trends = xr.concat(
        trend_maps, 
        dim=pd.Index(sigmas, name='sigma')
    )
    
    #cm/yr -> mm/yr
    combined_trends_mm_yr = combined_trends * 10
    
    print("Smoothing sensitivity analysis complete.")
    return combined_trends_mm_yr

@cache_result('valid_models_table')
def valid_models_table():
    """Finds all valid CMIP5 and CMIP6 models that have the required historical and future scenario files and displays them in a table."""

    def create_availability_df(cmip_version, target_models, scenarios_dict):
        """Helper function to generate an availability DataFrame for a given CMIP version."""
        
        print(f"\n Scanning {cmip_version} model availability...")
        inventory = get_cmip_files_inventory(cmip_version)
        all_files = inventory['all_files']

        #historical
        models_with_historical = set(all_files.get('historical', {}).keys())
        
        #future scenarios
        future_scenarios = [s for s in scenarios_dict if s != 'historical']

        results_data = []
        
        #check each model
        for model in sorted(target_models):
            row = {'Model': model}
            
            #historical
            has_historical = model in models_with_historical
            row['historical'] = has_historical
            
            #future scenario
            for scenario in future_scenarios:
                models_with_future_scenario = set(all_files.get(scenario, {}).keys())
                
                #both historical and future scenario only
                row[scenario] = has_historical and (model in models_with_future_scenario)
            
            results_data.append(row)
            
        #DataFrame
        df = pd.DataFrame(results_data)
        return df.set_index('Model')

    #CMIP5
    df_cmip5 = create_availability_df('CMIP5', TARGET_CMIP5_MODELS, CMIP_SCENARIOS['CMIP5'])

    #CMIP6
    df_cmip6 = create_availability_df('CMIP6', TARGET_CMIP6_MODELS, CMIP_SCENARIOS['CMIP6'])

    #combine DataFrames
    combined_df = pd.concat(
        [df_cmip5, df_cmip6], 
        keys=['CMIP5', 'CMIP6'], 
        names=['CMIP_Version', 'Model']
    )
    
    print("\nScan complete, returning combined availability table...")
    return combined_df

@cache_result('processed_CMIP_models')
def process_cmip_models():
    """Process CMIP models and return a single dataset."""
    
    print(f"\nProcessing {CMIP_VERSION} models...")
    
    inventory = get_cmip_files_inventory(CMIP_VERSION)
    all_files = inventory['all_files']
    
    #initialize
    model_names = []
    trends_list = []
    variability_list = []
    full_timeseries_list = []
    region_masks_list = []
    trend_stats_list = []
    
    for i, model_name in enumerate(TARGET_MODELS):
        hist_file = all_files['historical'].get(model_name)
        future_file = all_files[FUTURE_SCENARIO].get(model_name)
        
        if hist_file and future_file:
            print(f"Processing model: {model_name} ({i+1}/{len(TARGET_MODELS)})")

            try:
                #calculations per model
                combined_zos = load_cmip_model_data(model_name, future_scenario=FUTURE_SCENARIO, cmip_version=CMIP_VERSION)
                region_mask = create_region_mask(combined_zos.isel(time=0), EXTENT)
                
                period_data = combined_zos.sel(time=slice(START_YEAR, END_YEAR))
                trend_coeffs = period_data.polyfit(dim='time', deg=1)
                slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
                slope_mm_yr = slope * 10
                
                #variability
                detrended_data = detrend_timeseries(period_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
                model_variability = detrended_data.std(dim='time')

                trend_stats = calculate_weighted_stats(slope_mm_yr, region_mask)
                
                #append results
                model_names.append(model_name)
                trends_list.append(slope_mm_yr)
                variability_list.append(model_variability)
                full_timeseries_list.append(combined_zos)
                region_masks_list.append(region_mask)
                trend_stats_list.append(trend_stats)
                
            except Exception as e:
                print(f"Could not process model {model_name}: {e}")

    print(f"\nProcessed {len(model_names)} out of {len(TARGET_MODELS)} models")

    #concatenate
    model_trends = xr.concat(trends_list, dim=pd.Index(model_names, name='model'))
    model_variability = xr.concat(variability_list, dim=pd.Index(model_names, name='model'))
    full_timeseries = xr.concat(full_timeseries_list, dim=pd.Index(model_names, name='model'))
    region_masks = xr.concat(region_masks_list, dim=pd.Index(model_names, name='model'))

    #convert list
    trend_means = xr.DataArray([s['mean_x'] for s in trend_stats_list], coords={'model': model_names}, dims=['model'])
    trend_stds = xr.DataArray([s['std_x'] for s in trend_stats_list], coords={'model': model_names}, dims=['model'])
    
    #multi-model mean
    model_mean_trend_zos = model_trends.mean(dim='model', skipna=True)
    model_mean_variability = model_variability.mean(dim='model', skipna=True)
    
    #remove global mean
    weights = np.cos(np.deg2rad(model_mean_trend_zos.latitude))
    weights.name = "weights"
    model_global_mean = model_mean_trend_zos.weighted(weights).mean().item()
    
    model_mean_trend = model_mean_trend_zos - model_global_mean

    #dataset object
    output_ds = xr.Dataset(
        {
            'model_trend': model_trends,
            'model_mean_trend_zos' : model_mean_trend_zos,
            'model_mean_trend': model_mean_trend,
            'model_variability': model_variability,
            'model_mean_variability': model_mean_variability,
            'full_timeseries': full_timeseries,
            'region_mask': region_masks.astype('int8'),
            'trend_mean': trend_means,
            'trend_std': trend_stds
        }
    )
    
    #global attributes
    output_ds.attrs['description'] = f"Processed {CMIP_VERSION} model trends, variability, and timeseries."
    output_ds.attrs['valid_models_count'] = len(model_names)
    
    return output_ds

@cache_result('incrementing_window_results')
def perform_incrementing_window_analysis(obs_results, cmip_results):
    """Performs an analysis with an incrementally increasing window size, calculating PCC and RMSE between observed and modeled ODSL trends for each window."""
    
    print("\nPerforming incrementing window analysis...")

    #yearly timeseries data for observations and models
    obs_yearly_ts = obs_results['odsl_yearly'].rename({'year': 'time'})
    model_full_ts = cmip_results['full_timeseries']

    #sample model grid and region mask for regridding and weighting
    sample_model_grid = cmip_results['model_mean_trend']
    region_mask = create_region_mask(sample_model_grid, EXTENT)

    #regrid observed to model grid
    print("Regridding observed timeseries to model grid...")

    regridder_obs_to_model = xe.Regridder(obs_yearly_ts, sample_model_grid, 'bilinear', periodic=True)
    obs_yearly_ts_regridded = regridder_obs_to_model(obs_yearly_ts)

    try:
        regridder_obs_to_model.clean_weight_file()
    except AttributeError:
        pass

    #dictionary of all timeseries sources to analyze
    sources_ts = {'mmm': model_full_ts.mean(dim='model')}
    for model_name in cmip_results.model.values:
        sources_ts[model_name] = model_full_ts.sel(model=model_name)

    window_end_years = range(START_YEAR + 1, END_YEAR + 1)
    
    #dictionaries
    all_pcc = {source: [] for source in sources_ts.keys()}
    all_rmse = {source: [] for source in sources_ts.keys()}

    print(f"Calculating trends for {len(window_end_years)} incrementing windows...")
    for end_year in window_end_years:
        window_slice = slice(START_YEAR, end_year)
        
        #trend for the observed data for the current window
        obs_window_data = obs_yearly_ts_regridded.sel(time=window_slice)
        obs_trend_coeffs = obs_window_data.polyfit(dim='time', deg=1)
        obs_trend = obs_trend_coeffs.polyfit_coefficients.sel(degree=1)

        #loop through each model
        for source_name, source_ts in sources_ts.items():
            model_window_data = source_ts.sel(time=window_slice)
            model_trend_coeffs = model_window_data.polyfit(dim='time', deg=1)
            model_trend = model_trend_coeffs.polyfit_coefficients.sel(degree=1) * 10 #cm/yr -> mm/yr

            #weighted PCC and RMSE
            stats = calculate_weighted_stats(model_trend, region_mask, data_y=obs_trend)
            
            all_pcc[source_name].append(stats['pcc'])
            all_rmse[source_name].append(stats['rmse'])

    print("Incrementing window analysis complete.")

    #output
    source_coord = list(sources_ts.keys())
    pcc_da = xr.DataArray(
        [all_pcc[s] for s in source_coord],
        coords={'source': source_coord, 'end_year': window_end_years},
        dims=['source', 'end_year'],
        name='pcc'
    )
    rmse_da = xr.DataArray(
        [all_rmse[s] for s in source_coord],
        coords={'source': source_coord, 'end_year': window_end_years},
        dims=['source', 'end_year'],
        name='rmse'
    )

    output_ds = xr.Dataset({'pcc': pcc_da, 'rmse': rmse_da})
    output_ds.attrs['description'] = "PCC and RMSE from an incrementing window analysis of ODSL trends."
    
    return output_ds

@cache_result('sliding_window_results')
def perform_sliding_window_analysis():
    """Sliding window analysis on all models."""
    
    print("\nSliding window analysis...")
    
    #processed models
    cmip_results_ds = process_cmip_models() 
    obs_results = calculate_observed_odsl()
    odsl_mm_yr = obs_results['odsl']
    
    #sample grid
    sample_model_grid = cmip_results_ds['model_mean_trend']
    
    regridder_obs_to_model = xe.Regridder(odsl_mm_yr, sample_model_grid, 'bilinear', periodic=True)
    odsl_mm_yr_regridded = regridder_obs_to_model(odsl_mm_yr)
    
    #variability
    odsl_var_obs = obs_results['variability']
    regridder_var_to_model = xe.Regridder(odsl_var_obs, sample_model_grid, 'bilinear', periodic=True)
    odsl_var_obs_regridded = regridder_var_to_model(odsl_var_obs)

    try:
        regridder_obs_to_model.clean_weight_file()
    except AttributeError:
        pass
    
    #remove global mean from observations
    weights = np.cos(np.deg2rad(odsl_mm_yr_regridded.latitude))

    #centered trend
    global_mean = odsl_mm_yr_regridded.weighted(weights).mean(dim=("latitude", "longitude")).item()
    odsl_obs_dynamic = odsl_mm_yr_regridded - global_mean

    #centered variability
    mean_var_obs = odsl_var_obs_regridded.weighted(weights).mean(dim=("latitude", "longitude")).item()
    odsl_var_obs_centered = odsl_var_obs_regridded - mean_var_obs
    
    #sliding window analysis for each model
    all_pcc = []
    all_rmse = []
    all_mean_trends = []
    all_trends = []
    all_pcc_var = []
    all_rmse_var = []
    all_mean_variability = []
    all_variability = []
    all_windows = []
    model_names_for_sliding = []
    
    for model_name in cmip_results_ds.model.values:
        print(f"Processing sliding windows for {model_name}...")
        
        full_timeseries = cmip_results_ds['full_timeseries'].sel(model=model_name)
        region_mask = cmip_results_ds['region_mask'].sel(model=model_name)
        
        pcc_per_model = []
        rmse_per_model = []
        mean_trends_per_model = []
        trends_per_model = []
        mean_variability_per_model = []
        variability_per_model = []
        pcc_var_per_model = []
        rmse_var_per_model = []
        windows_per_model = []

        #slide window
        window_size = END_YEAR - START_YEAR + 1
        start_year = int(cmip_results_ds.full_timeseries.time.min().item())
        end_year = END_YEAR
        
        for window_start in range(start_year, end_year - window_size + 1):
            window_end = window_start + window_size - 1
            
            window_data = full_timeseries.sel(time=slice(window_start, window_end))
            
            if len(window_data.time) < window_size * 0.75:
                continue
            
            #linear trend
            trend_coeffs = window_data.polyfit(dim='time', deg=1)
            trend_mm_yr = trend_coeffs.polyfit_coefficients.sel(degree=1) * 10
            
            #variability
            detrended_window = detrend_timeseries(window_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
            variability_map = detrended_window.std(dim='time')
            
            #center the modeled variability
            mean_var_model = variability_map.weighted(weights).mean().item()
            variability_map_centered = variability_map - mean_var_model

            #statistics
            stats = calculate_weighted_stats(trend_mm_yr, region_mask, data_y=odsl_obs_dynamic)
            stats_var = calculate_weighted_stats(variability_map_centered, region_mask, data_y=odsl_var_obs_centered)
            stats_abs_var = calculate_weighted_stats(variability_map, region_mask)

            #append
            windows_per_model.append(window_start)
            pcc_per_model.append(stats['pcc'])
            rmse_per_model.append(stats['rmse'])
            mean_trends_per_model.append(stats['mean_x'])
            trends_per_model.append(trend_mm_yr)
            mean_variability_per_model.append(stats_abs_var['mean_x'])
            variability_per_model.append(variability_map)
            pcc_var_per_model.append(stats_var['pcc'])
            rmse_var_per_model.append(stats_var['rmse'])
        
        model_trends_da = xr.concat(trends_per_model, dim=pd.Index(windows_per_model, name='window_start_year'))
        all_trends.append(model_trends_da)
        model_variability_da = xr.concat(variability_per_model, dim=pd.Index(windows_per_model, name='window_start_year'))
        all_variability.append(model_variability_da)
        model_names_for_sliding.append(model_name)
        all_pcc.append(pcc_per_model)
        all_rmse.append(rmse_per_model)
        all_mean_trends.append(mean_trends_per_model)
        all_mean_variability.append(mean_variability_per_model)
        all_pcc_var.append(pcc_var_per_model)
        all_rmse_var.append(rmse_var_per_model)
        
        if not all_windows:
            all_windows = windows_per_model

    final_trends_da = xr.concat(all_trends, dim=pd.Index(model_names_for_sliding, name='model'))
    final_variability_da = xr.concat(all_variability, dim=pd.Index(model_names_for_sliding, name='model'))

    #assemble
    output_ds = xr.Dataset(
        {
            'pcc': (('model', 'window_start_year'), all_pcc),
            'rmse': (('model', 'window_start_year'), all_rmse),
            'mean_trend': (('model', 'window_start_year'), all_mean_trends),
            'sliding_trends': final_trends_da,
            'mean_variability': (('model', 'window_start_year'), all_mean_variability),
            'sliding_variability': final_variability_da,
            'odsl_obs_dynamic': odsl_obs_dynamic,
            'pcc_variability': (('model', 'window_start_year'), all_pcc_var),
            'rmse_variability': (('model', 'window_start_year'), all_rmse_var),
            'odsl_var_obs_centered': odsl_var_obs_centered, 
            'odsl_var_obs_regridded': odsl_var_obs_regridded,
        },
        coords={
            'model': model_names_for_sliding,
            'window_start_year': all_windows,
            'latitude': final_trends_da.latitude,
            'longitude': final_trends_da.longitude
        }
    )
    output_ds.attrs['description'] = "Sliding window mean trend / variability, PCC, RMSE, and trend map results."

    return output_ds

@cache_result('cmip_scenario_timeseries_results')
def process_cmip_scenario_data():
    """Process CMIP data to get ensemble timeseries for each scenario."""

    print("\nProcessing CMIP scenario ensembles")
    
    final_results = []
    
    #loop over cmip version scenarios
    for cmip_version, target_models in [("CMIP5", TARGET_CMIP5_MODELS), ("CMIP6", TARGET_CMIP6_MODELS)]:
        print(f"\nProcessing {cmip_version} scenarios")
        
        scenarios_to_process = list(CMIP_SCENARIOS[cmip_version].keys())
        historical_end_year = 2005 if cmip_version == "CMIP5" else 2014
        
        scenario_datasets = {}
        
        for scenario in scenarios_to_process:
            print(f"Processing scenario: {scenario}")
            
            model_timeseries_list = []
            
            #loop over each model
            for model_name in target_models:
                try:
                    end_year = historical_end_year if scenario == "historical" else 2100
                    future_scen = None if scenario == "historical" else scenario
                    
                    ts = load_cmip_model_data(model_name, hist_scenario='historical', future_scenario=future_scen, cmip_version=cmip_version, end_year=end_year)
                    if ts is None: continue
                    
                    #remove global mean
                    weights = np.cos(np.deg2rad(ts.latitude))
                    weights.name = "weights"
                    global_mean_ts = ts.weighted(weights).mean(dim=['latitude', 'longitude'])
                    odsl_ts = ts - global_mean_ts

                    region_mask = create_region_mask(odsl_ts.isel(time=0), EXTENT)
                    regional_ts = odsl_ts.where(region_mask).mean(dim=['latitude', 'longitude'])
                    regional_ts_mm = regional_ts * 10
                    regional_ts_mm = regional_ts_mm.rename({'time': 'year'})
                    model_timeseries_list.append(regional_ts_mm)

                except Exception as e:
                    print(f"Could not process model {model_name} for scenario {scenario}: {e}")

            #concatenate results
            if model_timeseries_list:
                ensemble_ts = xr.concat(model_timeseries_list, dim=pd.Index([f"model_{i}" for i in range(len(model_timeseries_list))], name='model'))
                ensemble_mean = ensemble_ts.mean(dim='model', skipna=True)
                ensemble_std = ensemble_ts.std(dim='model', skipna=True)

                scenario_datasets[scenario] = xr.Dataset({
                    'ensemble_mean': ensemble_mean,
                    'ensemble_std': ensemble_std,
                    'n_models': len(model_timeseries_list)
                })
                print(f"  Processed {len(model_timeseries_list)} models for {scenario}")

        if scenario_datasets:
            all_years = sorted({year for ds in scenario_datasets.values() for year in ds.year.values})
            
            ds_list = []
            for scenario_name, ds in scenario_datasets.items():
                ds = ds.reindex(year=all_years)
                ds_list.append(ds)

            combined_ds = xr.concat(ds_list, dim=pd.Index(list(scenario_datasets.keys()), name='scenario'))
            combined_ds.attrs['description'] = f"Ensemble timeseries for {cmip_version} scenarios"
            combined_ds.attrs['historical_end_year'] = historical_end_year
            combined_ds.attrs['valid_scenarios'] = list(scenario_datasets.keys())
            final_results.append(combined_ds)

    if not final_results:
        return xr.Dataset()

    cmip_coord = pd.Index(["CMIP5", "CMIP6"], name='cmip_version')
    combined_results = xr.concat(final_results, dim=cmip_coord)
    
    print(f"Successfully processed timeseries for {len(final_results)} CMIP versions")

    return combined_results

@cache_result('single_eof_result')
def calculate_single_eof(data_array, n_modes=EOF_N_MODES):
    """Helper function to perform EOF analysis on a single DataArray."""

    if data_array.time.size < n_modes:
        print(f"Skipping EOF; not enough time steps ({data_array.time.size})")
        return None
        
    coslat = np.cos(np.deg2rad(data_array['latitude'].values))
    weights = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(data_array, weights=weights)
    
    eofs = solver.eofs(neofs=n_modes)
    pcs = solver.pcs(npcs=n_modes, pcscaling=1)
    variance_fractions = solver.varianceFraction(neigs=n_modes)
    
    return xr.Dataset({
        'eofs': eofs,
        'pcs': pcs,
        'variance_fractions': variance_fractions
    })

@cache_result('eof_analysis_results')
def perform_eof_analysis(obs_results, cmip_results, n_modes=EOF_N_MODES):
    """Performs EOF analysis on observed data, the multi-model mean, and each individual CMIP model."""
    
    print("\nPerforming EOF analysis on all data sources...")
    
    #spatial extent
    lon_min, lon_max, lat_min, lat_max = EXTENT
    lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)

    sources_to_analyze = {}
    
    #observed data
    sources_to_analyze['observed'] = obs_results['odsl_yearly'].rename({'year': 'time'})

    #CMIP data with detrending to remove long term trend such as climate change (end with residual)
    model_full_ts = cmip_results['full_timeseries'].sel(time=slice(START_YEAR, END_YEAR))
    model_full_ts['time'] = model_full_ts['time'].astype(int)
    model_detrended = detrend_timeseries(model_full_ts, degree=VARIABILITY_DETREND_DEGREE, dim='time')
    
    #multimodel mean
    sources_to_analyze['mmm'] = model_detrended.mean(dim='model')
    
    #individual models
    for model_name in model_detrended.model.values:
        sources_to_analyze[model_name] = model_detrended.sel(model=model_name)

    #run EOF
    all_eof_results = {}
    for name, data_array in sources_to_analyze.items():
        print(f"Analyzing source: {name}...")
        try:
            data_regional = data_array.sel(latitude=lat_slice, longitude=lon_slice)
            eof_result_ds = calculate_single_eof(data_regional, n_modes)
            if eof_result_ds is not None:
                all_eof_results[name] = eof_result_ds
        except Exception as e:
            print(f"Could not perform EOF analysis for {name}: {e}")
            
    print("Completed EOF analysis for all sources.")

    return all_eof_results

@cache_result('indices_correlation_results')
def correlate_with_indices(all_eof_results):
    """Correlates the PCs from each EOF analysis result with climate indices."""

    print("\nCorrelating PCs with climate indices for all sources...")

    climate_indices = load_climate_indices_dict()
    all_correlation_results = {}
    
    for name, eof_result_ds in all_eof_results.items():
        print(f"Correlating for source: {name}...")
        pcs = eof_result_ds['pcs']
        correlations = calculate_pc_index_correlations(pcs, climate_indices)
        all_correlation_results[name] = correlations
        
        #print results
        for index_name, corr_da in correlations.items():
            print(f"Correlations with {index_name.upper()}: {np.round(corr_da.values, 2)}")
            
    return all_correlation_results

def main():
    """Run complete analysis."""

    print("ODSL analysis starting...")
    fig_dir = './figures/'

    #progress bar
    progress_columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()]

    with Progress(*progress_columns) as progress:

        total_steps = 11
        task_id = progress.add_task("[cyan]Overall progress", total=total_steps)

        #observed ODSL
        progress.update(task_id, description="[bold blue]Step 1/10:[/bold blue] Calculating the observed ODSL", advance=1)
        obs_results = calculate_observed_odsl()
        progress.console.print(f"Observed ODSL range: {obs_results['odsl'].min().item():.2f} to {obs_results['odsl'].max().item():.2f} mm/yr")
        progress.console.print("[green]✔ Completed observed ODSL calculation.[/green]")
        
        #smoothing comparison
        progress.update(task_id, description="[bold blue]Step 2/10:[/bold blue] Smoothing comparison", advance=1)
        smoothing_results = calculate_smoothing_sensitivity()
        
        #valid CMIP models
        progress.update(task_id, description="[bold blue]Step 3/9:[/bold blue] Finding valid CMIP models", advance=1)
        models_df = valid_models_table()
        progress.console.print(models_df)

        #CMIP models
        progress.update(task_id, description="[bold blue]Step 4/10:[/bold blue] Processing CMIP models", advance=1)
        cmip_results = process_cmip_models()
        progress.console.print(f"Processed {cmip_results.attrs['valid_models_count']} CMIP models")

        #incrementing window analysis
        progress.update(task_id, description="[bold blue]Step 5/10:[/bold blue] Performing incrementing window analysis", advance=1)
        incrementing_window_results = perform_incrementing_window_analysis(obs_results, cmip_results)
        progress.console.print("[green]✔ Completed incrementing window analysis.[/green]")

        #sliding window analysis
        progress.update(task_id, description="[bold blue]Step 6/10:[/bold blue] Performing sliding window analysis", advance=1)
        sliding_results = perform_sliding_window_analysis()
        progress.console.print("[green]✔ Completed sliding window analysis.[/green]")

        #scenario data
        progress.update(task_id, description="[bold blue]Step 7/10:[/bold blue] Processing scenario data", advance=1)
        scenario_results = process_cmip_scenario_data()
        progress.console.print("[green]✔ Completed scenario data processing.[/green]")

        #EOF analysis
        progress.update(task_id, description="[bold blue]Step 8/10:[/bold blue] EOF analysis", advance=1)
        eof_results = perform_eof_analysis(obs_results, cmip_results, n_modes=EOF_N_MODES)

        #EOF correlation with indices
        progress.update(task_id, description="[bold blue]Step 9/10:[/bold blue] Correlating with climate indices", advance=1)
        correlation_results = correlate_with_indices(eof_results)
        progress.console.print("[green]✔ Completed EOF analysis and correlations.[/green]")

        #figures
        progress.update(task_id, description="[bold blue]Step 10/10:[/bold blue] Generating all figures", advance=1)
        progress.console.print("\nAll calculations complete. Generating figures...")
        create_all_figures(obs_results=obs_results, smoothing_results=smoothing_results, cmip_results=cmip_results, incrementing_window_results=incrementing_window_results, sliding_results=sliding_results, scenario_results=scenario_results, eof_results=eof_results, correlation_results=correlation_results, fig_dir=fig_dir)
        progress.console.print("[bold green]✔ All figures generated![/bold green]")
        progress.update(task_id, description="[bold green]Analysis complete!", advance=1)

if __name__ == "__main__":
    main()