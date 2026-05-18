"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Execute the full ODSL calculations / analysis and plotting.
"""

from data_loader import (load_altimetry_data, load_budget_data, load_gia_data, load_cmip_model_data, get_cmip_files_inventory, load_climate_indices_dict, load_wind_stress_data)
from utils import (setup_esmf_environment, cache_result, calculate_weighted_stats, create_region_mask, detrend_timeseries, calculate_pc_index_correlations, remove_global_mean, calculate_single_eof, monte_carlo_significance_test, compute_field_significance, calculate_power_spectrum)
from plotting import create_all_figures 
from config import (CMIP_VERSION, PROCESS_ALL_SCENARIOS, START_YEAR, END_YEAR, SLIDING_START_YEAR, EXTENT, TARGET_CMIP5_MODELS, TARGET_CMIP6_MODELS, VARIABILITY_DETREND_DEGREE, CMIP_SCENARIOS, CMIP5_FUTURE_SCENARIO, CMIP6_FUTURE_SCENARIO, PROCESS_PICONTROL, N_MODES_OBSERVED, APPLY_SPATIAL_SMOOTHING, SPATIAL_SMOOTHING_SIGMA, ALPHA, USE_ROTATED_EOF, MONTE_CARLO_SIGNIFICANCE_TEST, N_REALIZATIONS_MONTE_CARLO, EOF_N_MODES)

#setup_esmf_environment()

import os
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import warnings
import time
from scipy import stats
from datetime import datetime

#correct configuration
if CMIP_VERSION == 'CMIP5':
    TARGET_MODELS   = TARGET_CMIP5_MODELS
    FUTURE_SCENARIO = CMIP5_FUTURE_SCENARIO
elif CMIP_VERSION == 'CMIP6':
    TARGET_MODELS   = TARGET_CMIP6_MODELS
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
    duacs_ds = load_altimetry_data()
    
    #budget
    asl_frederikse_ds = load_budget_data(extend_to_year=END_YEAR if END_YEAR > 2018 else None)
    
    #weird loading inconsistency
    if isinstance(asl_frederikse_ds, xr.Dataset):
        if '__xarray_dataarray_variable__' in asl_frederikse_ds:
            asl_frederikse_da = asl_frederikse_ds['__xarray_dataarray_variable__']
        else:
            var_name = list(asl_frederikse_ds.data_vars)[0]
            asl_frederikse_da = asl_frederikse_ds[var_name]
            print(f"Using budget variable: '{var_name}'")
    else:
        asl_frederikse_da = asl_frederikse_ds
    
    #GIA
    gia_data   = load_gia_data()
    gia_rad_da = gia_data['gia_rad']
    gia_sea_da = gia_data['gia_sea']

    #potential smoothing altimetry dataset
    if APPLY_SPATIAL_SMOOTHING:
        print(f"Applying Gaussian spatial smoothing with sigma={SPATIAL_SMOOTHING_SIGMA}...")
        
        def smooth_func(data):
            kernel = Gaussian2DKernel(x_stddev=SPATIAL_SMOOTHING_SIGMA)
            return convolve(data, kernel, boundary='extend', preserve_nan=True)

        duacs_ds['sla'] = xr.apply_ufunc(smooth_func, duacs_ds['sla'], input_core_dims=[['latitude', 'longitude']], output_core_dims=[['latitude', 'longitude']], vectorize=True, dask='parallelized', output_dtypes=[float])

    #yearly mean after optional smoothing
    duacs_yearly = duacs_ds.groupby('time.year').mean().compute()
    
    #common years
    year_slice   = slice(START_YEAR, END_YEAR)
    alt_years    = duacs_yearly.sel(year=year_slice).year.values
    fr_years     = asl_frederikse_da.sel(year=year_slice).year.values
    common_years = np.intersect1d(alt_years, fr_years)

    print(f"Analysis period: {common_years.min()}-{common_years.max()} ({len(common_years)} years)")

    #trends over common period
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

        #altimetry
        print("Computing altimetry trends...")
        polyfit_ds_sla = duacs_yearly.sla.sel(year=common_years).polyfit(dim='year', deg=1)
        trend_sla_alt  = polyfit_ds_sla['polyfit_coefficients'].sel(degree=1)

        #budget
        print("Computing budget trends...")
        polyfit_ds_asl = asl_frederikse_da.sel(year=common_years).polyfit(dim='year', deg=1)
        trend_asl_fr   = polyfit_ds_asl['polyfit_coefficients'].sel(degree=1)

    #regridding
    print("Creating regridders...")
    regridder_frederikse = xe.Regridder(trend_asl_fr, trend_sla_alt, 'bilinear', periodic=True, extrap_method='nearest_s2d')
    regridder_gia        = xe.Regridder(gia_rad_da, trend_sla_alt, 'bilinear', periodic=True, extrap_method='nearest_s2d')
    
    #regrid to altimetry grid
    print("Regridding data...")
    trend_asl_fr_regridded     = regridder_frederikse(trend_asl_fr)
    gia_rad_regridded          = regridder_gia(gia_rad_da)
    gia_sea_regridded          = regridder_gia(gia_sea_da)
    gia_geoid_correction_mm_yr = gia_sea_regridded + gia_rad_regridded
    
    #convert to mm/yr
    trend_sla_alt_mm_yr          = trend_sla_alt * 10  #cm/yr -> mm/yr
    trend_asl_fr_regridded_mm_yr = trend_asl_fr_regridded
    gia_regridded_mm_yr          = gia_geoid_correction_mm_yr
    
    #calculate ODSL
    odsl_mm_yr = trend_sla_alt_mm_yr - trend_asl_fr_regridded_mm_yr - gia_regridded_mm_yr

    assert isinstance(trend_sla_alt_mm_yr, xr.DataArray)
    assert isinstance(trend_asl_fr_regridded_mm_yr, xr.DataArray)
    assert isinstance(gia_regridded_mm_yr, xr.DataArray)
    assert isinstance(odsl_mm_yr, xr.DataArray)

    trend_sla_alt_mm_yr.name          = 'MSL'
    trend_asl_fr_regridded_mm_yr.name = 'Geoid'
    gia_regridded_mm_yr.name          = 'GIA'
    odsl_mm_yr.name                   = 'ODSL'

    #weighted global mean removal
    print("Removing global mean...")
    
    #apply removal global mean (to obtain regional ODSL per definition)
    odsl_mm_yr  = remove_global_mean(odsl_mm_yr)

    #select north atlantic only
    region_mask = create_region_mask(trend_sla_alt_mm_yr, EXTENT)

    trend_sla_alt_mm_yr          = trend_sla_alt_mm_yr.where(region_mask)
    trend_asl_fr_regridded_mm_yr = trend_asl_fr_regridded_mm_yr.where(region_mask)
    gia_regridded_mm_yr          = gia_regridded_mm_yr.where(region_mask)
    odsl_mm_yr                   = odsl_mm_yr.where(region_mask)

    odsl_mm_yr = odsl_mm_yr.compute()
    ODSL_regional = odsl_mm_yr.copy()

    print(f"ODSL range: {odsl_mm_yr.min().item():.2f} to {odsl_mm_yr.max().item():.2f} mm/yr")

    print("Calculating yearly ODSL fields...")
    
    #yearly ODSL fields
    sla_yearly_mm = duacs_yearly.sla.sel(year=common_years).compute() * 10  #cm -> mm
    asl_yearly    = asl_frederikse_da.sel(year=common_years)
    
    #regrid
    asl_yearly_regridded  = regridder_frederikse(asl_yearly)
    time_delta            = xr.DataArray(common_years - common_years[0], dims='year', coords={'year': common_years})
    gia_cumulative_signal = gia_geoid_correction_mm_yr * time_delta
    
    #subtract yearly budget and cumulative GIA
    odsl_yearly_fields = sla_yearly_mm - asl_yearly_regridded - gia_cumulative_signal

    print("Removing global mean from yearly fields...")
    odsl_yearly_fields = remove_global_mean(odsl_yearly_fields)
    
    #variability
    detrended_odsl = detrend_timeseries(odsl_yearly_fields, degree=VARIABILITY_DETREND_DEGREE, dim='year')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        observed_variability = detrended_odsl.std(dim='year') / 10    #mm -> cm

    #select north atlantic only
    observed_variability = observed_variability.where(region_mask)

    #anomaly from global mean for overview figure
    temporal_mean       = odsl_yearly_fields.mean(dim='year')
    odsl_yearly_anomaly = odsl_yearly_fields - temporal_mean

    #'regular' ODSL mean over observed period in cm 
    odsl_mean_m      = temporal_mean / 10  #mm -> cm
    odsl_mean_m      = odsl_mean_m.where(region_mask) 
    odsl_mean_m.name = 'odsl_mean'

    print(f"Yearly ODSL anomaly range: {odsl_yearly_anomaly.min().item():.2f} to {odsl_yearly_anomaly.max().item():.2f} mm")
    
    #signficance calculations
    print("Computing statistical significance...")

    #trend
    trend_sig           = compute_field_significance(odsl_yearly_fields, 'trend', region_mask=region_mask)
    odsl_trend_se       = trend_sig['std_error']
    odsl_trend_p_val    = trend_sig['p_value']
    odsl_trend_ci_lower = trend_sig['ci_lower']
    odsl_trend_ci_upper = trend_sig['ci_upper']

    #variability
    var_sig              = compute_field_significance(odsl_yearly_fields, 'variability', region_mask=region_mask)
    observed_variability = var_sig['field'] / 10  #mm -> cm
    variability_se       = var_sig['std_error'] / 10
    variability_ci_lower = var_sig['ci_lower'] / 10
    variability_ci_upper = var_sig['ci_upper'] / 10

    #mean ODSL
    odsl_mean_sig        = compute_field_significance(odsl_yearly_fields, 'ODSL', region_mask=region_mask)
    odsl_mean_se_m       = odsl_mean_sig['std_error'] / 10  #mm -> cm
    odsl_mean_p_val_m    = odsl_mean_sig['p_value']
    odsl_mean_ci_lower_m = odsl_mean_sig['ci_lower'] / 10
    odsl_mean_ci_upper_m = odsl_mean_sig['ci_upper'] / 10

    #clean up regridders
    try:
        regridder_frederikse.clean_weight_file() # type: ignore
        regridder_gia.clean_weight_file()        # type: ignore
    except AttributeError:
        pass

    odsl_yearly_fields.name  = 'odsl_yearly'
    odsl_yearly_anomaly.name = 'odsl_yearly_anomaly'

    output_ds = xr.Dataset({'odsl':                  odsl_mm_yr,

                            'odsl_mean':             odsl_mean_m,
                            'odsl_mean_std_error':   odsl_mean_se_m,
                            'odsl_mean_p_value':     odsl_mean_p_val_m,
                            'odsl_mean_ci_lower':    odsl_mean_ci_lower_m,
                            'odsl_mean_ci_upper':    odsl_mean_ci_upper_m,

                            'odsl_std_error':        odsl_trend_se,
                            'odsl_p_value':          odsl_trend_p_val,
                            'odsl_ci_lower':         odsl_trend_ci_lower,
                            'odsl_ci_upper':         odsl_trend_ci_upper,

                            'odsl_yearly':           odsl_yearly_fields,
                            'odsl_yearly_anomaly':   odsl_yearly_anomaly,

                            'msl':                   trend_sla_alt_mm_yr,
                            'geoid':                 trend_asl_fr_regridded_mm_yr,
                            'gia':                   gia_regridded_mm_yr,
                            'odsl_regional':         ODSL_regional,

                            'variability':           observed_variability,
                            'variability_std_error': variability_se,
                            'variability_ci_lower':  variability_ci_lower,
                            'variability_ci_upper':  variability_ci_upper})

    #global attributes
    output_ds.attrs['description']       = "Observed ODSL and its components."
    output_ds.attrs['analysis_period']   = f"{common_years.min()}-{common_years.max()}"
    output_ds.attrs['common_years_list'] = common_years.tolist() 

    return output_ds

@cache_result('smoothing_sensitivity_results')
def calculate_smoothing_sensitivity():
    """Calculate altimetry trends for different levels of Gaussian smoothing."""

    print("\nPerforming smoothing sensitivity analysis...")

    duacs_ds = load_altimetry_data()
    
    sigmas = [0, 3, 6, 9]
    trend_maps = []

    print(f"Calculating trends for sigma levels: {sigmas}")

    print("Calculating raw trend (Sigma=0)...")
    
    #yearly mean
    duacs_yearly = duacs_ds.groupby('time.year').mean()
    year_slice = slice(START_YEAR, END_YEAR)
    
    #trend raw data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
        
        polyfit_ds = duacs_yearly.sla.sel(year=year_slice).polyfit(dim='year', deg=1)
        
        #weird naming behaviour sometimes
        if 'polyfit_coefficients' in polyfit_ds:
             raw_trend = polyfit_ds['polyfit_coefficients'].sel(degree=1)
        else:
             raw_trend = polyfit_ds['__xarray_dataarray_variable___polyfit_coefficients'].sel(degree=1)

        raw_trend = raw_trend.compute()

    #smoothing
    for sigma in sigmas:
        if sigma == 0:
            trend_maps.append(raw_trend)
        else:
            print(f"Applying smoothing to trend map for sigma={sigma}...")
            
            kernel = Gaussian2DKernel(x_stddev=sigma)
            
            #convolution
            smoothed_values = convolve(raw_trend.values, kernel, boundary='extend', preserve_nan=True)
            
            smoothed_da = xr.DataArray(smoothed_values, coords=raw_trend.coords, dims=raw_trend.dims)
            
            trend_maps.append(smoothed_da)

    combined_trends = xr.concat(trend_maps, dim=pd.Index(sigmas, name='sigma'))

    #filter out north atlantic region only
    region_mask = create_region_mask(raw_trend, EXTENT)
    combined_trends = combined_trends.where(region_mask)
    
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
    combined_df = pd.concat([df_cmip5, df_cmip6], keys=['CMIP5', 'CMIP6'], names=['CMIP_Version', 'Model'])
    
    print("\nScan complete, returning combined availability table...")
    return combined_df

@cache_result('processed_CMIP_models')
def process_cmip_models():
    """Process CMIP models and return a single dataset."""
    
    print(f"\nProcessing {CMIP_VERSION} models...")
    
    inventory = get_cmip_files_inventory(CMIP_VERSION)
    all_files = inventory['all_files']
    
    #initialize
    model_names             = []
    trends_list             = []
    variability_list        = []
    variability_se_list     = []
    odsl_mean_list          = []
    full_timeseries_list    = []
    region_masks_list       = []
    trend_stats_list        = []
    trend_se_list           = []
    period_timeseries_list  = []
    p_values_list           = []
    odsl_mean_se_list       = []
    odsl_mean_p_val_list    = []
    odsl_mean_ci_lower_list = []
    odsl_mean_ci_upper_list = []
    
    for i, model_name in enumerate(TARGET_MODELS):
        hist_file = all_files['historical'].get(model_name)
        future_file = all_files[FUTURE_SCENARIO].get(model_name)
        
        if hist_file and future_file:
            print(f"Processing model: {model_name} ({i+1}/{len(TARGET_MODELS)})")

            try:
                #calculations per model
                #load cmip models
                combined_zos = load_cmip_model_data(model_name, future_scenario=FUTURE_SCENARIO, cmip_version=CMIP_VERSION)
                
                #time to integers
                if np.issubdtype(combined_zos.time.dtype, np.floating):
                    combined_zos['time'] = combined_zos['time'].astype(int)

                #remove global mean prior to masking North Atlantic
                combined_zos = remove_global_mean(combined_zos)
                
                #select North Atlantic region
                region_mask  = create_region_mask(combined_zos.isel(time=0), EXTENT)
                combined_zos = combined_zos.where(region_mask)
                
                period_data  = combined_zos.sel(time=slice(START_YEAR, END_YEAR))

                #signficance per model
                #trend
                trend_sig       = compute_field_significance(period_data, 'trend', region_mask=region_mask)
                slope_mm_yr     = trend_sig['field']
                se_slope_mm_yr  = trend_sig['std_error']
                p_val_da        = trend_sig['p_value']

                #variability
                var_sig              = compute_field_significance(period_data, 'variability', region_mask=region_mask)
                model_variability    = var_sig['field']
                model_variability_se = var_sig['std_error']

                #mean ODSL
                odsl_sig             = compute_field_significance(period_data, 'ODSL', region_mask=region_mask)
                model_odsl_mean_m    = odsl_sig['field']
                model_mean_se        = odsl_sig['std_error']
                model_mean_p_val     = odsl_sig['p_value']
                model_mean_ci_lower  = odsl_sig['ci_lower']
                model_mean_ci_upper  = odsl_sig['ci_upper']

                #trend statistics
                trend_stats = calculate_weighted_stats(slope_mm_yr, region_mask)
                
                #append results
                model_names.append(model_name)
                trends_list.append(slope_mm_yr)
                p_values_list.append(p_val_da)
                variability_list.append(model_variability)
                variability_se_list.append(model_variability_se)
                odsl_mean_list.append(model_odsl_mean_m)
                full_timeseries_list.append(combined_zos)
                period_timeseries_list.append(period_data)
                region_masks_list.append(region_mask)
                trend_stats_list.append(trend_stats)
                trend_se_list.append(se_slope_mm_yr)
                odsl_mean_se_list.append(model_mean_se)
                odsl_mean_p_val_list.append(model_mean_p_val)
                odsl_mean_ci_lower_list.append(model_mean_ci_lower)
                odsl_mean_ci_upper_list.append(model_mean_ci_upper)
                
            except Exception as e:
                print(f"Could not process model {model_name}: {e}")

    print(f"\nProcessed {len(model_names)} out of {len(TARGET_MODELS)} models")

    #concatenate
    model_trends             = xr.concat(trends_list, dim=pd.Index(model_names, name='model'))
    model_trend_p_values     = xr.concat(p_values_list, dim=pd.Index(model_names, name='model'))
    model_trend_se           = xr.concat(trend_se_list, dim=pd.Index(model_names, name='model'))

    model_variability        = xr.concat(variability_list, dim=pd.Index(model_names, name='model'))
    model_variability_se     = xr.concat(variability_se_list, dim=pd.Index(model_names, name='model'))

    model_odsl_means         = xr.concat(odsl_mean_list, dim=pd.Index(model_names, name='model'))
    model_odsl_mean_p_values = xr.concat(odsl_mean_p_val_list, dim=pd.Index(model_names, name='model'))
    model_odsl_mean_se       = xr.concat(odsl_mean_se_list, dim=pd.Index(model_names, name='model'))

    full_timeseries          = xr.concat(full_timeseries_list, dim=pd.Index(model_names, name='model'))
    region_masks             = xr.concat(region_masks_list, dim=pd.Index(model_names, name='model'))
    period_timeseries_concat = xr.concat(period_timeseries_list, dim=pd.Index(model_names, name='model'))

    #convert list
    trend_means = xr.DataArray([s['mean_x'] for s in trend_stats_list], coords={'model': model_names}, dims=['model'])
    trend_stds  = xr.DataArray([s['std_x'] for s in trend_stats_list], coords={'model': model_names}, dims=['model'])
    
    #multi-model mean
    model_mean_trend       = model_trends.mean(dim='model', skipna=True)
    model_mean_variability = model_variability.mean(dim='model', skipna=True)
    model_mean_odsl        = model_odsl_means.mean(dim='model', skipna=True)

    #significance multi-model mean
    print("Computing statistical significance for multi-model mean...")

    ensemble_mean_ts = period_timeseries_concat.mean(dim='model', skipna=True)
    valid_mask       = model_mean_trend.notnull()

    #trend
    mmm_trend_sig      = compute_field_significance(ensemble_mean_ts, 'trend', region_mask=valid_mask)
    mmm_trend_se       = mmm_trend_sig['std_error']
    mmm_trend_p_val    = mmm_trend_sig['p_value']
    mmm_trend_ci_lower = mmm_trend_sig['ci_lower']
    mmm_trend_ci_upper = mmm_trend_sig['ci_upper']

    #variability
    mmm_var_sig              = compute_field_significance(ensemble_mean_ts, 'variability', region_mask=valid_mask)
    mmm_variability_se       = mmm_var_sig['std_error']
    mmm_variability_ci_lower = mmm_var_sig['ci_lower']
    mmm_variability_ci_upper = mmm_var_sig['ci_upper']

    #mean ODSL
    mmm_odsl_sig           = compute_field_significance(ensemble_mean_ts, 'ODSL', region_mask=valid_mask)
    mmm_odsl_mean_se       = mmm_odsl_sig['std_error']
    mmm_odsl_mean_p_val    = mmm_odsl_sig['p_value']
    mmm_odsl_mean_ci_lower = mmm_odsl_sig['ci_lower']
    mmm_odsl_mean_ci_upper = mmm_odsl_sig['ci_upper']

    output_ds = xr.Dataset({'model_trend':                      model_trends,
                            'model_trend_p_values':             model_trend_p_values,
                            'model_trend_std_error':            model_trend_se,

                            'model_mean_trend_zos' :            model_mean_trend,
                            'model_mean_trend':                 model_mean_trend,
                            'model_mean_trend_std_error':       mmm_trend_se,
                            'model_mean_trend_p_value':         mmm_trend_p_val,
                            'model_mean_trend_ci_lower':        mmm_trend_ci_lower, 
                            'model_mean_trend_ci_upper':        mmm_trend_ci_upper,

                            'model_variability':                model_variability,
                            'model_variability_std_error':      model_variability_se,

                            'model_mean_variability':           model_mean_variability,
                            'model_mean_variability_std_error': mmm_variability_se,
                            'model_mean_variability_ci_lower':  mmm_variability_ci_lower,
                            'model_mean_variability_ci_upper':  mmm_variability_ci_upper,

                            'model_odsl_mean':                  model_odsl_means,
                            'model_mean_odsl':                  model_mean_odsl,
                            'model_odsl_mean_p_values':         model_odsl_mean_p_values,
                            'model_odsl_mean_std_error':        model_odsl_mean_se,

                            'model_mean_odsl_std_error':        mmm_odsl_mean_se, 
                            'model_mean_odsl_p_value':          mmm_odsl_mean_p_val,    
                            'model_mean_odsl_ci_lower':         mmm_odsl_mean_ci_lower,  
                            'model_mean_odsl_ci_upper':         mmm_odsl_mean_ci_upper,  

                            'full_timeseries':                  full_timeseries,
                            'region_mask':                      region_masks.astype('int8'),
                            'trend_mean':                       trend_means,
                            'trend_std':                        trend_stds})
    
    #global attributes
    output_ds.attrs['description']        = f"Processed {CMIP_VERSION} model trends, variability, and timeseries."
    output_ds.attrs['valid_models_count'] = len(model_names)
    
    return output_ds

@cache_result('processed_all_CMIP_scenarios')
def process_all_cmip_scenarios(default_result=None):
    """CMIP5 and CMIP6 for overview figure."""

    if not PROCESS_ALL_SCENARIOS:
        print("Skipping all CMIP scenario processing (PROCESS_ALL_SCENARIOS=False)")
        return None

    orig = (CMIP_VERSION, TARGET_MODELS, FUTURE_SCENARIO)

    cmip5_all = {}
    cmip6_all = {}

    if default_result is not None:
        default_scenario         = CMIP5_FUTURE_SCENARIO if CMIP_VERSION == 'CMIP5' else CMIP6_FUTURE_SCENARIO
        target                   = cmip5_all if CMIP_VERSION == 'CMIP5' else cmip6_all
        target[default_scenario] = default_result

    for scen in ['rcp26', 'rcp45', 'rcp85']:
        if scen not in cmip5_all:
            globals()['CMIP_VERSION']    = 'CMIP5'
            globals()['TARGET_MODELS']   = TARGET_CMIP5_MODELS
            globals()['FUTURE_SCENARIO'] = scen
            try:
                try:
                    cmip5_all[scen] = process_cmip_models.__wrapped__() # type: ignore
                except AttributeError:
                    cmip5_all[scen] = process_cmip_models()
            finally:
                globals()['CMIP_VERSION'], globals()['TARGET_MODELS'], globals()['FUTURE_SCENARIO'] = orig

    for scen in ['ssp126', 'ssp245', 'ssp585']:
        if scen not in cmip6_all:
            globals()['CMIP_VERSION']    = 'CMIP6'
            globals()['TARGET_MODELS']   = TARGET_CMIP6_MODELS
            globals()['FUTURE_SCENARIO'] = scen
            try:
                try:
                    cmip6_all[scen] = process_cmip_models.__wrapped__() # type: ignore
                except AttributeError:
                    cmip6_all[scen] = process_cmip_models()
            finally:
                globals()['CMIP_VERSION'], globals()['TARGET_MODELS'], globals()['FUTURE_SCENARIO'] = orig

    result = {}
    for scen, ds in cmip5_all.items():
        result[f'cmip5_{scen}'] = ds
    for scen, ds in cmip6_all.items():
        result[f'cmip6_{scen}'] = ds
    return result

@cache_result('incrementing_window_results')
def perform_incrementing_window_analysis(obs_results, cmip_results):
    """Performs an analysis with an incrementally increasing window size, calculating PCC and RMSE between observed and modeled ODSL trends for each window."""
    
    print("\nPerforming incrementing window analysis...")

    warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

    #yearly timeseries data for observations and models
    obs_yearly_ts = obs_results['odsl_yearly'].rename({'year': 'time'})
    model_full_ts = cmip_results['full_timeseries']

    #sample model grid and region mask for regridding and weighting
    sample_model_grid = cmip_results['model_mean_trend']
    region_mask       = create_region_mask(sample_model_grid, EXTENT)

    #regrid observed to model grid
    print("Regridding observed timeseries to model grid...")

    regridder_obs_to_model  = xe.Regridder(obs_yearly_ts, sample_model_grid, 'bilinear', periodic=True, extrap_method='nearest_s2d')
    obs_yearly_ts_regridded = regridder_obs_to_model(obs_yearly_ts)

    try:
        regridder_obs_to_model.clean_weight_file() # type: ignore
    except AttributeError:
        pass

    #dictionary of all timeseries sources to analyze
    sources_ts = {'multi model mean': model_full_ts.mean(dim='model')}

    for model_name in cmip_results.model.values:
        sources_ts[model_name] = model_full_ts.sel(model=model_name)

    window_end_years = range(START_YEAR + 1, END_YEAR + 1)
    
    #dictionaries
    variables = ['trend', 'variability', 'ODSL']
    all_pcc   = {var: {source: [] for source in sources_ts.keys()} for var in variables}
    all_rmse  = {var: {source: [] for source in sources_ts.keys()} for var in variables}
    all_sign  = {var: {source: [] for source in sources_ts.keys()} for var in variables}

    assert isinstance(obs_yearly_ts_regridded, xr.DataArray)

    print(f"Calculatin metrics for {len(window_end_years)} incrementing windows...")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
        warnings.filterwarnings('ignore', message='.*Polyfit may be poorly conditioned.*')
        warnings.filterwarnings('ignore', message='.*rank.*deficient.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')

        for end_year in window_end_years:
            window_slice = slice(START_YEAR, end_year)
            
            #observed window data
            obs_window_data = obs_yearly_ts_regridded.sel(time=window_slice)

            #observed metrics
            #trend (mm/yr)
            obs_trend_coeffs = obs_window_data.polyfit(dim='time', deg=1)
            obs_trend        = obs_trend_coeffs['polyfit_coefficients'].sel(degree=1)

            #variability (cm)
            obs_detrended   = detrend_timeseries(obs_window_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
            obs_variability = obs_detrended.std(dim='time') / 10
            
            #ODSL (cm)
            obs_mean_odsl = obs_window_data.mean(dim='time') / 10

            #loop through each model
            for source_name, source_ts in sources_ts.items():
                #model window data
                model_window_data = source_ts.sel(time=window_slice)

                #model metrics
                #trend (mm/yr)
                model_trend_coeffs = model_window_data.polyfit(dim='time', deg=1)
                model_trend        = model_trend_coeffs['polyfit_coefficients'].sel(degree=1) * 10 #cm/yr -> mm/yr

                #variability (cm)
                model_detrended   = detrend_timeseries(model_window_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
                model_variability = model_detrended.std(dim='time')
                
                #ODSL (cm)
                model_mean_odsl = model_window_data.mean(dim='time')

                #stats for each variable
                #trend
                stats_trend = calculate_weighted_stats(model_trend, region_mask, data_y=obs_trend)
                all_pcc['trend'][source_name].append(stats_trend['pcc'])
                all_rmse['trend'][source_name].append(stats_trend['rmse'])
                all_sign['trend'][source_name].append(stats_trend['sign_agreement'])
                
                #variability
                stats_var = calculate_weighted_stats(model_variability, region_mask, data_y=obs_variability)
                all_pcc['variability'][source_name].append(stats_var['pcc'])
                all_rmse['variability'][source_name].append(stats_var['rmse'])
                all_sign['variability'][source_name].append(stats_var['sign_agreement'])
                
                #ODSL
                stats_odsl = calculate_weighted_stats(model_mean_odsl, region_mask, data_y=obs_mean_odsl)
                all_pcc['ODSL'][source_name].append(stats_odsl['pcc'])
                all_rmse['ODSL'][source_name].append(stats_odsl['rmse'])
                all_sign['ODSL'][source_name].append(stats_odsl['sign_agreement'])

    print("Incrementing window analysis complete.")

    #output
    source_coord  = list(sources_ts.keys())
    end_year_list = list(window_end_years)

    #3D arrays (variable, source, end_year)
    pcc_data  = np.array([[all_pcc[var][s] for s in source_coord] for var in variables])
    rmse_data = np.array([[all_rmse[var][s] for s in source_coord] for var in variables])
    sign_data = np.array([[all_sign[var][s] for s in source_coord] for var in variables])
    
    pcc_da  = xr.DataArray(pcc_data,  coords={'variable': variables, 'source': source_coord, 'end_year': end_year_list}, dims=['variable', 'source', 'end_year'], name='pcc')
    rmse_da = xr.DataArray(rmse_data, coords={'variable': variables, 'source': source_coord, 'end_year': end_year_list}, dims=['variable', 'source', 'end_year'], name='rmse')
    sign_da = xr.DataArray(sign_data, coords={'variable': variables, 'source': source_coord, 'end_year': end_year_list}, dims=['variable', 'source', 'end_year'], name='sign_agreement')
    
    output_ds = xr.Dataset({'pcc': pcc_da, 'rmse': rmse_da, 'sign_agreement': sign_da})
    
    output_ds.attrs['description'] = "PCC, RMSE, and sign agreement from incrementing window analysis for trend, variability, and ODSL."
    
    return output_ds

@cache_result('piControl_incrementing_window_results')
def perform_piControl_incrementing_window():
    """Compute piControl incrementing window analysis, mirroring the historical approach."""
    
    if not PROCESS_PICONTROL:
        print("Skipping piControl incrementing window analysis (PROCESS_PICONTROL=False)")
        return None

    print("\npiControl incrementing window analysis...")
    
    obs_results   = calculate_observed_odsl()
    obs_yearly_ts = obs_results['odsl_yearly'].rename({'year': 'time'})
    
    cmip_results_ds   = process_cmip_models()
    sample_model_grid = cmip_results_ds['model_mean_trend']
    region_mask       = create_region_mask(sample_model_grid, EXTENT)
    
    #regridding
    regridder            = xe.Regridder(obs_yearly_ts, sample_model_grid, 'bilinear', periodic=True, extrap_method='nearest_s2d')
    obs_yearly_regridded = regridder(obs_yearly_ts)
    try:
        regridder.clean_weight_file()  # type: ignore
    except AttributeError:
        pass
    assert isinstance(obs_yearly_regridded, xr.DataArray)
    
    inventory       = get_cmip_files_inventory(CMIP_VERSION)
    all_files       = inventory['all_files']
    selected_models = list(cmip_results_ds.model.values)
    
    #first 32 years of each piControl model
    print("Loading piControl timeseries (first 32 years per model)...")
    pi_sources_ts = {}
    window_size   = END_YEAR - START_YEAR + 1  
    
    for model_name in selected_models:
        pi_file = all_files.get('piControl', {}).get(model_name)
        if not pi_file:
            continue
        
        try:
            combined_zos = load_cmip_model_data(model_name, hist_scenario='piControl', cmip_version=CMIP_VERSION)
            
            if np.issubdtype(combined_zos.time.dtype, np.floating):
                combined_zos['time'] = combined_zos['time'].astype(int)
            
            combined_zos = remove_global_mean(combined_zos)
            
            #first 32 years
            start_year_pi = int(combined_zos.time.min().item())
            combined_zos  = combined_zos.sel(time=slice(start_year_pi, start_year_pi + window_size - 1))
            
            if len(combined_zos.time) < window_size:
                print(f"Skipping {model_name}: only {len(combined_zos.time)} years available")
                continue
            
            #match historical axis
            combined_zos['time'] = np.arange(START_YEAR, END_YEAR + 1)
            
            #region mask
            pi_region_mask = create_region_mask(combined_zos.isel(time=0), EXTENT)
            combined_zos   = combined_zos.where(pi_region_mask)
            
            pi_sources_ts[model_name] = combined_zos
            print(f"Loaded piControl: {model_name}")
            
        except Exception as e:
            print(f"Could not process piControl {model_name}: {e}")
    
    if not pi_sources_ts:
        print("No piControl models loaded!")
        return None
    
    #multi-model mean
    pi_stack                          = xr.concat(list(pi_sources_ts.values()), dim='model')
    pi_sources_ts['multi model mean'] = pi_stack.mean(dim='model')
    del pi_stack
    
    #incrementing window analysis
    variables        = ['trend', 'variability', 'ODSL']
    window_end_years = list(range(START_YEAR + 1, END_YEAR + 1))
    
    all_pcc  = {var: {source: [] for source in pi_sources_ts.keys()} for var in variables}
    all_rmse = {var: {source: [] for source in pi_sources_ts.keys()} for var in variables}
    all_sign = {var: {source: [] for source in pi_sources_ts.keys()} for var in variables}
    
    print(f"Calculating metrics for {len(window_end_years)} incrementing windows...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for end_year in window_end_years:
            window_slice = slice(START_YEAR, end_year)
            
            #observed metrics
            obs_window      = obs_yearly_regridded.sel(time=window_slice)
            obs_trend       = obs_window.polyfit(dim='time', deg=1)['polyfit_coefficients'].sel(degree=1)
            obs_detrended   = detrend_timeseries(obs_window, degree=VARIABILITY_DETREND_DEGREE, dim='time')
            obs_variability = obs_detrended.std(dim='time') / 10
            obs_mean_odsl   = obs_window.mean(dim='time') / 10
            
            for source_name, source_ts in pi_sources_ts.items():
                model_window = source_ts.sel(time=window_slice)
                
                #model metrics
                model_trend       = model_window.polyfit(dim='time', deg=1)['polyfit_coefficients'].sel(degree=1) * 10
                model_detrended   = detrend_timeseries(model_window, degree=VARIABILITY_DETREND_DEGREE, dim='time')
                model_variability = model_detrended.std(dim='time')
                model_mean_odsl   = model_window.mean(dim='time')
                
                #stats
                st = calculate_weighted_stats(model_trend, region_mask, data_y=obs_trend)
                sv = calculate_weighted_stats(model_variability, region_mask, data_y=obs_variability)
                so = calculate_weighted_stats(model_mean_odsl, region_mask, data_y=obs_mean_odsl)
                
                all_pcc['trend'][source_name].append(st['pcc'])
                all_rmse['trend'][source_name].append(st['rmse'])
                all_sign['trend'][source_name].append(st['sign_agreement'])
                
                all_pcc['variability'][source_name].append(sv['pcc'])
                all_rmse['variability'][source_name].append(sv['rmse'])
                all_sign['variability'][source_name].append(sv['sign_agreement'])
                
                all_pcc['ODSL'][source_name].append(so['pcc'])
                all_rmse['ODSL'][source_name].append(so['rmse'])
                all_sign['ODSL'][source_name].append(so['sign_agreement'])
    
    print("piControl incrementing window analysis complete.")
    
    #regional mean and SE per model
    print("Computing full-period regional means and SEs...")

    bar_mean_trend = {}
    bar_se_trend   = {}
    bar_mean_var   = {}
    bar_se_var     = {}
    bar_mean_odsl  = {}
    bar_se_odsl    = {}

    for source_name, source_ts in pi_sources_ts.items():
        if source_name == 'multi model mean':
            continue

        period_data = source_ts.sel(time=slice(START_YEAR, END_YEAR))
        n_t         = len(period_data.time)
        pi_mask     = create_region_mask(period_data.isel(time=0), EXTENT)

        #trend + SE
        trend_coeffs = period_data.polyfit(dim='time', deg=1)
        slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
        slope_mm_yr = slope * 10

        fitted    = xr.polyval(period_data.time, trend_coeffs.polyfit_coefficients)
        residuals = period_data - fitted
        sse       = (residuals**2).sum(dim='time')
        time_vals = period_data.time.values.astype(float)
        ssx       = np.sum((time_vals - np.mean(time_vals))**2)
        se_field  = np.sqrt(sse / (n_t - 2)) / np.sqrt(ssx) * 10

        st                          = calculate_weighted_stats(slope_mm_yr, pi_mask)
        bar_mean_trend[source_name] = st['mean_x']
        w                           = np.cos(np.deg2rad(se_field.latitude))
        bar_se_trend[source_name]   = float(se_field.weighted(w).mean(dim=['latitude', 'longitude'], skipna=True).item())

        #variability + SE
        detrended                 = detrend_timeseries(period_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
        variability_map           = detrended.std(dim='time')
        sv                        = calculate_weighted_stats(variability_map, pi_mask)
        bar_mean_var[source_name] = sv['mean_x']
        bar_se_var[source_name]   = sv['mean_x'] / np.sqrt(2 * (n_t - 1))

        #mean ODSL + SE
        mean_odsl_map              = period_data.mean(dim='time')
        odsl_std_map               = period_data.std(dim='time')
        so                         = calculate_weighted_stats(mean_odsl_map, pi_mask)
        bar_mean_odsl[source_name] = so['mean_x']
        se_odsl_field              = odsl_std_map / np.sqrt(n_t)
        bar_se_odsl[source_name]   = float(se_odsl_field.weighted(w).mean(dim=['latitude', 'longitude'], skipna=True).item())

    bar_models = [m for m in selected_models if m in bar_mean_trend]

    #output
    source_coord = list(pi_sources_ts.keys())
    
    pcc_data  = np.array([[all_pcc[var][s] for s in source_coord] for var in variables])
    rmse_data = np.array([[all_rmse[var][s] for s in source_coord] for var in variables])
    sign_data = np.array([[all_sign[var][s] for s in source_coord] for var in variables])
    
    output_ds = xr.Dataset({'pcc':            xr.DataArray(pcc_data, coords={'variable': variables, 'source': source_coord, 'end_year': list(window_end_years)}, dims=['variable', 'source', 'end_year']),
                            'rmse':           xr.DataArray(rmse_data, coords={'variable': variables, 'source': source_coord, 'end_year': list(window_end_years)}, dims=['variable', 'source', 'end_year']),
                            'sign_agreement': xr.DataArray(sign_data, coords={'variable': variables, 'source': source_coord, 'end_year': list(window_end_years)}, dims=['variable', 'source', 'end_year'])})
    
    output_ds['bar_mean_trend']       = xr.DataArray([bar_mean_trend[m] for m in bar_models], coords={'bar_model': bar_models}, dims=['bar_model'])
    output_ds['bar_se_trend']         = xr.DataArray([bar_se_trend[m] for m in bar_models], coords={'bar_model': bar_models}, dims=['bar_model'])
    output_ds['bar_mean_variability'] = xr.DataArray([bar_mean_var[m] for m in bar_models], coords={'bar_model': bar_models}, dims=['bar_model'])
    output_ds['bar_se_variability']   = xr.DataArray([bar_se_var[m] for m in bar_models], coords={'bar_model': bar_models}, dims=['bar_model'])
    output_ds['bar_mean_odsl']        = xr.DataArray([bar_mean_odsl[m] for m in bar_models], coords={'bar_model': bar_models}, dims=['bar_model'])
    output_ds['bar_se_odsl']          = xr.DataArray([bar_se_odsl[m] for m in bar_models], coords={'bar_model': bar_models}, dims=['bar_model'])
    output_ds.attrs['description']    = "piControl incrementing window analysis (first 32 years per model)."
    
    return output_ds

@cache_result('sliding_window_results')
def perform_sliding_window_analysis():
    """Sliding window analysis on all models."""
    
    print("\nSliding window analysis...")
    
    #processed models
    cmip_results_ds = process_cmip_models() 
    obs_results     = calculate_observed_odsl()
    odsl_mm_yr      = obs_results['odsl']
    
    #regridding to model grids
    #sample grid
    sample_model_grid = cmip_results_ds['model_mean_trend']

    #trend
    odsl_filled = odsl_mm_yr.interpolate_na(dim='longitude', method='nearest', limit=5, fill_value="extrapolate")
    odsl_filled = odsl_filled.interpolate_na(dim='latitude', method='nearest', limit=5, fill_value="extrapolate")
    
    regridder_obs_to_model = xe.Regridder(odsl_filled, sample_model_grid, 'bilinear', periodic=True)
    odsl_mm_yr_regridded   = regridder_obs_to_model(odsl_filled)
    
    #variability
    var_filled = obs_results['variability'].interpolate_na(dim='longitude', method='nearest', limit=5, fill_value="extrapolate")
    var_filled = var_filled.interpolate_na(dim='latitude', method='nearest', limit=5, fill_value="extrapolate")
    
    regridder_var_to_model = xe.Regridder(var_filled, sample_model_grid, 'bilinear', periodic=True)
    odsl_var_obs_regridded = regridder_var_to_model(var_filled)

    #ODSL
    odsl_mean_filled = obs_results['odsl_mean'].interpolate_na(dim='longitude', method='nearest', limit=5, fill_value="extrapolate")
    odsl_mean_filled = odsl_mean_filled.interpolate_na(dim='latitude', method='nearest', limit=5, fill_value="extrapolate")

    regridder_odsl_mean = xe.Regridder(odsl_mean_filled, sample_model_grid, 'bilinear', periodic=True)
    odsl_obs_regridded  = regridder_odsl_mean(odsl_mean_filled)

    #yearly ODSL fields
    obs_yearly            = obs_results['odsl_yearly']
    obs_yearly_filled     = obs_yearly.interpolate_na(dim='longitude', method='nearest', limit=5, fill_value="extrapolate")
    obs_yearly_filled     = obs_yearly_filled.interpolate_na(dim='latitude', method='nearest', limit=5, fill_value="extrapolate")
    odsl_yearly_regridded = regridder_obs_to_model(obs_yearly_filled)

    try:
        regridder_obs_to_model.clean_weight_file() # type: ignore
        regridder_var_to_model.clean_weight_file() # type: ignore
        regridder_odsl_mean.clean_weight_file()    # type: ignore
    except AttributeError:
        pass
    
    assert isinstance(odsl_mm_yr_regridded, xr.DataArray)
    assert isinstance(odsl_var_obs_regridded, xr.DataArray)
    assert isinstance(odsl_obs_regridded, xr.DataArray)
    
    #sliding window analysis
    #initialize lists for each model
    all_pcc                 = []
    all_rmse                = []
    all_mean_trends         = []
    all_trends              = []
    all_pcc_var             = []
    all_rmse_var            = []
    all_mean_variability    = []
    all_variability         = []
    all_pcc_mean_odsl       = []
    all_rmse_mean_odsl      = []
    all_mean_odsl           = []
    all_odsl_mean_maps      = []
    all_windows             = []
    model_names_for_sliding = []
    
    for model_name in cmip_results_ds.model.values:
        print(f"Processing sliding windows for {model_name}...")
        
        full_timeseries = cmip_results_ds['full_timeseries'].sel(model=model_name)
        region_mask = cmip_results_ds['region_mask'].sel(model=model_name)
        
        pcc_per_model              = []
        rmse_per_model             = []
        mean_trends_per_model      = []
        trends_per_model           = []
        mean_variability_per_model = []
        variability_per_model      = []
        pcc_var_per_model          = []
        rmse_var_per_model         = []
        pcc_mean_odsl_per_model    = []
        rmse_mean_odsl_per_model   = []
        mean_odsl_per_model        = []
        odsl_mean_maps_per_model   = []
        windows_per_model          = []

        #slide window
        window_size = END_YEAR - START_YEAR + 1
        start_year  = max(int(cmip_results_ds.full_timeseries.time.min().item()), SLIDING_START_YEAR)
        end_year    = END_YEAR
        
        for window_start in range(start_year, end_year - window_size + 2):
            window_end  = window_start + window_size - 1
            window_data = full_timeseries.sel(time=slice(window_start, window_end))
            
            if len(window_data.time) < window_size * 0.75:
                continue
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
                warnings.filterwarnings('ignore', category=RuntimeWarning)
            
                #linear trend
                trend_coeffs = window_data.polyfit(dim='time', deg=1)
                trend_mm_yr  = trend_coeffs['polyfit_coefficients'].sel(degree=1) * 10
            
                #variability
                detrended_window = detrend_timeseries(window_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
                variability_map  = detrended_window.std(dim='time')

                #mean ODSL (cm)
                odsl_mean_map = window_data.mean(dim='time')

            #statistics
            stats_trend         = calculate_weighted_stats(trend_mm_yr, region_mask, data_y=odsl_mm_yr_regridded)
            stats_var           = calculate_weighted_stats(variability_map, region_mask, data_y=odsl_var_obs_regridded)
            stats_mean_odsl     = calculate_weighted_stats(odsl_mean_map, region_mask, data_y=odsl_obs_regridded)
            stats_abs_var       = calculate_weighted_stats(variability_map, region_mask)
            stats_abs_mean_odsl = calculate_weighted_stats(odsl_mean_map, region_mask)

            #append
            windows_per_model.append(window_start)
            pcc_per_model.append(stats_trend['pcc'])
            rmse_per_model.append(stats_trend['rmse'])
            mean_trends_per_model.append(stats_trend['mean_x'])
            trends_per_model.append(trend_mm_yr)
            mean_variability_per_model.append(stats_abs_var['mean_x'])
            variability_per_model.append(variability_map)
            pcc_var_per_model.append(stats_var['pcc'])
            rmse_var_per_model.append(stats_var['rmse'])
            pcc_mean_odsl_per_model.append(stats_mean_odsl['pcc'])
            rmse_mean_odsl_per_model.append(stats_mean_odsl['rmse'])
            mean_odsl_per_model.append(stats_abs_mean_odsl['mean_x'])
            odsl_mean_maps_per_model.append(odsl_mean_map)
        
        model_trends_da      = xr.concat(trends_per_model, dim=pd.Index(windows_per_model, name='window_start_year'))
        all_trends.append(model_trends_da)
        model_variability_da = xr.concat(variability_per_model, dim=pd.Index(windows_per_model, name='window_start_year'))
        all_variability.append(model_variability_da)
        model_odsl_mean_da   = xr.concat(odsl_mean_maps_per_model, dim=pd.Index(windows_per_model, name='window_start_year'))
        all_odsl_mean_maps.append(model_odsl_mean_da)
        
        model_names_for_sliding.append(model_name)
        all_pcc.append(pcc_per_model)
        all_rmse.append(rmse_per_model)
        all_mean_trends.append(mean_trends_per_model)
        all_mean_variability.append(mean_variability_per_model)
        all_pcc_var.append(pcc_var_per_model)
        all_rmse_var.append(rmse_var_per_model)
        all_pcc_mean_odsl.append(pcc_mean_odsl_per_model)
        all_rmse_mean_odsl.append(rmse_mean_odsl_per_model)
        all_mean_odsl.append(mean_odsl_per_model)
        
        if not all_windows:
            all_windows = windows_per_model

    final_trends_da      = xr.concat(all_trends, dim=pd.Index(model_names_for_sliding, name='model'))
    final_variability_da = xr.concat(all_variability, dim=pd.Index(model_names_for_sliding, name='model'))
    final_odsl_mean_da   = xr.concat(all_odsl_mean_maps, dim=pd.Index(model_names_for_sliding, name='model'))

    #best PCC and RMSE window multi-model mean
    print("Computing best PCC and RMSE window multi-model means and significance...")

    pcc_da       = xr.DataArray(all_pcc, coords={'model': model_names_for_sliding, 'window_start_year': all_windows})
    rmse_da      = xr.DataArray(all_rmse, coords={'model': model_names_for_sliding, 'window_start_year': all_windows})
    pcc_var_da   = xr.DataArray(all_pcc_var, coords={'model': model_names_for_sliding, 'window_start_year': all_windows})
    rmse_var_da  = xr.DataArray(all_rmse_var, coords={'model': model_names_for_sliding, 'window_start_year': all_windows})
    pcc_odsl_da  = xr.DataArray(all_pcc_mean_odsl, coords={'model': model_names_for_sliding, 'window_start_year': all_windows})
    rmse_odsl_da = xr.DataArray(all_rmse_mean_odsl, coords={'model': model_names_for_sliding, 'window_start_year': all_windows})

    window_size = END_YEAR - START_YEAR + 1

    #best PCC and RMSE window patterns per model
    def best_window_patterns(pcc_ts, rmse_ts, plot_var):
        best_pcc_years                = pcc_ts.idxmax('window_start_year')
        best_rmse_years               = rmse_ts.idxmin('window_start_year')
        best_pcc_pats, best_rmse_pats = [], []

        for model_name in cmip_results_ds.model.values:
            pcc_year  = best_pcc_years.sel(model=model_name).item()
            rmse_year = best_rmse_years.sel(model=model_name).item()

            if plot_var == 'trend':
                full_ts  = cmip_results_ds['full_timeseries'].sel(model=model_name)
                pcc_win  = full_ts.sel(time=slice(pcc_year, pcc_year + window_size - 1))
                rmse_win = full_ts.sel(time=slice(rmse_year, rmse_year + window_size - 1))
                best_pcc_pats.append(pcc_win.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
                best_rmse_pats.append(rmse_win.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
            elif plot_var == 'variability':
                best_pcc_pats.append(final_variability_da.sel(model=model_name, window_start_year=pcc_year))
                best_rmse_pats.append(final_variability_da.sel(model=model_name, window_start_year=rmse_year))
            elif plot_var == 'ODSL':
                best_pcc_pats.append(final_odsl_mean_da.sel(model=model_name, window_start_year=pcc_year))
                best_rmse_pats.append(final_odsl_mean_da.sel(model=model_name, window_start_year=rmse_year))

        stack_pcc  = xr.concat(best_pcc_pats, dim=pd.Index(cmip_results_ds.model.values, name='model'))
        stack_rmse = xr.concat(best_rmse_pats, dim=pd.Index(cmip_results_ds.model.values, name='model'))

        return stack_pcc, stack_rmse

    #significance from inter-model spread (different from utils function)
    def intermodel_significance(stack, valid_mask):

        n_models  = stack.sizes['model']
        mmm       = stack.mean(dim='model', skipna=True)
        model_std = stack.std(dim='model', skipna=True)
        se        = model_std / np.sqrt(n_models)

        t_stat       = mmm / se
        p_val_values = 2 * stats.t.sf(np.abs(t_stat.values), df=n_models - 1)
        p_val        = xr.DataArray(p_val_values, coords=mmm.coords, dims=mmm.dims).where(valid_mask)

        t_crit   = stats.t.ppf(1 - ALPHA / 2, df=n_models - 1)
        ci_lower = (mmm - t_crit * se).where(valid_mask)
        ci_upper = (mmm + t_crit * se).where(valid_mask)

        return mmm.where(valid_mask), se.where(valid_mask), p_val, ci_lower, ci_upper

    valid_mask = final_trends_da.mean(dim=['model', 'window_start_year']).notnull()

    #trend
    stack_pcc_trend, stack_rmse_trend                                                                        = best_window_patterns(pcc_da, rmse_da, 'trend')
    best_pcc_mmm_trend, best_pcc_trend_se, best_pcc_trend_p, best_pcc_trend_ci_lo, best_pcc_trend_ci_hi      = intermodel_significance(stack_pcc_trend, valid_mask)
    best_rmse_mmm_trend, best_rmse_trend_se, best_rmse_trend_p, best_rmse_trend_ci_lo, best_rmse_trend_ci_hi = intermodel_significance(stack_rmse_trend, valid_mask)

    #variability
    stack_pcc_var, stack_rmse_var                                                                  = best_window_patterns(pcc_var_da, rmse_var_da, 'variability')
    best_pcc_mmm_var, best_pcc_var_se, best_pcc_var_p, best_pcc_var_ci_lo, best_pcc_var_ci_hi      = intermodel_significance(stack_pcc_var, valid_mask)
    best_rmse_mmm_var, best_rmse_var_se, best_rmse_var_p, best_rmse_var_ci_lo, best_rmse_var_ci_hi = intermodel_significance(stack_rmse_var, valid_mask)

    #mean ODSL
    stack_pcc_odsl, stack_rmse_odsl                                                                     = best_window_patterns(pcc_odsl_da, rmse_odsl_da, 'ODSL')
    best_pcc_mmm_odsl, best_pcc_odsl_se, best_pcc_odsl_p, best_pcc_odsl_ci_lo, best_pcc_odsl_ci_hi      = intermodel_significance(stack_pcc_odsl, valid_mask)
    best_rmse_mmm_odsl, best_rmse_odsl_se, best_rmse_odsl_p, best_rmse_odsl_ci_lo, best_rmse_odsl_ci_hi = intermodel_significance(stack_rmse_odsl, valid_mask)

    #assemble
    output_ds = xr.Dataset({'pcc':                       (('model', 'window_start_year'), all_pcc),
                            'rmse':                      (('model', 'window_start_year'), all_rmse),
                            'mean_trend':                (('model', 'window_start_year'), all_mean_trends),
                            'sliding_trends':            final_trends_da,
                            'mean_variability':          (('model', 'window_start_year'), all_mean_variability),
                            'sliding_variability':       final_variability_da,
                            'odsl_mm_yr_regridded':      odsl_mm_yr_regridded,
                            'odsl_obs_regridded':        odsl_obs_regridded,
                            'odsl_yearly_regridded':     odsl_yearly_regridded,
                            'pcc_variability':           (('model', 'window_start_year'), all_pcc_var),
                            'rmse_variability':          (('model', 'window_start_year'), all_rmse_var),
                            'odsl_var_obs_regridded':    odsl_var_obs_regridded,
                            'pcc_mean_odsl':             (('model', 'window_start_year'), all_pcc_mean_odsl),
                            'rmse_mean_odsl':            (('model', 'window_start_year'), all_rmse_mean_odsl),
                            'mean_odsl':                 (('model', 'window_start_year'), all_mean_odsl),
                            'sliding_odsl_mean':         final_odsl_mean_da,
                            #best PCC and RMSE windows
                            #PCC trend
                            'best_pcc_mmm_trend':        best_pcc_mmm_trend,
                            'best_pcc_trend_se':         best_pcc_trend_se,
                            'best_pcc_trend_p_value':    best_pcc_trend_p,
                            'best_pcc_trend_ci_lower':   best_pcc_trend_ci_lo,
                            'best_pcc_trend_ci_upper':   best_pcc_trend_ci_hi,
                            #RMSE trend
                            'best_rmse_mmm_trend':       best_rmse_mmm_trend,
                            'best_rmse_trend_se':        best_rmse_trend_se,
                            'best_rmse_trend_p_value':   best_rmse_trend_p,
                            'best_rmse_trend_ci_lower':  best_rmse_trend_ci_lo,
                            'best_rmse_trend_ci_upper':  best_rmse_trend_ci_hi,
                            #PCC variability
                            'best_pcc_mmm_variability':  best_pcc_mmm_var,
                            'best_pcc_var_se':           best_pcc_var_se,
                            'best_pcc_var_p_value':      best_pcc_var_p,
                            'best_pcc_var_ci_lower':     best_pcc_var_ci_lo,
                            'best_pcc_var_ci_upper':     best_pcc_var_ci_hi,
                            #RMSE variability
                            'best_rmse_mmm_variability': best_rmse_mmm_var,
                            'best_rmse_var_se':          best_rmse_var_se,
                            'best_rmse_var_p_value':     best_rmse_var_p,
                            'best_rmse_var_ci_lower':    best_rmse_var_ci_lo,
                            'best_rmse_var_ci_upper':    best_rmse_var_ci_hi,
                            #PCC ODSL
                            'best_pcc_mmm_odsl':         best_pcc_mmm_odsl,
                            'best_pcc_odsl_se':          best_pcc_odsl_se,
                            'best_pcc_odsl_p_value':     best_pcc_odsl_p,
                            'best_pcc_odsl_ci_lower':    best_pcc_odsl_ci_lo,
                            'best_pcc_odsl_ci_upper':    best_pcc_odsl_ci_hi,
                            #RMSE ODSL
                            'best_rmse_mmm_odsl':        best_rmse_mmm_odsl,
                            'best_rmse_odsl_se':         best_rmse_odsl_se,
                            'best_rmse_odsl_p_value':    best_rmse_odsl_p,
                            'best_rmse_odsl_ci_lower':   best_rmse_odsl_ci_lo,
                            'best_rmse_odsl_ci_upper':   best_rmse_odsl_ci_hi},
                coords={'model': model_names_for_sliding, 'window_start_year': all_windows, 'latitude': final_trends_da.latitude, 'longitude': final_trends_da.longitude})
    output_ds.attrs['description'] = "Sliding window mean trend / variability, PCC, RMSE, and trend map results."

    return output_ds

@cache_result('piControl_sliding_window_results')
def perform_piControl_sliding_window(): 
    """Process piControl models and compute sliding window stats in one pass."""
    
    if not PROCESS_PICONTROL:
        print("Skipping piControl sliding window analysis (PROCESS_PICONTROL=False)")
        return None

    print("\npiControl sliding window analysis...")
    
    inventory = get_cmip_files_inventory(CMIP_VERSION)
    all_files = inventory['all_files']
    
    #observed fields
    sliding_results        = perform_sliding_window_analysis()
    odsl_mm_yr_regridded   = sliding_results['odsl_mm_yr_regridded']
    odsl_var_obs_regridded = sliding_results['odsl_var_obs_regridded']
    odsl_obs_regridded     = sliding_results['odsl_obs_regridded']
    
    window_size = END_YEAR - START_YEAR + 1
    
    all_pcc, all_rmse                     = [], []
    all_pcc_var, all_rmse_var             = [], []
    all_pcc_mean_odsl, all_rmse_mean_odsl = [], []
    all_mean_trends                       = []
    all_mean_var                          = []
    all_mean_odsl                         = []
    model_names_out                       = []
    
    #ensemble-mean piControl maps
    _pi_ens_coords       = None
    ens_map_sum          = {'trend': {}, 'var': {}, 'odsl': {}}
    ens_map_count        = {}
    ens_within_map_sum   = {'trend': {}, 'var': {}, 'odsl': {}}
    ens_within_map_count = {}
    
    cmip_results_ds = process_cmip_models()

    #within observations models
    obs_results        = calculate_observed_odsl()
    obs_trend_field    = obs_results['odsl']
    obs_trend_se_field = obs_results['odsl_std_error']

    obs_region_mask = create_region_mask(obs_trend_field, EXTENT)
    obs_stats_trend = calculate_weighted_stats(obs_trend_field, obs_region_mask)
    obs_trend_mean  = obs_stats_trend['mean_x']

    lat_obs      = 'latitude' if 'latitude' in obs_trend_se_field.dims else 'lat'
    lon_obs      = 'longitude' if 'longitude' in obs_trend_se_field.dims else 'lon'
    w_obs        = xr.DataArray(np.cos(np.deg2rad(obs_trend_se_field[lat_obs])), dims=[lat_obs])
    obs_trend_se = float(obs_trend_se_field.weighted(w_obs).mean(dim=[lat_obs, lon_obs], skipna=True).item())

    obs_lo = obs_trend_mean - obs_trend_se
    obs_hi = obs_trend_mean + obs_trend_se

    within_obs_models = []
    for model_name in cmip_results_ds.model.values:
        m_mean   = float(cmip_results_ds['trend_mean'].sel(model=model_name).item())
        se_field = cmip_results_ds['model_trend_std_error'].sel(model=model_name)
        w_m      = np.cos(np.deg2rad(se_field.latitude))
        m_se     = float(se_field.weighted(w_m).mean(dim=['latitude', 'longitude'], skipna=True).item())
        if (m_mean - m_se) <= obs_hi and (m_mean + m_se) >= obs_lo:
            within_obs_models.append(model_name)

    within_obs_set = set(within_obs_models)
    print(f"Within-obs models for piControl ensemble: {len(within_obs_models)}/{len(cmip_results_ds.model.values)}")

    selected_models = list(cmip_results_ds.model.values)
    
    for i, model_name in enumerate(selected_models):
        pi_file = all_files.get('piControl', {}).get(model_name)
        if not pi_file:
            continue
        
        print(f"Processing piControl sliding windows: {model_name} ({i+1}/{len(TARGET_MODELS)})")
        
        try:
            combined_zos = load_cmip_model_data(model_name, hist_scenario='piControl', cmip_version=CMIP_VERSION)
            
            if np.issubdtype(combined_zos.time.dtype, np.floating):
                combined_zos['time'] = combined_zos['time'].astype(int)
            
            combined_zos = remove_global_mean(combined_zos)
            region_mask  = create_region_mask(combined_zos.isel(time=0), EXTENT)
            combined_zos = combined_zos.where(region_mask)
            
            if _pi_ens_coords is None:
                _pi_ens_coords = {'latitude': combined_zos.latitude, 'longitude': combined_zos.longitude}
            
            start_year_pi = int(combined_zos.time.min().item())
            end_year_pi   = int(combined_zos.time.max().item())
            
            pcc_m, rmse_m           = [], []
            pcc_var_m, rmse_var_m   = [], []
            pcc_odsl_m, rmse_odsl_m = [], []
            mean_trends_m           = []
            mean_var_m              = []
            mean_odsl_m             = []
            
            window_counter = 0
            for ws in range(start_year_pi, end_year_pi - window_size + 2):
                window_data = combined_zos.sel(time=slice(ws, ws + window_size - 1))
                if len(window_data.time) < window_size * 0.75:
                    window_counter += 1
                    continue
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    
                    trend_mm_yr     = window_data.polyfit(dim='time', deg=1)['polyfit_coefficients'].sel(degree=1) * 10
                    detrended       = detrend_timeseries(window_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')
                    variability_map = detrended.std(dim='time')
                    odsl_mean_map   = window_data.mean(dim='time')
                
                st = calculate_weighted_stats(trend_mm_yr, region_mask, data_y=odsl_mm_yr_regridded)
                sv = calculate_weighted_stats(variability_map, region_mask, data_y=odsl_var_obs_regridded)
                so = calculate_weighted_stats(odsl_mean_map, region_mask, data_y=odsl_obs_regridded)
                
                pcc_m.append(st['pcc']); rmse_m.append(st['rmse'])
                pcc_var_m.append(sv['pcc']); rmse_var_m.append(sv['rmse'])
                pcc_odsl_m.append(so['pcc']); rmse_odsl_m.append(so['rmse'])
                mean_trends_m.append(st['mean_x'])
                mean_var_m.append(sv['mean_x'])
                mean_odsl_m.append(so['mean_x'])
                
                #spatial maps ensemble mean
                wi = window_counter
                window_counter += 1
                if wi not in ens_map_count:
                    ens_map_sum['trend'][wi] = trend_mm_yr.values.copy()
                    ens_map_sum['var'][wi]   = variability_map.values.copy()
                    ens_map_sum['odsl'][wi]  = odsl_mean_map.values.copy()
                    ens_map_count[wi]        = 1
                else:
                    ens_map_sum['trend'][wi] += trend_mm_yr.values
                    ens_map_sum['var'][wi]   += variability_map.values
                    ens_map_sum['odsl'][wi]  += odsl_mean_map.values
                    ens_map_count[wi]        += 1
                
                #spatial maps within observation ensemble
                if model_name in within_obs_set:
                    if wi not in ens_within_map_count:
                        ens_within_map_sum['trend'][wi] = trend_mm_yr.values.copy()
                        ens_within_map_sum['var'][wi]   = variability_map.values.copy()
                        ens_within_map_sum['odsl'][wi]  = odsl_mean_map.values.copy()
                        ens_within_map_count[wi]        = 1
                    else:
                        ens_within_map_sum['trend'][wi] += trend_mm_yr.values
                        ens_within_map_sum['var'][wi]   += variability_map.values
                        ens_within_map_sum['odsl'][wi]  += odsl_mean_map.values
                        ens_within_map_count[wi]        += 1
            
            model_names_out.append(model_name)
            all_pcc.append(pcc_m); all_rmse.append(rmse_m)
            all_pcc_var.append(pcc_var_m); all_rmse_var.append(rmse_var_m)
            all_pcc_mean_odsl.append(pcc_odsl_m); all_rmse_mean_odsl.append(rmse_odsl_m)
            all_mean_trends.append(mean_trends_m)
            all_mean_var.append(mean_var_m)
            all_mean_odsl.append(mean_odsl_m)
            
            #free memory
            del combined_zos, window_data # type: ignore
            
        except Exception as e:
            print(f"Could not process piControl {model_name}: {e}")
    
    print(f"\nCompleted piControl sliding windows for {len(model_names_out)} models")
    
    #ensemble mean piControl, RMSE/PCC from averaged spatial maps
    n_pi_models   = len(model_names_out)
    ensemble_mask = create_region_mask(odsl_mm_yr_regridded, EXTENT)
    
    pi_ens_pcc, pi_ens_rmse           = [], []
    pi_ens_pcc_var, pi_ens_rmse_var   = [], []
    pi_ens_pcc_odsl, pi_ens_rmse_odsl = [], []
    
    for wi in sorted(ens_map_count.keys()):
        if ens_map_count[wi] < n_pi_models:
            continue  
        n = ens_map_count[wi]
        
        mean_trend = xr.DataArray(ens_map_sum['trend'][wi] / n, coords=_pi_ens_coords, dims=['latitude', 'longitude'])
        mean_var   = xr.DataArray(ens_map_sum['var'][wi] / n, coords=_pi_ens_coords, dims=['latitude', 'longitude'])
        mean_odsl  = xr.DataArray(ens_map_sum['odsl'][wi] / n, coords=_pi_ens_coords, dims=['latitude', 'longitude'])
        
        st = calculate_weighted_stats(mean_trend, ensemble_mask, data_y=odsl_mm_yr_regridded)
        sv = calculate_weighted_stats(mean_var, ensemble_mask, data_y=odsl_var_obs_regridded)
        so = calculate_weighted_stats(mean_odsl, ensemble_mask, data_y=odsl_obs_regridded)
        
        pi_ens_pcc.append(st['pcc']); pi_ens_rmse.append(st['rmse'])
        pi_ens_pcc_var.append(sv['pcc']); pi_ens_rmse_var.append(sv['rmse'])
        pi_ens_pcc_odsl.append(so['pcc']); pi_ens_rmse_odsl.append(so['rmse'])
    
    #free memory
    del ens_map_sum 

    print(f"piControl ensemble mean: {len(pi_ens_pcc)} windows with all {n_pi_models} models contributing")

    #within observations ensemble piControl
    n_within_pi_models = len([m for m in within_obs_models if m in model_names_out])
    
    pi_within_pcc, pi_within_rmse           = [], []
    pi_within_pcc_var, pi_within_rmse_var   = [], []
    pi_within_pcc_odsl, pi_within_rmse_odsl = [], []

    for wi in sorted(ens_within_map_count.keys()):
        if ens_within_map_count[wi] < n_within_pi_models:
            continue
        n = ens_within_map_count[wi]

        mean_trend = xr.DataArray(ens_within_map_sum['trend'][wi] / n, coords=_pi_ens_coords, dims=['latitude', 'longitude'])
        mean_var   = xr.DataArray(ens_within_map_sum['var'][wi] / n, coords=_pi_ens_coords, dims=['latitude', 'longitude'])
        mean_odsl  = xr.DataArray(ens_within_map_sum['odsl'][wi] / n, coords=_pi_ens_coords, dims=['latitude', 'longitude'])

        st = calculate_weighted_stats(mean_trend, ensemble_mask, data_y=odsl_mm_yr_regridded)
        sv = calculate_weighted_stats(mean_var, ensemble_mask, data_y=odsl_var_obs_regridded)
        so = calculate_weighted_stats(mean_odsl, ensemble_mask, data_y=odsl_obs_regridded)

        pi_within_pcc.append(st['pcc']); pi_within_rmse.append(st['rmse'])
        pi_within_pcc_var.append(sv['pcc']); pi_within_rmse_var.append(sv['rmse'])
        pi_within_pcc_odsl.append(so['pcc']); pi_within_rmse_odsl.append(so['rmse'])

    del ens_within_map_sum

    print(f"piControl within-obs ensemble: {len(pi_within_pcc)} windows with all {n_within_pi_models} models contributing")
    
    #pad
    max_len = max(len(p) for p in all_pcc)
    def pad(lst): return lst + [np.nan] * (max_len - len(lst))
    
    output_ds = xr.Dataset({'pcc':              (('model', 'window_idx'), [pad(p) for p in all_pcc]),
                            'rmse':             (('model', 'window_idx'), [pad(p) for p in all_rmse]),
                            'pcc_variability':  (('model', 'window_idx'), [pad(p) for p in all_pcc_var]),
                            'rmse_variability': (('model', 'window_idx'), [pad(p) for p in all_rmse_var]),
                            'pcc_mean_odsl':    (('model', 'window_idx'), [pad(p) for p in all_pcc_mean_odsl]),
                            'rmse_mean_odsl':   (('model', 'window_idx'), [pad(p) for p in all_rmse_mean_odsl]),
                            'mean_trend':       (('model', 'window_idx'), [pad(p) for p in all_mean_trends]),
                            'mean_variability': (('model', 'window_idx'), [pad(p) for p in all_mean_var]),
                            'mean_odsl':        (('model', 'window_idx'), [pad(p) for p in all_mean_odsl]),}, 
                coords =   {'model': model_names_out, 'window_idx': np.arange(max_len)})
    
    #ensemble-mean
    output_ds['pi_ensemble_pcc']                = ('pi_ens_idx', pi_ens_pcc)
    output_ds['pi_ensemble_rmse']               = ('pi_ens_idx', pi_ens_rmse)
    output_ds['pi_ensemble_pcc_variability']    = ('pi_ens_idx', pi_ens_pcc_var)
    output_ds['pi_ensemble_rmse_variability']   = ('pi_ens_idx', pi_ens_rmse_var)
    output_ds['pi_ensemble_pcc_mean_odsl']      = ('pi_ens_idx', pi_ens_pcc_odsl)
    output_ds['pi_ensemble_rmse_mean_odsl']     = ('pi_ens_idx', pi_ens_rmse_odsl)

    output_ds['pi_within_obs_pcc']              = ('pi_within_idx', pi_within_pcc)
    output_ds['pi_within_obs_rmse']             = ('pi_within_idx', pi_within_rmse)
    output_ds['pi_within_obs_pcc_variability']  = ('pi_within_idx', pi_within_pcc_var)
    output_ds['pi_within_obs_rmse_variability'] = ('pi_within_idx', pi_within_rmse_var)
    output_ds['pi_within_obs_pcc_mean_odsl']    = ('pi_within_idx', pi_within_pcc_odsl)
    output_ds['pi_within_obs_rmse_mean_odsl']   = ('pi_within_idx', pi_within_rmse_odsl)
    
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
        historical_end_year  = 2005 if cmip_version == "CMIP5" else 2014
        
        scenario_datasets = {}
        
        for scenario in scenarios_to_process:
            print(f"Processing scenario: {scenario}")
            
            model_odsl_list        = []
            model_trend_list       = []
            model_variability_list = []
            valid_model_names      = []
            
            #loop over each model
            for model_name in target_models:
                try:
                    end_year    = historical_end_year + 1 if scenario == "historical" else 2100
                    future_scen = None if scenario == "historical" else scenario
                    
                    combined_zos = load_cmip_model_data(model_name, hist_scenario='historical', future_scenario=future_scen, cmip_version=cmip_version, end_year=end_year)
                    if combined_zos is None: 
                        continue
                    
                    #time to integers
                    if np.issubdtype(combined_zos.time.dtype, np.floating):
                        combined_zos['time'] = combined_zos['time'].astype(int)

                    #select only future period
                    if scenario != "historical":
                        combined_zos = combined_zos.sel(time=slice(historical_end_year + 1, None))
                    
                    #remove global mean prior to masking North Atlantic
                    combined_zos = remove_global_mean(combined_zos)
                    
                    #North Atlantic region and weighted regional mean
                    region_mask  = create_region_mask(combined_zos.isel(time=0), EXTENT)
                    weights      = np.cos(np.deg2rad(combined_zos.latitude))
                    weights.name = "weights"
                    regional_ts  = combined_zos.where(region_mask).weighted(weights).mean(dim=['latitude', 'longitude'])
                    
                    #convert to mm and rename time to year
                    regional_ts_mm = regional_ts * 10
                    regional_ts_mm = regional_ts_mm.rename({'time': 'year'})
                    
                    #ODSL timeseries
                    odsl_ts = regional_ts_mm
                    
                    #trend
                    trend_coeffs = regional_ts_mm.polyfit(dim='year', deg=1)
                    trend_ts     = xr.polyval(regional_ts_mm.year, trend_coeffs.polyfit_coefficients)
                    
                    #variability
                    variability_ts = odsl_ts - trend_ts
                    
                    model_odsl_list.append(odsl_ts)
                    model_trend_list.append(trend_ts)
                    model_variability_list.append(variability_ts)
                    valid_model_names.append(model_name)
                    
                except Exception as e:
                    print(f"Could not process model {model_name} for scenario {scenario}: {e}")
            
            #concatenate 
            if model_odsl_list:
                n_models  = len(model_odsl_list)
                model_dim = pd.Index(valid_model_names, name='model')
                
                odsl_ensemble        = xr.concat(model_odsl_list, dim=model_dim)
                trend_ensemble       = xr.concat(model_trend_list, dim=model_dim)
                variability_ensemble = xr.concat(model_variability_list, dim=model_dim)
                
                scenario_datasets[scenario] = xr.Dataset({'odsl_ensemble_mean':        odsl_ensemble.mean(dim='model', skipna=True), #ODSL (mm)
                                                          'odsl_ensemble_std':         odsl_ensemble.std(dim='model', skipna=True),
                                                          'trend_ensemble_mean':       trend_ensemble.mean(dim='model', skipna=True), #trend (mm)
                                                          'trend_ensemble_std':        trend_ensemble.std(dim='model', skipna=True),
                                                          'variability_ensemble_mean': variability_ensemble.mean(dim='model', skipna=True), #variability (mm)
                                                          'variability_ensemble_std':  variability_ensemble.std(dim='model', skipna=True),
                                                          'odsl_per_model':            odsl_ensemble,
                                                          'n_models':                  n_models}) #model count
                
                print(f"Processed {n_models} models for {scenario}")
        
        if scenario_datasets:
            all_years = sorted({year for ds in scenario_datasets.values() for year in ds.year.values})
            ds_list   = []
            for scenario_name, ds in scenario_datasets.items():
                ds = ds.reindex(year=all_years)
                ds_list.append(ds)
            
            combined_ds = xr.concat(ds_list, dim=pd.Index(list(scenario_datasets.keys()), name='scenario'))

            combined_ds.attrs['description']         = f"Ensemble timeseries for {cmip_version} scenarios"
            combined_ds.attrs['historical_end_year'] = historical_end_year
            combined_ds.attrs['valid_scenarios']     = list(scenario_datasets.keys())
            combined_ds.attrs['odsl_units']          = 'mm'
            combined_ds.attrs['trend_units']         = 'mm'
            combined_ds.attrs['variability_units']   = 'mm'

            final_results.append(combined_ds)
    
    if not final_results:
        return xr.Dataset()
    
    cmip_coord = pd.Index(["CMIP5", "CMIP6"], name='cmip_version')
    combined_results = xr.concat(final_results, dim=cmip_coord, join='outer')
    
    print(f"Successfully processed timeseries for {len(final_results)} CMIP versions")

    return combined_results

@cache_result('eof_analysis_results')
def perform_eof_analysis(cmip_results, sliding_results, n_modes=N_MODES_OBSERVED):
    """Performs EOF analysis on observed data, the multi-model mean, and each individual CMIP model."""
    
    print("\nPerforming EOF analysis on all data sources...")

    #spatial extent
    lon_min, lon_max, lat_min, lat_max = EXTENT
    lat_slice                          = slice(lat_min, lat_max)
    lon_slice                          = slice(lon_min, lon_max)

    sources_to_analyze = {}
    
    #observed data regridded to model grid (in sliding window analysis function)
    obs_da                         = sliding_results['odsl_yearly_regridded'].rename({'year': 'time'})
    obs_detrended                  = detrend_timeseries(obs_da, degree=VARIABILITY_DETREND_DEGREE, dim='time')
    obs_mask                       = create_region_mask(obs_detrended.isel(time=0), EXTENT)
    sources_to_analyze['observed'] = obs_detrended.where(obs_mask)

    #CMIP data
    model_full_ts         = cmip_results['full_timeseries'].sel(time=slice(START_YEAR, END_YEAR))
    model_full_ts['time'] = model_full_ts['time'].astype(int)
    model_detrended       = detrend_timeseries(model_full_ts, degree=VARIABILITY_DETREND_DEGREE, dim='time')
    model_mask            = create_region_mask(model_detrended.isel(time=0, model=0), EXTENT)
    
    #individual models
    for model_name in model_detrended.model.values:
        sources_to_analyze[model_name] = model_detrended.sel(model=model_name).where(model_mask)

    all_eof_results = {}

    #loop
    for name, data_array in sources_to_analyze.items():

        print(f"Analyzing source: {name}...")
        
        try:
            data_regional = data_array.sel(latitude=lat_slice, longitude=lon_slice)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                eof_result_ds = calculate_single_eof(data_regional, n_modes)

            if eof_result_ds is not None:

                n_samples = data_regional.sizes['time']
                
                #lambda
                if 'variance_fractions' in eof_result_ds:

                    lambdas = eof_result_ds['variance_fractions']
                    
                    #North's rule of thumb error (North et al. 1982, eq. 24)
                    north_error                               = lambdas * np.sqrt(2 / n_samples)
                    eof_result_ds['variance_fractions_error'] = north_error
                    
                    #summary which modes are distinct
                    significance_msg = []

                    l_vals = lambdas.values
                    e_vals = north_error.values
                    
                    for m in range(len(l_vals) - 1):

                        #mode m overlaps with mode m+1
                        lower_bound_current = l_vals[m] - e_vals[m]
                        upper_bound_next    = l_vals[m+1] + e_vals[m+1]
                        
                        if lower_bound_current > upper_bound_next:
                            status = "Distinct"
                        else:
                            status = "Degenerate (overlapping)"

                        significance_msg.append(f"Mode {m+1}-{m+2}: {status}")
                    
                    eof_result_ds.attrs['north_significance_test'] = "; ".join(significance_msg)

                    #Monte Carlo significance test
                    if MONTE_CARLO_SIGNIFICANCE_TEST:
                        n_modes_computed = len(eof_result_ds['variance_fractions'])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            mc_result = monte_carlo_significance_test(data_regional, n_realizations=N_REALIZATIONS_MONTE_CARLO, n_modes=n_modes_computed)  
                    
                        #thresholds
                        eof_result_ds['mc_threshold_95']   = mc_result['mc_threshold_95']
                        eof_result_ds['mc_threshold_99']   = mc_result['mc_threshold_99']
                        eof_result_ds['synthetic_lambdas'] = mc_result['synthetic_lambdas']
                        
                        #significance
                        observed_lambdas                   = eof_result_ds['variance_fractions']
                        eof_result_ds['is_significant_95'] = observed_lambdas > mc_result['mc_threshold_95']
                        eof_result_ds['is_significant_99'] = observed_lambdas > mc_result['mc_threshold_99']
                        
                        #summary message
                        n_sig_95                               = int(eof_result_ds['is_significant_95'].sum())
                        n_sig_99                               = int(eof_result_ds['is_significant_99'].sum())
                        eof_result_ds.attrs['mc_significance'] = f"{n_sig_95} modes significant at 95%, {n_sig_99} at 99%"
                        eof_result_ds.attrs['alpha']           = mc_result.attrs['alpha']
                        
                        print(f"Monte Carlo: {n_sig_95} significant modes (95% level)")

                    else:
                        print("Skipping Monte Carlo significance test.")

                all_eof_results[name] = eof_result_ds

        except Exception as e:
            print(f"Could not perform EOF analysis for {name}: {e}")

    #multi-model mean (EOF)
    model_keys = [k for k in all_eof_results.keys() if k != 'observed']
    
    if len(model_keys) > 1:
        print("Calculating Multi-Model Mean (simple average of model EOFs)...")
        
        eofs_list = []
        pcs_list  = []
        vars_list = []
        
        for name in model_keys:
            res = all_eof_results[name]
            
            mod_eofs = res['eofs'].copy().drop_vars('model', errors='ignore')
            mod_pcs  = res['pcs'].copy().drop_vars('model', errors='ignore')
            mod_vars = res['variance_fractions'].copy().drop_vars('model', errors='ignore')
            
            mod_eofs = mod_eofs.expand_dims(dim={'model': [name]})
            mod_pcs  = mod_pcs.expand_dims(dim={'model': [name]})
            mod_vars = mod_vars.expand_dims(dim={'model': [name]})
            
            eofs_list.append(mod_eofs)
            pcs_list.append(mod_pcs)
            vars_list.append(mod_vars)
        
        #simple mean across models
        mmm_eofs       = xr.concat(eofs_list, dim='model').mean(dim='model')
        mmm_pcs        = xr.concat(pcs_list, dim='model').mean(dim='model')
        mmm_vars       = xr.concat(vars_list, dim='model').mean(dim='model')
        mmm_vars_error = xr.concat(vars_list, dim='model').std(dim='model')
        
        mmm_ds = xr.Dataset({'eofs': mmm_eofs, 'pcs': mmm_pcs, 'variance_fractions': mmm_vars, 'variance_fractions_error': mmm_vars_error})
        
        mmm_ds.attrs['description'] = "Average of individual model EOFs"
        
        #monte carlo significance test multimodel mean
        if MONTE_CARLO_SIGNIFICANCE_TEST:
            print("Running Monte Carlo test for Multi-Model Mean...")
            try:
                #detrended multi-model mean
                mmm_detrended = model_detrended.mean(dim='model').where(model_mask)
                mmm_regional  = mmm_detrended.sel(latitude=lat_slice, longitude=lon_slice)
                
                n_modes_mmm   = len(mmm_ds['variance_fractions'])
                mc_result_mmm = monte_carlo_significance_test(mmm_regional, n_realizations=N_REALIZATIONS_MONTE_CARLO, n_modes=n_modes_mmm)
                
                #add to results
                mmm_ds['mc_threshold_95']   = mc_result_mmm['mc_threshold_95']
                mmm_ds['mc_threshold_99']   = mc_result_mmm['mc_threshold_99']
                mmm_ds['synthetic_lambdas'] = mc_result_mmm['synthetic_lambdas']
                
                observed_lambdas_mmm        = mmm_ds['variance_fractions']
                mmm_ds['is_significant_95'] = observed_lambdas_mmm > mc_result_mmm['mc_threshold_95']
                mmm_ds['is_significant_99'] = observed_lambdas_mmm > mc_result_mmm['mc_threshold_99']
                
                n_sig_95                        = int(mmm_ds['is_significant_95'].sum())
                n_sig_99                        = int(mmm_ds['is_significant_99'].sum())
                mmm_ds.attrs['mc_significance'] = f"{n_sig_95} modes significant at 95%, {n_sig_99} at 99%"
                mmm_ds.attrs['alpha']           = mc_result_mmm.attrs['alpha']
                
                print(f"MMM Monte Carlo: {n_sig_95} significant modes (95% level)")
                
            except Exception as e:
                print(f"Could not perform Monte Carlo test on MMM: {e}")
        
        else:
            print("Skipping Monte Carlo significance test for Multi-Model Mean.")
        
        all_eof_results['multi model mean'] = mmm_ds

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
        
        pcs                           = eof_result_ds['pcs']
        correlations                  = calculate_pc_index_correlations(pcs, climate_indices)
        correlations_ds               = xr.Dataset(correlations)
        all_correlation_results[name] = correlations_ds
        
        #print results
        for index_name, corr_da in correlations.items():
            print(f"Correlations with {index_name.upper()}: {np.round(corr_da.values, 2)}")
            
    return all_correlation_results

@cache_result('dual_eof_comparison_results')
def perform_dual_eof_comparison(cmip_results, sliding_results, n_modes=N_MODES_OBSERVED):
    """Comparison unrotated and rotated EOF (1993-2024)."""

    global USE_ROTATED_EOF
    original_setting = USE_ROTATED_EOF

    print("\nPerforming dual EOF comparison (unrotated + rotated)...")

    lon_min, lon_max, lat_min, lat_max = EXTENT
    lat_slice                          = slice(lat_min, lat_max)
    lon_slice                          = slice(lon_min, lon_max)

    sources_to_analyze = {}

    #observed regridded to model grid (from sliding window analysis function)
    obs_da                         = sliding_results['odsl_yearly_regridded'].rename({'year': 'time'})
    obs_detrended                  = detrend_timeseries(obs_da, degree=VARIABILITY_DETREND_DEGREE, dim='time')
    obs_mask                       = create_region_mask(obs_detrended.isel(time=0), EXTENT)
    sources_to_analyze['observed'] = obs_detrended.where(obs_mask)

    #CMIP individual models
    model_full_ts         = cmip_results['full_timeseries'].sel(time=slice(START_YEAR, END_YEAR))
    model_full_ts['time'] = model_full_ts['time'].astype(int)
    model_detrended       = detrend_timeseries(model_full_ts, degree=VARIABILITY_DETREND_DEGREE, dim='time')
    model_mask            = create_region_mask(model_detrended.isel(time=0, model=0), EXTENT)

    for model_name in model_detrended.model.values:
        sources_to_analyze[model_name] = model_detrended.sel(model=model_name).where(model_mask)

    #run both regular and rotated EOF
    all_results = {}

    for name, data_array in sources_to_analyze.items():
        print(f"Analyzing source: {name} ...")
        try:
            data_regional = data_array.sel(latitude=lat_slice, longitude=lon_slice)

            USE_ROTATED_EOF = False
            unrotated_ds    = calculate_single_eof(data_regional, n_modes)

            USE_ROTATED_EOF = True
            rotated_ds      = calculate_single_eof(data_regional, n_modes)

            if unrotated_ds is not None and rotated_ds is not None:
                all_results[f'{name}__unrotated'] = unrotated_ds
                all_results[f'{name}__rotated']   = rotated_ds

        except Exception as e:
            print(f"Could not perform dual EOF for {name}: {e}")

    USE_ROTATED_EOF = original_setting

    #multi-model mean
    model_keys = [k for k in sources_to_analyze.keys() if k != 'observed']

    for eof_type in ['unrotated', 'rotated']:
        eofs_list = []
        pcs_list  = []
        vars_list = []

        for name in model_keys:
            key = f'{name}__{eof_type}'
            if key not in all_results:
                continue
            res = all_results[key]

            mod_eofs = res['eofs'].copy().drop_vars('model', errors='ignore').expand_dims(dim={'model': [name]})
            mod_pcs  = res['pcs'].copy().drop_vars('model', errors='ignore').expand_dims(dim={'model': [name]})
            mod_vars = res['variance_fractions'].copy().drop_vars('model', errors='ignore').expand_dims(dim={'model': [name]})

            eofs_list.append(mod_eofs)
            pcs_list.append(mod_pcs)
            vars_list.append(mod_vars)

        if eofs_list:
            mmm_ds = xr.Dataset({'eofs':               xr.concat(eofs_list, dim='model').mean(dim='model'),
                                 'pcs':                xr.concat(pcs_list, dim='model').mean(dim='model'),
                                 'variance_fractions': xr.concat(vars_list, dim='model').mean(dim='model')})
            all_results[f'multi model mean__{eof_type}'] = mmm_ds

    print("Dual EOF comparison complete.\n")
    return all_results

@cache_result('dual_eof_comparison_results_historical')
def perform_dual_eof_comparison_historical(cmip_results, n_modes=N_MODES_OBSERVED):
    """Comparison unrotated and rotated EOF (1850-2024)."""

    global USE_ROTATED_EOF
    original_setting = USE_ROTATED_EOF

    print("\nPerforming dual EOF comparison over historical period (1850-END_YEAR)...")

    lon_min, lon_max, lat_min, lat_max = EXTENT
    lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)

    model_full_ts         = cmip_results['full_timeseries'].sel(time=slice(1850, END_YEAR))
    model_full_ts['time'] = model_full_ts['time'].astype(int)
    model_detrended       = detrend_timeseries(model_full_ts, degree=VARIABILITY_DETREND_DEGREE, dim='time')
    model_mask            = create_region_mask(model_detrended.isel(time=0, model=0), EXTENT)

    sources_to_analyze = {
        m: model_detrended.sel(model=m).where(model_mask)
        for m in model_detrended.model.values
    }

    all_results = {}
    for name, data_array in sources_to_analyze.items():
        print(f"Analyzing source (historical): {name} ...")
        try:
            data_regional = data_array.sel(latitude=lat_slice, longitude=lon_slice)

            USE_ROTATED_EOF = False
            unrotated_ds    = calculate_single_eof(data_regional, n_modes)
            USE_ROTATED_EOF = True
            rotated_ds      = calculate_single_eof(data_regional, n_modes)

            if unrotated_ds is not None and rotated_ds is not None:
                all_results[f'{name}__unrotated'] = unrotated_ds
                all_results[f'{name}__rotated']   = rotated_ds
        except Exception as e:
            print(f"Could not perform historical dual EOF for {name}: {e}")

    USE_ROTATED_EOF = original_setting

    #multi-model mean
    model_keys = list(sources_to_analyze.keys())
    for eof_type in ['unrotated', 'rotated']:
        eofs_list, pcs_list, vars_list = [], [], []
        for name in model_keys:
            key = f'{name}__{eof_type}'
            if key not in all_results:
                continue
            res = all_results[key]
            eofs_list.append(res['eofs'].copy().drop_vars('model', errors='ignore').expand_dims(dim={'model': [name]}))
            pcs_list.append( res['pcs'].copy().drop_vars('model', errors='ignore').expand_dims(dim={'model': [name]}))
            vars_list.append(res['variance_fractions'].copy().drop_vars('model', errors='ignore').expand_dims(dim={'model': [name]}))

        if eofs_list:
            mmm_ds = xr.Dataset({
                'eofs':               xr.concat(eofs_list, dim='model').mean(dim='model'),
                'pcs':                xr.concat(pcs_list,  dim='model').mean(dim='model'),
                'variance_fractions': xr.concat(vars_list, dim='model').mean(dim='model'),
            })
            all_results[f'multi model mean__{eof_type}'] = mmm_ds

    print("Historical dual EOF comparison complete.\n")
    return all_results

@cache_result('spectral_analysis_results')
def perform_spectral_analysis(cmip_results, obs_results, n_modes=None):
    """Power spectra of EOF PCs for observed, multi-model mean, and individual models over 1850-2024 and 1993-2024."""

    if n_modes is None:
        n_modes = EOF_N_MODES

    print(f"\nPerforming spectral analysis for top {n_modes} EOF modes...")

    #observed data (1993–2024)
    obs_yearly     = obs_results['odsl_yearly']

    #regrid observed yearly fields to model grid
    sample_grid   = cmip_results['model_mean_trend']
    obs_filled    = obs_yearly.interpolate_na(dim='longitude', method='nearest', limit=5, fill_value="extrapolate")
    obs_filled    = obs_filled.interpolate_na(dim='latitude', method='nearest', limit=5, fill_value="extrapolate")
    regridder     = xe.Regridder(obs_filled, sample_grid, 'bilinear', periodic=True)
    obs_regridded = regridder(obs_filled)

    assert isinstance(obs_regridded, xr.DataArray)
    try:
        regridder.clean_weight_file() # type: ignore
    except AttributeError:
        pass

    #mask and detrend
    base_mask     = create_region_mask(sample_grid, EXTENT)
    obs_detrended = detrend_timeseries(obs_regridded.where(base_mask), degree=VARIABILITY_DETREND_DEGREE, dim='year')

    full_data      = cmip_results['full_timeseries'].sel(time=slice(1850, END_YEAR)).where(base_mask)
    full_detrended = detrend_timeseries(full_data, degree=VARIABILITY_DETREND_DEGREE, dim='time')

    mmm_detrended = full_detrended.mean(dim='model', skipna=True)

    #spectra single EOF
    def spectra_from_data(da, n_modes, label=''):
        """Run EOF on da, return power spectra for each mode."""

        try:
            eof_res = calculate_single_eof(da, n_modes=n_modes)
        except Exception as e:
            print(f"ERROR during EOF for {label}: {e}")
            return None

        if eof_res is None:
            print(f"WARNING: EOF returned None for {label}")
            return None

        pcs          = eof_res['pcs']
        mode_spectra = []
        for mode in pcs.mode.values[:n_modes]:
            psd = calculate_power_spectrum(pcs.sel(mode=mode))
            mode_spectra.append(psd)

        return xr.concat(mode_spectra, dim=pd.Index(range(1, len(mode_spectra) + 1), name='mode'))

    #observed
    print("Computing observed spectra...")
    obs_spectra = spectra_from_data(obs_detrended.rename({'year': 'time'}), n_modes, label='Observed')
    print(f"obs_spectra result: {obs_spectra is not None}")
    if obs_spectra is None:
        print("WARNING: Observed spectra computation returned None!")

    #multi-model mean
    print("Computing multi-model mean spectra...")
    mmm_spectra = spectra_from_data(mmm_detrended, n_modes, label='Multi-model mean')

    #individual models
    all_model_spectra = []
    model_names       = []
    for model_name in full_detrended.model.values:
        print(f"Computing spectra for {model_name}...")
        da_model = full_detrended.sel(model=model_name)
        spec     = spectra_from_data(da_model, n_modes, label=model_name)
        if spec is not None:
            all_model_spectra.append(spec)
            model_names.append(model_name)

    if len(all_model_spectra) == 0:
        raise ValueError("No models produced valid spectra.")

    model_spectra = xr.concat(all_model_spectra, dim=pd.Index(model_names, name='model'))
    output        = xr.Dataset({'model_spectra': model_spectra, 'mmm_spectrum': mmm_spectra})

    if obs_spectra is not None:
        output['obs_spectrum'] = obs_spectra.rename({'frequency': 'frequency_obs'})

    #altimetry era
    print("Computing spectra over altimetry era...")
    obs_period_detrended     = detrend_timeseries(full_data.sel(time=slice(START_YEAR, END_YEAR)), degree=VARIABILITY_DETREND_DEGREE, dim='time')
    obs_period_mmm_detrended = obs_period_detrended.mean(dim='model', skipna=True)
    obs_period_mmm_spectra   = spectra_from_data(obs_period_mmm_detrended, n_modes, label='MMM (obs period)')

    obs_period_model_spectra = []
    obs_period_model_names   = []
    for model_name in obs_period_detrended.model.values:
        spec = spectra_from_data(obs_period_detrended.sel(model=model_name), n_modes, label=f'{model_name} (obs period)')
        if spec is not None:
            obs_period_model_spectra.append(spec)
            obs_period_model_names.append(model_name)

    if obs_period_model_spectra:
        combined = xr.concat(obs_period_model_spectra, dim=pd.Index(obs_period_model_names, name='model'))
        output['obs_period_model_spectra'] = combined.rename({'frequency': 'frequency_obs'})

    if obs_period_mmm_spectra is not None:
        output['obs_period_mmm_spectrum']  = obs_period_mmm_spectra.rename({'frequency': 'frequency_obs'})

    output.attrs['description'] = 'Power spectra of EOF PCs for observed, MMM, and individual models.'
    output.attrs['n_modes']     = n_modes

    return output


def main():
    """Run complete analysis."""

    print("ODSL analysis starting...")
    fig_dir = './figures/'

    r = {}

    #description, function, result keys
    pipeline = [("Calculating the observed ODSL",                     lambda: calculate_observed_odsl(),                                                                      'obs_results'),
                ("Smoothing comparison",                              lambda: calculate_smoothing_sensitivity(),                                                              'smoothing_results'),
                ("Finding valid CMIP models",                         lambda: valid_models_table(),                                                                           'models_df'),
                ("Processing CMIP models",                            lambda: process_cmip_models(),                                                                          'cmip_results'),
                ("Processing all CMIP scenarios",                     lambda: process_all_cmip_scenarios(default_result=r['cmip_results']),                                   'scenario_data_all'),
                ("Performing incrementing window analysis",           lambda: perform_incrementing_window_analysis(r['obs_results'], r['cmip_results']),                      'incrementing_window_results'),
                ("Performing piControl incrementing window analysis", lambda: perform_piControl_incrementing_window(),                                                        'picontrol_incrementing_results'),
                ("Performing sliding window analysis CMIP",           lambda: perform_sliding_window_analysis(),                                                              'sliding_results'),
                ("Performing sliding window analysis piControl",      lambda: perform_piControl_sliding_window(),                                                             'picontrol_results'),
                ("Processing scenario data",                          lambda: process_cmip_scenario_data(),                                                                   'scenario_results'),
                ("EOF analysis",                                      lambda: perform_eof_analysis(r['cmip_results'], r['sliding_results'], n_modes=N_MODES_OBSERVED),        'eof_results'),
                ("Correlating with climate indices",                  lambda: correlate_with_indices(r['eof_results']),                                                       'correlation_results'),
                ("Standard and rotated EOF comparison",               lambda: perform_dual_eof_comparison(r['cmip_results'], r['sliding_results'], n_modes=N_MODES_OBSERVED), 'dual_eof_results'),
                ("Standard and rotated EOF comparison (historical)",  lambda: perform_dual_eof_comparison_historical(r['cmip_results'], n_modes=N_MODES_OBSERVED),            'dual_eof_results_historical'),
                ("Frequency analysis",                                lambda: perform_spectral_analysis(r['cmip_results'], r['obs_results'], n_modes=EOF_N_MODES),            'spectral_results'),
                ("Wind analysis",                                     lambda: load_wind_stress_data(),                                                                        'wind_results')]

    progress_columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()]

    total_steps = len(pipeline) + 1 

    with Progress(*progress_columns) as progress:
        task_id = progress.add_task("[cyan]Overall progress", total=total_steps)

        for i, (description, func, keys) in enumerate(pipeline, 1):
            progress.update(task_id, description=f"[bold blue]Step {i}/{total_steps}:[/bold blue] {description}", advance=1)

            start_time = time.time()
            result = func()
            elapsed = time.time() - start_time

            #store results
            if isinstance(keys, tuple):
                for key, val in zip(keys, result):
                    r[key] = val
            else:
                r[keys] = result

            progress.console.print(f"[green]✔ {description} ({elapsed:.2f}s)[/green]")

        #unpack CMIP5 vs CMIP6 scenario results for comparison figure
        sd = r.get('scenario_data_all')
        if sd is not None:
            r['cmip5_all']     = {k.replace('cmip5_', ''): v for k, v in sd.items() if k.startswith('cmip5_')}
            r['cmip6_all']     = {k.replace('cmip6_', ''): v for k, v in sd.items() if k.startswith('cmip6_')}
            r['cmip5_results'] = r['cmip5_all'][CMIP5_FUTURE_SCENARIO]
            r['cmip6_results'] = r['cmip6_all'][CMIP6_FUTURE_SCENARIO]
        else:
            r['cmip5_all']     = None
            r['cmip6_all']     = None
            r['cmip5_results'] = None
            r['cmip6_results'] = None

        #figures
        step_n = total_steps
        progress.update(task_id, description=f"[bold blue]Step {step_n}/{total_steps}:[/bold blue] Generating all figures", advance=1)
        progress.console.print("\nAll calculations complete. Generating figures...")
        
        create_all_figures(obs_results=r['obs_results'], smoothing_results=r['smoothing_results'], cmip_results=r['cmip_results'], cmip5_results=r['cmip5_results'], cmip6_results=r['cmip6_results'], cmip5_all=r['cmip5_all'], cmip6_all=r['cmip6_all'], incrementing_window_results=r['incrementing_window_results'], sliding_results=r['sliding_results'], picontrol_incrementing_results=r['picontrol_incrementing_results'], picontrol_results=r['picontrol_results'], scenario_results=r['scenario_results'], eof_results=r['eof_results'], dual_eof_results=r['dual_eof_results'], dual_eof_results_historical=r['dual_eof_results_historical'], correlation_results=r['correlation_results'], fig_dir=fig_dir, spectral_results=r['spectral_results'], wind_results=r['wind_results'])
    
        progress.console.print("[bold green]✔ All figures generated![/bold green]")
        progress.update(task_id, description="[bold green]Analysis complete!")
        progress.console.print(f"[bold green]✔ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold green]")

if __name__ == "__main__":
    main()
