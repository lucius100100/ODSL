import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd
import traceback
from scipy import stats

from data_loader import (load_altimetry_data, load_budget_data, load_gia_data, 
                        load_cmip_model_data, load_multiple_cmip_models, get_cmip_files_inventory, find_folder_by_name)
from utils import cache_result, calculate_weighted_stats, create_region_mask, rotate_longitude
from plotting import create_all_figures 
from config import (START_YEAR, END_YEAR, EXTENT, PROJECTION_PARAMS, 
                   TARGET_CMIP5_MODELS)

#figures directory
fig_dir = './figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

@cache_result('observed_odsl')
def calculate_observed_odsl():
    """Calculate observed ODSL from components."""
    print("\n=== Calculating Observed ODSL ===")
    
    #load data
    duacs_yearly = load_altimetry_data()
    asl_frederikse = load_budget_data()
    gia_data = load_gia_data()
    gia_rad_da = gia_data['gia_rad']
    gia_sea_da = gia_data['gia_sea']
    
    #common years
    year_slice = slice(START_YEAR, END_YEAR)
    alt_years = duacs_yearly.sel(year=year_slice).year.values
    fr_years = asl_frederikse.sel(year=year_slice).year.values
    common_years = np.intersect1d(alt_years, fr_years)
    
    print(f"Analysis period: {common_years.min()}-{common_years.max()} ({len(common_years)} years)")
    
    #trends over common period
    trend_sla_alt = duacs_yearly.sla.sel(year=common_years).polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)
    trend_asl_fr = asl_frederikse.sel(year=common_years).polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)
    
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
    
    print(f"ODSL range: {odsl_mm_yr.min().item():.2f} to {odsl_mm_yr.max().item():.2f} mm/yr")
    
    #clean up regridders
    try:
        regridder_frederikse.clean_weight_file()
        regridder_gia.clean_weight_file()
    except AttributeError:
        pass

    output_ds = xr.Dataset(
        {
            'odsl': odsl_mm_yr,
            'msl': trend_sla_alt_mm_yr,
            'geoid': trend_asl_fr_regridded_mm_yr,
            'gia': gia_regridded_mm_yr,
        }
    )

    #global attributes
    output_ds.attrs['description'] = "Observed ODSL and its components."
    output_ds.attrs['analysis_period'] = f"{common_years.min()}-{common_years.max()}"
    output_ds.attrs['common_years_list'] = common_years.tolist() 

    return output_ds

@cache_result('cmip5_processed_models')
def process_cmip5_models():
    """Process CMIP5 models and return a single dataset."""
    print("\n=== Processing CMIP5 Models ===")
    
    inventory = get_cmip_files_inventory("CMIP5")
    all_files = inventory['all_files']
    
    #initialize
    model_names = []
    trends_list = []
    full_timeseries_list = []
    region_masks_list = []
    trend_stats_list = []
    
    for i, model_name in enumerate(TARGET_CMIP5_MODELS):
        hist_file = all_files['historical'].get(model_name)
        rcp45_file = all_files['rcp45'].get(model_name)
        
        if hist_file and rcp45_file:
            print(f"Processing model: {model_name} ({i+1}/{len(TARGET_CMIP5_MODELS)})")
            
            try:
                #calculations per model
                combined_zos = load_cmip_model_data(model_name, future_scenario='rcp45')
                region_mask = create_region_mask(combined_zos.isel(time=0), EXTENT)
                
                period_data = combined_zos.sel(time=slice(START_YEAR, END_YEAR))
                trend_coeffs = period_data.polyfit(dim='time', deg=1)
                slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
                slope_mm_yr = slope * 10
                
                trend_stats = calculate_weighted_stats(slope_mm_yr, region_mask)
                
                #append results
                model_names.append(model_name)
                trends_list.append(slope_mm_yr)
                full_timeseries_list.append(combined_zos)
                region_masks_list.append(region_mask)
                trend_stats_list.append(trend_stats)
                
            except Exception as e:
                print(f"Could not process model {model_name}: {e}")

    print(f"\nProcessed {len(model_names)} out of {len(TARGET_CMIP5_MODELS)} models")

    #concatenate
    model_trends = xr.concat(trends_list, dim=pd.Index(model_names, name='model'))
    full_timeseries = xr.concat(full_timeseries_list, dim=pd.Index(model_names, name='model'))
    region_masks = xr.concat(region_masks_list, dim=pd.Index(model_names, name='model'))

    #convert list
    trend_means = xr.DataArray([s['mean_x'] for s in trend_stats_list], coords={'model': model_names}, dims=['model'])
    trend_stds = xr.DataArray([s['std_x'] for s in trend_stats_list], coords={'model': model_names}, dims=['model'])
    
    #multi-model mean
    model_mean_trend = model_trends.mean(dim='model', skipna=True)
    
    #dataset object
    output_ds = xr.Dataset(
        {
            'model_trend': model_trends,
            'model_mean_trend': model_mean_trend,
            'full_timeseries': full_timeseries,
            'region_mask': region_masks.astype('int8'),
            'trend_mean': trend_means,
            'trend_std': trend_stds
        }
    )
    
    #global attributes
    output_ds.attrs['description'] = "Processed CMIP5 model trends and timeseries."
    output_ds.attrs['valid_models_count'] = len(model_names)
    
    return output_ds

@cache_result('sliding_window_results')
def perform_sliding_window_analysis():
    """Sliding window analysis on all models."""
    print("\n=== Sliding Window Analysis ===")
    
    #processed models
    cmip_results_ds = process_cmip5_models() 
    obs_results = calculate_observed_odsl()
    odsl_mm_yr = obs_results['odsl']
    
    #sample grid
    sample_model_grid = cmip_results_ds['model_mean_trend']
    
    regridder_obs_to_model = xe.Regridder(odsl_mm_yr, sample_model_grid, 'bilinear', periodic=True)
    odsl_mm_yr_regridded = regridder_obs_to_model(odsl_mm_yr)
    
    try:
        regridder_obs_to_model.clean_weight_file()
    except AttributeError:
        pass
    
    #remove global mean from observations
    weights = np.cos(np.deg2rad(odsl_mm_yr_regridded.latitude))
    global_mean = odsl_mm_yr_regridded.weighted(weights).mean(dim=("latitude", "longitude")).item()
    odsl_obs_dynamic = odsl_mm_yr_regridded - global_mean
    
    #sliding window analysis for each model
    all_pcc = []
    all_rmse = []
    all_windows = []
    model_names_for_sliding = []
    
    for model_name in cmip_results_ds.model.values:
        print(f"Processing sliding windows for {model_name}...")
        
        full_timeseries = cmip_results_ds['full_timeseries'].sel(model=model_name)
        region_mask = cmip_results_ds['region_mask'].sel(model=model_name)
        
        pcc_per_model = []
        rmse_per_model = []
        windows_per_model = []

        #slide window
        window_size = 20
        start_year = 1850
        end_year = 2012
        
        for window_start in range(start_year, end_year - window_size + 1):
            window_end = window_start + window_size - 1
            
            window_data = full_timeseries.sel(time=slice(window_start, window_end))
            
            if len(window_data.time) < window_size * 0.75:
                continue
            
            #linear trend
            trend_coeffs = window_data.polyfit(dim='time', deg=1)
            trend_mm_yr = trend_coeffs.polyfit_coefficients.sel(degree=1) * 10
            
            #statistics
            stats = calculate_weighted_stats(trend_mm_yr, region_mask, data_y=odsl_obs_dynamic)
            
            windows_per_model.append(window_start)
            pcc_per_model.append(stats['pcc'])
            rmse_per_model.append(stats['rmse'])
        
        model_names_for_sliding.append(model_name)
        all_pcc.append(pcc_per_model)
        all_rmse.append(rmse_per_model)
        
        if not all_windows:
            all_windows = windows_per_model
            
    #assemble
    output_ds = xr.Dataset(
        {
            'pcc': (('model', 'window_start_year'), all_pcc),
            'rmse': (('model', 'window_start_year'), all_rmse),
            'odsl_obs_dynamic': odsl_obs_dynamic, 
        },
        coords={
            'model': model_names_for_sliding,
            'window_start_year': all_windows
        }
    )
    output_ds.attrs['description'] = "Sliding window PCC and RMSE results."

    return output_ds

@cache_result('steric_comparison')
def compare_with_steric_record():
    """Compare global mean removed ODSL with steric record."""
    print("\n=== Comparing with Steric Record ===")
    
    #observed ODSL
    obs_results = calculate_observed_odsl()
    odsl_mm_yr = obs_results['odsl']
    
    common_years = np.array(obs_results.attrs['common_years_list'])
    
    #global mean of ODSL
    weights = np.cos(np.deg2rad(odsl_mm_yr.latitude))
    global_mean_odsl = odsl_mm_yr.weighted(weights).mean(dim=("latitude", "longitude")).item()
    
    print(f"\nGlobal mean ODSL trend: {global_mean_odsl:.3f} mm/yr")
    
    #remove global mean
    odsl_global_mean_removed = odsl_mm_yr - global_mean_odsl
    
    #regional mean after removing global mean
    region_mask = create_region_mask(odsl_mm_yr, EXTENT)
    regional_stats_original = calculate_weighted_stats(odsl_mm_yr, region_mask)
    regional_stats_detrended = calculate_weighted_stats(odsl_global_mean_removed, region_mask)
    
    #steric record
    try:
        
        #excel
        budget_dir = find_folder_by_name("Budget")
        frederikse_dir = os.path.join(budget_dir, "Frederikse")
        excel_path = os.path.join(frederikse_dir, "global_basin_timeseries.xlsx")
        df_steric = pd.read_excel(excel_path, sheet_name='Global')
        
        #filter for common years and reset index
        steric_data = df_steric[df_steric['Year'].isin(common_years)][['Year', 'Steric [mean]']].copy()
        steric_data = steric_data.reset_index(drop=True)
        
        #steric trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            steric_data['Year'], 
            steric_data['Steric [mean]']
        )
        steric_trend_mm_yr = slope 
        
        print(f"\nSteric trend ({common_years.min()}-{common_years.max()}): {steric_trend_mm_yr:.3f} mm/yr")
        
        #comparison table
        print("\n" + "="*60)
        print("COMPARISON TABLE: ODSL vs Steric Record")
        print("="*60)
        print(f"Analysis period: {common_years.min()}-{common_years.max()}")
        print(f"Region: North Atlantic ({EXTENT[0]}°E to {EXTENT[1]}°E, {EXTENT[2]}°N to {EXTENT[3]}°N)")
        print("-"*60)
        
        print("\nGLOBAL TRENDS:")
        print(f"  Global mean ODSL:           {global_mean_odsl:6.3f} mm/yr")
        print(f"  Global steric (Frederikse):  {steric_trend_mm_yr:6.3f} mm/yr")
        print(f"  Difference:                 {global_mean_odsl - steric_trend_mm_yr:6.3f} mm/yr")
        
        print("\nREGIONAL NORTH ATLANTIC TRENDS:")
        print(f"  Original ODSL:              {regional_stats_original['mean_x']:6.3f} mm/yr")
        print(f"  ODSL (global mean removed):  {regional_stats_detrended['mean_x']:6.3f} mm/yr")
        print(f"  Change after detrending:     {regional_stats_detrended['mean_x'] - regional_stats_original['mean_x']:6.3f} mm/yr")
        
        print("\nYEAR-BY-YEAR STERIC DATA:")
        print("-"*40)
        print("Year  | Steric [mm] | Annual Change [mm]")
        print("-"*40)
        
        for idx in range(len(steric_data)):
            year = steric_data.loc[idx, 'Year']
            steric_value = steric_data.loc[idx, 'Steric [mean]']
            
            if idx > 0:
                annual_change = steric_value - steric_data.loc[idx-1, 'Steric [mean]']
                print(f"{int(year)} | {steric_value:11.2f} | {annual_change:17.2f}")
            else:
                print(f"{int(year)} | {steric_value:11.2f} | {'N/A':>17}")
        
        print("-"*40)
        
        #summary statistics
        total_change = steric_data['Steric [mean]'].iloc[-1] - steric_data['Steric [mean]'].iloc[0]
        n_years = len(steric_data) - 1
        
        print(f"Total change: {total_change:.2f} mm")
        print(f"Average annual change: {total_change/n_years:.3f} mm/yr")
        print(f"Linear trend (from regression): {steric_trend_mm_yr:.3f} mm/yr")
        
        #ODSL without global mean to steric
        print(f"\nCOMPARISON CHECK:")
        print(f"Global mean ODSL trend: {global_mean_odsl:.3f} mm/yr")
        print(f"Global steric trend: {steric_trend_mm_yr:.3f} mm/yr")
        print(f"Ratio (ODSL/Steric): {global_mean_odsl/steric_trend_mm_yr:.3f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error loading steric record: {e}")
        print("Please ensure the Excel file exists at the expected location")
        traceback.print_exc()
    
    return {
        'global_mean_odsl': global_mean_odsl,
        'odsl_detrended': odsl_global_mean_removed
    }

def main():
    """Run complete analysis."""
    print("=== ODSL Analysis ===")
    fig_dir = './figures/'

    #observed ODSL
    obs_results = calculate_observed_odsl()
    print(f"Observed ODSL range: {obs_results['odsl'].min().item():.2f} to {obs_results['odsl'].max().item():.2f} mm/yr")
    
    #steric record
    steric_comparison = compare_with_steric_record()
    
    #CMIP models
    cmip_results = process_cmip5_models()
    print(f"Processed {cmip_results.attrs['valid_models_count']} CMIP5 models")
    
    #sliding window analysis
    sliding_results = perform_sliding_window_analysis()
    print("Completed sliding window analysis")
    
    print("\n=== All calculations complete. Generating figures... ===")

    #figures
    create_all_figures(obs_results=obs_results, cmip_results=cmip_results, sliding_results=sliding_results, fig_dir=fig_dir)
    print("All figures generated!")

if __name__ == "__main__":
    main()