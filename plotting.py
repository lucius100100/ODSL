import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd

from utils import calculate_weighted_stats, create_region_mask
from config import (START_YEAR, END_YEAR, EXTENT, PROJECTION_PARAMS)

def create_all_figures(obs_results, cmip_results, sliding_results, fig_dir):
    """Generate all figures for the analysis."""
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    plot_observed_odsl_components(obs_results, fig_dir)
    plot_cmip5_multimodel_mean(cmip_results, fig_dir)
    plot_observed_vs_modeled(cmip_results, sliding_results, fig_dir)
    plot_sliding_window_timeseries(sliding_results, fig_dir)
    plot_best_matching_periods(sliding_results, fig_dir)
    plot_model_comparison_summary(cmip_results, sliding_results, fig_dir)

def add_map_features(ax, extent, is_left=False, is_bottom=False):
    """Add standard map features to axis."""
    lon_min, lon_max, lat_min, lat_max = extent
    boundary_path = mpath.Path([
        [lon_min, lat_min], [lon_max, lat_min],
        [lon_max, lat_max], [lon_min, lat_max],
        [lon_min, lat_min]
    ]).interpolated(50)
    
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    boundary_in_proj_coords = proj_to_data.transform_path(boundary_path)
    ax.set_boundary(boundary_in_proj_coords)
    
    verts = boundary_in_proj_coords.vertices
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = is_left
    gl.bottom_labels = is_bottom

def plot_observed_odsl_components(obs_results, fig_dir):
    """Plot the components of observed ODSL."""
    print("Plotting observed ODSL components...")
    
    #projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=PROJECTION_PARAMS['central_longitude'],
        central_latitude=PROJECTION_PARAMS['central_latitude'],
        standard_parallels=PROJECTION_PARAMS['standard_parallels']
    )
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 10), 
                            subplot_kw={'projection': proj})
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    #data
    msl = obs_results['msl']
    geoid = obs_results['geoid']
    gia = obs_results['gia']
    odsl = obs_results['odsl']

    #common years
    common_years = np.array(obs_results.attrs['common_years_list'])
    
    #statistics
    region_mask = create_region_mask(msl, EXTENT)
    stats_msl = calculate_weighted_stats(msl, region_mask)
    stats_geoid = calculate_weighted_stats(geoid, region_mask)
    stats_gia = calculate_weighted_stats(gia, region_mask)
    stats_odsl = calculate_weighted_stats(odsl, region_mask)
    
    #color scale
    vmax = max(abs(msl.quantile(0.98)), abs(geoid.quantile(0.98)))
    
    #plotting
    im1 = ax1.pcolormesh(msl.longitude, msl.latitude, msl, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    add_map_features(ax1, EXTENT, is_left=True, is_bottom=True)
    ax1.set_title(f'a) MSL (Altimetry SLA)\nMean: {stats_msl["mean_x"]:.2f} mm/yr, RMS: {stats_msl["std_x"]:.2f} mm/yr')
    
    im2 = ax2.pcolormesh(geoid.longitude, geoid.latitude, geoid, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    add_map_features(ax2, EXTENT, is_left=True, is_bottom=True)
    ax2.set_title(f'b) Geoid (Frederikse budget ASL)\nMean: {stats_geoid["mean_x"]:.2f} mm/yr, RMS: {stats_geoid["std_x"]:.2f} mm/yr')
    
    im3 = ax3.pcolormesh(gia.longitude, gia.latitude, gia, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    add_map_features(ax3, EXTENT, is_left=True, is_bottom=True)
    ax3.set_title(f'c) GIA\nMean: {stats_gia["mean_x"]:.2f} mm/yr, RMS: {stats_gia["std_x"]:.2f} mm/yr')
    
    im4 = ax4.pcolormesh(odsl.longitude, odsl.latitude, odsl, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    add_map_features(ax4, EXTENT, is_left=True, is_bottom=True)
    ax4.set_title(f'd) ODSL (MSL - Geoid - GIA)\nMean: {stats_odsl["mean_x"]:.2f} mm/yr, RMS: {stats_odsl["std_x"]:.2f} mm/yr')
    
    #colorbar
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.025])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Sea level trend (mm/yr)', fontsize=14)
    
    plt.suptitle(f'Observed ODSL trend ({common_years.min()}-{common_years.max()})',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(fig_dir, f'ODSL_components_{START_YEAR}_{END_YEAR}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_cmip5_multimodel_mean(cmip_results, fig_dir):
    """Plot CMIP5 multi-model mean ODSL."""
    print("Plotting CMIP5 multi-model mean...")
    
    model_mean_trend = cmip_results['model_mean_trend']
    valid_models_count = cmip_results.attrs['valid_models_count']
    
    #projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=PROJECTION_PARAMS['central_longitude'],
        central_latitude=PROJECTION_PARAMS['central_latitude'],
        standard_parallels=PROJECTION_PARAMS['standard_parallels']
    )
    
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': proj})
    
    add_map_features(ax, EXTENT, is_left=True, is_bottom=True)
    
    #symmetric color range
    vmax = abs(model_mean_trend.quantile(0.98, skipna=True).item())
    vmin = -vmax
    
    mesh = model_mean_trend.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False 
    )
    
    #colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)
    cbar.set_label('ODSL trend (mm/year)', fontsize=10)
    
    #regional statistics
    region_mask = create_region_mask(model_mean_trend, EXTENT)
    stats_model = calculate_weighted_stats(model_mean_trend, region_mask)
    
    ax.set_title(
        f'CMIP5 multi-model mean ({valid_models_count} models (historical + RCP4.5))\n'
        f'ODSL trend ({START_YEAR}-{END_YEAR}) Mean: {stats_model["mean_x"]:.2f} mm/yr, '
        f'RMS: {stats_model["std_x"]:.2f} mm/yr',
        fontsize=12, pad=15
    )
    
    plt.savefig(os.path.join(fig_dir, f'CMIP5_multimodel_mean_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_observed_vs_modeled(cmip_results, sliding_results, fig_dir):
    """Plot observed vs modeled ODSL comparison."""
    print("Plotting observed vs modeled ODSL...")
    
    #data
    model_mean_trend = cmip_results['model_mean_trend']
    odsl_obs_dynamic = sliding_results['odsl_obs_dynamic']
    
    #difference
    difference = model_mean_trend - odsl_obs_dynamic
    
    #statistics
    region_mask = create_region_mask(model_mean_trend, EXTENT)
    stats_comparison = calculate_weighted_stats(model_mean_trend, region_mask, data_y=odsl_obs_dynamic)
    stats_difference = calculate_weighted_stats(difference, region_mask)
    pcc_w = stats_comparison['pcc']
    
    #projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=PROJECTION_PARAMS['central_longitude'],
        central_latitude=PROJECTION_PARAMS['central_latitude'],
        standard_parallels=PROJECTION_PARAMS['standard_parallels']
    )
    
    #subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3,
        figsize=(22, 7),
        subplot_kw={'projection': proj}
    )
    
    vmax_unified = max(
        abs(odsl_obs_dynamic.quantile(0.98, skipna=True).item()),
        abs(model_mean_trend.quantile(0.98, skipna=True).item()),
        abs(difference.quantile(0.98, skipna=True).item())
    )
    vmin_unified = -vmax_unified
    
    #subplot 1: observed ODSL
    add_map_features(ax1, EXTENT, is_left=True, is_bottom=True)
    mesh1 = odsl_obs_dynamic.plot.pcolormesh(
        ax=ax1, transform=ccrs.PlateCarree(), cmap='coolwarm',
        vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False
    )
    ax1.set_title(f'a) Observed ODSL (Detrended)\nMean: {stats_comparison["mean_y"]:.2f} mm/yr, 'f'RMS: {stats_comparison["std_y"]:.2f} mm/yr', fontsize=11)

    #subplot 2: modelled ODSL
    add_map_features(ax2, EXTENT, is_left=False, is_bottom=True)
    mesh2 = model_mean_trend.plot.pcolormesh(
        ax=ax2, transform=ccrs.PlateCarree(), cmap='coolwarm',
        vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False
    )
    ax2.set_title(f'b) CMIP5 mean ODSL\nMean: {stats_comparison["mean_x"]:.2f} mm/yr, '
                  f'RMS: {stats_comparison["std_x"]:.2f} mm/yr', fontsize=11)
    
    #subplot 3: difference (model - observed)
    add_map_features(ax3, EXTENT, is_left=False, is_bottom=True)
    mesh3 = difference.plot.pcolormesh(
        ax=ax3, transform=ccrs.PlateCarree(), cmap='coolwarm',
        vmin=-vmax_unified, vmax=vmax_unified, add_colorbar=False
    )
    ax3.set_title(f'c) Difference (model - obs)\nMean: {stats_difference["mean_x"]:.2f} mm/yr, '
                  f'RMS: {stats_difference["std_x"]:.2f} mm/yr', fontsize=11)
    
    #title and layout
    fig.suptitle(
        f'Observed vs. modeled ODSL trend ({START_YEAR}-{END_YEAR})\n'
        f'North Atlantic PCC = {pcc_w:.2f}',
        fontsize=16, y=1.02
    )
    
    #shared colorbar
    cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.03])
    cbar = fig.colorbar(mesh1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Trend (mm/year)', fontsize=12)
    
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.15)
    
    plt.savefig(os.path.join(fig_dir, f'observed_vs_modeled_{START_YEAR}_{END_YEAR}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_sliding_window_timeseries(sliding_results, fig_dir):
    """Plot PCC and RMSE time series from sliding window analysis."""
    print("Plotting PCC and RMSE time series...")
    
    pcc = sliding_results['pcc']
    rmse = sliding_results['rmse']
    
    #model names and window centers
    model_names = pcc.model.values
    window_centers = pcc.window_start_year.values + 10
    n_models = len(model_names)
    
    #color range
    colors = plt.colormaps['tab20'](np.linspace(0, 1, n_models))
    
    #line styles
    line_styles = ['-', '--', '-.', ':']
    line_widths = [3, 3, 3, 3]
    
    #plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    #all PCC values for y-limits
    all_pcc_values = []
    all_rmse_values = []
    
    #time series
    #PCC
    for i, model_name in enumerate(model_names):
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        linewidth = line_widths[i % len(line_widths)]
        
        ax1.plot(window_centers, pcc.sel(model=model_name), 
                 color=color, linestyle=linestyle, linewidth=linewidth,
                 alpha=0.8, label=model_name)
    
    ax1.axvline(1993 + 9.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Obs period center') 
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('PCC', fontsize=12)
    ax1.set_title('Model-observation comparison\n20-year sliding windows (1900-2012)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(pcc.min() - 0.1, pcc.max() + 0.1)
    
    #RMSE
    for i, model_name in enumerate(model_names):
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        linewidth = line_widths[i % len(line_widths)]
        
        ax2.plot(window_centers, rmse.sel(model=model_name), 
                 color=color, linestyle=linestyle, linewidth=linewidth,
                 alpha=0.8, label=model_name)
    
    #observation period
    ax2.axvline(1993 + 9.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Obs period center')
    ax2.set_xlabel('Window center year', fontsize=12)
    ax2.set_ylabel('RMSE (mm/yr)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(rmse.min() * 0.9, rmse.max() * 1.1)
    
    #shared legend
    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), 
                     fontsize=10, framealpha=0.9, title='$\\bf{Models}$', title_fontsize=12)
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)
    
    plt.tight_layout()
    
    #grid lines
    for ax in [ax1, ax2]:
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.minorticks_on()
    
    plt.savefig(os.path.join(fig_dir, 'sliding_window_timeseries.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_best_matching_periods(sliding_results, fig_dir):
    """Calculates and plots the best matching 20-year periods."""
    print("Plotting best matching periods...")
    
    model_names = sliding_results.model.values
    
    best_pcc_windows = []
    best_pcc_values = []
    best_rmse_windows = []
    best_rmse_values = []

    #loop through models
    for model_name in model_names:
        pcc_model = sliding_results['pcc'].sel(model=model_name)
        rmse_model = sliding_results['rmse'].sel(model=model_name)
        
        #find best window
        #PCC
        try:
            best_pcc_year = pcc_model.idxmax('window_start_year').item()
            max_pcc_value = pcc_model.max('window_start_year').item()
            best_pcc_windows.append((best_pcc_year, best_pcc_year + 19))
            best_pcc_values.append(max_pcc_value)
        except ValueError: 
            best_pcc_windows.append((np.nan, np.nan))
            best_pcc_values.append(np.nan)

        #RMSE
        try:
            best_rmse_year = rmse_model.idxmin('window_start_year').item()
            min_rmse_value = rmse_model.min('window_start_year').item()
            best_rmse_windows.append((best_rmse_year, best_rmse_year + 19))
            best_rmse_values.append(min_rmse_value)
        except ValueError: 
            best_rmse_windows.append((np.nan, np.nan))
            best_rmse_values.append(np.nan)

    #plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    observed_period = (1993, 2012)
    observed_center = (observed_period[0] + observed_period[1]) / 2
    combined_data = list(zip(model_names, best_pcc_windows, best_rmse_windows, best_pcc_values, best_rmse_values))

    def sort_key(item):
        rmse_window = item[2]
        if np.isnan(rmse_window[0]):
            return float('inf')
        rmse_center = (rmse_window[0] + rmse_window[1]) / 2
        return abs(rmse_center - observed_center)

    sorted_combined_data = sorted(combined_data, key=sort_key, reverse=True)

    if not sorted_combined_data:
        print("Warning: No valid data to plot for best matching periods.")
        plt.show()
        return

    sorted_model_list, sorted_pcc_windows, sorted_rmse_windows, sorted_pcc_values, sorted_rmse_values = zip(*sorted_combined_data)

    y_positions = np.arange(len(sorted_model_list))
    bar_height = 0.4

    for i, (model, pcc_window, rmse_window, pcc_value, rmse_value) in enumerate(
        zip(sorted_model_list, sorted_pcc_windows, sorted_rmse_windows, sorted_pcc_values, sorted_rmse_values)
    ):
        if not np.isnan(pcc_window[0]):
            ax.barh(y_positions[i] + bar_height/2, pcc_window[1] - pcc_window[0], left=pcc_window[0], height=bar_height, color='red', alpha=0.7, label='Best PCC' if i == 0 else "")
            bar_center = (pcc_window[0] + pcc_window[1]) / 2
            ax.text(bar_center, y_positions[i] + bar_height/2, f'{pcc_value:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        
        if not np.isnan(rmse_window[0]):
            ax.barh(y_positions[i] - bar_height/2, rmse_window[1] - rmse_window[0], left=rmse_window[0], height=bar_height, color='black', alpha=0.7, label='Best RMSE' if i == 0 else "")
            bar_center = (rmse_window[0] + rmse_window[1]) / 2
            ax.text(bar_center, y_positions[i] - bar_height/2, f'{rmse_value:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    ax.axvspan(observed_period[0], observed_period[1], alpha=0.2, color='red', label='Observation period')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_model_list)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_xlim(1900, 2015)
    ax.set_title('Best matching 20-year periods by model (vs. 1993-2012 Observations)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'best_matching_periods.png'), dpi=300, bbox_inches='tight')

    plt.show()

def plot_model_comparison_summary(cmip_results_ds, sliding_results_ds, fig_dir):
    """Calculates and plots a summary of model-observation comparison statistics, Richter et al., 2017, fig. 8."""
    print("Creating model-observation comparison statistics figure...")

    #stats for each model
    mean_rmse_per_model = sliding_results_ds['rmse'].mean(dim='window_start_year')
    min_rmse_per_model = sliding_results_ds['rmse'].min(dim='window_start_year')
    max_rmse_per_model = sliding_results_ds['rmse'].max(dim='window_start_year')
    mean_pcc_per_model = sliding_results_ds['pcc'].mean(dim='window_start_year')
    min_pcc_per_model = sliding_results_ds['pcc'].min(dim='window_start_year')
    max_pcc_per_model = sliding_results_ds['pcc'].max(dim='window_start_year')

    #stats 1993-2012 observation period
    obs_period_rmse_list, obs_period_pcc_list = [], []
    odsl_obs_dynamic = sliding_results_ds['odsl_obs_dynamic']
    for model_name in cmip_results_ds.model.values:
        obs_period_trend = cmip_results_ds['model_trend'].sel(model=model_name)
        region_mask = cmip_results_ds['region_mask'].sel(model=model_name).astype(bool)
        stats = calculate_weighted_stats(obs_period_trend, region_mask, data_y=odsl_obs_dynamic)
        obs_period_rmse_list.append(stats['rmse'])
        obs_period_pcc_list.append(stats['pcc'])

    #pandas dataframe    
    df = pd.DataFrame({
        'mean_rmse': mean_rmse_per_model.values, 'min_rmse': min_rmse_per_model.values, 'max_rmse': max_rmse_per_model.values,
        'obs_period_rmse': obs_period_rmse_list,
        'mean_pcc': mean_pcc_per_model.values, 'min_pcc': min_pcc_per_model.values, 'max_pcc': max_pcc_per_model.values,
        'obs_period_pcc': obs_period_pcc_list,
    }, index=cmip_results_ds.model.values)
    df_sorted = df.sort_values('mean_rmse')

    #ensemble stats    
    ensemble_region_mask = create_region_mask(cmip_results_ds['model_mean_trend'], EXTENT)
    stats_obs_ensemble = calculate_weighted_stats(cmip_results_ds['model_mean_trend'], ensemble_region_mask, data_y=odsl_obs_dynamic)
    
    best_pcc_years = sliding_results_ds['pcc'].idxmax('window_start_year')
    best_rmse_years = sliding_results_ds['rmse'].idxmin('window_start_year')
    best_pcc_trends, best_rmse_trends = [], []
    for model_name in cmip_results_ds.model.values:
        full_ts = cmip_results_ds['full_timeseries'].sel(model=model_name)
        pcc_year = best_pcc_years.sel(model=model_name).item()
        rmse_year = best_rmse_years.sel(model=model_name).item()
        pcc_window_data = full_ts.sel(time=slice(pcc_year, pcc_year + 19))
        rmse_window_data = full_ts.sel(time=slice(rmse_year, rmse_year + 19))
        best_pcc_trends.append(pcc_window_data.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
        best_rmse_trends.append(rmse_window_data.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
    
    best_pcc_ensemble_mean = xr.concat(best_pcc_trends, dim='model').mean(dim='model')
    best_rmse_ensemble_mean = xr.concat(best_rmse_trends, dim='model').mean(dim='model')
    stats_pcc_ensemble = calculate_weighted_stats(best_pcc_ensemble_mean, ensemble_region_mask, data_y=odsl_obs_dynamic)
    stats_rmse_ensemble = calculate_weighted_stats(best_rmse_ensemble_mean, ensemble_region_mask, data_y=odsl_obs_dynamic)

    ensemble_data = {'Obs Period\nEnsemble': stats_obs_ensemble, 'Best RMSE\nEnsemble': stats_rmse_ensemble, 'Best PCC\nEnsemble': stats_pcc_ensemble}
    for name, stats in ensemble_data.items():
        df_sorted.loc[name] = {'mean_rmse': stats['rmse'], 'min_rmse': stats['rmse'], 'max_rmse': stats['rmse'], 'obs_period_rmse': stats['rmse'],
                               'mean_pcc': stats['pcc'], 'min_pcc': stats['pcc'], 'max_pcc': stats['pcc'], 'obs_period_pcc': stats['pcc']}

    #plotting    
    model_names = df_sorted.index.tolist()
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(14, 8))
    
    #final data lists
    mean_rmse_all, min_rmse_all, max_rmse_all, obs_period_rmse = df_sorted['mean_rmse'], df_sorted['min_rmse'], df_sorted['max_rmse'], df_sorted['obs_period_rmse']
    mean_pcc_all, min_pcc_all, max_pcc_all, obs_period_pcc = df_sorted['mean_pcc'], df_sorted['min_pcc'], df_sorted['max_pcc'], df_sorted['obs_period_pcc']
    
    #scaling and offset
    y_min, y_max = 0, 1
    rmse_min_val, rmse_max_val = 0, np.nanmax(max_rmse_all) * 1.1
    rmse_range = rmse_max_val - rmse_min_val
    rmse_scale_factor = 0.5 / rmse_range if rmse_range > 0 else 0

    pcc_min_val, pcc_max_val = np.nanmin(min_pcc_all) - 0.1, np.nanmax(max_pcc_all) + 0.1
    pcc_range = pcc_max_val - pcc_min_val
    pcc_scale_factor = 0.5 / pcc_range if pcc_range > 0 else 0
    pcc_offset = 0.5

    #scale RMSE
    mean_rmse_scaled = mean_rmse_all * rmse_scale_factor
    min_rmse_scaled = min_rmse_all * rmse_scale_factor
    max_rmse_scaled = max_rmse_all * rmse_scale_factor
    obs_rmse_scaled = obs_period_rmse * rmse_scale_factor
    
    #scale PCC
    mean_pcc_scaled = (mean_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    min_pcc_scaled = (min_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    max_pcc_scaled = (max_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    obs_pcc_scaled = (obs_period_pcc - pcc_min_val) * pcc_scale_factor + pcc_offset

    #RMSE bars (lower half)
    ax.bar(x, mean_rmse_scaled, 0.8, color='grey', alpha=0.5, edgecolor='black', linewidth=1.5, label='Mean RMSE (all windows)')
    rmse_errors_scaled = [mean_rmse_scaled - min_rmse_scaled, max_rmse_scaled - mean_rmse_scaled]
    ax.errorbar(x, mean_rmse_scaled, yerr=rmse_errors_scaled, fmt='none', color='black', capsize=6, capthick=1.5, label='RMSE range (all windows)')
    ax.scatter(x, obs_rmse_scaled, color='black', s=80, zorder=5, label='RMSE (observed)')
    
    #PCC lines and dots (upper half)
    mean_line_plotted = False
    for i, (x_pos, mean_val) in enumerate(zip(x, mean_pcc_scaled)):
        if not np.isnan(mean_val):
            label = 'Mean PCC (all windows)' if not mean_line_plotted else ""
            ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val], color='red', linewidth=2, zorder=3, label=label)
            mean_line_plotted = True
    
    pcc_errors_scaled = [mean_pcc_scaled - min_pcc_scaled, max_pcc_scaled - mean_pcc_scaled]
    ax.errorbar(x, mean_pcc_scaled, yerr=pcc_errors_scaled, fmt='none', color='red', capsize=6, capthick=1.5, label='PCC range (all windows)')
    ax.scatter(x, obs_pcc_scaled, color='red', s=80, zorder=5, label='PCC (observed)')

    #axis and label
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax2 = ax.twinx()
    ax2.set_ylim(y_min, y_max)
    
    rmse_tick_values = np.linspace(rmse_min_val, rmse_max_val, 5)
    ax.set_yticks(rmse_tick_values * rmse_scale_factor)
    ax.set_yticklabels([f'{v:.1f}' for v in rmse_tick_values])
    ax.set_ylabel('RMSE (mm/yr)', fontsize=12, color='black', y=0.25)
    ax.tick_params(axis='y', labelcolor='black')
    
    pcc_tick_values = np.linspace(pcc_min_val, pcc_max_val, 5)
    ax2.set_yticks((pcc_tick_values - pcc_min_val) * pcc_scale_factor + pcc_offset)
    ax2.set_yticklabels([f'{v:.1f}' for v in pcc_tick_values])
    ax2.set_ylabel('PCC', fontsize=12, color='red', y=0.75)
    ax2.tick_params(axis='y', labelcolor='red')

    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)
    separator_pos = len(df_sorted) - 3.5
    ax.axvline(separator_pos, color='gray', linestyle='--')
    
    #x-axis setup
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_xlabel('Models')
    ax.set_title('Model-observation ODSL comparison\nMean statistics over all 20-yr sliding windows', fontsize=14, pad=20)

    #model names inside plot area
    for i, model_name in enumerate(model_names):
        ax.text(i, 0.02, model_name.replace('\n', ' '), rotation=90, ha='center', va='bottom', fontsize=10)
    
    #gridlines
    for tick in ax.get_yticks():
        ax.axhline(y=tick, color='gray', linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)
    for tick in ax2.get_yticks():
        ax.axhline(y=tick, color='gray', linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)

    ax.legend(loc='center right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()