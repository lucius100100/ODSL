from utils import calculate_weighted_stats, create_region_mask
from config import (START_YEAR, END_YEAR, EXTENT, PROJECTION_PARAMS, PLOT_VARIABLE, PLOT_CONFIG)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd
import xesmf as xe
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

def create_all_figures(lowess_results_df, obs_results, cmip_results, sliding_results, scenario_results, fig_dir):
    """Generate all figures for the analysis."""

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    variable_fig_dir = os.path.join(fig_dir, PLOT_VARIABLE)
    if not os.path.exists(variable_fig_dir):
        os.makedirs(variable_fig_dir)

    plot_model_comparison_summary(cmip_results, sliding_results, variable_fig_dir)
    plot_lowess_residuals_spatially(sliding_results, cmip_results, lowess_results_df, variable_fig_dir)
    plot_lowess_fit(lowess_results_df, variable_fig_dir)
    plot_scenario_comparison(scenario_results, variable_fig_dir)

    print(f"\nGenerating figures for {PLOT_VARIABLE.upper()}")
    print(f"Figures will be saved in: {variable_fig_dir}")

    if PLOT_VARIABLE == 'trend':
        plot_observed_odsl_components(obs_results, variable_fig_dir)
    elif PLOT_VARIABLE == 'variability':
        plot_observed_variability(obs_results, variable_fig_dir)

    plot_cmip_multimodel_mean(cmip_results, variable_fig_dir)
    plot_observed_vs_modeled(cmip_results, sliding_results, variable_fig_dir)
    plot_sliding_window_timeseries(sliding_results, variable_fig_dir)
    plot_best_matching_periods(sliding_results, variable_fig_dir)
    #plot_model_comparison_summary(cmip_results, sliding_results, variable_fig_dir)

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
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
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
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 10), subplot_kw={'projection': proj})
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
    
    plt.suptitle(f'Observed ODSL trend ({common_years.min()}-{common_years.max()})', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(fig_dir, f'ODSL_components_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_cmip_multimodel_mean(cmip_results, fig_dir):
    """Plot CMIP multi-model mean ODSL."""

    print("Plotting CMIP multi-model mean...")
    
    cfg = PLOT_CONFIG[PLOT_VARIABLE]
    
    if PLOT_VARIABLE == 'trend':
        data_to_plot = cmip_results['model_mean_trend']
    else:
        data_to_plot = cmip_results['model_mean_variability']
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
    if PLOT_VARIABLE == 'trend':
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = -vmax
    else:
        vmax = data_to_plot.quantile(0.98, skipna=True).item()
        vmin = data_to_plot.quantile(0.02, skipna=True).item()

    mesh = data_to_plot.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cfg['cmap'],
        vmin=vmin, vmax=vmax, add_colorbar=False 
    )
    
    #colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)
    cbar.set_label(f'ODSL {cfg["name"]} ({cfg["units"]})', fontsize=10)
    
    #regional statistics
    region_mask = create_region_mask(data_to_plot, EXTENT)
    stats_model = calculate_weighted_stats(data_to_plot, region_mask)
    
    ax.set_title(
        f'CMIP multi-model mean ({valid_models_count} models (historical + RCP4.5))\n' f'ODSL trend ({START_YEAR}-{END_YEAR}) Mean: {stats_model["mean_x"]:.2f} mm/yr,' f'RMS: {stats_model["std_x"]:.2f} {cfg["units"]}',fontsize=12, pad=15, fontweight='bold')
    
    plt.savefig(os.path.join(fig_dir, f'CMIP_multimodel_mean_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_observed_vs_modeled(cmip_results, sliding_results, fig_dir):
    """Plot observed vs modeled ODSL comparison for trend or variability."""

    cfg = PLOT_CONFIG[PLOT_VARIABLE]

    print(f"Plotting observed vs modeled ODSL {cfg['name']}...")

    if PLOT_VARIABLE == 'trend':
        model_data = cmip_results['model_mean_trend']
        obs_data = sliding_results['odsl_obs_dynamic']
        cmap_unified = 'coolwarm'
        
        vmax_unified = max(
            abs(obs_data.quantile(0.98, skipna=True).item()),
            abs(model_data.quantile(0.98, skipna=True).item())
        )
        vmin_unified = -vmax_unified

    else:
        model_data = cmip_results['model_mean_variability']
        obs_data = sliding_results['odsl_var_obs_regridded']
        cmap_unified = cfg['cmap']

        vmax_unified = max(
            obs_data.quantile(0.98, skipna=True).item(),
            model_data.quantile(0.98, skipna=True).item()
        )
        vmin_unified = 0

    difference = model_data - obs_data
    vmax_diff = abs(difference.quantile(0.98, skipna=True).item())
    
    region_mask = create_region_mask(model_data, EXTENT)
    stats_comparison = calculate_weighted_stats(model_data, region_mask, data_y=obs_data)
    stats_difference = calculate_weighted_stats(difference, region_mask)
    pcc_w = stats_comparison['pcc']
    
    proj = ccrs.AlbersEqualArea(
        central_longitude=PROJECTION_PARAMS['central_longitude'],
        central_latitude=PROJECTION_PARAMS['central_latitude'],
        standard_parallels=PROJECTION_PARAMS['standard_parallels']
    )
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(22, 7), subplot_kw={'projection': proj})

    #subplot 1: observed ODSL
    add_map_features(ax1, EXTENT, is_left=True, is_bottom=True)
    mesh1 = obs_data.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False)
    ax1.set_title(f'a) Observed ODSL ({cfg["name"]})\nMean: {stats_comparison["mean_y"]:.2f} {cfg["units"]}, RMS: {stats_comparison["std_y"]:.2f} {cfg["units"]}', fontsize=11)

    #subplot 2: modelled ODSL
    add_map_features(ax2, EXTENT, is_left=False, is_bottom=True)
    mesh2 = model_data.plot.pcolormesh(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False)
    ax2.set_title(f'b) CMIP mean ODSL ({cfg["name"]})\nMean: {stats_comparison["mean_x"]:.2f} {cfg["units"]}, RMS: {stats_comparison["std_x"]:.2f} {cfg["units"]}', fontsize=11)
    
    #subplot 3: difference (model - observed)
    add_map_features(ax3, EXTENT, is_left=False, is_bottom=True)
    mesh3 = difference.plot.pcolormesh(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False)
    ax3.set_title(f'c) Difference (model - obs)\nMean: {stats_difference["mean_x"]:.2f} {cfg["units"]}, RMS: {stats_difference["std_x"]:.2f} {cfg["units"]}', fontsize=11)
    
    fig.suptitle(f'Observed vs. modeled ODSL {cfg["name"]} ({START_YEAR}-{END_YEAR})\n' f'North Atlantic PCC = {pcc_w:.2f}', fontsize=16, y=1.02, fontweight='bold')
    
    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    cbar = fig.colorbar(mesh1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=12)
    
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.15)
    
    plt.savefig(os.path.join(fig_dir, f'observed_vs_modeled_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_sliding_window_timeseries(sliding_results, fig_dir):
    """Plot PCC and RMSE time series from sliding window analysis."""

    cfg = PLOT_CONFIG[PLOT_VARIABLE]
    window_size = END_YEAR - START_YEAR + 1

    print(f"Plotting PCC and RMSE time series for {cfg['name']}...")

    if PLOT_VARIABLE == 'trend':
        pcc = sliding_results['pcc']
        rmse = sliding_results['rmse']
        mean_ts_data = sliding_results['mean_trend']
        y_label_top = f'Mean {cfg["name"]} ({cfg["units"]})'
    else: 
        pcc = sliding_results['pcc_variability']
        rmse = sliding_results['rmse_variability']
        mean_ts_data = sliding_results['mean_variability']
        y_label_top = f'Mean {cfg["name"]} ({cfg["units"]})'
    
    pcc_mean = pcc.mean(dim='model')
    rmse_mean = rmse.mean(dim='model')
    mean_ts_ensemble = mean_ts_data.mean(dim='model')

    model_names = pcc.model.values
    window_center_offset = (window_size - 1) / 2
    window_centers = pcc.window_start_year.values + window_center_offset
    title_start_year = int(window_centers.min())
    title_end_year = int(window_centers.max())
    n_models = len(model_names)
    
    colors = plt.colormaps['tab20'](np.linspace(0, 1, n_models))
    line_styles = ['-', '--', '-.', ':']
    line_widths = [3, 3, 3, 3]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    #plot 1: mean trend or variability
    for i, model_name in enumerate(model_names):
        model_ts = mean_ts_data.sel(model=model_name)
        ax1.plot(window_centers, model_ts, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], 
                 linewidth=line_widths[i % len(line_widths)], alpha=0.8, label=model_name)
        
    ax1.plot(window_centers, mean_ts_ensemble, color='black', linewidth=3.5, linestyle='-', label='Ensemble mean', zorder=10, alpha=0.9)
    ax1.axvline(START_YEAR + window_center_offset, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Observed period center')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel(y_label_top, fontsize=12)
    ax1.set_title(f'Model-observation comparison ({cfg["name"]})\n{window_size}-year sliding windows ({title_start_year} to {title_end_year})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(mean_ts_data.min() * 1.1, mean_ts_data.max() * 1.1)

    #plot 2: PCC
    for i, model_name in enumerate(model_names):
        ax2.plot(window_centers, pcc.sel(model=model_name), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)],
                 linewidth=line_widths[i % len(line_widths)], alpha=0.8, label=model_name)
        
    ax2.plot(window_centers, pcc_mean, color='black', linewidth=3.5, linestyle='-', label='Ensemble mean', zorder=10, alpha=0.9)
    ax2.axvline(START_YEAR + window_center_offset, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Observed period center')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.set_ylabel('PCC', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(pcc.min() - 0.1, pcc.max() + 0.1)
    
    #plot 3: RMSE
    for i, model_name in enumerate(model_names):
        ax3.plot(window_centers, rmse.sel(model=model_name), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)],
                 linewidth=line_widths[i % len(line_widths)], alpha=0.8, label=model_name)
        
    ax3.plot(window_centers, rmse_mean, color='black', linewidth=3.5, linestyle='-', label='Ensemble mean', zorder=10, alpha=0.9)
    ax3.axvline(START_YEAR + window_center_offset, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Observed period center')
    ax3.set_xlabel('Window center year', fontsize=12)
    ax3.set_ylabel(f'RMSE ({cfg["units"]})', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(rmse.min() * 0.9, rmse.max() * 1.1)
    
    #dynamic x-lim
    min_limit = window_centers.min() - 5
    max_limit = window_centers.max() + 5
    ax3.set_xlim(min_limit, max_limit)

    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, framealpha=0.9, title='$\\bf{Models}$', title_fontsize=14)
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)
    
    plt.tight_layout()
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.minorticks_on()
    
    plt.savefig(os.path.join(fig_dir, f'sliding_window_timeseries_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_best_matching_periods(sliding_results, fig_dir):
    """Calculates and plots the best matching observed window periods."""

    cfg = PLOT_CONFIG[PLOT_VARIABLE]
    window_size = END_YEAR - START_YEAR + 1

    print(f"Plotting best matching periods for {cfg['name']}...")
    
    if PLOT_VARIABLE == 'trend':
        pcc_data = sliding_results['pcc']
        rmse_data = sliding_results['rmse']
    else:
        pcc_data = sliding_results['pcc_variability']
        rmse_data = sliding_results['rmse_variability']

    model_names = sliding_results.model.values
    
    best_pcc_windows, best_pcc_values = [], []
    best_rmse_windows, best_rmse_values = [], []

    for model_name in model_names:
        pcc_model = pcc_data.sel(model=model_name)
        rmse_model = rmse_data.sel(model=model_name)
        
        try:
            best_pcc_year = pcc_model.idxmax('window_start_year').item()
            best_pcc_windows.append((best_pcc_year, best_pcc_year + window_size - 1))
            best_pcc_values.append(pcc_model.max('window_start_year').item())
        except ValueError: 
            best_pcc_windows.append((np.nan, np.nan)); best_pcc_values.append(np.nan)

        try:
            best_rmse_year = rmse_model.idxmin('window_start_year').item()
            best_rmse_windows.append((best_rmse_year, best_rmse_year + window_size - 1))
            best_rmse_values.append(rmse_model.min('window_start_year').item())
        except ValueError: 
            best_rmse_windows.append((np.nan, np.nan)); best_rmse_values.append(np.nan)

    pcc_mean_ts = pcc_data.mean(dim='model')
    rmse_mean_ts = rmse_data.mean(dim='model')

    ens_best_pcc_year = pcc_mean_ts.idxmax('window_start_year').item()
    ens_best_pcc_window = (ens_best_pcc_year, ens_best_pcc_year + window_size - 1)
    ens_max_pcc_value = pcc_mean_ts.max('window_start_year').item()
    
    ens_best_rmse_year = rmse_mean_ts.idxmin('window_start_year').item()
    ens_best_rmse_window = (ens_best_rmse_year, ens_best_rmse_year + window_size - 1)
    ens_min_rmse_value = rmse_mean_ts.min('window_start_year').item()

    #plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    observed_period = (START_YEAR, END_YEAR)
    observed_center = (observed_period[0] + observed_period[1]) / 2
    combined_data = list(zip(model_names, best_pcc_windows, best_rmse_windows, best_pcc_values, best_rmse_values))

    def sort_key(item):
        rmse_window = item[2]
        if np.isnan(rmse_window[0]): return float('inf')
        return abs(((rmse_window[0] + rmse_window[1]) / 2) - observed_center)
    
    sorted_combined_data = sorted(combined_data, key=sort_key, reverse=True)
    
    ensemble_entry = ('Ensemble mean', ens_best_pcc_window, ens_best_rmse_window, ens_max_pcc_value, ens_min_rmse_value)
    sorted_combined_data.append(ensemble_entry)

    if not sorted_combined_data:
        plt.show(); return

    sorted_model_list, sorted_pcc_windows, sorted_rmse_windows, sorted_pcc_values, sorted_rmse_values = zip(*sorted_combined_data)

    y_positions = np.arange(len(sorted_model_list))
    bar_height = 0.4

    for i, (pcc_window, rmse_window, pcc_value, rmse_value) in enumerate(zip(sorted_pcc_windows, sorted_rmse_windows, sorted_pcc_values, sorted_rmse_values)):
        if not np.isnan(rmse_window[0]):
            ax.barh(y_positions[i] - bar_height/2, rmse_window[1] - rmse_window[0], left=rmse_window[0], height=bar_height, color='black', alpha=0.7, label='Best RMSE' if i == 0 else "")
            bar_center = (rmse_window[0] + rmse_window[1]) / 2
            ax.text(bar_center, y_positions[i] - bar_height/2, f'{rmse_value:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        if not np.isnan(pcc_window[0]):
            ax.barh(y_positions[i] + bar_height/2, pcc_window[1] - pcc_window[0], left=pcc_window[0], height=bar_height, color='red', alpha=0.7, label='Best PCC' if i == 0 else "")
            bar_center = (pcc_window[0] + pcc_window[1]) / 2
            ax.text(bar_center, y_positions[i] + bar_height/2, f'{pcc_value:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    ax.axvspan(observed_period[0], observed_period[1], alpha=0.2, color='red', label='Observation period')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_model_list)
    ax.set_ylim(-0.5, len(sorted_model_list) - 0.5)
    ax.set_xlabel('Year', fontsize=12)

    #dynamic x-lim
    all_windows = list(sorted_pcc_windows) + list(sorted_rmse_windows)
    all_years = [year for window in all_windows if not np.isnan(window[0]) for year in window]
    all_years.extend(observed_period)
    if all_years:
        min_limit = min(all_years) - 5
        max_limit = max(all_years) + 5
        ax.set_xlim(min_limit, max_limit)

    separator_pos = len(model_names) - 0.5
    ax.axhline(y=separator_pos, color='gray', linestyle='--', xmin=-0.12, clip_on=False)

    ax.set_title(f'Best matching {window_size}-year periods by model ({cfg["name"]} vs. observations)', fontsize=14, fontweight='bold')
    ax.legend(loc='center left')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'best_matching_periods_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison_summary(cmip_results_ds, sliding_results_ds, fig_dir):
    """Calculates and plots a summary of model-observation comparison statistics."""

    cfg = PLOT_CONFIG[PLOT_VARIABLE]
    unit = cfg['units']
    window_size = END_YEAR - START_YEAR + 1

    print(f"Creating model-observation comparison statistics figure for {cfg['name']}...")

    if PLOT_VARIABLE == 'trend':
        rmse_ts = sliding_results_ds['rmse']
        pcc_ts = sliding_results_ds['pcc']
        obs_period_model_data = cmip_results_ds['model_trend']
        obs_pattern_data = sliding_results_ds['odsl_obs_dynamic']
    else:
        rmse_ts = sliding_results_ds['rmse_variability']
        pcc_ts = sliding_results_ds['pcc_variability']
        obs_period_model_data = cmip_results_ds['model_variability']
        obs_pattern_data = sliding_results_ds['odsl_var_obs_centered']

    #calculations
    mean_rmse_per_model = rmse_ts.mean(dim='window_start_year')
    min_rmse_per_model = rmse_ts.min(dim='window_start_year')
    max_rmse_per_model = rmse_ts.max(dim='window_start_year')
    mean_pcc_per_model = pcc_ts.mean(dim='window_start_year')
    min_pcc_per_model = pcc_ts.min(dim='window_start_year')
    max_pcc_per_model = pcc_ts.max(dim='window_start_year')

    obs_period_rmse_list, obs_period_pcc_list = [], []
    for model_name in cmip_results_ds.model.values:
        model_pattern = obs_period_model_data.sel(model=model_name)
        region_mask = cmip_results_ds['region_mask'].sel(model=model_name).astype(bool)
        stats = calculate_weighted_stats(model_pattern, region_mask, data_y=obs_pattern_data)
        obs_period_rmse_list.append(stats['rmse'])
        obs_period_pcc_list.append(stats['pcc'])
    
    df = pd.DataFrame({
        'mean_rmse': mean_rmse_per_model.values, 'min_rmse': min_rmse_per_model.values, 'max_rmse': max_rmse_per_model.values,
        'obs_period_rmse': obs_period_rmse_list,
        'mean_pcc': mean_pcc_per_model.values, 'min_pcc': min_pcc_per_model.values, 'max_pcc': max_pcc_per_model.values,
        'obs_period_pcc': obs_period_pcc_list,
    }, index=cmip_results_ds.model.values)
    df_sorted = df.sort_values('mean_rmse')

    ensemble_region_mask = create_region_mask(cmip_results_ds['model_mean_trend'], EXTENT)
    if PLOT_VARIABLE == 'trend':
        stats_obs_ensemble = calculate_weighted_stats(cmip_results_ds['model_mean_trend'], ensemble_region_mask, data_y=obs_pattern_data)
    else:
        stats_obs_ensemble = calculate_weighted_stats(cmip_results_ds['model_mean_variability'], ensemble_region_mask, data_y=obs_pattern_data)

    best_pcc_years = pcc_ts.idxmax('window_start_year')
    best_rmse_years = rmse_ts.idxmin('window_start_year')
    best_pcc_patterns, best_rmse_patterns = [], []
    
    if PLOT_VARIABLE == 'trend':
        for model_name in cmip_results_ds.model.values:
            full_ts = cmip_results_ds['full_timeseries'].sel(model=model_name)
            pcc_year = best_pcc_years.sel(model=model_name).item()
            rmse_year = best_rmse_years.sel(model=model_name).item()
            pcc_window = full_ts.sel(time=slice(pcc_year, pcc_year + window_size - 1))
            rmse_window = full_ts.sel(time=slice(rmse_year, rmse_year + window_size - 1))
            best_pcc_patterns.append(pcc_window.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
            best_rmse_patterns.append(rmse_window.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
    else:
        sliding_var_maps = sliding_results_ds['sliding_variability']
        for model_name in cmip_results_ds.model.values:
            pcc_year = best_pcc_years.sel(model=model_name).item()
            rmse_year = best_rmse_years.sel(model=model_name).item()
            best_pcc_patterns.append(sliding_var_maps.sel(model=model_name, window_start_year=pcc_year))
            best_rmse_patterns.append(sliding_var_maps.sel(model=model_name, window_start_year=rmse_year))

    best_pcc_ensemble_mean = xr.concat(best_pcc_patterns, dim='model').mean(dim='model')
    best_rmse_ensemble_mean = xr.concat(best_rmse_patterns, dim='model').mean(dim='model')
    stats_pcc_ensemble = calculate_weighted_stats(best_pcc_ensemble_mean, ensemble_region_mask, data_y=obs_pattern_data)
    stats_rmse_ensemble = calculate_weighted_stats(best_rmse_ensemble_mean, ensemble_region_mask, data_y=obs_pattern_data)

    ensemble_data = {'Obs Period\nEnsemble': stats_obs_ensemble, 'Best RMSE\nEnsemble': stats_rmse_ensemble, 'Best PCC\nEnsemble': stats_pcc_ensemble}
    for name, stats in ensemble_data.items():
        df_sorted.loc[name] = {'mean_rmse': stats['rmse'], 'min_rmse': stats['rmse'], 'max_rmse': stats['rmse'], 'obs_period_rmse': stats['rmse'],'mean_pcc': stats['pcc'], 'min_pcc': stats['pcc'], 'max_pcc': stats['pcc'], 'obs_period_pcc': stats['pcc']}

    #plotting
    model_names = df_sorted.index.tolist()
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(16, 10))
    
    mean_rmse_all, min_rmse_all, max_rmse_all, obs_period_rmse = df_sorted['mean_rmse'], df_sorted['min_rmse'], df_sorted['max_rmse'], df_sorted['obs_period_rmse']
    mean_pcc_all, min_pcc_all, max_pcc_all, obs_period_pcc = df_sorted['mean_pcc'], df_sorted['min_pcc'], df_sorted['max_pcc'], df_sorted['obs_period_pcc']
    
    y_min, y_max = 0, 1
    rmse_min_val, rmse_max_val = 0, np.nanmax(max_rmse_all) * 1.1
    rmse_range = rmse_max_val - rmse_min_val
    rmse_scale_factor = 0.5 / rmse_range if rmse_range > 0 else 0

    pcc_min_val, pcc_max_val = np.nanmin(min_pcc_all) - 0.1, np.nanmax(max_pcc_all) + 0.1
    pcc_range = pcc_max_val - pcc_min_val
    pcc_scale_factor = 0.5 / pcc_range if pcc_range > 0 else 0
    pcc_offset = 0.5

    mean_rmse_scaled = mean_rmse_all * rmse_scale_factor
    min_rmse_scaled = min_rmse_all * rmse_scale_factor
    max_rmse_scaled = max_rmse_all * rmse_scale_factor
    obs_rmse_scaled = obs_period_rmse * rmse_scale_factor
    
    mean_pcc_scaled = (mean_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    min_pcc_scaled = (min_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    max_pcc_scaled = (max_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    obs_pcc_scaled = (obs_period_pcc - pcc_min_val) * pcc_scale_factor + pcc_offset

    #violin
    df_rmse_long = rmse_ts.to_dataframe(name='rmse').reset_index()
    df_pcc_long = pcc_ts.to_dataframe(name='pcc').reset_index()
    df_rmse_long['rmse_scaled'] = df_rmse_long['rmse'] * rmse_scale_factor
    df_pcc_long['pcc_scaled'] = (df_pcc_long['pcc'] - pcc_min_val) * pcc_scale_factor + pcc_offset
    model_order = df_sorted.index.tolist()

    #violin plots
    sns.violinplot(data=df_rmse_long, x='model', y='rmse_scaled', order=model_order, ax=ax, color='grey', alpha=0.4, inner=None, saturation=0.7, zorder=1, cut=0)
    sns.violinplot(data=df_pcc_long, x='model', y='pcc_scaled', order=model_order, ax=ax, color='red', alpha=0.4, inner=None, saturation=0.7, zorder=1, cut=0)

    #mean RMSE
    mean_rmse_line_plotted = False
    for i, (x_pos, mean_val) in enumerate(zip(x, mean_rmse_scaled)):
        if not np.isnan(mean_val):
            label = 'Mean RMSE (all sliding windows)' if not mean_rmse_line_plotted else ""
            ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val], color='black', linewidth=2, zorder=3, label=label)
            mean_rmse_line_plotted = True

    #range RMSE
    ax.errorbar(x, mean_rmse_scaled, yerr=[mean_rmse_scaled - min_rmse_scaled, max_rmse_scaled - mean_rmse_scaled], fmt='none', color='black', capsize=6, capthick=1.5, label='RMSE range (all sliding windows)')
    ax.scatter(x, obs_rmse_scaled, color='black', s=80, zorder=5, label='RMSE over observed period')
    
    #mean PCC
    mean_line_plotted = False
    for i, (x_pos, mean_val) in enumerate(zip(x, mean_pcc_scaled)):
        if not np.isnan(mean_val):
            label = 'Mean PCC (all sliding windows)' if not mean_line_plotted else ""
            ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val], color='red', linewidth=2, zorder=3, label=label)
            mean_line_plotted = True
    
    #range PCC
    ax.errorbar(x, mean_pcc_scaled, yerr=[mean_pcc_scaled - min_pcc_scaled, max_pcc_scaled - mean_pcc_scaled], fmt='none', color='red', capsize=6, capthick=1.5, label='PCC range (all sliding windows)')
    ax.scatter(x, obs_pcc_scaled, color='red', s=80, zorder=5, label='PCC over observed period')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax2 = ax.twinx()
    ax2.set_ylim(y_min, y_max)
    
    locator = mticker.MaxNLocator(nbins=5, prune='both')
    rmse_tick_values = locator.tick_values(rmse_min_val, rmse_max_val)
    ax.set_yticks(rmse_tick_values * rmse_scale_factor)
    ax.set_yticklabels([f'{v:.1f}' for v in rmse_tick_values])
    ax.set_ylabel(f'RMSE ({unit})', fontsize=12, color='black', y=0.25)
    ax.tick_params(axis='y', labelcolor='black')
    
    locator = mticker.MaxNLocator(nbins=5, prune='both')
    pcc_tick_values = locator.tick_values(pcc_min_val, pcc_max_val)
    ax2.set_yticks((pcc_tick_values - pcc_min_val) * pcc_scale_factor + pcc_offset)
    ax2.set_yticklabels([f'{v:.1f}' for v in pcc_tick_values])
    ax2.set_ylabel('PCC', fontsize=12, color='red', y=0.75)
    ax2.tick_params(axis='y', labelcolor='red')

    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)
    separator_pos = len(df_sorted) - 3.5
    ax.axvline(separator_pos, color='gray', linestyle='--', ymin=-0.17, clip_on=False)
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=90, ha='center', va='top', fontsize=10)
    ax.set_xlabel('Models')
    ax.set_xlim(-0.5, len(model_names) - 0.5)
    ax.set_title(f'Model-observation ODSL comparison ({cfg["name"]})\nMean statistics over all {window_size}-yr sliding windows', fontsize=14, pad=20, fontweight='bold')
    
    for tick in ax.get_yticks(): ax.axhline(y=tick, color='gray', linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)
    for tick in ax2.get_yticks(): ax.axhline(y=tick, color='gray', linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)

    #ordered legend
    handles, labels = ax.get_legend_handles_labels()

    rmse_violin_patch = mpatches.Patch(color='grey', alpha=0.5, label='RMSE distribution (all sliding windows)')
    pcc_violin_patch = mpatches.Patch(color='red', alpha=0.5, label='PCC distribution (all sliding windows)')
    handles.extend([rmse_violin_patch, pcc_violin_patch])
    labels.extend(['RMSE distribution (all sliding windows)', 'PCC distribution (all sliding windows)'])

    desired_order = [
    'PCC over observed period',
    'Mean PCC (all sliding windows)',
    'PCC range (all sliding windows)',
    'PCC distribution (all sliding windows)',
    'RMSE over observed period',
    'Mean RMSE (all sliding windows)',
    'RMSE range (all sliding windows)',
    'RMSE distribution (all sliding windows)'
    ]

    label_handle_map = dict(zip(labels, handles))
    reordered_handles = [label_handle_map[label] for label in desired_order if label in label_handle_map]
    reordered_labels = [label for label in desired_order if label in label_handle_map]
    ax.legend(reordered_handles, reordered_labels, loc='upper left', bbox_to_anchor=(0, 0.5), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'model_comparison_summary_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_observed_variability(obs_results, fig_dir):
    """Plot the observed ODSL variability."""
    
    cfg = PLOT_CONFIG['variability']

    print("Plotting observed ODSL variability...")
    
    #projection
    proj = ccrs.AlbersEqualArea(
        central_longitude=PROJECTION_PARAMS['central_longitude'],
        central_latitude=PROJECTION_PARAMS['central_latitude'],
        standard_parallels=PROJECTION_PARAMS['standard_parallels']
    )
    
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': proj})
    add_map_features(ax, EXTENT, is_left=True, is_bottom=True)
    
    #data
    variability = obs_results['variability']
    common_years = np.array(obs_results.attrs['common_years_list'])

    vmax = variability.quantile(0.98, skipna=True).item()
    vmin = variability.quantile(0.02, skipna=True).item()

    #plotting
    mesh = variability.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin, vmax=vmax, add_colorbar=False)
    
    #colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)
    cbar.set_label(f'ODSL {cfg["name"]} ({cfg["units"]})', fontsize=10)
    
    #statistics
    region_mask = create_region_mask(variability, EXTENT)
    stats_obs = calculate_weighted_stats(variability, region_mask)
    
    #title
    ax.set_title(f'Observed ODSL {cfg["name"]} ({common_years.min()}-{common_years.max()})\n' f'Mean: {stats_obs["mean_x"]:.2f} {cfg["units"]}, RMS: {stats_obs["std_x"]:.2f} {cfg["units"]}', fontsize=12, pad=15, fontweight='bold')
    
    #save figure
    plt.savefig(os.path.join(fig_dir, f'Observed_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_scenario_comparison(scenario_results, fig_dir):
    """Plot timeseries comparison of CMIP5 and CMIP6 ensemble scenarios."""

    print("Plotting CMIP scenario timeseries comparison...")

    if not scenario_results or "cmip_version" not in scenario_results.coords:
        print("No valid scenario results to plot.")
        return

    #plotting
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), sharex=True)
    gs = axes[0, 0].get_gridspec()
    gs.update(hspace=0.3, wspace=0.12)
    ax_trend_cmip5, ax_trend_cmip6 = axes[0, 0], axes[0, 1]
    ax_var_cmip5, ax_var_cmip6 = axes[1, 0], axes[1, 1]

    axes_map = {"CMIP5": {"trend": ax_trend_cmip5, "var": ax_var_cmip5}, "CMIP6": {"trend": ax_trend_cmip6, "var": ax_var_cmip6}}

    #scenario colors and labels
    scenario_colors = {'historical': 'black', 'rcp26': '#377eb8', 'ssp126': '#377eb8', 'rcp45': '#4daf4a', 'ssp245': '#4daf4a', 'rcp85': '#e41a1c', 'ssp585': '#e41a1c'}     
    scenario_labels = {'rcp26': 'RCP2.6 ± 1 std. dev.', 'ssp126': 'SSP1-2.6 ± 1 std. dev.', 'rcp45': 'RCP4.5 ± 1 std. dev.', 'ssp245': 'SSP2-4.5 ± 1 std. dev.', 'rcp85': 'RCP8.5 ± 1 std. dev.', 'ssp585': 'SSP5-8.5 ± 1 std. dev.', 'historical': 'Historical ± 1 std. dev.'}

    cmip5_data = scenario_results.sel(cmip_version="CMIP5")
    cmip6_data = scenario_results.sel(cmip_version="CMIP6")
    
    #y-axis limits
    trend_min = min(((cmip5_data.ensemble_mean - cmip5_data.ensemble_std) / 10).min(), ((cmip6_data.ensemble_mean - cmip6_data.ensemble_std) / 10).min())
    trend_max = max(((cmip5_data.ensemble_mean + cmip5_data.ensemble_std) / 10).max(), ((cmip6_data.ensemble_mean + cmip6_data.ensemble_std) / 10).max())
    var_min = 0
    var_max = max((cmip5_data.ensemble_std / 10).max(), (cmip6_data.ensemble_std / 10).max())

    for cmip_version, data in [("CMIP5", cmip5_data), ("CMIP6", cmip6_data)]:

        start_year = data.year.min().item()
        end_year = data.year.max().item()

        ax_trend = axes_map[cmip_version]["trend"]
        ax_var = axes_map[cmip_version]["var"]
        
        historical_end_year = 2005 if cmip_version == "CMIP5" else 2014

        hist_data = data.sel(scenario='historical', year=slice(None, historical_end_year)).dropna(dim='year')
        
        ax_trend.plot(hist_data.year, hist_data.ensemble_mean / 10, color='black', linewidth=2.5, label='Historical ± 1 std. dev.', zorder=10)
        ax_trend.fill_between(hist_data.year, (hist_data.ensemble_mean - hist_data.ensemble_std) / 10, (hist_data.ensemble_mean + hist_data.ensemble_std) / 10, color='black', alpha=0.2, zorder=5)
        ax_var.plot(hist_data.year, hist_data.ensemble_std / 10, color='black', linewidth=2.5, label='Historical ± 1 std. dev.', zorder=10)
        ax_var.fill_between(hist_data.year, 0, hist_data.ensemble_std / 10, color='black', alpha=0.1, zorder=4)

        for scenario in data.scenario.values:
            if scenario == 'historical':
                continue
            
            future_data = data.sel(scenario=scenario, year=slice(historical_end_year + 1, None)).dropna(dim='year')
            
            if future_data.year.size > 0:
                color = scenario_colors.get(scenario, 'grey')
                label = scenario_labels.get(scenario, scenario.upper())

                #last point of historical data
                last_historical_point = hist_data.isel(year=-1)
                
                #prepend to the future data
                connected_data = xr.concat([last_historical_point, future_data], dim='year')
                
                #connected data for plotting
                ax_trend.plot(connected_data.year, connected_data.ensemble_mean / 10, color=color, linewidth=2, label=label, alpha=0.8)
                ax_trend.fill_between(connected_data.year, (connected_data.ensemble_mean - connected_data.ensemble_std) / 10, (connected_data.ensemble_mean + connected_data.ensemble_std) / 10, color=color, alpha=0.15)
                ax_var.plot(connected_data.year, connected_data.ensemble_std / 10, color=color, linewidth=2, label=label, alpha=0.8)
                ax_var.fill_between(connected_data.year, 0, connected_data.ensemble_std / 10, color=color, alpha=0.1)

        #formatting
        ax_trend.set_title(f'{cmip_version} regional mean sea level', fontsize=14)
        ax_var.set_title(f'{cmip_version} ensemble variability', fontsize=14)
        
        for ax in [ax_trend, ax_var]:
            ax.grid(True, alpha=0.3)
            ax.set_xlim(start_year, end_year)

        #legend
        handles, labels = ax_trend.get_legend_handles_labels()
        observed_patch = mpatches.Patch(color='red', alpha=0.2, label='Observed period')
        handles.append(observed_patch)
        ax_trend.legend(handles=handles, loc='upper left')

    for ax in axes.flat:
        ax.axvline(START_YEAR, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.axvline(END_YEAR, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.axvspan(START_YEAR, END_YEAR, color='red', alpha=0.2, zorder=0)

    #labels and limits
    ax_trend_cmip5.set_ylabel('Sea level anomaly (cm)', fontsize=12)
    ax_var_cmip5.set_ylabel('Variability (cm)', fontsize=12)
    ax_var_cmip5.set_xlabel('Year', fontsize=12)
    ax_var_cmip6.set_xlabel('Year', fontsize=12)
    
    #y-axis limits
    ax_trend_cmip5.set_ylim(trend_min * 1.05, trend_max * 1.05)
    ax_trend_cmip6.set_ylim(trend_min * 1.05, trend_max * 1.05)
    ax_var_cmip5.set_ylim(var_min, var_max * 1.05)
    ax_var_cmip6.set_ylim(var_min, var_max * 1.05)
    ax_trend_cmip6.tick_params(axis='y', labelleft=False)
    ax_var_cmip6.tick_params(axis='y', labelleft=False)

    plt.suptitle(f'ODSL CMIP scenario comparison\nNorth Atlantic region ({EXTENT[0]}°E-{EXTENT[1]}°E, {EXTENT[2]}°N-{EXTENT[3]}°N)', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, 'cmip_scenario_timeseries_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_lowess_fit(lowess_results_df, fig_dir):
    """Scatter plot of observed vs. modeled data with a LOWESS fit line."""
    
    cfg = PLOT_CONFIG['variability']

    print("Plotting LOWESS fit...")

    #get data
    obs_points = lowess_results_df['obs_points'].dropna().values
    model_points = lowess_results_df['model_points'].dropna().values
    lowess_df = lowess_results_df[['x_fit', 'y_fit']].dropna()
    frac = lowess_results_df.attrs.get('frac', 'N/A')

    #plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(obs_points, model_points, alpha=0.3, s=10, color='gray', label='Grid-point comparison')
    ax.plot(lowess_df['x_fit'], lowess_df['y_fit'], color='red', linewidth=3, label=f'LOWESS Fit (frac={frac})')
    
    #1:1 reference line
    lim_min = min(np.min(obs_points), np.min(model_points)) * 0.9
    lim_max = max(np.max(obs_points), np.max(model_points)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=2, label='1:1 Line')
    
    #formatting
    ax.set_xlabel(f'Observed {cfg["name"]} ({cfg["units"]})', fontsize=12)
    ax.set_ylabel(f'Modeled {cfg["name"]} ({cfg["units"]})', fontsize=12)
    ax.set_title(f'Modeled vs. observed {cfg["name"]}\nGrid-point comparison with LOWESS smoothing', fontsize=14, fontweight='bold')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    filename = os.path.join(fig_dir, f'lowess_fit_{cfg["name"]}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"LOWESS plot saved to: {filename}")
    plt.show()

def plot_lowess_residuals_spatially(sliding_results, cmip_results, lowess_results_df, fig_dir):
    """Plots the spatial distribution of the residuals from the LOWESS fit."""
    
    cfg = PLOT_CONFIG['variability']

    print("Plotting LOWESS fit residuals spatially...")

    #data arrays
    obs_data = sliding_results['odsl_var_obs_regridded']
    model_data = cmip_results['model_mean_variability']
    
    #flatten and mask
    x_flat = obs_data.values.flatten()
    y_flat = model_data.values.flatten()
    valid_mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)

    #smoothed fit line
    lowess_df = lowess_results_df[['x_fit', 'y_fit']].dropna()
    
    #interpolation
    fit_values = np.interp(
        x=lowess_results_df['obs_points'].dropna().values,
        xp=lowess_df['x_fit'],
        fp=lowess_df['y_fit']
    )
    
    #residual (model value - expected trend value)
    residuals_1d = lowess_results_df['model_points'].dropna().values - fit_values

    #empty 2D array and filling it with the residuals at the correct locations
    residuals_2d_flat = np.full(x_flat.shape, np.nan)
    residuals_2d_flat[valid_mask] = residuals_1d
    residuals_map = xr.DataArray(residuals_2d_flat.reshape(obs_data.shape), coords=obs_data.coords,name='lowess_residual')

    #plotting
    proj = ccrs.AlbersEqualArea(
        central_longitude=PROJECTION_PARAMS['central_longitude'],
        central_latitude=PROJECTION_PARAMS['central_latitude'],
        standard_parallels=PROJECTION_PARAMS['standard_parallels']
    )
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': proj})
    add_map_features(ax, EXTENT, is_left=True, is_bottom=True)

    #diverging colormap and a symmetric color scale
    vmax = abs(residuals_map.quantile(0.98, skipna=True).item())
    mesh = residuals_map.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)
    cbar.set_label(f'LOWESS fit residual ({cfg["units"]})', fontsize=10)
    
    ax.set_title(f'Spatial pattern of LOWESS fit residuals\n(Model variability - expected trend)', fontsize=12, pad=15, fontweight='bold')
    
    filename = os.path.join(fig_dir, f'lowess_residuals_spatial_{cfg["name"]}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
