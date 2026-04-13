"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Plotting functions for ODSL analysis.
"""

from utils import calculate_weighted_stats, create_region_mask, add_map_features, compute_field_significance, make_aligned_levels
from data_loader import load_climate_indices_dict
from config import (CMIP_VERSION, START_YEAR, END_YEAR, EXTENT, PROJECTION_PARAMS, PLOT_VARIABLE, PLOT_CONFIG, EOF_N_MODES, PLOT_ALL_VARIABLES, USE_ROTATED_EOF, N_REALIZATIONS_MONTE_CARLO, N_MODES_OBSERVED, KNMI_DUTCH_COAST, NA_REGIONS)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import os
import pandas as pd
import warnings
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle, FancyArrow
from matplotlib.text import Text
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, TwoSlopeNorm, LightSource
from scipy import stats
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from scipy import signal
import regionmask
from PIL import Image

warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)

def create_all_figures(obs_results, smoothing_results, cmip_results, cmip5_results, cmip6_results, cmip5_all, cmip6_all, incrementing_window_results, picontrol_incrementing_results, sliding_results, picontrol_results, scenario_results, eof_results, dual_eof_results, correlation_results, fig_dir, spectral_results, wind_results):
    """Generate all figures for the analysis."""

    #general figure directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    #EOF-specific figure directory
    eof_fig_dir = os.path.join(fig_dir, "eof_analysis")
    if not os.path.exists(eof_fig_dir):
        os.makedirs(eof_fig_dir)

    print(f"\nGenerating figures for {PLOT_VARIABLE.upper()}")
    print(f"Figures will be saved in: {fig_dir}")

    #variable-specific figures
    if PLOT_ALL_VARIABLES:
        variables_to_plot = ['ODSL', 'trend', 'variability']
    else:
        variables_to_plot = [PLOT_VARIABLE]

    for plot_var in variables_to_plot:
        #variable-specific figure directory
        variable_fig_dir = os.path.join(fig_dir, plot_var)
        if not os.path.exists(variable_fig_dir):
            os.makedirs(variable_fig_dir)
        
        plot_all_models_overview_best_windows(cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var, metric='pcc')
        plot_all_models_overview_best_windows(cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var, metric='rmse')
        plot_regional_overview(obs_results, cmip_results, variable_fig_dir, plot_var=plot_var)
        # plot_cmip_multimodel_mean(cmip_results, variable_fig_dir, plot_var=plot_var)
        # plot_observed_regridded(sliding_results, variable_fig_dir, plot_var=plot_var)
        plot_observed_vs_modeled(obs_results, cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var)
        plot_observed_vs_best_windows(obs_results, cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var)
        # plot_scatter_comparison(cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var)
        # plot_scatter_comparison_individual_models(cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var)
        plot_all_models_overview(cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var)
        # plot_model_comparison_summary(cmip_results, sliding_results, variable_fig_dir, plot_var=plot_var, picontrol_results=picontrol_results)
        # plot_sliding_window_timeseries(sliding_results, variable_fig_dir, plot_var=plot_var)
        # plot_best_and_worst_matching_periods(sliding_results, variable_fig_dir, plot_var=plot_var)
        plot_incrementing_window_skill(incrementing_window_results, variable_fig_dir, plot_var=plot_var, picontrol_incrementing_results=picontrol_incrementing_results)
        # plot_odsl_trend_bar_comparison(obs_results, cmip_results, variable_fig_dir, plot_var=plot_var)
        if cmip5_all is not None and cmip6_all is not None:
            plot_cmip5_vs_cmip6_scenarios(cmip5_all, cmip6_all, variable_fig_dir, plot_var=plot_var)
        
        if plot_var == 'trend':
            plot_observed_odsl_components(obs_results, variable_fig_dir, wind_results=wind_results)
        elif plot_var == 'variability':
            plot_observed_variability(obs_results, variable_fig_dir)

    #EOF figures
    plot_dual_eof_comparison(dual_eof_results, eof_fig_dir, num_modes_to_plot=10, source='observed')
    plot_spatial_eofs(eof_results, eof_fig_dir, num_modes_to_plot=EOF_N_MODES)
    plot_pc_timeseries(dual_eof_results, eof_fig_dir, num_modes_to_plot=5, normalize=False)
    plot_pc_timeseries(dual_eof_results, eof_fig_dir, num_modes_to_plot=5, normalize=True)
    plot_eof_maps_comparison(eof_results, eof_fig_dir, num_modes=EOF_N_MODES)
    plot_eof_monte_carlo_significance(eof_results, eof_fig_dir)
    plot_obs_vs_mmm_significance(eof_results, eof_fig_dir)
    plot_eof_summary_table(eof_results, correlation_results, eof_fig_dir)
    plot_spectral_analysis(spectral_results, eof_fig_dir, n_modes=EOF_N_MODES)
    export_eof_results_to_csv(eof_results, correlation_results, eof_fig_dir)
    for mx, my in [(0, 1), (0, 2), (1, 2)]:
        plot_correlation_biplot(eof_results, correlation_results, eof_fig_dir, mode_x=mx, mode_y=my, normalize=True)
        plot_correlation_biplot(eof_results, correlation_results, eof_fig_dir, mode_x=mx, mode_y=my, normalize=False)

    #general figures
    # plot_yearly_odsl_anomaly(obs_results, fig_dir)
    plot_scenario_timeseries(scenario_results, obs_results, fig_dir)
    plot_smoothing_sensitivity(smoothing_results, fig_dir)
    # plot_3d_odsl_cover_art(obs_results, fig_dir, variable='odsl_mean')

def plot_observed_odsl_components(obs_results, fig_dir, wind_results=None):
    """Plot the components of observed ODSL."""

    print("Plotting observed ODSL components...")
    
    cfg = PLOT_CONFIG['trend']

    #projection
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 10), subplot_kw={'projection': proj})
    plt.subplots_adjust(hspace=-0.05, wspace=0.05)
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    #data
    msl   = obs_results['msl']
    geoid = obs_results['geoid']
    gia   = obs_results['gia']
    odsl  = obs_results['odsl_regional']

    #common years
    common_years = np.array(obs_results.attrs['common_years_list'])
    
    #statistics
    region_mask = create_region_mask(msl, EXTENT)
    stats_msl   = calculate_weighted_stats(msl, region_mask)
    stats_geoid = calculate_weighted_stats(geoid, region_mask)
    stats_gia   = calculate_weighted_stats(gia, region_mask)
    stats_odsl  = calculate_weighted_stats(odsl, region_mask)
    
    #color scale
    vmax                                          = max(abs(msl.quantile(0.98, skipna=True).item()), abs(geoid.quantile(0.98, skipna=True).item()), abs(gia.quantile(0.98, skipna=True).item()), abs(odsl.quantile(0.98, skipna=True).item()))
    levels_main, ticks_main, vmin_comp, vmax_comp = make_aligned_levels(-vmax, vmax)

    #plotting
    im1 = msl.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), cmap=cfg['cmap'],  vmin=vmin_comp, vmax=vmax_comp, levels=levels_main, extend='both', add_colorbar=False)
    add_map_features(ax1, EXTENT, is_left=True, is_bottom=True)
    ax1.set_title(r'$\bf{a)\ MSL\ (altimetry)}$' + f'\nMean: {stats_msl["mean_x"]:.2f} mm/yr | RMS: {stats_msl["std_x"]:.2f} mm/yr')

    im2 = geoid.plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin_comp, vmax=vmax_comp, levels=levels_main, extend='both', add_colorbar=False)
    add_map_features(ax2, EXTENT, is_left=True, is_bottom=True)
    ax2.set_title(r'$\bf{b)\ Contributors\ (steric + barystatic)}$' + f'\nMean: {stats_geoid["mean_x"]:.2f} mm/yr | RMS: {stats_geoid["std_x"]:.2f} mm/yr')

    im3 = gia.plot.contourf(ax=ax3, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin_comp, vmax=vmax_comp, levels=levels_main, extend='both', add_colorbar=False)
    add_map_features(ax3, EXTENT, is_left=True, is_bottom=True)
    ax3.set_title(r'$\bf{c)\ GIA}$' + f'\nMean: {stats_gia["mean_x"]:.2f} mm/yr | RMS: {stats_gia["std_x"]:.2f} mm/yr')

    im4 = odsl.plot.contourf(ax=ax4, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin_comp, vmax=vmax_comp, levels=levels_main, extend='both', add_colorbar=False)
    add_map_features(ax4, EXTENT, is_left=True, is_bottom=True)
    ax4.set_title(r'$\bf{d)\ ODSL\ (a - b - c)}$' + f'\nMean: {stats_odsl["mean_x"]:.2f} mm/yr | RMS: {stats_odsl["std_x"]:.2f} mm/yr')
    
    #wind stress vectors
    if wind_results is not None:
        tau_x_mean_wind = wind_results['tau_x_mean']
        tau_y_mean_wind = wind_results['tau_y_mean']

        lon_min, lon_max, lat_min, lat_max = EXTENT
        tau_x_sub                          = tau_x_mean_wind.sel(longitude=slice(lon_min - 5, lon_max + 5), latitude=slice(lat_min - 5, lat_max + 5))
        tau_y_sub                          = tau_y_mean_wind.sel(longitude=slice(lon_min - 5, lon_max + 5), latitude=slice(lat_min - 5, lat_max + 5))

        skip               = 30
        tx                 = tau_x_sub.values[::skip, ::skip]
        ty                 = tau_y_sub.values[::skip, ::skip]
        lons               = tau_x_sub.longitude.values[::skip]
        lats               = tau_x_sub.latitude.values[::skip]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        ax4.quiver(lon_grid, lat_grid, tx, ty, transform=ccrs.PlateCarree(), scale=3, width=0.002, headwidth=3, headlength=3, color='black', alpha=0.7, zorder=3)
        ax4.add_feature(cfeature.LAND, facecolor='lightgray', zorder=4)
        ax4.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=5)
    
    #colorbar
    cbar_ax = fig.add_axes((0.2, 0.08, 0.6, 0.025))
    cbar    = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_ticks(ticks_main.tolist())
    cbar.set_label('Trend (mm/yr)', fontsize=14)
    
    plt.suptitle(f'Observed ODSL components ({common_years.min()}-{common_years.max()})', fontsize=16, fontweight='bold', y=0.91)
    
    plt.savefig(os.path.join(fig_dir, f'ODSL_components_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_regional_overview(obs_results, cmip_results, fig_dir, plot_var=None):
    """Overview figure of some relevant North Atlantic regions."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Creating North Atlantic regional overview ({cfg['name']})...")

    region_names_ordered = list(NA_REGIONS.keys())
    n_regions            = len(region_names_ordered)

    #mask for square box grid and polygon
    def make_mask(data, region_def):

        lat_n = 'latitude' if 'latitude' in data.dims else 'lat'
        lon_n = 'longitude' if 'longitude' in data.dims else 'lon'
        if 'polygon' in region_def:
            poly = region_def['polygon']
            rgn  = regionmask.Regions([poly], names=['R'], abbrevs=['R'])
            return rgn.mask_3D(data[lon_n], data[lat_n]).sel(region=0)
        else:
            ext                = region_def['extent']
            lons               = data[lon_n].values
            lats               = data[lat_n].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            return xr.DataArray((lon_grid >= ext[0]) & (lon_grid <= ext[1]) & (lat_grid >= ext[2]) & (lat_grid <= ext[3]), coords={lat_n: lats, lon_n: lons}, dims=[lat_n, lon_n])

    #variables
    if plot_var == 'trend':
        obs_map         = obs_results['odsl']
        obs_se_field    = obs_results['odsl_std_error']
        model_stack_key = 'model_trend'
        model_se_key    = 'model_trend_std_error'
        mmm_field       = cmip_results['model_mean_trend']
        mmm_se_field    = cmip_results['model_mean_trend_std_error']
    elif plot_var == 'variability':
        obs_map         = obs_results['variability']
        obs_se_field    = obs_results['variability_std_error']
        model_stack_key = 'model_variability'
        model_se_key    = 'model_variability_std_error'
        mmm_field       = cmip_results['model_mean_variability']
        mmm_se_field    = cmip_results['model_mean_variability_std_error']
    elif plot_var == 'ODSL':
        obs_map         = obs_results['odsl_mean']
        obs_se_field    = obs_results['odsl_mean_std_error']
        model_stack_key = 'model_odsl_mean'
        model_se_key    = 'model_odsl_mean_std_error'
        mmm_field       = cmip_results['model_mean_odsl']
        mmm_se_field    = cmip_results['model_mean_odsl_std_error']
    else:
        raise ValueError(f"Unknown plot_var: {plot_var}")

    model_names = cmip_results.model.values.tolist()

    #bar data per region
    region_bar_data = {}

    for region_name in region_names_ordered:
        region_def = NA_REGIONS[region_name]
        print(f"Computing stats for {region_name}...")
        try:
            #observed
            obs_mask  = make_mask(obs_map, region_def)
            obs_stats = calculate_weighted_stats(obs_map, obs_mask)
            obs_mean  = obs_stats['mean_x']

            lat_obs = 'latitude' if 'latitude' in obs_se_field.dims else 'lat'
            lon_obs = 'longitude' if 'longitude' in obs_se_field.dims else 'lon'
            w_obs   = np.cos(np.deg2rad(obs_se_field[lat_obs]))
            obs_se  = float(obs_se_field.where(obs_mask).weighted(w_obs).mean(dim=[lat_obs, lon_obs], skipna=True).item())

            #individual models
            model_means = []
            model_ses   = []

            for mn in model_names:
                field  = cmip_results[model_stack_key].sel(model=mn)
                m_mask = make_mask(field, region_def)
                s      = calculate_weighted_stats(field, m_mask)
                model_means.append(s['mean_x'])

                if model_se_key is not None:
                    se_f  = cmip_results[model_se_key].sel(model=mn)
                    lat_m = 'latitude' if 'latitude' in se_f.dims else 'lat'
                    lon_m = 'longitude' if 'longitude' in se_f.dims else 'lon'
                    w_m   = np.cos(np.deg2rad(se_f[lat_m]))
                    model_ses.append(float(se_f.where(m_mask).weighted(w_m).mean(dim=[lat_m, lon_m], skipna=True).item()))
                else:
                    model_ses.append(0.0)

            #multi-model mean
            mmm_mask  = make_mask(mmm_field, region_def)
            mmm_stats = calculate_weighted_stats(mmm_field, mmm_mask)
            mmm_mean  = mmm_stats['mean_x']

            lat_mmm = 'latitude' if 'latitude' in mmm_se_field.dims else 'lat'
            lon_mmm = 'longitude' if 'longitude' in mmm_se_field.dims else 'lon'
            w_cmip  = np.cos(np.deg2rad(mmm_se_field[lat_mmm]))
            mmm_se  = float(mmm_se_field.where(mmm_mask).weighted(w_cmip).mean(dim=[lat_mmm, lon_mmm], skipna=True).item())

            region_bar_data[region_name] = {'obs_mean': obs_mean, 'obs_se': obs_se, 'mmm_mean': mmm_mean, 'mmm_se': mmm_se, 'model_means': model_means, 'model_ses': model_ses}

        except Exception as e:
            print(f"Warning: could not compute {region_name}: {e}")

    #csv results, showing which models fall within range of observed
    entries  = ['Observed', 'Multi-Model Mean'] + model_names
    csv_data = {'source': entries}

    for region_name in region_names_ordered:
        if region_name not in region_bar_data:
            csv_data[f'{region_name}_mean']       = [np.nan] * len(entries)
            csv_data[f'{region_name}_se']         = [np.nan] * len(entries)
            csv_data[f'{region_name}_within_obs'] = [np.nan] * len(entries)
            continue

        data   = region_bar_data[region_name]
        obs_lo = data['obs_mean'] - data['obs_se']
        obs_hi = data['obs_mean'] + data['obs_se']

        means_col  = [data['obs_mean'], data['mmm_mean']] + data['model_means']
        ses_col    = [data['obs_se'], data['mmm_se']] + data['model_ses']
        within_col = []

        for m_val, m_se in zip(means_col, ses_col):
            m_lo = m_val - m_se
            m_hi = m_val + m_se
            within_col.append((m_lo <= obs_hi) and (m_hi >= obs_lo))

        #observed always within range
        within_col[0] = True

        csv_data[f'{region_name}_mean']       = [round(v, 3) for v in means_col]
        csv_data[f'{region_name}_se']         = [round(v, 3) for v in ses_col]
        csv_data[f'{region_name}_within_obs'] = within_col

    csv_df   = pd.DataFrame(csv_data)
    csv_path = os.path.join(fig_dir, f'regional_model_selection_{cfg["name"]}_{START_YEAR}_{END_YEAR}.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    #plotting
    bar_cols = 4
    bar_rows = int(np.ceil(n_regions / bar_cols))

    fig = plt.figure(figsize=(22, 8 + bar_rows * 3.5))

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.5, bar_rows], hspace=0.25)

    #overview map (top)
    proj   = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    ax_map = fig.add_subplot(outer[0], projection=proj)

    #bar charts (below)
    inner = gridspec.GridSpecFromSubplotSpec(bar_rows, bar_cols, subplot_spec=outer[1], wspace=0.3, hspace=0.5)

    bar_axes = {}
    for idx, region_name in enumerate(region_names_ordered):
        row                   = idx // bar_cols
        col                   = idx % bar_cols
        bar_axes[region_name] = fig.add_subplot(inner[row, col])

    #unused bar slots empty
    for idx in range(n_regions, bar_rows * bar_cols):
        row      = idx // bar_cols
        col      = idx % bar_cols
        ax_empty = fig.add_subplot(inner[row, col])
        ax_empty.axis('off')

    #overview map
    add_map_features(ax_map, EXTENT, is_left=True, is_bottom=True)

    if plot_var == 'variability':
        vmax = obs_map.quantile(0.98, skipna=True).item()
        vmin = obs_map.min(skipna=True).item()
    else:
        vmax = abs(obs_map.quantile(0.98, skipna=True).item())
        vmin = -vmax

    levels_main, ticks_main, vmin, vmax = make_aligned_levels(vmin, vmax, force_zero_min=(plot_var == 'variability'))
    extend_main                         = 'max' if plot_var == 'variability' else 'both'

    mesh = obs_map.plot.contourf(ax=ax_map, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin, vmax=vmax, add_colorbar=False, levels=levels_main, extend=extend_main)

    #200m bathymnetry line
    try:
        shelf_200m = cfeature.NaturalEarthFeature('physical', 'bathymetry_K_200', '10m')
        ax_map.add_feature(shelf_200m, facecolor='none', edgecolor='darkgray', linewidth=0.8) # type: ignore[attr-defined]
    except Exception:
        print("200m bathymetry line unsuccesful")

    #subregion boxes and polygons
    for region_name, region_def in NA_REGIONS.items():
        if region_name == 'North Atlantic':
            continue
        color = region_def['color']
        if 'polygon' in region_def:
            poly      = region_def['polygon']
            poly_lons = [p[0] for p in poly] + [poly[0][0]]
            poly_lats = [p[1] for p in poly] + [poly[0][1]]
            ax_map.plot(poly_lons, poly_lats, transform=ccrs.PlateCarree(), color=color, linewidth=2.0, zorder=6)
            ax_map.fill(poly_lons, poly_lats, transform=ccrs.PlateCarree(), color=color, alpha=0.15, zorder=4)
        else:
            ext      = region_def['extent']
            box_lons = [ext[0], ext[1], ext[1], ext[0], ext[0]]
            box_lats = [ext[2], ext[2], ext[3], ext[3], ext[2]]
            ax_map.plot(box_lons, box_lats, transform=ccrs.PlateCarree(), color=color, linewidth=2.0, zorder=5)

    #colorbar
    pos_map = ax_map.get_position()
    cbar_ax = fig.add_axes((pos_map.x0 + 0.05, pos_map.y0 - 0.03, pos_map.width - 0.1, 0.012))
    cbar    = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', extend=extend_main)
    cbar.set_ticks(ticks_main.tolist())
    cbar.set_label(f'Observed {cfg["name"]} ({cfg["units"]})', fontsize=11)

    #legend
    map_legend_handles = []
    for region_name, region_def in NA_REGIONS.items():
        if 'polygon' in region_def:
            map_legend_handles.append(mpatches.Patch(facecolor=region_def['color'], alpha=0.15, edgecolor=region_def['color'], linewidth=2, label=region_name))
        else:
            map_legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=region_def['color'], linewidth=2, label=region_name))
    map_legend_handles.append(Line2D([0], [0], color='darkgray', linewidth=1.0, label='200m bathymetry'))

    ax_map.legend(handles=map_legend_handles, loc='center left', bbox_to_anchor=(-0.25, 0.5), fontsize=16, framealpha=0.9, ncol=1)

    ax_map.set_title(f'Observed ODSL {cfg["name"]} ({START_YEAR}-{END_YEAR})', fontsize=13, fontweight='bold', pad=10)

    #bar subplots lower half
    for region_name in region_names_ordered:
        if region_name not in region_bar_data:
            continue
        ax    = bar_axes[region_name]
        data  = region_bar_data[region_name]
        color = NA_REGIONS[region_name]['color']

        sorted_idx   = np.argsort(data['model_means'])
        mm_sorted    = [data['model_means'][i] for i in sorted_idx]
        se_sorted    = [data['model_ses'][i] for i in sorted_idx]
        names_sorted = [model_names[i] for i in sorted_idx]

        x_labels = ['Observed', 'Multi-model mean'] + names_sorted
        y_vals   = [data['obs_mean'], data['mmm_mean']] + mm_sorted
        y_errs   = [data['obs_se'], data['mmm_se']] + se_sorted
        x_pos    = np.arange(len(y_vals))
        n_p      = 2

        #individual models
        ax.errorbar(x_pos[n_p:], y_vals[n_p:], yerr=se_sorted, fmt='o', capsize=2, c='#164273', markersize=3, markeredgecolor='#164273', markeredgewidth=0.3, zorder=2)
        #observed
        ax.errorbar(x_pos[0], y_vals[0], yerr=y_errs[0], fmt='o', capsize=2, c='green', markersize=4, zorder=3)
        #multi-model mean
        ax.errorbar(x_pos[1], y_vals[1], yerr=y_errs[1], fmt='o', capsize=2, c='black', markersize=4, zorder=3)

        #top and lower observed reference lines
        ax.axhline(y=data['obs_mean'] + data['obs_se'], color='grey', ls='--', lw=0.6)
        ax.axhline(y=data['obs_mean'] - data['obs_se'], color='grey', ls='--', lw=0.6)
        ax.axhline(y=0, color='grey', ls='-', lw=0.4, alpha=0.5)

        #x-axis
        ax.set_xlim(-0.5, len(y_vals) - 0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=5, rotation=90, ha='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

        #title
        ax.set_title(region_name, fontsize=9, fontweight='bold', color=color, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

        if sorted_idx is not None:
            ax.set_ylabel(f'{cfg["units"]}', fontsize=7)

    #legend bar charts
    bar_legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=7, label=r'Observed ($\pm$ 1 SE)'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=7, label=r'Multi-model mean ($\pm$ 1 SE)'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='#164273', markersize=7, label=r'CMIP models ($\pm$ 1 SE)')]

    #legend location
    empty_start = n_regions
    if empty_start < bar_rows * bar_cols:
        row = empty_start // bar_cols
        col = empty_start % bar_cols
        ax_leg = fig.add_subplot(inner[row, col])
        ax_leg.axis('off')
        ax_leg.legend(handles=bar_legend_handles, loc='center', fontsize=9, frameon=False)
    else:
        last_ax = bar_axes[region_names_ordered[0]]
        last_ax.legend(handles=bar_legend_handles, fontsize=7, loc='upper left')

    output_path = os.path.join(fig_dir, f'regional_overview_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
    
def plot_odsl_trend_bar_comparison(obs_results, cmip_results, fig_dir, plot_var=None):
    """ODSL North Atlantic trend bar over altimetry era for observed, multi-model mean, and individual models."""

    print("Plotting ODSL bar comparison...")

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    #variables
    if plot_var == 'trend':
        obs_field = obs_results['odsl']
        obs_se_field = obs_results['odsl_std_error']
        model_means_da = cmip_results['trend_mean']
        model_se_key = 'model_trend_std_error'
        mmm_field = cmip_results['model_mean_trend']
        mmm_se_field = cmip_results['model_mean_trend_std_error']

    elif plot_var == 'variability':
        obs_field = obs_results['variability']
        obs_se_field = obs_results['variability_std_error']
        model_means_da = None 
        model_se_key = 'model_variability_std_error'
        mmm_field = cmip_results['model_mean_variability']
        mmm_se_field = cmip_results['model_mean_variability_std_error']

    elif plot_var == 'ODSL':
        obs_field = obs_results['odsl_mean']
        obs_se_field = obs_results['odsl_mean_std_error']
        model_means_da = None
        model_se_key = 'model_odsl_mean_std_error'
        mmm_field = cmip_results['model_mean_odsl']
        mmm_se_field = cmip_results['model_mean_odsl_std_error']

    else:
        raise ValueError(f"Unknown plot_var: {plot_var}. Must be 'trend', 'variability', or 'ODSL'")

    #observed
    obs_region_mask = create_region_mask(obs_field, EXTENT)
    obs_stats = calculate_weighted_stats(obs_field, obs_region_mask)
    obs_mean = obs_stats['mean_x']

    lat_obs = 'latitude' if 'latitude' in obs_se_field.dims else 'lat'
    lon_obs = 'longitude' if 'longitude' in obs_se_field.dims else 'lon'
    w_obs = np.cos(np.deg2rad(obs_se_field[lat_obs]))
    obs_se = float(obs_se_field.weighted(w_obs).mean(dim=[lat_obs, lon_obs], skipna=True).item())

    #individual means and SE
    model_names = cmip_results.model.values.tolist()

    #compute if necessary
    if model_means_da is not None:
        model_means = model_means_da.values.tolist()
    else:
        if plot_var == 'variability':
            model_stack = cmip_results['model_variability']
        elif plot_var == 'ODSL':
            model_stack = cmip_results['model_odsl_mean']

        model_means = []
        for model_name in model_names:
            field = model_stack.sel(model=model_name) # type: ignore
            mask = create_region_mask(field, EXTENT)
            s = calculate_weighted_stats(field, mask)
            model_means.append(s['mean_x'])

    model_ses = []
    if model_se_key is not None:
        for model_name in model_names:
            se_field = cmip_results[model_se_key].sel(model=model_name)
            w_m = np.cos(np.deg2rad(se_field.latitude))
            model_ses.append(float(se_field.weighted(w_m).mean(dim=['latitude', 'longitude'], skipna=True).item()))
    else:
        #no error bar if no SE
        model_ses = [0.0] * len(model_names)

    #multi-model mean
    mmm_region_mask = create_region_mask(mmm_field, EXTENT)
    mmm_stats = calculate_weighted_stats(mmm_field, mmm_region_mask)
    mmm_mean = mmm_stats['mean_x']

    w_cmip = np.cos(np.deg2rad(mmm_se_field.latitude))
    mmm_se = float(mmm_se_field.weighted(w_cmip).mean(dim=['latitude', 'longitude'], skipna=True).item())

    #sort individual models by avg
    sorted_idx = np.argsort(model_means)
    model_names_sorted = [model_names[i] for i in sorted_idx]
    model_means_sorted = [model_means[i] for i in sorted_idx]
    model_ses_sorted   = [model_ses[i]   for i in sorted_idx]

    #labels and values
    x_labels = ['Observed', 'Multi-model\nmean'] + model_names_sorted
    y_values = [obs_mean, mmm_mean] + model_means_sorted
    y_errors = [obs_se, mmm_se] + model_ses_sorted
    n_priority = 2
    x_pos = np.arange(len(x_labels))

    #plotting
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.4), 5))

    #individual CMIP models
    ax.errorbar(x_pos[n_priority:], y_values[n_priority:], yerr=model_ses_sorted, fmt='o', capsize=3, c='#164273', markeredgecolor='#164273', markeredgewidth=0.5, zorder=2, label=r'CMIP models ($\pm$ 1 SE)')

    #multi-model mean
    ax.errorbar(x_pos[1], y_values[1], yerr=y_errors[1], fmt='o', capsize=3, c='black', markersize=6, zorder=3, label=r'Multi-model mean ($\pm$ 1 SE)')

    #observed
    ax.errorbar(x_pos[0], y_values[0], yerr=y_errors[0], fmt='o', capsize=3, c='green', markersize=6, zorder=3, label=r'Observed ($\pm$ 1 SE)')

    #horizontal lines \pm 1 SE of observed
    ax.axhline(y=obs_mean + obs_se, color='grey', linestyle='--', linewidth=0.8)
    ax.axhline(y=obs_mean - obs_se, color='grey', linestyle='--', linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=90)

    #background box for individual models within \pm 1 SE of observed
    obs_lo = obs_mean - obs_se
    obs_hi = obs_mean + obs_se
    bbox_props = dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="none", alpha=0.6)
    for i, label in enumerate(ax.get_xticklabels()):
        if i < 2:
            #observed and multi-model mean bold
            label.set_fontweight('bold')
        else:
            #individual models within observed range background box
            m_lo = y_values[i] - y_errors[i]
            m_hi = y_values[i] + y_errors[i]
            if m_lo <= obs_hi and m_hi >= obs_lo:
                label.set_bbox(bbox_props)

    #separator line multi-model mean and individual models
    ax.axvline(x=n_priority - 0.5, color='gray', linestyle='--', linewidth=0.8)

    #other layout
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.grid(axis='y', alpha=0.5)
    ax.set_ylabel(f'{cfg["name"]} ({cfg["units"]})')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f'Regional mean ODSL {cfg["name"]} ({START_YEAR}-{END_YEAR})', fontsize=12, fontweight='bold')
    fig.tight_layout()

    output_path = os.path.join(fig_dir, f'odsl_bar_comparison_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_cmip_multimodel_mean(cmip_results, fig_dir, plot_var=None):
    """Plot CMIP multi-model mean ODSL."""

    print("Plotting CMIP multi-model mean...")
    
    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]
    
    if plot_var == 'trend':
        data_to_plot = cmip_results['model_mean_trend']
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = -vmax
    elif plot_var == 'variability':
        data_to_plot = cmip_results['model_mean_variability']
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = 0
    elif plot_var == 'ODSL':
        data_to_plot = cmip_results['model_mean_odsl']
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = -vmax
    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    valid_models_count = cmip_results.attrs['valid_models_count']
    
    #projection
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': proj})
    
    add_map_features(ax, EXTENT, is_left=True, is_bottom=True)

    mesh = data_to_plot.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin, vmax=vmax, add_colorbar=False, levels=50)
    
    #colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)

    #dynamic ticks based on data range
    data_range = vmax - vmin
    if data_range < 2:
        n_ticks = 7
        tick_values = np.linspace(vmin, vmax, n_ticks)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])
    else:
        cbar.set_ticks(np.arange(np.ceil(vmin), np.floor(vmax) + 1))
    
    cbar.set_label(f'ODSL {cfg["name"]} ({cfg["units"]})', fontsize=10)
    
    #regional statistics
    region_mask = create_region_mask(data_to_plot, EXTENT)
    stats_model = calculate_weighted_stats(data_to_plot, region_mask)
    
    ax.set_title(f'CMIP multi-model mean ({valid_models_count} models)\nODSL {cfg["name"]} ({START_YEAR}-{END_YEAR}), Mean: {stats_model["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_model["std_x"]:.2f} {cfg["units"]}', fontsize=12, pad=15, fontweight='bold')
    
    plt.savefig(os.path.join(fig_dir, f'CMIP_multimodel_mean_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_observed_regridded(sliding_results, fig_dir, plot_var=None):
    """Plot the observed ODSL after regridding to the CMIP grid."""

    print("Plotting observed ODSL (regridded to CMIP grid)...")
    
    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]
    
    if plot_var == 'trend':
        data_to_plot = sliding_results['odsl_mm_yr_regridded']
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = -vmax
    elif plot_var == 'ODSL':
        data_to_plot = sliding_results['odsl_obs_regridded']
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = -vmax
    elif plot_var == 'variability':
        data_to_plot = sliding_results['odsl_var_obs_regridded']
        vmax = abs(data_to_plot.quantile(0.98, skipna=True).item())
        vmin = 0
    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}. Must be 'trend', 'variability', or 'ODSL'")

    #projection
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': proj})
    
    add_map_features(ax, EXTENT, is_left=True, is_bottom=True)

    mesh = data_to_plot.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin, vmax=vmax, add_colorbar=False, levels=50)
    
    #colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)

    #dynamic ticks based on data range
    data_range = vmax - vmin
    if data_range < 1:
        #small values decimal ticks
        tick_step = data_range / 5
        magnitude = 10 ** np.floor(np.log10(tick_step))
        tick_step = np.ceil(tick_step / magnitude) * magnitude
        tick_values = np.arange(np.ceil(vmin / tick_step) * tick_step, vmax + tick_step/2, tick_step)
        cbar.set_ticks(tick_values)
    else:
        #larger values integer ticks
        tick_values = np.arange(np.ceil(vmin), np.floor(vmax) + 1)
        cbar.set_ticks(tick_values)

    cbar.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=10)
    
    #regional statistics
    region_mask = create_region_mask(data_to_plot, EXTENT)
    stats_obs = calculate_weighted_stats(data_to_plot, region_mask)
    
    ax.set_title(f'Observed ODSL ({cfg["name"]}), regridded to CMIP grid\n({START_YEAR}-{END_YEAR}) Mean: {stats_obs["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_obs["std_x"]:.2f} {cfg["units"]}', fontsize=12, pad=15, fontweight='bold')
    
    plt.savefig(os.path.join(fig_dir, f'Observed_regridded_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cmip5_vs_cmip6_scenarios(cmip5_all, cmip6_all, fig_dir, plot_var=None):
    """Plot CMIP5 vs CMIP6 multi-model mean for low/medium/high scenarios (3x3 grid)."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Plotting CMIP5 vs CMIP6 scenario comparison ({cfg['name']})...")

    #scenario pairs
    scenario_pairs = [('Low',    'rcp26',  'ssp126'),
                      ('Medium', 'rcp45',  'ssp245'),
                      ('High',   'rcp85',  'ssp585')]

    CMIP5_LABELS = {'rcp26': 'RCP2.6', 'rcp45': 'RCP4.5', 'rcp85': 'RCP8.5'}
    CMIP6_LABELS = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}

    if plot_var == 'trend':
        mmm_key = 'model_mean_trend'
        stack_key = 'model_trend'
        p_key = 'model_mean_trend_p_value'
    elif plot_var == 'variability':
        mmm_key = 'model_mean_variability'
        stack_key = 'model_variability'
        p_key = None
    elif plot_var == 'ODSL':
        mmm_key = 'model_mean_odsl'
        stack_key = 'model_odsl_mean'
        p_key = 'model_mean_odsl_p_value'
    else:
        raise ValueError(f"Unknown plot_var: {plot_var}")

    #data per scenario pair
    rows_data   = []
    global_vmax = 0

    for label, c5_key, c6_key in scenario_pairs:
        c5 = cmip5_all[c5_key]
        c6 = cmip6_all[c6_key]

        c5_data  = c5[mmm_key]
        c6_data  = c6[mmm_key]
        c5_stack = c5[stack_key]
        c6_stack = c6[stack_key]
        c5_pval  = c5.get(p_key) if p_key else None
        c6_pval  = c6.get(p_key) if p_key else None

        diff = c6_data - c5_data

        #global color limits
        if plot_var == 'variability':
            local_max = max(c5_data.quantile(0.98, skipna=True).item(), c6_data.quantile(0.98, skipna=True).item())
        else:
            local_max = max(abs(c5_data.quantile(0.98, skipna=True).item()), abs(c6_data.quantile(0.98, skipna=True).item()))
        global_vmax = max(global_vmax, local_max)

        rows_data.append({'label': label, 'c5_key': c5_key, 'c6_key': c6_key, 'c5_data': c5_data, 'c6_data': c6_data, 'diff': diff, 'c5_stack': c5_stack, 'c6_stack': c6_stack, 'c5_pval': c5_pval, 'c6_pval': c6_pval, 'n5': len(c5.model), 'n6': len(c6.model)})

    #global color limits
    if plot_var == 'variability':
        vmin_main = 0
        vmax_main = global_vmax
    else:
        vmax_main = global_vmax
        vmin_main = -vmax_main

    #diff color limits
    all_diff_max = max(abs(r['diff'].quantile(0.98, skipna=True).item()) for r in rows_data)
    if all_diff_max == 0:
        all_diff_max = 1.0
    vmax_diff = all_diff_max
    vmin_diff = -all_diff_max

    cmap_main = 'coolwarm'
    cmap_diff = 'PiYG'

    levels_main, ticks_main, vmin_main, vmax_main = make_aligned_levels(vmin_main, vmax_main, force_zero_min=(plot_var == 'variability'))
    levels_diff, ticks_diff, vmin_diff, vmax_diff = make_aligned_levels(vmin_diff, vmax_diff)
    extend_main                                   = 'max' if plot_var == 'variability' else 'both'

    #plotting
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(22, 15), subplot_kw={'projection': proj})

    mesh1 = None
    mesh3 = None
    for row_idx, rd in enumerate(rows_data):
        ax_c5   = axes[row_idx, 0]
        ax_c6   = axes[row_idx, 1]
        ax_diff = axes[row_idx, 2]

        c5_label = CMIP5_LABELS[rd['c5_key']]
        c6_label = CMIP6_LABELS[rd['c6_key']]

        #statistics
        region_mask = create_region_mask(rd['c5_data'], EXTENT)
        stats_cmip  = calculate_weighted_stats(rd['c5_data'], region_mask, data_y=rd['c6_data'])
        stats_diff  = calculate_weighted_stats(rd['diff'], region_mask)

        #SNR
        c5_std   = rd['c5_stack'].std(dim='model', skipna=True)
        c6_std   = rd['c6_stack'].std(dim='model', skipna=True)
        snr_c5   = abs(rd['c5_data']) / c5_std
        snr_c6   = abs(rd['c6_data']) / c6_std
        snr_diff = abs(rd['diff']) / c6_std

        #CMIP5
        add_map_features(ax_c5, EXTENT, is_left=True, is_bottom=(row_idx == 2))
        mesh1 = rd['c5_data'].plot.contourf(ax=ax_c5, transform=ccrs.PlateCarree(), cmap=cmap_main, vmin=vmin_main, vmax=vmax_main, add_colorbar=False, levels=levels_main, extend=extend_main)
        if rd['c5_pval'] is not None:
            ax_c5.contourf(rd['c5_pval'].longitude, rd['c5_pval'].latitude, rd['c5_pval'] < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
        ax_c5.contour(snr_c5.longitude, snr_c5.latitude, snr_c5, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())

        row_letter = chr(ord('a') + row_idx * 3)
        ax_c5.set_title(f'{row_letter}) CMIP5 {c5_label} ({rd["n5"]} models)\nMean: {stats_cmip["mean_x"]:.2f} | RMS: {stats_cmip["std_x"]:.2f} {cfg["units"]}', fontsize=10)

        #CMIP6
        add_map_features(ax_c6, EXTENT, is_left=False, is_bottom=(row_idx == 2))
        mesh2 = rd['c6_data'].plot.contourf(ax=ax_c6, transform=ccrs.PlateCarree(), cmap=cmap_main, vmin=vmin_main, vmax=vmax_main, add_colorbar=False, levels=levels_main, extend=extend_main)
        if rd['c6_pval'] is not None:
            ax_c6.contourf(rd['c6_pval'].longitude, rd['c6_pval'].latitude, rd['c6_pval'] < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
        ax_c6.contour(snr_c6.longitude, snr_c6.latitude, snr_c6, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())

        row_letter_b = chr(ord('a') + row_idx * 3 + 1)
        bold_line = fr"$\bf{{PCC = {stats_cmip['pcc']:.2f}\ |\ RMSE = {stats_cmip['rmse']:.2f}}}$"
        if plot_var != 'variability':
            sign_val = stats_cmip["sign_agreement"] * 100
            bold_line += ' | ' + fr"$\bf{{Sign\ agreement = {sign_val:.0f}\%}}$"
        title_b = f'{bold_line}\n{row_letter_b}) CMIP6 {c6_label} ({rd["n6"]} models)\nMean: {stats_cmip["mean_y"]:.2f} | RMS: {stats_cmip["std_y"]:.2f} {cfg["units"]}'
        ax_c6.set_title(title_b, fontsize=10)

        #difference
        add_map_features(ax_diff, EXTENT, is_left=False, is_bottom=(row_idx == 2))
        mesh3 = rd['diff'].plot.contourf(ax=ax_diff, transform=ccrs.PlateCarree(), cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff, add_colorbar=False, levels=levels_diff, extend='both')
        ax_diff.contour(snr_diff.longitude, snr_diff.latitude, snr_diff, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())

        row_letter_c = chr(ord('a') + row_idx * 3 + 2)
        ax_diff.set_title(f'{row_letter_c}) CMIP6 $-$ CMIP5 ({rd["label"]})\nMean: {stats_diff["mean_x"]:.2f} | RMS: {stats_diff["std_x"]:.2f} {cfg["units"]}', fontsize=10)

        #row label
        ax_diff.annotate(rd['label'], xy=(1.08, 0.5), xycoords='axes fraction', fontsize=20, fontweight='bold', rotation=-90, ha='center', va='center')

    fig.suptitle(f'CMIP5 vs CMIP6 ODSL {cfg["name"]} ({START_YEAR}-{END_YEAR})', fontsize=16, y=1.01, fontweight='bold')

    #colorbars
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.08, top=0.93, wspace=0.12, hspace=0.25)

    #shared colorbar CMIP5 and CMIP6
    pos0 = axes[2, 0].get_position()
    pos1 = axes[2, 1].get_position()
    pos2 = axes[2, 2].get_position()
    cbar_bottom = pos0.y0 - 0.05
    cbar_height = 0.015

    if mesh1 is not None:
        cbar_ax1 = fig.add_axes((pos0.x0, cbar_bottom, pos1.x1 - pos0.x0, cbar_height))
        cbar1 = fig.colorbar(mesh1, cax=cbar_ax1, orientation='horizontal', extend=extend_main)
        cbar1.set_ticks(ticks_main.tolist())
        cbar1.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=12)

    #difference colorbar
    if mesh3 is not None:
        cbar_ax2 = fig.add_axes((pos2.x0, cbar_bottom, pos2.width, cbar_height))
        cbar2 = fig.colorbar(mesh3, cax=cbar_ax2, orientation='horizontal', extend='both')
        cbar2.set_ticks(ticks_diff.tolist())
        cbar2.set_label(f'Difference ({cfg["units"]})', fontsize=12)

    output_path = os.path.join(fig_dir, f'cmip5_vs_cmip6_scenarios_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_observed_vs_modeled(obs_results, cmip_results, sliding_results, fig_dir, plot_var=None):
    """Plot observed vs modeled ODSL comparison for trend or variability."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Plotting observed vs modeled ODSL {cfg['name']}...")

    if plot_var == 'trend':
        model_data = cmip_results['model_mean_trend']
        model_stack = cmip_results['model_trend']
        obs_data = sliding_results['odsl_mm_yr_regridded']
        obs_data_high_res = obs_results['odsl']
        obs_p_value = obs_results.get('odsl_p_value')
        model_p_value = cmip_results.get('model_mean_trend_p_value')
        cmap_unified = cfg['cmap']
        
        vmax_unified = max(abs(obs_data.quantile(0.98, skipna=True).item()), abs(model_data.quantile(0.98, skipna=True).item()))
        vmin_unified = -vmax_unified

    elif plot_var == 'variability':
        model_data = cmip_results['model_mean_variability']
        model_stack = cmip_results['model_variability']
        obs_data = sliding_results['odsl_var_obs_regridded']
        obs_data_high_res = obs_results['variability']
        obs_p_value = None 
        model_p_value = None
        cmap_unified = cfg['cmap']

        vmax_unified = max(obs_data.quantile(0.98, skipna=True).item(), model_data.quantile(0.98, skipna=True).item())
        vmin_unified = min(obs_data.min(skipna=True).item(), model_data.min(skipna=True).item())

    elif plot_var == 'ODSL':
        model_data = cmip_results['model_mean_odsl'] 
        model_stack = cmip_results['model_odsl_mean']
        obs_data = sliding_results['odsl_obs_regridded']
        obs_data_high_res = obs_results['odsl_mean']
        obs_p_value = obs_results.get('odsl_mean_p_value')
        model_p_value = cmip_results.get('model_mean_odsl_p_value')
        cmap_unified = cfg['cmap']
        
        vmax_unified = max(abs(obs_data).quantile(0.98, skipna=True).item(), abs(model_data).quantile(0.98, skipna=True).item())
        vmin_unified = -vmax_unified

    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}. Must be 'trend', 'variability', or 'ODSL'")

    difference = model_data - obs_data
    vmax_diff = abs(difference.quantile(0.98, skipna=True).item())
    vmin_diff = -vmax_diff
    cmap_diff = 'BrBG'

    levels_main, ticks_main, vmin_unified, vmax_unified = make_aligned_levels(vmin_unified, vmax_unified, force_zero_min=(plot_var == 'variability'))
    levels_diff, ticks_diff, vmin_diff, vmax_diff       = make_aligned_levels(vmin_diff, vmax_diff)
    extend_main                                         = 'max' if plot_var == 'variability' else 'both'

    #Signal-to-Noise Ratio (SNR)
    model_std = model_stack.std(dim='model', skipna=True)
    #model mean
    snr_model = abs(model_data) / model_std
    #difference
    snr_diff = abs(difference) / model_std
        
    region_mask = create_region_mask(model_data, EXTENT)
    stats_comparison = calculate_weighted_stats(model_data, region_mask, data_y=obs_data)
    stats_difference = calculate_weighted_stats(difference, region_mask)
    pcc_w = stats_comparison['pcc']
    rmse_w = stats_comparison['rmse']
    sign_agr = stats_comparison['sign_agreement']
    
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(22, 5.5), subplot_kw={'projection': proj})

    #subplot 1: observed ODSL
    add_map_features(ax1, EXTENT, is_left=True, is_bottom=True)
    mesh1 = obs_data.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False, levels=levels_main, extend=extend_main)

    if obs_p_value is not None:
        ax1.contourf(obs_p_value.longitude, obs_p_value.latitude, obs_p_value < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
        
    ax1.set_title(f'a) Observed ODSL ({cfg["name"]})\nMean: {stats_comparison["mean_y"]:.2f} {cfg["units"]} | RMS: {stats_comparison["std_y"]:.2f} {cfg["units"]}', fontsize=11)

    #subplot 2: modelled ODSL
    add_map_features(ax2, EXTENT, is_left=False, is_bottom=True)
    mesh2 = model_data.plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False, levels=levels_main, extend=extend_main)
    
    if model_p_value is not None:
        ax2.contourf(model_p_value.longitude, model_p_value.latitude, model_p_value < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])   
    
    ax2.contour(snr_model.longitude, snr_model.latitude, snr_model, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())
    ax2.set_title(f'b) CMIP mean ODSL ({cfg["name"]})\nMean: {stats_comparison["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_comparison["std_x"]:.2f} {cfg["units"]}', fontsize=11)
    
    #subplot 3: difference (model - observed)
    add_map_features(ax3, EXTENT, is_left=False, is_bottom=True)
    mesh3 = difference.plot.contourf(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff, add_colorbar=False, levels=levels_diff, extend='both')
    ax3.contour(snr_diff.longitude, snr_diff.latitude, snr_diff, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())
    ax3.set_title(f'c) Difference (model - obs)\nMean: {stats_difference["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_difference["std_x"]:.2f} {cfg["units"]}', fontsize=11)
    
    #overall figure title
    title_str = f'Observed vs. modeled ODSL ({START_YEAR}-{END_YEAR})\nPCC = {pcc_w:.2f} | RMSE = {rmse_w:.2f} {cfg["units"]}'
    #sign agreement if not variability (variability \geq 0)
    if plot_var != 'variability':
        title_str += f' | Sign agreement = {sign_agr:.0%}'
    fig.suptitle(title_str, fontsize=16, y=1.02, fontweight='bold')

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.85, wspace=0.15)

    #colorbars position
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()

    cbar_bottom = pos1.y0 - 0.12 
    cbar_height = 0.03

    #shared colorbar observed and modeled
    cbar_ax1 = fig.add_axes((pos1.x0, cbar_bottom, pos2.x1 - pos1.x0, cbar_height))
    cbar1 = fig.colorbar(mesh1, cax=cbar_ax1, orientation='horizontal', extend=extend_main)
    cbar1.set_ticks(ticks_main.tolist())
    cbar1.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=12)
    
    #colorbar difference
    cbar_ax2 = fig.add_axes((pos3.x0, cbar_bottom, pos3.width, cbar_height))
    cbar2 = fig.colorbar(mesh3, cax=cbar_ax2, orientation='horizontal', extend='both')
    cbar2.set_ticks(ticks_diff.tolist())
    cbar2.set_label(f'Difference ({cfg["units"]})', fontsize=12)
    
    plt.savefig(os.path.join(fig_dir, f'observed_vs_modeled_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_sliding_window_timeseries(sliding_results, fig_dir, plot_var=None):
    """Plot PCC and RMSE time series from sliding window analysis."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]
    window_size = END_YEAR - START_YEAR + 1

    print(f"Plotting PCC and RMSE time series for {cfg['name']}...")

    if plot_var == 'trend':
        pcc = sliding_results['pcc']
        rmse = sliding_results['rmse']
        mean_ts_data = sliding_results['mean_trend']
        y_label_top = f'Mean {cfg["name"]} ({cfg["units"]})'

    elif plot_var == 'ODSL':
        pcc = sliding_results['pcc_mean_odsl']
        rmse = sliding_results['rmse_mean_odsl']
        mean_ts_data = sliding_results['mean_odsl']
        y_label_top = f'Mean {cfg["name"]} ({cfg["units"]})'

    elif plot_var == 'variability':
        pcc = sliding_results['pcc_variability']
        rmse = sliding_results['rmse_variability']
        mean_ts_data = sliding_results['mean_variability']
        y_label_top = f'Mean {cfg["name"]} ({cfg["units"]})'

    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}. Must be 'trend', 'variability', or 'ODSL'")
    
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
        ax2.plot(window_centers, pcc.sel(model=model_name), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=line_widths[i % len(line_widths)], alpha=0.8, label=model_name)
        
    ax2.plot(window_centers, pcc_mean, color='black', linewidth=3.5, linestyle='-', label='Ensemble mean', zorder=10, alpha=0.9)
    ax2.axvline(START_YEAR + window_center_offset, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Observed period center')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.set_ylabel('PCC', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(pcc.min() - 0.1, pcc.max() + 0.1)
    
    #plot 3: RMSE
    for i, model_name in enumerate(model_names):
        ax3.plot(window_centers, rmse.sel(model=model_name), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=line_widths[i % len(line_widths)], alpha=0.8, label=model_name)
        
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

    #legend
    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, framealpha=0.9, title='$\\bf{Models}$', title_fontsize=14)
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0) # type: ignore
    
    plt.tight_layout()
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.minorticks_on()
    
    plt.savefig(os.path.join(fig_dir, f'sliding_window_timeseries_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_best_and_worst_matching_periods(sliding_results, fig_dir, plot_var=None):
    """Timeline of the best and worst matching periods to observed period following from sliding window analysis."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]
    window_size = END_YEAR - START_YEAR + 1

    print(f"Plotting best matching periods for {cfg['name']}...")
    
    if plot_var == 'trend':
        pcc_data = sliding_results['pcc']
        rmse_data = sliding_results['rmse']

    elif plot_var == 'variability':
        pcc_data = sliding_results['pcc_variability']
        rmse_data = sliding_results['rmse_variability']

    elif plot_var == 'ODSL':
        pcc_data = sliding_results['pcc_mean_odsl']
        rmse_data = sliding_results['rmse_mean_odsl']

    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    model_names = sliding_results.model.values
    
    best_pcc_windows, best_pcc_values = [], []
    worst_pcc_windows, worst_pcc_values = [], []
    best_rmse_windows, best_rmse_values = [], []
    worst_rmse_windows, worst_rmse_values = [], []

    for model_name in model_names:
        pcc_model = pcc_data.sel(model=model_name)
        rmse_model = rmse_data.sel(model=model_name)
        
        try:
            #best PCC
            best_pcc_year = pcc_model.idxmax('window_start_year').item()
            best_pcc_windows.append((best_pcc_year, best_pcc_year + window_size - 1))
            best_pcc_values.append(pcc_model.max('window_start_year').item())

            #worst PCC
            worst_pcc_year = pcc_model.idxmin('window_start_year').item()
            worst_pcc_windows.append((worst_pcc_year, worst_pcc_year + window_size - 1))
            worst_pcc_values.append(pcc_model.min('window_start_year').item())
        except ValueError: 
            best_pcc_windows.append((np.nan, np.nan)); best_pcc_values.append(np.nan)

        try:
            #best RMSE
            best_rmse_year = rmse_model.idxmin('window_start_year').item()
            best_rmse_windows.append((best_rmse_year, best_rmse_year + window_size - 1))
            best_rmse_values.append(rmse_model.min('window_start_year').item())

            #worst RMSE
            worst_rmse_year = rmse_model.idxmax('window_start_year').item()
            worst_rmse_windows.append((worst_rmse_year, worst_rmse_year + window_size - 1))
            worst_rmse_values.append(rmse_model.max('window_start_year').item())
        except ValueError: 
            best_rmse_windows.append((np.nan, np.nan)); best_rmse_values.append(np.nan)

    pcc_mean_ts = pcc_data.mean(dim='model')
    rmse_mean_ts = rmse_data.mean(dim='model')

    #best PCC ensemble
    ens_best_pcc_year = pcc_mean_ts.idxmax('window_start_year').item()
    ens_best_pcc_window = (ens_best_pcc_year, ens_best_pcc_year + window_size - 1)
    ens_max_pcc_value = pcc_mean_ts.max('window_start_year').item()

    #worst PCC ensemble
    ens_worst_pcc_year = pcc_mean_ts.idxmin('window_start_year').item()
    ens_worst_pcc_window = (ens_worst_pcc_year, ens_worst_pcc_year + window_size - 1)
    ens_min_pcc_value = pcc_mean_ts.min('window_start_year').item()
    
    #best RMSE ensemble
    ens_best_rmse_year = rmse_mean_ts.idxmin('window_start_year').item()
    ens_best_rmse_window = (ens_best_rmse_year, ens_best_rmse_year + window_size - 1)
    ens_min_rmse_value = rmse_mean_ts.min('window_start_year').item()

    #worst RMSE ensemble
    ens_worst_rmse_year = rmse_mean_ts.idxmax('window_start_year').item()
    ens_worst_rmse_window = (ens_worst_rmse_year, ens_worst_rmse_year + window_size - 1)
    ens_max_rmse_value = rmse_mean_ts.max('window_start_year').item()

    #plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    observed_period = (START_YEAR, END_YEAR)
    observed_center = (observed_period[0] + observed_period[1]) / 2

    #combined list
    combined_data = list(zip(model_names, best_pcc_windows, best_rmse_windows, best_pcc_values, best_rmse_values, worst_pcc_windows, worst_rmse_windows, worst_pcc_values, worst_rmse_values))

    def sort_key(item):
        rmse_window = item[2]
        if np.isnan(rmse_window[0]): return float('inf')
        return abs(((rmse_window[0] + rmse_window[1]) / 2) - observed_center)
    
    sorted_combined_data = sorted(combined_data, key=sort_key, reverse=True)
    
    ensemble_entry = ('Ensemble mean', ens_best_pcc_window, ens_best_rmse_window, ens_max_pcc_value, ens_min_rmse_value, ens_worst_pcc_window, ens_worst_rmse_window, ens_min_pcc_value, ens_max_rmse_value)
    sorted_combined_data.append(ensemble_entry)

    if not sorted_combined_data:
        plt.show(); return

    sorted_model_list, sorted_best_pcc_windows, sorted_best_rmse_windows, sorted_best_pcc_values, sorted_best_rmse_values, sorted_worst_pcc_windows, sorted_worst_rmse_windows, sorted_worst_pcc_values, sorted_worst_rmse_values = zip(*sorted_combined_data)

    y_positions = np.arange(len(sorted_model_list))
    bar_height = 0.4

    for i in range(len(sorted_model_list)):
        
        #PCC bars
        pcc_y_pos = y_positions[i] + bar_height / 2
        best_pcc_win = sorted_best_pcc_windows[i]
        worst_pcc_win = sorted_worst_pcc_windows[i]
        
        #best PCC
        if not np.isnan(best_pcc_win[0]):
            value = sorted_best_pcc_values[i]
            ax.barh(pcc_y_pos, best_pcc_win[1] - best_pcc_win[0], left=best_pcc_win[0], height=bar_height, color='red', alpha=0.8, label='Best PCC' if i == 0 else "", edgecolor='red', linewidth=1)
            ax.text((best_pcc_win[0] + best_pcc_win[1]) / 2, pcc_y_pos, f'{value:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=7)

        #worst PCC
        if not np.isnan(worst_pcc_win[0]):
            value = sorted_worst_pcc_values[i]
            ax.barh(pcc_y_pos, worst_pcc_win[1] - worst_pcc_win[0], left=worst_pcc_win[0], height=bar_height, color='red', alpha=0.2, label='Worst PCC' if i == 0 else "", edgecolor='red', linestyle='--', linewidth=1)
            ax.text((worst_pcc_win[0] + worst_pcc_win[1]) / 2, pcc_y_pos, f'{value:.2f}', ha='center', va='center', color='black', fontsize=7)

        #line between best and worst PCC window
        if not np.isnan(best_pcc_win[0]) and not np.isnan(worst_pcc_win[0]):
            left_end = min(best_pcc_win[1], worst_pcc_win[1])
            right_start = max(best_pcc_win[0], worst_pcc_win[0])
            if left_end < right_start:
                ax.plot([left_end, right_start], [pcc_y_pos, pcc_y_pos], color='red', linestyle='-', linewidth=1.5)
            
        #RMSE bars
        rmse_y_pos = y_positions[i] - bar_height / 2
        best_rmse_win = sorted_best_rmse_windows[i]
        worst_rmse_win = sorted_worst_rmse_windows[i]
        
        #best RMSE
        if not np.isnan(best_rmse_win[0]):
            value = sorted_best_rmse_values[i]
            ax.barh(rmse_y_pos, best_rmse_win[1] - best_rmse_win[0], left=best_rmse_win[0], height=bar_height, color='black', alpha=0.8, label='Best RMSE' if i == 0 else "", edgecolor='black', linewidth=1)
            ax.text((best_rmse_win[0] + best_rmse_win[1]) / 2, rmse_y_pos, f'{value:.2f}', ha='center', va='center', color='white', fontweight='bold', fontsize=7)

        #worst RMSE
        if not np.isnan(worst_rmse_win[0]):
            value = sorted_worst_rmse_values[i]
            ax.barh(rmse_y_pos, worst_rmse_win[1] - worst_rmse_win[0], left=worst_rmse_win[0], height=bar_height, color='black', alpha=0.2, label='Worst RMSE' if i == 0 else "", edgecolor='black', linestyle='--', linewidth=1)
            ax.text((worst_rmse_win[0] + worst_rmse_win[1]) / 2, rmse_y_pos, f'{value:.2f}', ha='center', va='center', color='black', fontsize=7)

        #line between best and worst RMSE window
        if not np.isnan(best_rmse_win[0]) and not np.isnan(worst_rmse_win[0]):
            left_end = min(best_rmse_win[1], worst_rmse_win[1])
            right_start = max(best_rmse_win[0], worst_rmse_win[0])
            if left_end < right_start:
                ax.plot([left_end, right_start], [rmse_y_pos, rmse_y_pos], color='black', linestyle='-', linewidth=1.5)

    ax.axvline(observed_period[0], color='green', linestyle='--', linewidth=1.5, label='Observation period')
    ax.axvline(observed_period[1], color='green', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_model_list)
    ax.set_ylim(-0.5, len(sorted_model_list) - 0.5)
    ax.set_xlabel('Year', fontsize=12)

    #dynamic x-lim
    all_windows = list(sorted_best_pcc_windows) + list(sorted_best_rmse_windows) + list(sorted_worst_pcc_windows) + list(sorted_worst_rmse_windows)
    all_years = [year for window in all_windows if not np.isnan(window[0]) for year in window]
    all_years.extend(observed_period)
    if all_years:
        min_limit = min(all_years) - 5
        max_limit = max(all_years) + 5
        ax.set_xlim(min_limit, max_limit)

    separator_pos = len(model_names) - 0.5
    ax.axhline(y=separator_pos, color='gray', linestyle='--', xmin=-0.12, clip_on=False)

    ax.set_title(f'Best and worst matching {window_size}-year periods by model ({cfg["name"]} vs. observations)', fontsize=14, fontweight='bold')
    ax.legend(loc='center left')
    ax.grid(True, axis='x', linestyle='--', alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'best_and_worst_matching_periods_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_model_comparison_summary(cmip_results_ds, sliding_results_ds, fig_dir, plot_var=None, picontrol_results=None):
    """Model-observation comparison using sliding windows, following Richter et al., 2017 figure 4."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]
    unit = cfg['units']
    window_size = END_YEAR - START_YEAR + 1

    print(f"Creating model-observation comparison statistics figure for {cfg['name']}...")

    #select data depending on plotting variable
    if plot_var == 'trend':
        rmse_ts = sliding_results_ds['rmse']
        pcc_ts = sliding_results_ds['pcc']
        obs_period_model_data = cmip_results_ds['model_trend']
        obs_pattern_data = sliding_results_ds['odsl_mm_yr_regridded']
        obs_period_mmm_data = cmip_results_ds['model_mean_trend']
        sliding_maps = sliding_results_ds['sliding_trends']

    elif plot_var == 'variability':
        rmse_ts = sliding_results_ds['rmse_variability']
        pcc_ts = sliding_results_ds['pcc_variability']
        obs_period_model_data = cmip_results_ds['model_variability']
        obs_pattern_data = sliding_results_ds['odsl_var_obs_regridded']
        obs_period_mmm_data = cmip_results_ds['model_mean_variability']
        sliding_maps = sliding_results_ds['sliding_variability']

    elif plot_var == 'ODSL':
        rmse_ts = sliding_results_ds['rmse_mean_odsl']
        pcc_ts = sliding_results_ds['pcc_mean_odsl']
        obs_period_model_data = cmip_results_ds['model_odsl_mean']
        obs_pattern_data = sliding_results_ds['odsl_obs_regridded']
        obs_period_mmm_data = cmip_results_ds['model_mean_odsl']
        sliding_maps = sliding_results_ds['sliding_odsl_mean']

    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    #calculations
    mean_rmse_per_model = rmse_ts.mean(dim='window_start_year')
    min_rmse_per_model = rmse_ts.min(dim='window_start_year')
    max_rmse_per_model = rmse_ts.max(dim='window_start_year')
    mean_pcc_per_model = pcc_ts.mean(dim='window_start_year')
    min_pcc_per_model = pcc_ts.min(dim='window_start_year')
    max_pcc_per_model = pcc_ts.max(dim='window_start_year')

    #generate list
    obs_period_rmse_list, obs_period_pcc_list = [], []
    for model_name in cmip_results_ds.model.values:
        obs_period_rmse_list.append(rmse_ts.sel(model=model_name, window_start_year=START_YEAR).item())
        obs_period_pcc_list.append(pcc_ts.sel(model=model_name, window_start_year=START_YEAR).item())
    
    #transform into dataframe
    df = pd.DataFrame({'mean_rmse': mean_rmse_per_model.values, 'min_rmse': min_rmse_per_model.values, 'max_rmse': max_rmse_per_model.values, 'obs_period_rmse': obs_period_rmse_list, 'mean_pcc': mean_pcc_per_model.values, 'min_pcc': min_pcc_per_model.values, 'max_pcc': max_pcc_per_model.values, 'obs_period_pcc': obs_period_pcc_list}, index=cmip_results_ds.model.values)
    df_sorted = df.sort_values('mean_rmse')

    #mask
    ensemble_region_mask = create_region_mask(cmip_results_ds['model_mean_trend'], EXTENT)

    #calculate ensemble stats for observed period
    stats_obs_ensemble = calculate_weighted_stats(obs_period_mmm_data, ensemble_region_mask, data_y=obs_pattern_data)

    best_pcc_years = pcc_ts.idxmax('window_start_year')
    best_rmse_years = rmse_ts.idxmin('window_start_year')
    best_pcc_patterns, best_rmse_patterns = [], []
    
    #calculate best matching patterns depending on plotting variable
    if plot_var == 'trend':
        for model_name in cmip_results_ds.model.values:
            full_ts = cmip_results_ds['full_timeseries'].sel(model=model_name)
            pcc_year = best_pcc_years.sel(model=model_name).item()
            rmse_year = best_rmse_years.sel(model=model_name).item()
            pcc_window = full_ts.sel(time=slice(pcc_year, pcc_year + window_size - 1))
            rmse_window = full_ts.sel(time=slice(rmse_year, rmse_year + window_size - 1))
            best_pcc_patterns.append(pcc_window.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)
            best_rmse_patterns.append(rmse_window.polyfit(dim='time', deg=1).polyfit_coefficients.sel(degree=1) * 10)

    elif plot_var == 'variability':
        sliding_var_maps = sliding_results_ds['sliding_variability']
        for model_name in cmip_results_ds.model.values:
            pcc_year = best_pcc_years.sel(model=model_name).item()
            rmse_year = best_rmse_years.sel(model=model_name).item()
            best_pcc_patterns.append(sliding_var_maps.sel(model=model_name, window_start_year=pcc_year))
            best_rmse_patterns.append(sliding_var_maps.sel(model=model_name, window_start_year=rmse_year))

    elif plot_var == 'ODSL':
        sliding_odsl_maps = sliding_results_ds['sliding_odsl_mean']
        for model_name in cmip_results_ds.model.values:
            pcc_year = best_pcc_years.sel(model=model_name).item()
            rmse_year = best_rmse_years.sel(model=model_name).item()
            best_pcc_patterns.append(sliding_odsl_maps.sel(model=model_name, window_start_year=pcc_year))
            best_rmse_patterns.append(sliding_odsl_maps.sel(model=model_name, window_start_year=rmse_year))

    #ensemble calculations
    best_pcc_ensemble_mean = xr.concat(best_pcc_patterns, dim='model').mean(dim='model')
    best_rmse_ensemble_mean = xr.concat(best_rmse_patterns, dim='model').mean(dim='model')
    stats_pcc_ensemble = calculate_weighted_stats(best_pcc_ensemble_mean, ensemble_region_mask, data_y=obs_pattern_data)
    stats_rmse_ensemble = calculate_weighted_stats(best_rmse_ensemble_mean, ensemble_region_mask, data_y=obs_pattern_data)

    #multi-model mean ensemble
    ens_mean_maps = sliding_maps.mean(dim='model')
    ensemble_mean_label = 'Multi-model\nensemble'
    ens_rmse_per_window = []
    ens_pcc_per_window = []
    for ws in ens_mean_maps.window_start_year.values:
        stats = calculate_weighted_stats(ens_mean_maps.sel(window_start_year=ws), ensemble_region_mask, data_y=obs_pattern_data)
        ens_rmse_per_window.append(stats['rmse'])
        ens_pcc_per_window.append(stats['pcc'])

    ens_rmse_arr = np.array(ens_rmse_per_window)
    ens_pcc_arr = np.array(ens_pcc_per_window)

    #ensemble labels
    ensemble_data = {}
    ensemble_data[ensemble_mean_label] = {'mean_rmse': np.mean(ens_rmse_arr), 'min_rmse': np.min(ens_rmse_arr), 'max_rmse': np.max(ens_rmse_arr), 'obs_period_rmse': stats_obs_ensemble['rmse'], 'mean_pcc': np.mean(ens_pcc_arr), 'min_pcc': np.min(ens_pcc_arr), 'max_pcc': np.max(ens_pcc_arr),'obs_period_pcc': stats_obs_ensemble['pcc']}

    #models within observed range
    def get_models_within_obs_range(fig_dir):
        csv_path = os.path.join(fig_dir, f'regional_model_selection_{cfg["name"]}_{START_YEAR}_{END_YEAR}.csv')
        df = pd.read_csv(csv_path)
        mask = (df['North Atlantic_within_obs'] == True) & (~df['source'].isin(['Observed', 'Multi-Model Mean']))
        return df.loc[mask, 'source'].tolist()

    selected_models = get_models_within_obs_range(fig_dir)
    valid_models = [m for m in selected_models if m in cmip_results_ds.model.values]
    within_obs_label = None
    within_rmse_arr = np.array([])
    within_pcc_arr = np.array([])

    if len(valid_models) > 0:
        within_obs_label = f'Within obs range\nensemble ({len(valid_models)})'

        within_maps = sliding_maps.sel(model=valid_models).mean(dim='model')
        within_rmse_per_window = []
        within_pcc_per_window = []
        for ws in within_maps.window_start_year.values:
            s = calculate_weighted_stats(within_maps.sel(window_start_year=ws), ensemble_region_mask, data_y=obs_pattern_data)
            within_rmse_per_window.append(s['rmse'])
            within_pcc_per_window.append(s['pcc'])

        within_rmse_arr = np.array(within_rmse_per_window)
        within_pcc_arr = np.array(within_pcc_per_window)

        within_obs_period_pattern = obs_period_model_data.sel(model=valid_models).mean(dim='model')
        within_obs_stats = calculate_weighted_stats(within_obs_period_pattern, ensemble_region_mask, data_y=obs_pattern_data)

        ensemble_data[within_obs_label] = {'mean_rmse': np.mean(within_rmse_arr), 'min_rmse': np.min(within_rmse_arr), 'max_rmse': np.max(within_rmse_arr), 'obs_period_rmse': within_obs_stats['rmse'], 'mean_pcc': np.mean(within_pcc_arr), 'min_pcc': np.min(within_pcc_arr), 'max_pcc': np.max(within_pcc_arr), 'obs_period_pcc': within_obs_stats['pcc']}

        print(f"Within observed range ensemble: {len(valid_models)}/{len(cmip_results_ds.model.values)} models selected")

    #best rmse and pcc window ensembles
    ensemble_data['Best RMSE\nensemble'] = stats_rmse_ensemble
    ensemble_data['Best PCC\nensemble'] = stats_pcc_ensemble

    for name, stats in ensemble_data.items():
        if 'mean_rmse' in stats:
            #multi-model ensemble and within observations ensemble
            df_sorted.loc[name] = stats
        else:
            #best RMSE and PCC ensembles
            df_sorted.loc[name] = {'mean_rmse': stats['rmse'], 'min_rmse': stats['rmse'], 'max_rmse': stats['rmse'], 'obs_period_rmse': stats['rmse'], 'mean_pcc': stats['pcc'], 'min_pcc': stats['pcc'], 'max_pcc': stats['pcc'], 'obs_period_pcc': stats['pcc']}

    #plotting
    model_names = df_sorted.index.tolist()
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(14, 10))
    
    mean_rmse_all, min_rmse_all, max_rmse_all, obs_period_rmse = df_sorted['mean_rmse'], df_sorted['min_rmse'], df_sorted['max_rmse'], df_sorted['obs_period_rmse']
    mean_pcc_all, min_pcc_all, max_pcc_all, obs_period_pcc = df_sorted['mean_pcc'], df_sorted['min_pcc'], df_sorted['max_pcc'], df_sorted['obs_period_pcc']
    
    #piControl loading and range
    pi_rmse_min, pi_rmse_max = np.nan, np.nan
    pi_pcc_min, pi_pcc_max = np.nan, np.nan
    pi_pcc_pre = None
    pi_rmse_pre = None
    if picontrol_results is not None:
        if plot_var == 'trend':
            pi_pcc_pre = picontrol_results['pcc']
            pi_rmse_pre = picontrol_results['rmse']
        elif plot_var == 'variability':
            pi_pcc_pre = picontrol_results['pcc_variability']
            pi_rmse_pre = picontrol_results['rmse_variability']
        elif plot_var == 'ODSL':
            pi_pcc_pre = picontrol_results['pcc_mean_odsl']
            pi_rmse_pre = picontrol_results['rmse_mean_odsl']
        if pi_pcc_pre is not None and pi_rmse_pre is not None:
            pi_rmse_min = float(np.nanmin(pi_rmse_pre.values))
            pi_rmse_max = float(np.nanmax(pi_rmse_pre.values))
            pi_pcc_min = float(np.nanmin(pi_pcc_pre.values))
            pi_pcc_max = float(np.nanmax(pi_pcc_pre.values))

    #y-axes plotting range
    y_min, y_max = 0, 1

    #dynamic range RMSE axis
    actual_rmse_min = np.nanmin([np.nanmin(min_rmse_all), pi_rmse_min])
    actual_rmse_max = np.nanmax([np.nanmax(max_rmse_all), pi_rmse_max])
    rmse_data_range = actual_rmse_max - actual_rmse_min
    padding = rmse_data_range * 0.05 if rmse_data_range > 0 else 0.1
    rmse_min_val = max(0, actual_rmse_min - padding)
    rmse_max_val = actual_rmse_max + padding
    rmse_range = rmse_max_val - rmse_min_val
    rmse_scale_factor = 0.5 / rmse_range if rmse_range > 0 else 0

    #dynamic range PCC axis
    actual_pcc_min = np.nanmin([np.nanmin(min_pcc_all), pi_pcc_min])
    actual_pcc_max = np.nanmax([np.nanmax(max_pcc_all), pi_pcc_max])
    pcc_data_range = actual_pcc_max - actual_pcc_min
    padding = pcc_data_range * 0.05 if pcc_data_range > 0 else 0.1
    pcc_min_val = actual_pcc_min - padding
    pcc_max_val = actual_pcc_max + padding
    pcc_range = pcc_max_val - pcc_min_val
    pcc_scale_factor = 0.5 / pcc_range if pcc_range > 0 else 0
    pcc_offset = 0.5

    mean_rmse_scaled = (mean_rmse_all - rmse_min_val) * rmse_scale_factor
    min_rmse_scaled = (min_rmse_all - rmse_min_val) * rmse_scale_factor
    max_rmse_scaled = (max_rmse_all - rmse_min_val) * rmse_scale_factor
    obs_rmse_scaled = (obs_period_rmse - rmse_min_val) * rmse_scale_factor
    
    mean_pcc_scaled = (mean_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    min_pcc_scaled = (min_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    max_pcc_scaled = (max_pcc_all - pcc_min_val) * pcc_scale_factor + pcc_offset
    obs_pcc_scaled = (obs_period_pcc - pcc_min_val) * pcc_scale_factor + pcc_offset

    #'regular' violin plots RMSE and PCC
    df_rmse_long = rmse_ts.to_dataframe(name='rmse').reset_index()
    df_pcc_long = pcc_ts.to_dataframe(name='pcc').reset_index()
    df_rmse_long['rmse_scaled'] = (df_rmse_long['rmse'] - rmse_min_val) * rmse_scale_factor
    df_pcc_long['pcc_scaled'] = (df_pcc_long['pcc'] - pcc_min_val) * pcc_scale_factor + pcc_offset

    #ensemble mean distribution
    window_starts = ens_mean_maps.window_start_year.values
    ens_rmse_rows = pd.DataFrame({'model': ensemble_mean_label, 'rmse': ens_rmse_arr, 'window_start_year': window_starts})
    ens_pcc_rows = pd.DataFrame({'model': ensemble_mean_label, 'pcc': ens_pcc_arr, 'window_start_year': window_starts})
    ens_rmse_rows['rmse_scaled'] = (ens_rmse_rows['rmse'] - rmse_min_val) * rmse_scale_factor
    ens_pcc_rows['pcc_scaled'] = (ens_pcc_rows['pcc'] - pcc_min_val) * pcc_scale_factor + pcc_offset

    df_rmse_long = pd.concat([df_rmse_long, ens_rmse_rows], ignore_index=True)
    df_pcc_long = pd.concat([df_pcc_long, ens_pcc_rows], ignore_index=True)

    #within observations ensemble violin data
    if within_obs_label is not None:
        window_starts = ens_mean_maps.window_start_year.values
        ens_within_rmse_rows = pd.DataFrame({'model': within_obs_label, 'rmse': within_rmse_arr, 'window_start_year': window_starts})
        ens_within_pcc_rows = pd.DataFrame({'model': within_obs_label, 'pcc': within_pcc_arr, 'window_start_year': window_starts})
        ens_within_rmse_rows['rmse_scaled'] = (ens_within_rmse_rows['rmse'] - rmse_min_val) * rmse_scale_factor
        ens_within_pcc_rows['pcc_scaled'] = (ens_within_pcc_rows['pcc'] - pcc_min_val) * pcc_scale_factor + pcc_offset

        df_rmse_long = pd.concat([df_rmse_long, ens_within_rmse_rows], ignore_index=True)
        df_pcc_long = pd.concat([df_pcc_long, ens_within_pcc_rows], ignore_index=True)

    model_order = df_sorted.index.tolist()

    #violin plots
    #regular cmip models
    sns.violinplot(data=df_rmse_long, x='model', y='rmse_scaled', order=model_order, ax=ax, color='grey', alpha=0.7, inner=None, saturation=0.7, zorder=1, cut=0, width=1.0)
    sns.violinplot(data=df_pcc_long, x='model', y='pcc_scaled', order=model_order, ax=ax, color='red', alpha=0.7, inner=None, saturation=0.7, zorder=1, cut=0, width=1.0)
        
    #piControl violin plots
    if picontrol_results is not None and pi_pcc_pre is not None and pi_rmse_pre is not None:
        pi_rmse_rows = []
        pi_pcc_rows = []
        for model_name in model_order:
            if model_name in pi_pcc_pre.model.values: # type: ignore
                pcc_vals = pi_pcc_pre.sel(model=model_name).values # type: ignore
                rmse_vals = pi_rmse_pre.sel(model=model_name).values # type: ignore
                pcc_vals = pcc_vals[~np.isnan(pcc_vals)]
                rmse_vals = rmse_vals[~np.isnan(rmse_vals)]
                for v in rmse_vals:
                    pi_rmse_rows.append({'model': model_name, 'rmse': v})
                for v in pcc_vals:
                    pi_pcc_rows.append({'model': model_name, 'pcc': v})
        
        #piControl ensemble mean 
        pi_ens_rmse_vals = np.array([])
        pi_ens_pcc_vals = np.array([])
        if ensemble_mean_label in model_order:
            if plot_var == 'trend':
                pi_ens_rmse_vals = picontrol_results['pi_ensemble_rmse'].values
                pi_ens_pcc_vals = picontrol_results['pi_ensemble_pcc'].values
            elif plot_var == 'variability':
                pi_ens_rmse_vals = picontrol_results['pi_ensemble_rmse_variability'].values
                pi_ens_pcc_vals = picontrol_results['pi_ensemble_pcc_variability'].values
            elif plot_var == 'ODSL':
                pi_ens_rmse_vals = picontrol_results['pi_ensemble_rmse_mean_odsl'].values
                pi_ens_pcc_vals = picontrol_results['pi_ensemble_pcc_mean_odsl'].values
            
            for v in pi_ens_rmse_vals[~np.isnan(pi_ens_rmse_vals)]:
                pi_rmse_rows.append({'model': ensemble_mean_label, 'rmse': v})
            for v in pi_ens_pcc_vals[~np.isnan(pi_ens_pcc_vals)]:
                pi_pcc_rows.append({'model': ensemble_mean_label, 'pcc': v})

        #piControl within-obs ensemble
        pi_within_rmse_vals = np.array([])
        pi_within_pcc_vals = np.array([])
        if within_obs_label is not None and within_obs_label in model_order:
            if plot_var == 'trend':
                pi_within_rmse_vals = picontrol_results['pi_within_obs_rmse'].values
                pi_within_pcc_vals = picontrol_results['pi_within_obs_pcc'].values
            elif plot_var == 'variability':
                pi_within_rmse_vals = picontrol_results['pi_within_obs_rmse_variability'].values
                pi_within_pcc_vals = picontrol_results['pi_within_obs_pcc_variability'].values
            elif plot_var == 'ODSL':
                pi_within_rmse_vals = picontrol_results['pi_within_obs_rmse_mean_odsl'].values
                pi_within_pcc_vals = picontrol_results['pi_within_obs_pcc_mean_odsl'].values

            for v in pi_within_rmse_vals[~np.isnan(pi_within_rmse_vals)]:
                pi_rmse_rows.append({'model': within_obs_label, 'rmse': v})
            for v in pi_within_pcc_vals[~np.isnan(pi_within_pcc_vals)]:
                pi_pcc_rows.append({'model': within_obs_label, 'pcc': v})

        if pi_rmse_rows:
            df_pi_rmse = pd.DataFrame(pi_rmse_rows)
            df_pi_pcc = pd.DataFrame(pi_pcc_rows)
            df_pi_rmse['rmse_scaled'] = (df_pi_rmse['rmse'] - rmse_min_val) * rmse_scale_factor
            df_pi_pcc['pcc_scaled'] = (df_pi_pcc['pcc'] - pcc_min_val) * pcc_scale_factor + pcc_offset

            n_before = len(ax.collections)
            
            sns.violinplot(data=df_pi_rmse, x='model', y='rmse_scaled', order=model_order, ax=ax, color='lightblue', alpha=0.5, inner=None, saturation=0.5, zorder=0, cut=0, width=1.0)
            sns.violinplot(data=df_pi_pcc, x='model', y='pcc_scaled', order=model_order, ax=ax, color='lightblue', alpha=0.5, inner=None, saturation=0.5, zorder=0, cut=0, width=1.0)

            #edge style
            for collection in ax.collections[n_before:]:
                collection.set_edgecolor('black')
                collection.set_linewidth(1.2)
                collection.set_linestyle('dashed')

    #mean RMSE
    mean_rmse_line_plotted = False
    n_ensembles = 4 if within_obs_label is not None else 3
    for i, (x_pos, mean_val) in enumerate(zip(x, mean_rmse_scaled)):
        if not np.isnan(mean_val) and i < len(x) - n_ensembles:
            label = 'Mean RMSE (all sliding windows)' if not mean_rmse_line_plotted else ""
            ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val], color='black', linewidth=2, zorder=3, label=label)
            mean_rmse_line_plotted = True

    #mean line and range ensemble multi-model mean
    ens_mean_idx = len(model_names) - n_ensembles
    ens_x = x[ens_mean_idx]
    if not np.isnan(mean_rmse_scaled.iloc[ens_mean_idx]):
        ax.plot([ens_x - 0.2, ens_x + 0.2], [mean_rmse_scaled.iloc[ens_mean_idx]] * 2, color='black', linewidth=2, zorder=3)
        ax.errorbar(ens_x, mean_rmse_scaled.iloc[ens_mean_idx], yerr=[[mean_rmse_scaled.iloc[ens_mean_idx] - min_rmse_scaled.iloc[ens_mean_idx]], [max_rmse_scaled.iloc[ens_mean_idx] - mean_rmse_scaled.iloc[ens_mean_idx]]], fmt='none', color='black', capsize=6, capthick=1.5)

    #select distinction between models and ensembles
    obs_ensemble_idx = len(model_names) - 3
    x_models = x[:-n_ensembles]

    #range RMSE
    ax.errorbar(x_models, mean_rmse_scaled.iloc[:-n_ensembles], yerr=[mean_rmse_scaled.iloc[:-n_ensembles] - min_rmse_scaled.iloc[:-n_ensembles], max_rmse_scaled.iloc[:-n_ensembles] - mean_rmse_scaled.iloc[:-n_ensembles]], fmt='none', color='black', capsize=6, capthick=1.5, label='RMSE range (all sliding windows)')
    n_best = 2    
    ax.scatter(x[:-n_best], obs_rmse_scaled.iloc[:-n_best], color='green', s=60, zorder=5)
    ax.scatter(x[-n_best:], obs_rmse_scaled.iloc[-n_best:], color='black', s=60, zorder=5)
    
    #mean PCC
    mean_line_plotted = False
    for i, (x_pos, mean_val) in enumerate(zip(x, mean_pcc_scaled)):
        if not np.isnan(mean_val) and i < len(x) - n_ensembles:
            label = 'Mean PCC (all sliding windows)' if not mean_line_plotted else ""
            ax.plot([x_pos - 0.2, x_pos + 0.2], [mean_val, mean_val], color='red', linewidth=2, zorder=3, label=label)
            mean_line_plotted = True

    #mean and range PCC best PCC windows
    if not np.isnan(mean_pcc_scaled.iloc[ens_mean_idx]):
        ax.plot([ens_x - 0.2, ens_x + 0.2], [mean_pcc_scaled.iloc[ens_mean_idx]] * 2, color='red', linewidth=2, zorder=3)
        ax.errorbar(ens_x, mean_pcc_scaled.iloc[ens_mean_idx], yerr=[[mean_pcc_scaled.iloc[ens_mean_idx] - min_pcc_scaled.iloc[ens_mean_idx]], [max_pcc_scaled.iloc[ens_mean_idx] - mean_pcc_scaled.iloc[ens_mean_idx]]], fmt='none', color='red', capsize=6, capthick=1.5)

    #mean line and range for within-obs ensemble
    if within_obs_label is not None:
        within_idx = model_names.index(within_obs_label)
        within_x = x[within_idx]
        if not np.isnan(mean_rmse_scaled.iloc[within_idx]):
            ax.plot([within_x - 0.2, within_x + 0.2], [mean_rmse_scaled.iloc[within_idx]] * 2, color='black', linewidth=2, zorder=3)
            ax.errorbar(within_x, mean_rmse_scaled.iloc[within_idx], yerr=[[mean_rmse_scaled.iloc[within_idx] - min_rmse_scaled.iloc[within_idx]], [max_rmse_scaled.iloc[within_idx] - mean_rmse_scaled.iloc[within_idx]]], fmt='none', color='black', capsize=6, capthick=1.5)
        if not np.isnan(mean_pcc_scaled.iloc[within_idx]):
            ax.plot([within_x - 0.2, within_x + 0.2], [mean_pcc_scaled.iloc[within_idx]] * 2, color='red', linewidth=2, zorder=3)
            ax.errorbar(within_x, mean_pcc_scaled.iloc[within_idx], yerr=[[mean_pcc_scaled.iloc[within_idx] - min_pcc_scaled.iloc[within_idx]], [max_pcc_scaled.iloc[within_idx] - mean_pcc_scaled.iloc[within_idx]]], fmt='none', color='red', capsize=6, capthick=1.5)
    
    #range PCC
    ax.errorbar(x_models, mean_pcc_scaled.iloc[:-n_ensembles], yerr=[mean_pcc_scaled.iloc[:-n_ensembles] - min_pcc_scaled.iloc[:-n_ensembles], max_pcc_scaled.iloc[:-n_ensembles] - mean_pcc_scaled.iloc[:-n_ensembles]], fmt='none', color='red', capsize=6, capthick=1.5, label='PCC range (all sliding windows)')
    ax.scatter(x[:-n_best], obs_pcc_scaled.iloc[:-n_best], color='green', s=60, zorder=5)
    ax.scatter(x[-n_best:], obs_pcc_scaled.iloc[-n_best:], color='red', s=60, zorder=5)

    ax.set_ylim(y_min, y_max)
    ax2 = ax.twinx()
    ax2.set_ylim(y_min, y_max)
    ax.set_xlabel('')
    
    #y axes tickers
    num_ticks = 5

    #RMSE y-axis ticks
    scaled_rmse_ticks = np.linspace(0, 0.5, num_ticks)
    original_rmse_labels = scaled_rmse_ticks / rmse_scale_factor + rmse_min_val
    ax.set_yticks(scaled_rmse_ticks)
    ax.set_yticklabels([f'{val:.1f}' for val in original_rmse_labels])
    ax.set_ylabel(f'RMSE ({unit})', fontsize=12, color='black', y=0.25)
    ax.tick_params(axis='y', labelcolor='black')

    #PCC y-axis ticks
    scaled_pcc_ticks = np.linspace(0.5, 1.0, num_ticks)
    original_pcc_labels = (scaled_pcc_ticks - pcc_offset) / pcc_scale_factor + pcc_min_val
    ax2.set_yticks(scaled_pcc_ticks)
    ax2.set_yticklabels([f'{val:.1f}' for val in original_pcc_labels])
    ax2.set_ylabel('PCC', fontsize=12, color='red', y=0.75)
    ax2.tick_params(axis='y', labelcolor='red')

    #separation line individual models and ensembles
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)
    separator_pos = len(df_sorted) - n_ensembles - 0.5
    ax.axvline(separator_pos, color='gray', linestyle='--', ymin=-0.17, clip_on=False)
    
    #x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=90, ha='center', va='top', fontsize=10)
    ax.set_xlim(-0.5, len(model_names) - 0.5)

    ax.set_title(f'Model-observation ODSL comparison ({cfg["name"]})\nMean statistics over all {window_size}-yr sliding windows', fontsize=14, pad=20, fontweight='bold')
    
    #grid lines
    for tick in ax.get_yticks():
        ax.axhline(y=tick, color='gray', linestyle='-', alpha=0.3, linewidth=0.7, zorder=0)
    for tick in ax2.get_yticks():
        ax.axhline(y=tick, color='grey', linestyle='-', alpha=0.3, linewidth=0.7, zorder=0)

    #ordered legend
    handles, labels = ax.get_legend_handles_labels()

    rmse_violin_patch = mpatches.Patch(color='grey', alpha=0.7, label='RMSE distribution (all sliding windows)')
    pcc_violin_patch = mpatches.Patch(color='red', alpha=0.7, label='PCC distribution (all sliding windows)')
    handles.extend([rmse_violin_patch, pcc_violin_patch])
    labels.extend(['RMSE distribution (all sliding windows)', 'PCC distribution (all sliding windows)'])
    obs_period_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=8, label='PCC & RMSE over observed period')
    handles.append(obs_period_dot)
    labels.append('PCC & RMSE over observed period')

    desired_order = ['PCC & RMSE over observed period',
                     'Mean PCC (all sliding windows)',
                     'PCC range (all sliding windows)',
                     'PCC distribution (all sliding windows)',
                     'Mean RMSE (all sliding windows)',
                     'RMSE range (all sliding windows)',
                     'RMSE distribution (all sliding windows)']

    if picontrol_results is not None:
        pi_patch = mpatches.Patch(color='lightblue', alpha=0.5, edgecolor='black', linewidth=1.2, linestyle='dashed', label='RMSE & PCC distribution (piControl)')
        handles.extend([pi_patch])
        labels.extend(['RMSE & PCC distribution (piControl)'])
        
        desired_order.extend(['RMSE & PCC distribution (piControl)'])

    label_handle_map = dict(zip(labels, handles))
    reordered_handles = [label_handle_map[label] for label in desired_order if label in label_handle_map]
    reordered_labels = [label for label in desired_order if label in label_handle_map]
    ax.legend(reordered_handles, reordered_labels, loc='upper left', bbox_to_anchor=(0, 0.53), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'model_comparison_summary_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_observed_variability(obs_results, fig_dir):
    """Plot the observed ODSL variability."""
    
    cfg = PLOT_CONFIG['variability']

    print("Plotting observed ODSL variability...")
    
    #projection
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': proj})
    add_map_features(ax, EXTENT, is_left=True, is_bottom=True)
    
    #data
    variability = obs_results['variability']
    common_years = np.array(obs_results.attrs['common_years_list'])

    vmax = variability.quantile(0.98, skipna=True).item()
    vmin = variability.quantile(0.02, skipna=True).item()

    #plotting
    mesh = variability.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cfg['cmap'], vmin=vmin, vmax=vmax, add_colorbar=False, levels=50)

    #colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.08)
    cbar.set_ticks(np.arange(np.ceil(vmin), np.floor(vmax) + 1))
    cbar.set_ticks([], minor=True)
    cbar.set_label(f'ODSL {cfg["name"]} ({cfg["units"]})', fontsize=10)
    
    #statistics
    region_mask = create_region_mask(variability, EXTENT)
    stats_obs = calculate_weighted_stats(variability, region_mask)
    
    #title
    ax.set_title(f'Observed ODSL {cfg["name"]} ({common_years.min()}-{common_years.max()})\n' f'Mean: {stats_obs["mean_x"]:.2f} {cfg["units"]}, RMS: {stats_obs["std_x"]:.2f} {cfg["units"]}', fontsize=12, pad=15, fontweight='bold')
    
    #save figure
    plt.savefig(os.path.join(fig_dir, f'Observed_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_scenario_timeseries(scenario_results, obs_results, fig_dir):
    """Regional mean ODSL for CMIP historical and future scenarios and compare to observed."""

    print("Plotting CMIP scenario timeseries...")

    if not scenario_results or "cmip_version" not in scenario_results.coords:
        print("No valid scenario results to plot.")
        return

    data = scenario_results.sel(cmip_version=CMIP_VERSION)

    #scenario config
    if CMIP_VERSION == 'CMIP6':
        future_scenarios = ['ssp126', 'ssp245', 'ssp585']
        scenario_colors = {'historical': 'black',
                           'ssp126': '#377eb8',
                           'ssp245': '#e41a1c',
                           'ssp585': '#f2ff00'}
        scenario_labels = {'historical': 'Historical',
                           'ssp126': 'SSP1-2.6',
                           'ssp245': 'SSP2-4.5',
                           'ssp585': 'SSP5-8.5'}
    else:  #CMIP5
        future_scenarios = ['rcp26', 'rcp45', 'rcp85']
        scenario_colors = {'historical': 'black',
                           'rcp26': '#377eb8',
                           'rcp45': '#e41a1c',
                           'rcp85': '#f2ff00'}
        scenario_labels = {'historical': 'Historical',
                           'rcp26': 'RCP 2.6',
                           'rcp45': 'RCP 4.5',
                           'rcp85': 'RCP 8.5'}

    #plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    #historical
    hist_data = data.sel(scenario='historical').dropna(dim='year')
    n_models_hist = int(hist_data.n_models.values)
    historical_end_year = int(hist_data.year.values[-1])
    
    ax.plot(hist_data.year, hist_data['odsl_ensemble_mean'] / 10, color=scenario_colors['historical'], linewidth=2.5, label=f'Historical (n={n_models_hist})') #mm -> cm
    ax.fill_between(hist_data.year, (hist_data['odsl_ensemble_mean'] - hist_data['odsl_ensemble_std']) / 10, (hist_data['odsl_ensemble_mean'] + hist_data['odsl_ensemble_std']) / 10, color='black', alpha=0.15)
    
    upper_bound = (hist_data['odsl_ensemble_mean'] + hist_data['odsl_ensemble_std']) / 10
    lower_bound = (hist_data['odsl_ensemble_mean'] - hist_data['odsl_ensemble_std']) / 10
    ax.plot(hist_data.year, upper_bound, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.plot(hist_data.year, lower_bound, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    
    #historical mean
    hist_mean = float(hist_data['odsl_ensemble_mean'].mean().item() / 10)
    
    #future scenarios
    for scenario in future_scenarios:
        future_data = data.sel(scenario=scenario, year=slice(historical_end_year + 1, None)).dropna(dim='year')
        
        if future_data.year.size > 0:
            color    = scenario_colors[scenario]
            n_models = int(future_data.n_models.values)
            label    = f'{scenario_labels[scenario]} (n={n_models})'
            
            last_hist = hist_data.isel(year=-1)
            connected = xr.concat([last_hist, future_data], dim='year', data_vars='minimal', coords='minimal', compat='override')
            
            ax.plot(connected.year, connected['odsl_ensemble_mean'] / 10, color=color, linewidth=2, label=label)
            ax.fill_between(connected.year, (connected['odsl_ensemble_mean'] - connected['odsl_ensemble_std']) / 10, (connected['odsl_ensemble_mean'] + connected['odsl_ensemble_std']) / 10, color=color, alpha=0.15)
            #uncertainty bounds
            upper_bound = (connected['odsl_ensemble_mean'] + connected['odsl_ensemble_std']) / 10
            lower_bound = (connected['odsl_ensemble_mean'] - connected['odsl_ensemble_std']) / 10
            ax.plot(connected.year, upper_bound, color=color, linewidth=0.8, linestyle=':', alpha=0.5)
            ax.plot(connected.year, lower_bound, color=color, linewidth=0.8, linestyle=':', alpha=0.5)
    
    #zero reference line
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.8, alpha=0.5)
    
    #observed period shading
    ax.axvline(START_YEAR, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(END_YEAR, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvspan(START_YEAR, END_YEAR, color='green', alpha=0.1, zorder=0)
    
    #formatting
    ax.set_xlim(data.year.min().item(), 2100)
    ax.set_ylabel('ODSL (cm)', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    #legend with std dev explanation
    handles, labels = ax.get_legend_handles_labels()
    observed_patch = mpatches.Patch(color='green', alpha=0.2, label='Observed period')
    handles.append(observed_patch)
    labels.append('Observed period')
    
    ax.legend(handles, labels, loc='upper left', fontsize=10, framealpha=0.9, title=r'Ensemble mean $\pm$ 1 std. dev.', title_fontsize=10)
    
    ax.set_title(f'{CMIP_VERSION} North Atlantic ODSL ensemble projections', fontsize=14, fontweight='bold')
    
    #INSET PLOT
    ax_inset = ax.inset_axes((0.25, 0.49, 0.5, 0.45))

    period_averages = {}
    period_trends   = {}
    final_y_values  = {}
    
    #historical
    obs_period_hist = hist_data.sel(year=slice(START_YEAR, min(END_YEAR, historical_end_year)))
    ax_inset.plot(obs_period_hist.year, obs_period_hist['odsl_ensemble_mean'] / 10, color=scenario_colors['historical'], linewidth=2)
    ax_inset.fill_between(obs_period_hist.year, (obs_period_hist['odsl_ensemble_mean'] - obs_period_hist['odsl_ensemble_std']) / 10, (obs_period_hist['odsl_ensemble_mean'] + obs_period_hist['odsl_ensemble_std']) / 10, color='black', alpha=0.15)
    final_y_values['historical'] = float(obs_period_hist['odsl_ensemble_mean'].isel(year=-1) / 10)

    #uncertainty bounds
    upper_bound = (obs_period_hist['odsl_ensemble_mean'] + obs_period_hist['odsl_ensemble_std']) / 10
    lower_bound = (obs_period_hist['odsl_ensemble_mean'] - obs_period_hist['odsl_ensemble_std']) / 10
    ax_inset.plot(obs_period_hist.year, upper_bound, color='black', linewidth=0.6, linestyle=':', alpha=0.5)
    ax_inset.plot(obs_period_hist.year, lower_bound, color='black', linewidth=0.6, linestyle=':', alpha=0.5)
    
    #historical trends
    hist_years    = obs_period_hist.year.values
    hist_values   = obs_period_hist['odsl_ensemble_mean'].values / 10
    hist_trend, _ = np.polyfit(hist_years, hist_values, 1)
    period_trends['historical'] = hist_trend * 10

    #future scenarios inset
    for scenario in future_scenarios:
        future_data = data.sel(scenario=scenario, year=slice(historical_end_year + 1, END_YEAR)).dropna(dim='year')
        
        if future_data.year.size > 0:
            color = scenario_colors[scenario]
            
            #connect to last historical point
            last_hist = obs_period_hist.isel(year=-1)
            connected = xr.concat([last_hist, future_data], dim='year', data_vars='minimal', coords='minimal', compat='override')
            
            ax_inset.plot(connected.year, connected['odsl_ensemble_mean'] / 10, color=color, linewidth=2)
            ax_inset.fill_between(connected.year, (connected['odsl_ensemble_mean'] - connected['odsl_ensemble_std']) / 10, (connected['odsl_ensemble_mean'] + connected['odsl_ensemble_std']) / 10, color=color, alpha=0.15)
            
            #uncertainty bounds
            upper_bound = (connected['odsl_ensemble_mean'] + connected['odsl_ensemble_std']) / 10
            lower_bound = (connected['odsl_ensemble_mean'] - connected['odsl_ensemble_std']) / 10
            ax_inset.plot(connected.year, upper_bound, color=color, linewidth=0.6, linestyle=':', alpha=0.5)
            ax_inset.plot(connected.year, lower_bound, color=color, linewidth=0.6, linestyle=':', alpha=0.5)

            #average over observed period
            combined_data             = xr.concat([obs_period_hist, future_data], dim='year')
            period_avg                = float(combined_data['odsl_ensemble_mean'].mean().item() / 10) 
            period_averages[scenario] = period_avg

            #trend over observed period
            combined_years           = combined_data.year.values
            combined_values          = combined_data['odsl_ensemble_mean'].values / 10 # type: ignore[operator]
            trend, _                 = np.polyfit(combined_years, combined_values, 1)
            period_trends[scenario]  = trend * 10
            final_y_values[scenario] = float(combined_data['odsl_ensemble_mean'].isel(year=-1).item() / 10)

    #observed data inset
    obs_avg   = None
    obs_trend = None
    if obs_results is not None and 'odsl_yearly' in obs_results:

        odsl_yearly = obs_results['odsl_yearly']
        
        #regional mean
        #weighing for latitude
        weights      = np.cos(np.deg2rad(odsl_yearly.latitude))
        weights.name = "weights"
        
        #select North Atlantic region
        mask              = create_region_mask(odsl_yearly.isel(year=0), EXTENT)
        obs_regional_mean = odsl_yearly.where(mask).weighted(weights).mean(dim=['latitude', 'longitude'])

        #period
        obs_period_obs = obs_regional_mean.sel(year=slice(START_YEAR, END_YEAR))

        #reference to 1993-1997 period
        ref_period     = obs_period_obs.sel(year=slice(1993, 1997))
        obs_ref_mean   = float(ref_period.mean().item())
        obs_period_obs = obs_period_obs - obs_ref_mean
        
        ax_inset.plot(obs_period_obs.year, obs_period_obs / 10, color='green', linewidth=2, linestyle='-', label='Observed', zorder=15)
        
        #avg
        obs_avg = float(obs_period_obs.mean() / 10)

        #trend
        obs_years                  = obs_period_obs.year.values
        obs_values                 = obs_period_obs.values / 10
        obs_trend_val, _           = np.polyfit(obs_years, obs_values, 1)
        obs_trend                  = obs_trend_val * 10
        final_y_values['observed'] = float(obs_period_obs.isel(year=-1) / 10)
    
    #table in figure
    #average text box inset
    avg_text_lines = [r"                             $\bf{{Mean\ |\ Trend}}$"] # type: ignore
    
    #observed avg and trend
    if obs_avg is not None:
        avg_text_lines.append(f"                 Observed: {obs_avg:.2f} cm | {obs_trend:+.1f} mm/yr")

    #scenario avg and trend
    for scenario in future_scenarios:
        if scenario in period_averages:
            avg_text_lines.append(f"Historical + {scenario_labels[scenario]}:  {period_averages[scenario]:.2f} cm | {period_trends[scenario]:+.1f} mm/yr")
    
    avg_text = '\n'.join(avg_text_lines)
    
    #avg text box inset
    ax_inset.text(0.3, 0.65, avg_text, transform=ax_inset.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    #formatting inset
    ax_inset.set_xlim(START_YEAR, END_YEAR)
    ax_inset.set_xlabel('Year', fontsize=9)
    ax_inset.set_ylabel('ODSL (cm)', fontsize=9)
    ax_inset.tick_params(axis='both', labelsize=8)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_inset.set_title(f'Observed period ({START_YEAR}-{END_YEAR})', fontsize=10, fontweight='bold')
    ax_inset.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    #rectangle indicating inset
    ax.indicate_inset((START_YEAR, ax_inset.get_ylim()[0], END_YEAR - START_YEAR, ax_inset.get_ylim()[1] - ax_inset.get_ylim()[0]), edgecolor='green', linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{CMIP_VERSION.lower()}_scenario_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_yearly_odsl_anomaly(obs_results, fig_dir):
    """Yearly observed ODSL anomaly for each year in the analysis period."""
    
    print("Generating yearly observed ODSL anomaly figure...")
    
    odsl_anomaly = obs_results['odsl_yearly_anomaly'] /10 #mm -> cm
    
    vmax = odsl_anomaly.quantile(0.99, skipna=True).item()
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 51)
    
    num_years = len(odsl_anomaly.year)
    ncols = 4
    nrows = (num_years + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3.2), subplot_kw={'projection': ccrs.Robinson(central_longitude=-40)})
    axes = axes.flatten()
    
    im = None
    for i, year in enumerate(odsl_anomaly.year.values):
        ax = axes[i]
        data_for_year = odsl_anomaly.sel(year=year)
        
        #fill NaNs with a value outside the colormap range to help contourf
        data_filled = data_for_year.fillna(0)
        
        im = data_filled.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=levels, extend='both', add_colorbar=False)
        
        #land mask covers the filled ocean NaNs
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=3)
        ax.coastlines(linewidth=0.5, zorder=4)
        ax.set_global()
        ax.set_title(f'{int(year)}', fontsize=10)
    
    for j in range(num_years, len(axes)):
        axes[j].set_visible(False)
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, hspace=0.15, wspace=0.05)
    
    if im:
        cbar_ax = fig.add_axes((0.2, 0.08, 0.6, 0.015))
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
        n_ticks = 7
        tick_values = np.linspace(vmin, vmax, n_ticks)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{v:.0f}' for v in tick_values])
        cbar.set_label('ODSL anomaly (cm)', fontsize=11)
    
    fig.suptitle('Yearly observed ODSL anomaly', fontsize=16, fontweight='bold', y=0.95)
    
    output_path = os.path.join(fig_dir, 'observed_odsl_yearly_anomaly.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_spatial_eofs(all_eof_results, fig_dir, num_modes_to_plot=3):
    """Visualizes the spatial patterns of the EOFs for each specified data source."""

    sources_to_plot = ['observed', 'multi model mean']

    for source_name in sources_to_plot:
        if source_name not in all_eof_results:
            print(f"Skipping spatial EOF plot for '{source_name}'; results not found.")
            continue

        print(f"Generating spatial EOF plot for: {source_name}")

        eof_results = all_eof_results[source_name]
        eofs = eof_results['eofs']
        variance = eof_results['variance_fractions']

        proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])

        #plot
        fig, axes = plt.subplots(nrows=num_modes_to_plot, figsize=(7, 5 * num_modes_to_plot), subplot_kw={'projection': proj})

        if num_modes_to_plot == 1: 
            axes = [axes]

        mesh = None
        for i in range(num_modes_to_plot):
            ax = axes[i]
            is_bottom = (i == num_modes_to_plot - 1)
            add_map_features(ax, EXTENT, is_left=True, is_bottom=is_bottom)
            mode_data = eofs.sel(mode=i)

            mode_vmax = abs(mode_data).max(skipna=True).compute().item()
            
            mesh = ax.contourf(mode_data.longitude, mode_data.latitude, mode_data, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=-mode_vmax, vmax=mode_vmax, levels=50)
            ax.set_title(f"EOF mode {i+1} ({variance.sel(mode=i).item()*100:.1f}% variance)")

        fig.suptitle(f'Spatial EOF patterns for {source_name.upper()}', fontsize=16, fontweight='bold', y=0.92)

        #shared colorbar
        if mesh:
            cbar_ax = fig.add_axes((0.88, 0.125, 0.025, 0.7))
            cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Amplitude', fontsize=14)
        
        output_path = os.path.join(fig_dir, f'spatial_eofs_{source_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close(fig)

def plot_pc_timeseries(dual_eof_results, fig_dir, num_modes_to_plot=5, normalize=False):
    """PC time series for all sources, standard (solid) vs rotated (dashed)."""

    print(f"Generating PC time series plot ({'normalised' if normalize else 'raw'})...")

    PRIORITY           = ['observed', 'multi model mean']
    individual_sources = sorted([s for s in set(k.rsplit('__', 1)[0] for k in dual_eof_results) if s not in PRIORITY])
    source_order       = individual_sources + [s for s in PRIORITY if f'{s}__unrotated' in dual_eof_results]

    fig, axes = plt.subplots(num_modes_to_plot, 1, figsize=(14, num_modes_to_plot * 3), sharex=True)
    if num_modes_to_plot == 1:
        axes = [axes]

    for mode_idx in range(num_modes_to_plot):
        ax = axes[mode_idx]
        subplot_letter = chr(97 + mode_idx)

        for source_name in source_order:
            unrot_key = f'{source_name}__unrotated'
            rot_key   = f'{source_name}__rotated'

            if unrot_key not in dual_eof_results or rot_key not in dual_eof_results:
                continue

            pc_unrot = dual_eof_results[unrot_key]['pcs'].sel(mode=mode_idx)
            pc_rot   = dual_eof_results[rot_key]['pcs'].sel(mode=mode_idx)

            if normalize:
                pc_unrot = (pc_unrot - pc_unrot.mean()) / pc_unrot.std()
                pc_rot   = (pc_rot - pc_rot.mean()) / pc_rot.std()

            time_vals = pc_unrot.time.values

            if source_name == 'observed':
                color, lw, zorder = 'green', 2.0, 10
                label_unrot = 'Observed (standard)'
                label_rot   = 'Observed (rotated)'
            elif source_name == 'multi model mean':
                color, lw, zorder = 'black', 2.0, 9
                label_unrot = 'Multi-model mean (standard)'
                label_rot   = 'Multi-model mean (rotated)'
            else:
                color, lw, zorder = 'grey', 0.6, 1
                if source_name == individual_sources[0]:
                    label_unrot = 'Individual models (standard)'
                    label_rot   = 'Individual models (rotated)'
                else:
                    label_unrot = ''
                    label_rot   = ''

            ax.plot(time_vals, pc_unrot, color=color, linewidth=lw, linestyle='-', alpha=0.8 if source_name in PRIORITY else 0.4, zorder=zorder, label=label_unrot)
            ax.plot(time_vals, pc_rot, color=color, linewidth=lw, linestyle='--', alpha=0.8 if source_name in PRIORITY else 0.4, zorder=zorder, label=label_rot)

        ax.axhline(0, color='grey', linewidth=0.5)
        suffix = ' (normalised)' if normalize else ''
        var_unrot = dual_eof_results[f'observed__unrotated']['variance_fractions'].sel(mode=mode_idx).item() * 100
        var_rot   = dual_eof_results[f'observed__rotated']['variance_fractions'].sel(mode=mode_idx).item() * 100
        ax.set_title(f'{subplot_letter}) PC {mode_idx + 1} — Obs. var: {var_unrot:.1f}% (std) / {var_rot:.1f}% (rot)', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'Amplitude{suffix}', fontsize=9)
        ax.set_xlim(START_YEAR, END_YEAR)
        ax.grid(True, linestyle='--', alpha=0.4)

        if mode_idx == 0:
            legend_handles = [Line2D([0], [0], color='green', lw=2, ls='-', label='Observed (standard)'),
                              Line2D([0], [0], color='green', lw=2, ls='--', label='Observed (rotated)'),
                              Line2D([0], [0], color='black', lw=2, ls='-', label='Multi-model mean (standard)'),
                              Line2D([0], [0], color='black', lw=2, ls='--', label='Multi-model mean (rotated)'),
                              Line2D([0], [0], color='grey', lw=1, ls='-', alpha=0.6, label='Models (standard)'),
                              Line2D([0], [0], color='grey', lw=1, ls='--', alpha=0.6, label='Models (rotated)')]
            ax.legend(handles=legend_handles, loc='lower right', fontsize=8, ncol=3)

    axes[-1].set_xlabel('Year', fontsize=11)

    eof_label = 'Rotated EOF' if USE_ROTATED_EOF else 'EOF'
    norm_label = ' (normalised)' if normalize else ''
    fig.suptitle(f'Principal component time series: standard vs rotated{norm_label}', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    norm_tag    = '_normalised' if normalize else ''
    output_path = os.path.join(fig_dir, f'pc_timeseries{norm_tag}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_correlation_biplot(all_eof_results, all_correlation_results, fig_dir, mode_x=0, mode_y=1, normalize=True):
    """Biplot of PCs (normalised to unit variance) and observed index correlations."""

    print(f"Generating combined correlation biplot (PC{mode_x+1} vs PC{mode_y+1})...")

    try:
        index_names = sorted([k for k in next(iter(all_correlation_results.values())).keys() if k != 'expl_var'])
    except StopIteration:
        index_names = []

    PRIORITY           = ['observed', 'multi model mean']
    individual_sources = sorted([s for s in all_eof_results if s not in PRIORITY])
    source_order       = individual_sources + [s for s in PRIORITY if s in all_eof_results]

    #color list climate indices
    _index_palette = ['#e41a1c', '#377eb8', '#ff7f00', '#984ea3', '#a65628', '#f781bf', "#dee602", "#00eeff", "#0059ff", "#ff00ff"]
    index_colors = {name: _index_palette[i % len(_index_palette)] for i, name in enumerate(index_names)}

    fig, ax = plt.subplots(figsize=(10, 10))

    global_max_pc = 0

    year_min, year_max = None, None
    for source_name in source_order:

        if source_name not in all_eof_results:
            continue

        eof_results = all_eof_results[source_name]
        pc_x_raw    = eof_results['pcs'].sel(mode=mode_x)
        pc_y_raw    = eof_results['pcs'].sel(mode=mode_y)

        #normalisation PCs
        if normalize:
            pc_x = (pc_x_raw - pc_x_raw.mean()) / pc_x_raw.std()
            pc_y = (pc_y_raw - pc_y_raw.mean()) / pc_y_raw.std()
        else:
            pc_x = pc_x_raw
            pc_y = pc_y_raw

        #time
        if 'time' in pc_x_raw.coords:
            try:
                years = pc_x_raw.time.dt.year.values
            except AttributeError:
                years = pc_x_raw.time.values
        elif 'year' in pc_x_raw.coords:
            years = pc_x_raw.year.values
        else:
            years = np.arange(len(pc_x))

        #normalisation time for alpha fading for chronology
        year_min  = years.min() if year_min is None else min(year_min, years.min())
        year_max  = years.max() if year_max is None else max(year_max, years.max())
        year_norm = (years - year_min) / (year_max - year_min) if year_max > year_min else np.ones_like(years)

        if source_name == 'observed':
            base_color        = 'green'
            marker, s, zorder = 'D', 70, 10
            label             = 'Observed'
        elif source_name == 'multi model mean':
            base_color        = 'black'
            marker, s, zorder = 's', 70, 9
            label             = 'Multi-model mean'
        else:
            base_color        = 'grey'
            marker, s, zorder = 'o', 40, 2
            label             = 'Individual models' if source_name == individual_sources[0] else None

        #fading
        alpha_min = 0.3
        alphas    = alpha_min + (1.0 - alpha_min) * year_norm
        rgba      = np.array([mcolors.to_rgba(base_color, alpha=a) for a in alphas])

        edge = 'none' if source_name not in PRIORITY else 'k'
        edge_lw = 0.0 if source_name not in PRIORITY else 0.3

        ax.scatter(pc_x, pc_y, c=rgba, marker=marker, s=s, edgecolor=edge, linewidth=edge_lw, zorder=zorder, label=label)

        local_max     = max(np.max(np.abs(pc_x)), np.max(np.abs(pc_y)))
        global_max_pc = max(global_max_pc, local_max)

    scale_factor = global_max_pc * 0.9
    #reference circle
    circle = Circle((0, 0), scale_factor, color='gray', fill=False, linestyle=':', linewidth=0.8, zorder=1)
    ax.add_patch(circle)
    angle  = np.deg2rad(45)
    ax.text(scale_factor * np.cos(angle), scale_factor * np.sin(angle), 'r = 1.0', color='gray', rotation=315, ha='left', va='center', fontsize=9)

    #correlation arrows
    obs_correlations = all_correlation_results.get('observed', {})
    for index_name in index_names:
        if index_name not in obs_correlations:
            continue
        corr_x = obs_correlations[index_name].sel(mode=mode_x).item()
        corr_y = obs_correlations[index_name].sel(mode=mode_y).item()

        vec_x = corr_x * scale_factor
        vec_y = corr_y * scale_factor

        ax.annotate('', xy=(vec_x, vec_y), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=index_colors[index_name], lw=3.0), zorder=10)

        r = np.sqrt(vec_x**2 + vec_y**2)
        if r > 0:
            offset_x = vec_x / r * global_max_pc * 0.08
            offset_y = vec_y / r * global_max_pc * 0.08
        else:
            offset_x = offset_y = 0
        ax.text(vec_x + offset_x, vec_y + offset_y, index_name.upper(), color=index_colors[index_name], fontweight='bold', fontsize=9, ha='center', va='center', zorder=11)

    #formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    limit = global_max_pc * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')

    suffix = ' (normalised)' if normalize else ''
    ax.set_xlabel(f'PC {mode_x + 1}{suffix}', fontsize=12)
    ax.set_ylabel(f'PC {mode_y + 1}{suffix}', fontsize=12)
    ax.set_title(f'EOF correlation biplot (PC{mode_x+1} vs PC{mode_y+1})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    #legend entries
    #sources
    source_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=6, markeredgecolor='k', markeredgewidth=0.3, label='Individual models'),
                      Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=8, markeredgecolor='k', markeredgewidth=0.3, label='Observed'),
                      Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, markeredgecolor='k', markeredgewidth=0.3, label='Multi-model mean'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=6, alpha=0.15, markeredgecolor='k', markeredgewidth=0.3, label=f'Early ({year_min or "?"})'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=6, alpha=1.0, markeredgecolor='k', markeredgewidth=0.3, label=f'Late ({year_max or "?"})')]

    source_legend = ax.legend(handles=source_handles, loc='upper left', fontsize=9, title='Sources', title_fontproperties={'weight': 'bold'})
    ax.add_artist(source_legend)

    #climate indices
    index_handles = []
    for n in index_names:
        if n not in obs_correlations:
            continue
        handle = Line2D([0], [0], color=index_colors[n], lw=3, marker='>', markersize=8, markeredgecolor=index_colors[n], label=n.upper())
        index_handles.append(handle)
    ax.legend(handles=index_handles, loc='lower left', fontsize=9, title='Climate indices (observed)', title_fontproperties={'weight': 'bold'})
        
    plt.tight_layout()
    norm_tag = '_normalised' if normalize else ''
    output_path = os.path.join(fig_dir, f'correlation_biplot_pc{mode_x+1}_vs_pc{mode_y+1}{norm_tag}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def export_eof_results_to_csv(all_eof_results, all_correlation_results, fig_dir):
    """CSV file summarizing the results of the EOF analysis for all data sources."""

    print("Exporting full EOF analysis results to CSV...")
    
    table_data = []
    
    source_order = sorted(all_eof_results.keys())
    
    try:
        index_names = sorted(list(next(iter(all_correlation_results.values())).keys()))
    except StopIteration:
        print("No correlation results found to export.")
        index_names = []

    for source_name in source_order:
        if source_name not in all_eof_results: continue
        
        eof_res = all_eof_results[source_name]
        corr_res = all_correlation_results.get(source_name, {})
        
        for mode in eof_res['pcs'].mode.values:
            row_data = {'Source': source_name, 'Mode': mode + 1, 'Variance fraction (%)': eof_res['variance_fractions'].sel(mode=mode).item() * 100}
            
            for index_name in index_names:
                col_name = f'{index_name.upper()} correlation'
                if index_name in corr_res:
                    row_data[col_name] = corr_res[index_name].sel(mode=mode).item()
                else:
                    row_data[col_name] = None
            
            table_data.append(row_data)

    if not table_data:
        print("No EOF data available to export to CSV.")
        return

    #DataFrame
    df = pd.DataFrame(table_data)
    
    #save to csv
    output_path = os.path.join(fig_dir, 'full_eof_analysis_summary.csv')
    df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"Successfully saved EOF summary to {output_path}")

def plot_eof_summary_table(all_eof_results, all_correlation_results, fig_dir):
    """Colored table with variance fraction and index correlations from EOF analysis."""

    print("Generating EOF summary table figure...")

    table_data = []

    #top rows observed and multi-model mean, after individual models in alphabetical order
    PRIORITY_ROWS = ['observed', 'multi model mean'] 
    priority_order = [s for s in PRIORITY_ROWS if s in all_eof_results]
    remaining_order = sorted([s for s in all_eof_results.keys() if s not in PRIORITY_ROWS])
    source_order = priority_order + remaining_order
    n_priority = len(priority_order)

    try:
        index_names = sorted([k for k in next(iter(all_correlation_results.values())).keys() if k != 'expl_var'])
    except StopIteration:
        index_names = []

    for source_name in source_order:
        if source_name not in all_eof_results: continue
            
        eof_res = all_eof_results[source_name]
        corr_res = all_correlation_results.get(source_name, {})
        
        row_data = {'Source': source_name}
        
        available_modes = eof_res['pcs'].mode.values
        limit = min(len(available_modes), EOF_N_MODES)
        modes_to_process = available_modes[:limit]

        for mode in modes_to_process:

            #variance fraction
            var_frac = eof_res['variance_fractions'].sel(mode=mode).item() * 100
            row_data[f'Mode {mode+1}\nVar. (%)'] = var_frac
            
            #correlations
            for index_name in index_names:
                col_name = f'Mode {mode+1}\n{index_name.upper()}'
                if index_name in corr_res:
                    row_data[col_name] = corr_res[index_name].sel(mode=mode).item()
                else:
                    row_data[col_name] = None # type: ignore
        
        table_data.append(row_data)

    if not table_data:
        print("No EOF data to create a summary table.")
        return

    df = pd.DataFrame(table_data).set_index('Source')

    short_col_labels = [c.split('\n')[-1] for c in df.columns]

    #plotting
    fig, ax = plt.subplots(figsize=(20, 0.1))
    ax.axis('off')

    #darker background observed and multi-model mean
    table = ax.table(cellText=df.round(2).values.tolist(), colLabels=short_col_labels, rowLabels=df.index.tolist(), loc='center', cellLoc='center', rowColours=['#d0d0d0' if i < n_priority else '#f2f2f2' for i in range(len(df))])

    #mode number header row
    cols_per_mode = 1 + len(index_names)
    header_height = table[0, 0].get_height()

    for j, col_name in enumerate(df.columns):

        mode_idx = j // cols_per_mode
        is_center_col = (j % cols_per_mode) == (cols_per_mode // 2)
        
        cell = table.add_cell(row=-1, col=j, width=table[0, j].get_width(), height=header_height, text=f"Mode {mode_idx+1}" if is_center_col else "", loc='center', facecolor='#e0e0e0')
        cell.get_text().set_weight('bold')  # type: ignore

        edges = 'TB'
        if (j % cols_per_mode) == 0: edges += 'L'
        if (j % cols_per_mode) == (cols_per_mode - 1): edges += 'R'
        cell.visible_edges = edges  

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)

    #colormaps for cell scaling
    var_cmap = plt.get_cmap('YlGn')
    var_norm = Normalize(vmin=0, vmax=100)
    corr_cmap = plt.get_cmap('coolwarm')
    corr_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)

    for i in range(len(df)):
        is_priority = i < n_priority

        #bold text observed and multi-model mean rows and column header modes
        row_label_cell = table[i + 1, -1]
        t = row_label_cell.get_text()
        if is_priority:
            t.set_fontweight('bold')

        for j in range(len(df.columns)):
            cell = table[i + 1, j]
            val = df.iloc[i, j]
            col_name = df.columns[j]

            t = cell.get_text()
            if is_priority:
                t.set_fontweight('bold')

            if pd.isna(val):
                cell.set_facecolor('#d0d0d0' if is_priority else 'white')
                continue

            val_float = float(val)  # type: ignore

            if 'Var' in col_name:
                cell.set_facecolor(var_cmap(var_norm(val_float)))
            else:
                cell.set_facecolor(corr_cmap(corr_norm(val_float)))
                if abs(val_float) > 0.6:
                    cell.get_text().set_color('white')

    fig.canvas.draw()
    renderer = fig.canvas.renderer  # type: ignore

    table_top = table.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
    fig.text(0.5, table_top + 9, 'EOF variance and climate indices correlations', fontsize=16, fontweight='bold', ha='center', va='bottom')
    fig.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98)

    output_path = os.path.join(fig_dir, 'eof_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    
def plot_scatter_comparison(cmip_results, sliding_results, fig_dir, plot_var=None):
    """2D density scatter plot comparing modeled with observed ODSL for each grid cell."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Generating scatter plot for ODSL {cfg['name']}...")

    if plot_var == 'trend':
        model_data = cmip_results['model_mean_trend']
        obs_data = sliding_results['odsl_mm_yr_regridded']
    elif plot_var == 'variability':
        model_data = cmip_results['model_mean_variability']
        obs_data = sliding_results['odsl_var_obs_regridded']
    elif plot_var == 'ODSL':
        model_data = cmip_results['model_mean_odsl']
        obs_data = sliding_results['odsl_obs_regridded']
    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    #flatten
    obs_flat = obs_data.values.flatten()
    model_flat = model_data.values.flatten()

    #mask
    valid_mask = ~np.isnan(obs_flat) & ~np.isnan(model_flat)
    x_points = model_flat[valid_mask]
    y_points = obs_flat[valid_mask]

    #statistics
    region_mask = create_region_mask(model_data, EXTENT)
    stats_w = calculate_weighted_stats(model_data, region_mask, data_y=obs_data)
    bias = stats_w['bias'] 
    pct_5_model, pct_95_model = np.percentile(x_points, [5, 95])
    pct_5_obs, pct_95_obs = np.percentile(y_points, [5, 95])
    range_ratio = (pct_95_model - pct_5_model) / (pct_95_obs - pct_5_obs)
    median_model = np.median(x_points)
    median_obs = np.median(y_points)

    #linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x_points, y_points)
    r_squared = r_value**2 # type: ignore
    normalized_std_dev = stats_w['std_x'] / stats_w['std_y']
    num_points = len(x_points)

    #plotting
    fig, ax = plt.subplots(figsize=(9, 8))

    abs_max = max(abs(np.nanmin(x_points)), abs(np.nanmax(x_points)), abs(np.nanmin(y_points)), abs(np.nanmax(y_points)))
    lims = (-abs_max * 1.1, abs_max * 1.1)

    #hexbin with logarithmic scale
    hb = ax.hexbin(x_points, y_points, gridsize=100, cmap='inferno', norm=LogNorm(), extent=(lims[0], lims[1], lims[0], lims[1]))
    
    #colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('Number of grid cells')

    #axes limits
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    #lines
    ax.plot(lims, lims, 'k--', linewidth=1, label='1:1 line', alpha=0.6)
    ax.plot(x_points, slope * x_points + intercept, 'r-', linewidth=2, alpha=0.9, label=f'Best fit (y={slope:.2f}x + {intercept:.2f})')

    #statistics
    stats_text = (f"N = {num_points}\n"
                  r"R$^2$ = {r_squared:.2f}\n"
                  f"Bias = {bias:.2f} {cfg['units']}\n"
                  f"Std ratio (mod/obs) = {normalized_std_dev:.2f}\n"
                  f"Range ratio (5-95%) = {range_ratio:.2f}\n"
                  f"Median obs = {median_obs:.2f}\n"
                  f"Median model = {median_model:.2f}"
                 )

    #text box
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    #formatting
    ax.set_xlabel(f"Modeled {cfg['name']} ({cfg['units']})", fontsize=12)
    ax.set_ylabel(f"Observed {cfg['name']} ({cfg['units']})", fontsize=12)
    ax.set_title(f"Grid-cell comparison of {cfg['name']} ({START_YEAR}-{END_YEAR})", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'scatter_comparison_{cfg["name"]}.png'), dpi=300)
    #plt.show()
    plt.close(fig)

def plot_scatter_comparison_individual_models(cmip_results, sliding_results, fig_dir, plot_var=None):
    """Generate a grid of scatter plots, comparing each individual CMIP model to the observed ODSL for the common period."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Generating individual model scatter plot grid for ODSL {cfg['name']}...")

    if plot_var == 'trend':
        all_model_data = cmip_results['model_trend']
        obs_data = sliding_results['odsl_mm_yr_regridded']
    elif plot_var == 'variability':
        all_model_data = cmip_results['model_variability']
        obs_data = sliding_results['odsl_var_obs_regridded']
    elif plot_var == 'ODSL':
        all_model_data = cmip_results['model_odsl_mean']
        obs_data = sliding_results['odsl_obs_regridded']
    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    model_names = all_model_data.model.values
    num_models = len(model_names)

    #flatten
    all_model_flat = all_model_data.values.flatten()
    obs_flat_for_lims = obs_data.values.flatten()
    all_points = np.concatenate([all_model_flat, obs_flat_for_lims])

    #axes limits
    plot_max = max(abs(np.nanmin(all_points)), abs(np.nanmax(all_points)))
    lims = [-plot_max * 1.1, plot_max * 1.1]

    #subplot grid
    ncols = 4
    nrows = (num_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4.5), sharex=True, sharey=True)
    axes = axes.flatten()

    hb = None
    #loop
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_data_single = all_model_data.sel(model=model_name)

        #1D data arrays
        obs_flat = obs_data.values.flatten()
        model_flat = model_data_single.values.flatten()
        valid_mask = ~np.isnan(obs_flat) & ~np.isnan(model_flat)
        x_points = model_flat[valid_mask]
        y_points = obs_flat[valid_mask]

        #statistics
        region_mask = create_region_mask(model_data_single, EXTENT)
        stats_w = calculate_weighted_stats(model_data_single, region_mask, data_y=obs_data)
        pcc_w = stats_w['pcc']
        rmse_w = stats_w['rmse']

        #linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x_points, y_points)
        r_squared = r_value**2 # type: ignore
        normalized_std_dev = stats_w['std_x'] / stats_w['std_y']

        #hexbin
        hb = ax.hexbin(x_points, y_points, gridsize=100, cmap='inferno', norm=LogNorm(), extent=[lims[0], lims[1], lims[0], lims[1]])

        #lines
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6)
        ax.plot(x_points, slope * x_points + intercept, 'r-', linewidth=2, alpha=0.9)

        #statistics box
        stats_text = (f"PCC = {pcc_w:.2f}\n" f"RMSE = {rmse_w:.2f}\n" f"R² = {r_squared:.2f}\n" f"Std ratio = {normalized_std_dev:.2f}\n" f"y = {slope:.2f}x + {intercept:.2f}")
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        ax.text(0.5, 0.97, model_name, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='center', va='top')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.6)

    #formatting
    for j in range(num_models, len(axes)):
        axes[j].set_visible(False)

    #labels and title
    fig.supxlabel(f"Modeled ODSL {cfg['name']} ({cfg['units']})", fontsize=14, y=0.05)
    fig.supylabel(f"Observed ODSL {cfg['name']} ({cfg['units']})", fontsize=14, x=0.02)
    fig.suptitle(f"Grid-cell comparison for individual CMIP models vs. observations ({cfg['name']})", fontsize=16, fontweight='bold', y=0.94)

    fig.subplots_adjust(left=0.05, bottom=0.07, right=0.92, top=0.92, wspace=0.02, hspace=0.1)

    #shared colorbar
    pos_top_right = axes[ncols - 1].get_position()
    pos_bottom = axes[num_models - 1].get_position()
    if hb:
        cbar_ax = fig.add_axes((pos_top_right.x1 + 0.015, pos_bottom.y0, 0.02, pos_top_right.y1 - pos_bottom.y0))
        fig.colorbar(hb, cax=cbar_ax, label='Number of grid cells')

    plt.savefig(os.path.join(fig_dir, f'scatter_comparison_individual_models_{cfg["name"]}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

def plot_incrementing_window_skill(incrementing_window_results, fig_dir, plot_var=None, picontrol_incrementing_results=None):
    """Plots the evolution of PCC and RMSE from the incrementing window analysis."""

    print("Generating plot for incrementing window skill analysis...")

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    #data
    pcc_data = incrementing_window_results['pcc'].sel(variable=plot_var)
    rmse_data = incrementing_window_results['rmse'].sel(variable=plot_var)

    end_years = pcc_data.end_year.values
    sources = pcc_data.source.values

    #sign agreement only if available and relevant
    has_sign_agreement = 'sign_agreement' in incrementing_window_results
    include_sign_agreement = has_sign_agreement and plot_var in ['trend', 'ODSL']
    n_subplots = 3 if include_sign_agreement else 2

    #plotting
    subplot_height = 3.5
    fig_height = n_subplots * subplot_height
    fig, axes = plt.subplots(nrows=n_subplots, ncols=1, figsize=(12, fig_height), sharex=True)
    
    all_pcc_vals = []
    all_rmse_vals = []

    #PCC
    ax1 = axes[0]
    is_first_model = True
    for source in sources:
        if source != 'multi model mean':
            vals = pcc_data.sel(source=source).values
            all_pcc_vals.extend(vals[~np.isnan(vals)])
            if is_first_model:
                ax1.plot(end_years, vals, color='lightgrey', linewidth=1, label='Individual models')
                is_first_model = False
            else:
                ax1.plot(end_years, vals, color='lightgrey', linewidth=1)
            
    mmm_pcc = pcc_data.sel(source='multi model mean').values
    all_pcc_vals.extend(mmm_pcc[~np.isnan(mmm_pcc)])
    ax1.plot(end_years, mmm_pcc, color='black', linewidth=2.5, label='Multi-model mean')

    ax1.set_title('PCC vs. time window', fontweight='bold')
    ax1.set_ylabel('PCC')
    ax1.grid(True, linestyle='--', alpha=0.6)

    #RMSE
    ax2 = axes[1]
    is_first_model = True
    for source in sources:
        if source != 'multi model mean':
            vals = rmse_data.sel(source=source).values
            all_rmse_vals.extend(vals[~np.isnan(vals)])
            if is_first_model:
                ax2.plot(end_years, vals, color='lightgrey', linewidth=1, label='Individual models')
                is_first_model = False
            else:
                ax2.plot(end_years, vals, color='lightgrey', linewidth=1)
    
    mmm_rmse = rmse_data.sel(source='multi model mean').values
    all_rmse_vals.extend(mmm_rmse[~np.isnan(mmm_rmse)])
    ax2.plot(end_years, mmm_rmse, color='black', linewidth=2.5, label='Multi-model mean')
    
    ax2.set_title('RMSE vs. time window', fontweight='bold')
    ax2.set_ylabel(f'RMSE ({cfg["units"]})')
    ax2.grid(True, linestyle='--', alpha=0.6)

    #picontrol
    if picontrol_incrementing_results is not None:
        pi_pcc_data = picontrol_incrementing_results['pcc'].sel(variable=plot_var)
        pi_rmse_data = picontrol_incrementing_results['rmse'].sel(variable=plot_var)
        pi_end_years = picontrol_incrementing_results.end_year.values
        
        #multi-model mean \pm std across individual models
        pi_sources = [s for s in pi_pcc_data.source.values if s != 'multi model mean']
        pi_pcc_individuals = pi_pcc_data.sel(source=pi_sources)
        pi_rmse_individuals = pi_rmse_data.sel(source=pi_sources)
        
        pi_pcc_mean = pi_pcc_individuals.mean(dim='source').values
        pi_pcc_std = pi_pcc_individuals.std(dim='source').values
        pi_rmse_mean = pi_rmse_individuals.mean(dim='source').values
        pi_rmse_std = pi_rmse_individuals.std(dim='source').values
        
        #y-axis scaling
        all_pcc_vals.extend((pi_pcc_mean - 2*pi_pcc_std).tolist())
        all_pcc_vals.extend((pi_pcc_mean + 2*pi_pcc_std).tolist())
        all_rmse_vals.extend(np.maximum(0, pi_rmse_mean - 2*pi_rmse_std).tolist())
        all_rmse_vals.extend((pi_rmse_mean + 2*pi_rmse_std).tolist())
        
        #PCC
        ax1.fill_between(pi_end_years, pi_pcc_mean - 2*pi_pcc_std, pi_pcc_mean + 2*pi_pcc_std, color='lightblue', alpha=0.4, zorder=0, label=r'piControl $\pm$2$\sigma$')
        ax1.plot(pi_end_years, pi_pcc_mean, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.7, label='piControl multi-model mean')
        
        #RMSE
        ax2.fill_between(pi_end_years, np.maximum(0, pi_rmse_mean - 2*pi_rmse_std), pi_rmse_mean + 2*pi_rmse_std, color='lightblue', alpha=0.4, zorder=0, label=r'piControl $\pm$2$\sigma$')
        ax2.plot(pi_end_years, pi_rmse_mean, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.7, label='piControl multi-model mean')

    #dynamic y-axis scaling 
    pcc_min, pcc_max = np.nanmin(all_pcc_vals), np.nanmax(all_pcc_vals)
    pcc_range = pcc_max - pcc_min
    pcc_center = (pcc_max + pcc_min) / 2
    ax1.set_ylim(pcc_center - pcc_range * 0.55, pcc_center + pcc_range * 0.55)

    rmse_min, rmse_max = np.nanmin(all_rmse_vals), np.nanmax(all_rmse_vals)
    rmse_range = rmse_max - rmse_min
    ax2.set_ylim(max(0, rmse_min - rmse_range * 0.05), rmse_max + rmse_range * 0.05)

    #legend
    if plot_var == 'variability':
        ax1.legend(loc='best')
    else:
        ax2.legend(loc='best')

    #sign agreement
    if include_sign_agreement:
        sign_data = incrementing_window_results['sign_agreement'].sel(variable=plot_var)
        ax3 = axes[2]
        all_sign_vals = []

        is_first_model = True
        for source in sources:
            if source != 'multi model mean':
                vals = sign_data.sel(source=source).values * 100
                all_sign_vals.extend(vals[~np.isnan(vals)])
                if is_first_model:
                    ax3.plot(end_years, vals, color='lightgrey', linewidth=1)
                    is_first_model = False
                else:
                    ax3.plot(end_years, vals, color='lightgrey', linewidth=1)
        
        mmm_sign = sign_data.sel(source='multi model mean').values * 100
        all_sign_vals.extend(mmm_sign[~np.isnan(mmm_sign)])
        ax3.plot(end_years, mmm_sign, color='black', linewidth=2.5)

        #piControl sign agreement
        if picontrol_incrementing_results is not None and 'sign_agreement' in picontrol_incrementing_results:
            pi_sign_data = picontrol_incrementing_results['sign_agreement'].sel(variable=plot_var)
            pi_end_years_sign = picontrol_incrementing_results.end_year.values
            pi_sources_sign = [s for s in pi_sign_data.source.values if s != 'multi model mean']
            pi_sign_individuals = pi_sign_data.sel(source=pi_sources_sign)
            pi_sign_mean = pi_sign_individuals.mean(dim='source').values * 100
            pi_sign_std = pi_sign_individuals.std(dim='source').values * 100
            
            all_sign_vals.extend((pi_sign_mean - 2*pi_sign_std).tolist())
            all_sign_vals.extend((pi_sign_mean + 2*pi_sign_std).tolist())
            
            ax3.fill_between(pi_end_years_sign, pi_sign_mean - 2*pi_sign_std, pi_sign_mean + 2*pi_sign_std, color='lightblue', alpha=0.4, zorder=0)
            ax3.plot(pi_end_years_sign, pi_sign_mean, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.7)
    
        sign_min, sign_max = np.nanmin(all_sign_vals), np.nanmax(all_sign_vals)
        sign_range = sign_max - sign_min
        sign_center = (sign_max + sign_min) / 2
        ax3.set_ylim(sign_center - sign_range * 0.55, sign_center + sign_range * 0.55)

        ax3.set_title('Sign agreement vs. time window', fontweight='bold')
        ax3.set_ylabel('Sign agreement (%)')
        ax3.set_xlabel('End year of time window')
        ax3.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.set_xlabel('End year of time window')

    #title and save
    fig.suptitle(f'Model skill evolution with increasing data availability ({cfg["name"]})', fontsize=16, y=0.96, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    output_path = os.path.join(fig_dir, f'incrementing_window_skill_{cfg["name"]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_smoothing_sensitivity(smoothing_results, fig_dir):
    """Altimetry trend with different levels of smoothing and a shared colorbar."""

    print("Generating smoothing sensitivity plot...")

    if isinstance(smoothing_results, xr.Dataset):
        var_name = list(smoothing_results.data_vars)[0]
        smoothing_results = smoothing_results[var_name]

    sigmas = smoothing_results.sigma.values

    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 7), subplot_kw={'projection': proj})

    #shared color scale
    vmax = abs(smoothing_results).quantile(0.98).item()

    #dynamic colorbar levels
    locator = MaxNLocator(nbins=7, integer=True, prune='both')
    major_ticks = locator.tick_values(-vmax, vmax)
    n_minor = 8                   
    levels = np.concatenate([np.linspace(major_ticks[i], major_ticks[i + 1], n_minor, endpoint=False) for i in range(len(major_ticks) - 1)] + [np.array([major_ticks[-1]])])



    titles = ['a) No Smoothing', 'b) $\\sigma_{x,y}$ = 3', 'c) $\\sigma_{x,y}$ = 6', 'd) $\\sigma_{x,y}$ = 9']

    #stats compared to no smoothing
    reference_data = smoothing_results.sel(sigma=0)
    region_mask = create_region_mask(reference_data, EXTENT)
    
    mesh = None
    for i, (ax, sigma, title) in enumerate(zip(axes, sigmas, titles)):

        is_left = (i == 0)
        add_map_features(ax, EXTENT, is_left=is_left, is_bottom=True)
        data_to_plot = smoothing_results.sel(sigma=sigma)

        mesh = data_to_plot.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=levels, vmin=major_ticks[0], vmax=major_ticks[-1], add_colorbar=False, extend='both')
        
        #statistics
        if sigma > 0:

            stats = calculate_weighted_stats(data_to_plot, region_mask, data_y=reference_data)
            subtitle = f"\nPCC: {stats['pcc']:.2f} | RMSE: {stats['rmse']:.2f} mm/yr"
            ax.set_title(title + subtitle, fontsize=11)
        else:
            ax.set_title(title, fontsize=11)

    fig.suptitle('Sensitivity of altimetry trend to Gaussian smoothing', fontsize=18, fontweight='bold', y=0.85)

    fig.subplots_adjust(bottom=0.2)

    #colorbar
    cbar_ax = fig.add_axes((0.25, 0.25, 0.5, 0.03))
    if mesh:
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks(major_ticks)
        cbar.set_label('Trend (mm/yr)', fontsize=12)
    
    #save
    output_path = os.path.join(fig_dir, 'smoothing_sensitivity_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_all_models_overview(cmip_results, sliding_results, fig_dir, plot_var=None):
    """Overview figure all models."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Plotting individual model overview for {cfg['name']}...")

    if plot_var == 'trend':
        models_stack = cmip_results['model_trend'] 
        obs_ref = sliding_results['odsl_mm_yr_regridded']
        cmap_unified = 'coolwarm'
        vmax_val = abs(models_stack.quantile(0.98, skipna=True).item())
        vmin_val = -vmax_val
    elif plot_var == 'variability':
        models_stack = cmip_results['model_variability']
        obs_ref = sliding_results['odsl_var_obs_regridded']
        cmap_unified = cfg['cmap']
        vmax_val = models_stack.quantile(0.98, skipna=True).item()
        vmin_val = 0
    elif plot_var == 'ODSL':
        models_stack = cmip_results['model_odsl_mean']
        obs_ref = sliding_results['odsl_obs_regridded']
        cmap_unified = cfg['cmap']
        vmax_val = abs(models_stack.quantile(0.98, skipna=True).item())
        vmin_val = -vmax_val
    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    #list of models
    model_names = models_stack.model.values
    n_models = len(model_names)
    
    #grid layout
    ncols = 5
    nrows = math.ceil(n_models / ncols)
    
    #projection
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 22), subplot_kw={'projection': proj})
    axes = axes.flatten()

    region_mask = create_region_mask(models_stack.isel(model=0), EXTENT)

    #colorbar limits
    levels, major_ticks, vmin_val, vmax_val = make_aligned_levels(vmin_val, vmax_val, force_zero_min=(plot_var == 'variability'))
    extend                                  = 'max' if plot_var == 'variability' else 'both'

    #loop
    plot_handle = None
    for i, ax in enumerate(axes):
        if i < n_models:
            model_name = model_names[i]
            data_model = models_stack.sel(model=model_name)
            
            #stats
            stats = calculate_weighted_stats(data_model, region_mask, data_y=obs_ref)
            pcc = stats['pcc']
            rmse = stats['rmse']
            sign_agreement = stats['sign_agreement']
            
            #axes
            row_idx = i // ncols
            col_idx = i % ncols
            is_left = (col_idx == 0)
            is_bottom = (row_idx == nrows - 1) or (i + ncols >= n_models)

            add_map_features(ax, EXTENT, is_left=is_left, is_bottom=is_bottom)
            extend = 'max' if plot_var == 'variability' else 'both'
            plot_handle = data_model.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_val, vmax=vmax_val, add_colorbar=False, levels=levels, extend=extend)

            #significance
            p_val_map = None
            if plot_var == 'trend' and 'model_trend_p_values' in cmip_results:
                p_val_map = cmip_results['model_trend_p_values'].sel(model=model_name)
            elif plot_var == 'ODSL' and 'model_odsl_mean_p_values' in cmip_results:
                p_val_map = cmip_results['model_odsl_mean_p_values'].sel(model=model_name)

            if p_val_map is not None:
                x_coord = p_val_map.longitude if 'longitude' in p_val_map.coords else p_val_map.lon
                y_coord = p_val_map.latitude if 'latitude' in p_val_map.coords else p_val_map.lat
                ax.contourf(x_coord, y_coord, p_val_map < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
            
            #title
            if plot_var in ('trend', 'variability'):
                title = fr"$\bf{{{model_name}}}$" + f"\nPCC: {pcc:.2f} | RMSE: {rmse:.2f} | Sign agreement: {sign_agreement:.0%}"
            else:
                title = fr"$\bf{{{model_name}}}$"

            ax.set_title(title, fontsize=8)
            
        else:
            ax.set_visible(False)

    #layout and colorbar
    fig.subplots_adjust(bottom=0.08, top=0.95, wspace=0.1, hspace=0)
    cbar_ax = fig.add_axes((0.15, 0.04, 0.7, 0.015))
    
    if plot_handle:
        cbar = fig.colorbar(plot_handle, cax=cbar_ax, orientation='horizontal', extend=extend)
        cbar.set_ticks(major_ticks.tolist())
        cbar.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=14)
    
    fig.suptitle(f"Individual model performance: {cfg['name']} ({START_YEAR}-{END_YEAR})", fontsize=20, fontweight='bold', y=0.95)

    filename = f'all_models_overview_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'
    output_path = os.path.join(fig_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_eof_maps_comparison(eof_results, fig_dir, num_modes=3):
    """Plots the Observed vs Multi-Model Mean EOF maps side-by-side for the first n modes."""

    if 'observed' not in eof_results or 'multi model mean' not in eof_results:
        print("Cannot plot EOF comparison: missing data.")
        return

    print(f"Plotting EOF comparison maps (raw) for first {num_modes} modes...")

    obs_res = eof_results['observed']
    mmm_res = eof_results['multi model mean']
    
    obs_eofs = obs_res['eofs']
    mmm_eofs = mmm_res['eofs']
    
    obs_vars = obs_res['variance_fractions'] * 100
    mmm_vars = mmm_res['variance_fractions'] * 100
    
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])
    
    fig, axes = plt.subplots(nrows=num_modes, ncols=2, figsize=(14, 4 * num_modes), subplot_kw={'projection': proj})
    if num_modes == 1: axes = np.array([axes])

    #global color limits
    obs_subset = obs_eofs.isel(mode=slice(0, num_modes))
    mmm_subset = mmm_eofs.isel(mode=slice(0, num_modes))

    val_obs = abs(obs_subset).quantile(0.99).item()
    global_lim_obs = np.ceil(val_obs * 100) / 100.0

    val_mmm = abs(mmm_subset).quantile(0.99).item()
    global_lim_mmm = np.ceil(val_mmm * 100) / 100.0

    im_obs = None
    im_mmm = None

    for i in range(num_modes):
        ax_obs = axes[i, 0]
        ax_mmm = axes[i, 1]
        
        mode_idx = i
        mode_num = i + 1
        
        #load data
        map_obs = obs_eofs.sel(mode=mode_idx)
        map_mmm = mmm_eofs.sel(mode=mode_idx)
        
        #mask
        combined_mask = map_mmm.notnull() & map_obs.notnull()
        
        #weighted Stats
        stats = calculate_weighted_stats(map_mmm, combined_mask, data_y=map_obs)
        pcc = stats['pcc']
        sign_agr = stats['sign_agreement']

        #plot observed left column
        add_map_features(ax_obs, EXTENT, is_left=True, is_bottom=(i == num_modes-1))
        im_obs = map_obs.plot.contourf(ax=ax_obs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-global_lim_obs, vmax=global_lim_obs,  add_colorbar=False, levels=50, extend='both')
        
        #observed title
        var_txt = f"{obs_vars.sel(mode=mode_idx).item():.1f}%"
        if 'variance_fractions_error' in obs_res:
            err = obs_res['variance_fractions_error'].sel(mode=mode_idx).item() * 100
            var_txt += f" ± {err:.1f}%"
        ax_obs.set_title(f"Observed mode {mode_num}\nExp. Var: {var_txt}", fontsize=11, fontweight='bold')

        #plot multi-model mean right column
        add_map_features(ax_mmm, EXTENT, is_left=False, is_bottom=(i == num_modes-1))
        im_mmm = map_mmm.plot.contourf(ax=ax_mmm, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-global_lim_mmm, vmax=global_lim_mmm,  add_colorbar=False, levels=50, extend='both')
        
        #mmm title
        var_txt_m = f"{mmm_vars.sel(mode=mode_idx).item():.1f}%"
        if 'variance_fractions_error' in mmm_res:
            err = mmm_res['variance_fractions_error'].sel(mode=mode_idx).item() * 100
            var_txt_m += f" ± {err:.1f}%"
        
        ax_mmm.set_title(f"PCC: {pcc:.2f} | Sign agreement: {sign_agr:.0%}\nMulti-model mean mode {mode_num}\nExp. var: {var_txt_m}", fontsize=11, fontweight='bold')

    scalar_fmt = mticker.ScalarFormatter(useMathText=True)
    scalar_fmt.set_powerlimits((-1, 1))

    #get positions
    pos_top_left = axes[0, 0].get_position()
    pos_bot_left = axes[-1, 0].get_position()
    pos_top_right = axes[0, 1].get_position()
    pos_bot_right = axes[-1, 1].get_position()
    
    #height colorbar
    cbar_height = pos_top_left.y1 - pos_bot_left.y0

    #observed colorbar
    cbar_ax_obs = fig.add_axes((pos_bot_left.x0 - 0.08, pos_bot_left.y0, 0.02, cbar_height))
    if im_obs:
        cb_obs = fig.colorbar(im_obs, cax=cbar_ax_obs, orientation='vertical', format=scalar_fmt)
        cb_obs.ax.yaxis.set_ticks_position('left')
        cb_obs.ax.yaxis.set_label_position('left')
        cb_obs.set_label('Amplitude', fontsize=12)

    #mmm colorbar
    cbar_ax_mmm = fig.add_axes((pos_bot_right.x1 + 0.02, pos_bot_right.y0, 0.02, cbar_height))
    if im_mmm:
        cb_mmm = fig.colorbar(im_mmm, cax=cbar_ax_mmm, orientation='vertical', format=scalar_fmt)
        cb_mmm.set_label('Amplitude', fontsize=12, rotation=270, labelpad=15)

    #title
    fig.suptitle(f'EOF mode comparison: observed vs. multi-model mean ({START_YEAR}-{END_YEAR})', fontsize=16, fontweight='bold', y=0.93)

    #layout
    fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.05, hspace=0.3, wspace=0.15)

    output_path = os.path.join(fig_dir, f'eof_comparison_obs_vs_mmm_{START_YEAR}_{END_YEAR}.png')
    print(f"Saving EOF comparison figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_eof_monte_carlo_significance(eof_results, fig_dir):
    """EOF eigenvalues with Monte Carlo significance thresholds."""

    print("\nCreating Monte Carlo significance plots...")
    
    #select data
    sources_with_mc_unordered = {name: res for name, res in eof_results.items() if 'mc_threshold_99' in res}
    
    #observed first, multi-model mean second, then individual models alphabetical
    priority = {'observed': 0, 'multi model mean': 1}
    sorted_keys = sorted(sources_with_mc_unordered.keys(), key=lambda k: (priority.get(k, 2), k))
    sources_with_mc = {k: sources_with_mc_unordered[k] for k in sorted_keys}
    
    if not sources_with_mc:
        print("No Monte Carlo results found. Skipping.")
        return
    
    #plotting
    n_sources = len(sources_with_mc)
    n_cols = min(4, n_sources)
    n_rows = int(np.ceil(n_sources / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_sources == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(sources_with_mc.items()):
        ax = axes[idx]
        
        n_plot = 12
        observed = result['variance_fractions'].values[:n_plot] * 100 
        modes = np.arange(1, n_plot + 1)
        threshold_95 = result['mc_threshold_95'].values[:n_plot] * 100
        is_sig_95 = result['is_significant_95'].values[:n_plot]
        is_sig_99 = result['is_significant_99'].values[:n_plot]
        
        #colors
        colors = ['red' if s99 else 'black' if s95 else 'grey' for s95, s99 in zip(is_sig_95, is_sig_99)]
        
        ax.bar(modes, observed, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        #thresholds
        ax.plot(modes, threshold_95, color='black', linestyle='--', linewidth=1.5, label='95% threshold')
        
        #synthetic distribution range if available
        if 'synthetic_lambdas' in result:
            synthetic = result['synthetic_lambdas'].values[:, :n_plot] * 100
            p00 = np.percentile(synthetic, 0, axis=0)
            p05 = np.percentile(synthetic, 5, axis=0)
            p95 = np.percentile(synthetic, 95, axis=0)
            p100 = np.percentile(synthetic, 100, axis=0)
            ax.fill_between(modes, p00, p100, alpha=0.4, color='gray', label='Synthetic range (min-max)')
        
        #formatting
        if idx < 26:
            subplot_label = chr(97 + idx)
        else:
            subplot_label = chr(97 + (idx // 26) - 1) + chr(97 + (idx % 26))
        
        ax.set_xlabel('EOF mode', fontsize=12)
        ax.set_ylabel('Variance explained (%)', fontsize=12)
        
        title_name = 'Multi-model mean' if name == 'multi model mean' else name.title()
        ax.set_title(f'{subplot_label}) {title_name}\n({is_sig_95.sum()} significant at 95%)', fontsize=13, fontweight='bold')        
        
        ax.set_xticks(modes)
        ax.grid(True, alpha=0.5, linestyle='--')
        
        #alpha value
        if 'alpha' in result.attrs:
            alpha_val = result.attrs.get('alpha', 'N/A')
            ax.plot([], [], ' ', label=f'$\\alpha$ = {alpha_val:.3f}')
        
        ax.legend(loc='upper right', fontsize=10)

    #unused subplots
    for idx in range(n_sources, len(axes)):
        axes[idx].axis('off')
    
    eof_label = 'Rotated EOF' if USE_ROTATED_EOF else 'Standard EOF'
    fig.suptitle(f'{eof_label} Monte Carlo significance test (n={N_REALIZATIONS_MONTE_CARLO})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    
    save_path = os.path.join(fig_dir, 'eof_monte_carlo_significance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_obs_vs_mmm_significance(sources_with_mc, fig_dir):
    """Observed vs multi-model mean significance."""
    
    if 'observed' not in sources_with_mc or 'multi model mean' not in sources_with_mc:
        print("Skipping obs vs multi-model mean comparison.""")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (ax, source_name) in enumerate(zip([ax1, ax2], ['observed', 'multi model mean'])):
        result = sources_with_mc[source_name]
        subplot_letter = chr(97 + idx)  

        if 'mc_threshold_95' not in result:
            ax.text(0.5, 0.5, f'{source_name}\n(No MC data)', ha='center', va='center', transform=ax.transAxes, fontsize=14, style='italic')
            continue

        n_plot = 12
        observed = result['variance_fractions'].values[:n_plot] * 100 
        modes = np.arange(1, n_plot + 1)
        threshold_95 = result['mc_threshold_95'].values[:n_plot] * 100
        is_sig_95 = result['is_significant_95'].values[:n_plot]
        is_sig_99 = result['is_significant_99'].values[:n_plot]

        #colors
        colors = ['red' if s99 else 'black' if s95 else 'grey' for s95, s99 in zip(is_sig_95, is_sig_99)]

        ax.bar(modes, observed, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        #threshold line
        ax.plot(modes, threshold_95, color='black', linestyle='--', linewidth=1.5, label='95% threshold', zorder=10)

        #North's rule error bars
        if 'variance_fractions_error' in result:
            errors = result['variance_fractions_error'].values[:n_plot] * 100
            ax.errorbar(modes, observed, yerr=errors, fmt='none', ecolor='black', capsize=4, capthick=1.5, alpha=0.7, label="North's error", zorder=5)

        if 'synthetic_lambdas' in result:
            synthetic = result['synthetic_lambdas'].values[:, :n_plot] * 100
            p00 = np.percentile(synthetic, 0, axis=0)
            p100 = np.percentile(synthetic, 100, axis=0)
            ax.fill_between(modes, p00, p100, alpha=0.4, color='gray', label='Synthetic range (min-max)')

        #alpha in legend
        if 'alpha' in result.attrs:
            alpha_val = result.attrs.get('alpha', 'N/A')
            ax.plot([], [], ' ', label=f'$\\alpha$ = {alpha_val:.3f}')

        ax.set_xlabel('EOF mode', fontsize=12)
        ax.set_ylabel('Variance explained (%)', fontsize=12)
        display_name = 'Multi-model mean' if source_name == 'multi model mean' else source_name.title()
        ax.set_title(f'{subplot_letter}) {display_name}\n({is_sig_95.sum()} significant at 95%)', fontsize=11)        
        ax.set_xticks(modes)
        ax.grid(True, alpha=0.5, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)

    eof_label = 'Rotated EOF' if USE_ROTATED_EOF else 'Standard EOF'
    fig.suptitle(f'{eof_label} Monte Carlo significance\nObserved vs multi-model mean (n={N_REALIZATIONS_MONTE_CARLO})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'eof_significance_obs_vs_mmm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_odsl_cover_art(dataset, fig_dir, variable='odsl'):
    """3D representation of observed ODSL for cover page."""

    print(f"Generating 3D cover art for '{variable}'...")

    da = dataset[variable]
    mask = ~np.isnan(da.values)
    
    #valid indices
    valid_rows = np.any(mask, axis=1) 
    valid_cols = np.any(mask, axis=0) 
    
    #slice
    data = da.values[valid_rows][:, valid_cols]
    lon = da.longitude.values[valid_cols]
    lat = da.latitude.values[valid_rows]
    
    #coordinate mesh
    X, Y = np.meshgrid(lon, lat)
    Z = data

    #setup
    z_scale = 100.0   
    cmap_name = 'turbo'  

    #plotting
    fig = plt.figure(figsize=(16, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_axis_off()
    ax.set_facecolor('white')

    #lighting and color
    ls = LightSource(azdeg=315, altdeg=50)
    
    vmin = np.nanpercentile(Z, 2)
    vmax = np.nanpercentile(Z, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    #shade
    rgb = ls.shade(Z, cmap=plt.get_cmap(cmap_name), norm=norm, vert_exag=0.5, blend_mode='overlay')

    #surface
    surf = ax.plot_surface(X, Y, Z * z_scale, facecolors=rgb, linewidth=0, antialiased=True, shade=False, rstride=1, cstride=1)

    x_range = lon.max() - lon.min()
    y_range = lat.max() - lat.min()
    aspect_ratio = x_range / y_range
    
    #aspect ratio
    ax.set_box_aspect((aspect_ratio, 1, 0.25))

    #view settings
    ax.view_init(elev=50, azim=-90)
    
    #zoom
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    plt.tight_layout()
    save_path_png = os.path.join(fig_dir, 'cover_art_3D.png')
    plt.savefig(save_path_png, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    #crop whitespace
    img = Image.open(save_path_png)
    img_array = np.array(img)
    
    #non-white pixels
    if img_array.shape[2] == 4:
        non_empty = img_array[:, :, 3] > 0
    else:
        non_empty = np.any(img_array < 250, axis=2)
    
    rows = np.any(non_empty, axis=1)
    cols = np.any(non_empty, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        padding = 10
        rmin = max(0, rmin - padding)
        rmax = min(img_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_array.shape[1], cmax + padding)
        
        img_cropped = img.crop((cmin, rmin, cmax, rmax))
        img_cropped.save(save_path_png)
        
        #pdf
        save_path_pdf = os.path.join(fig_dir, 'cover_art_3D.pdf')
        img_cropped.convert('RGB').save(save_path_pdf, 'PDF', resolution=300)
        
        print(f"Cropped from {img.size} to {img_cropped.size}")

def plot_dual_eof_comparison(dual_eof_results, fig_dir, num_modes_to_plot=10, source='observed'):
    """Standard vs rotated EOF variance."""

    os.makedirs(fig_dir, exist_ok=True)
    
    #sources
    all_sources = sorted(set(k.rsplit('__', 1)[0] for k in dual_eof_results.keys()))
    if source in all_sources:
        ref_key = source
        model_keys = [k for k in all_sources if k != source]
        has_ref = True
    else:
        ref_key = None
        model_keys = all_sources
        has_ref = False

    if not model_keys:
        print("Not enough data to perform multimodel comparison.")
        return

    #data helper
    def get_variance_curve(key, eof_type):
        try:
            ds = dual_eof_results[f'{key}__{eof_type}']
            var_data = ds['variance_fractions'].values * 100
            if len(var_data) >= num_modes_to_plot:
                return var_data[:num_modes_to_plot]
            else:
                padded = np.full(num_modes_to_plot, np.nan)
                padded[:len(var_data)] = var_data
                return padded
        except KeyError:
            return np.full(num_modes_to_plot, np.nan)

    #matrices
    unrot_ensemble = np.array([get_variance_curve(m, 'unrotated') for m in model_keys])
    rot_ensemble   = np.array([get_variance_curve(m, 'rotated')   for m in model_keys])

    mmm_unrot = np.nanmean(unrot_ensemble, axis=0)
    mmm_rot   = np.nanmean(rot_ensemble, axis=0)

    ref_unrot = get_variance_curve(ref_key, 'unrotated') if has_ref else None
    ref_rot   = get_variance_curve(ref_key, 'rotated')   if has_ref else None

    #plotting
    fig, ax = plt.subplots(figsize=(12, 7))

    modes_idx = np.arange(1, num_modes_to_plot + 1)
    width     = 0.35
    pos_unrot = modes_idx - width / 2
    pos_rot   = modes_idx + width / 2

    #unrotated
    box_data_unrot = [unrot_ensemble[:, i][~np.isnan(unrot_ensemble[:, i])] for i in range(num_modes_to_plot)]
    bp_u           = ax.boxplot(box_data_unrot, positions=pos_unrot, patch_artist=True, widths=width * 0.8, showfliers=False)
    for patch in bp_u['boxes']:
        patch.set_facecolor('#c9b2d6'); patch.set_alpha(0.6); patch.set_edgecolor('#7b4f9e')
    for element in ['medians', 'whiskers', 'caps']:
        plt.setp(bp_u[element], color='#7b4f9e')
    ax.plot(pos_unrot, mmm_unrot, color='black', marker='o', markersize=6, linestyle='None', zorder=4)
    if ref_unrot is not None:
        ax.plot(pos_unrot, ref_unrot, color='green', marker='s', markersize=11, linestyle='None', zorder=5)

    #rotated
    box_data_rot = [rot_ensemble[:, i][~np.isnan(rot_ensemble[:, i])] for i in range(num_modes_to_plot)]
    bp_r         = ax.boxplot(box_data_rot, positions=pos_rot, patch_artist=True, widths=width * 0.8, showfliers=False)
    for patch in bp_r['boxes']:
        patch.set_facecolor('#fdbf6f'); patch.set_alpha(0.6); patch.set_edgecolor('#ff7f0e')
    for element in ['medians', 'whiskers', 'caps']:
        plt.setp(bp_r[element], color='#ff7f0e')
    ax.plot(pos_rot, mmm_rot, color='black', marker='o', markersize=6, linestyle='None', zorder=4)
    if ref_rot is not None:
        ax.plot(pos_rot, ref_rot, color='green', marker='s', markersize=11, linestyle='None', zorder=5)

    #formatting
    ax.set_title(f'Explained variance per mode: standard vs rotated\n{source.capitalize()} vs {len(model_keys)} {CMIP_VERSION} models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mode number', fontsize=12)
    ax.set_ylabel('Explained variance (%)', fontsize=12)
    ax.set_xticks(modes_idx)
    ax.set_xticklabels(modes_idx)
    ax.grid(True, alpha=0.3)

    #legend
    legend_handles = [Line2D([0], [0], color='#7b4f9e', marker='*', linestyle='None', markersize=11, label=f'Standard {source}'),
                      Line2D([0], [0], color='#7b4f9e', marker='o', linestyle='None', markersize=6, label='Standard Multi-model mean'),
                      Patch(facecolor='#c9b2d6', edgecolor='#7b4f9e', alpha=0.6, label='Standard spread'),
                      Line2D([0], [0], color='#ff7f0e', marker='*', linestyle='None', markersize=11, label=f'Rotated {source}'),
                      Line2D([0], [0], color='#ff7f0e', marker='o', linestyle='None', markersize=6, label='Rotated Multi-model mean'),
                      Patch(facecolor='#fdbf6f', edgecolor='#ff7f0e', alpha=0.6, label='Rotated spread')]
    ax.legend(handles=legend_handles, loc='upper right', ncol=2, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'dual_eof_combined_variance_{source}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_observed_vs_best_windows(obs_results, cmip_results, sliding_results, fig_dir, plot_var=None):
    """Plot observed vs best-PCC and best-RMSE window multimodel means."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    print(f"Plotting observed vs best-window multimodel means for ODSL {cfg['name']}...")

    #data selection
    if plot_var == 'trend':
        obs_data = sliding_results['odsl_mm_yr_regridded']
        obs_p_value = obs_results.get('odsl_p_value')
        model_stack = cmip_results['model_trend']
        best_pcc_mmm = sliding_results['best_pcc_mmm_trend']
        best_rmse_mmm = sliding_results['best_rmse_mmm_trend']
        best_pcc_p_value = sliding_results['best_pcc_trend_p_value']
        best_rmse_p_value = sliding_results['best_rmse_trend_p_value']

        vmax_unified = max(abs(obs_data.quantile(0.98, skipna=True).item()), abs(best_pcc_mmm.quantile(0.98, skipna=True).item()), abs(best_rmse_mmm.quantile(0.98, skipna=True).item()))
        vmin_unified = -vmax_unified

    elif plot_var == 'variability':
        obs_data = sliding_results['odsl_var_obs_regridded']
        obs_p_value = None
        model_stack = cmip_results['model_variability']
        best_pcc_mmm = sliding_results['best_pcc_mmm_variability']
        best_rmse_mmm = sliding_results['best_rmse_mmm_variability']
        best_pcc_p_value = None
        best_rmse_p_value = None

        vmax_unified = max(obs_data.quantile(0.98, skipna=True).item(), best_pcc_mmm.quantile(0.98, skipna=True).item(), best_rmse_mmm.quantile(0.98, skipna=True).item())
        vmin_unified = min(obs_data.min(skipna=True).item(), best_pcc_mmm.min(skipna=True).item(), best_rmse_mmm.min(skipna=True).item())

    elif plot_var == 'ODSL':
        obs_data = sliding_results['odsl_obs_regridded']
        obs_p_value = obs_results.get('odsl_mean_p_value')
        model_stack = cmip_results['model_odsl_mean']
        best_pcc_mmm = sliding_results['best_pcc_mmm_odsl']
        best_rmse_mmm = sliding_results['best_rmse_mmm_odsl']
        best_pcc_p_value = sliding_results['best_pcc_odsl_p_value']
        best_rmse_p_value = sliding_results['best_rmse_odsl_p_value']

        vmax_unified = max(abs(obs_data).quantile(0.98, skipna=True).item(), abs(best_pcc_mmm).quantile(0.98, skipna=True).item(), abs(best_rmse_mmm).quantile(0.98, skipna=True).item())
        vmin_unified = -vmax_unified

    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}. Must be 'trend', 'variability', or 'ODSL'")

    cmap_unified = cfg['cmap']

    #difference
    diff_pcc = best_pcc_mmm - obs_data
    diff_rmse = best_rmse_mmm - obs_data
    vmax_diff = max(abs(diff_pcc.quantile(0.98, skipna=True).item()), abs(diff_rmse.quantile(0.98, skipna=True).item()))
    vmin_diff = -vmax_diff
    cmap_diff = 'BrBG'

    #colorbar ticks
    levels_main, ticks_main, vmin_unified, vmax_unified = make_aligned_levels(vmin_unified, vmax_unified, force_zero_min=(plot_var == 'variability'))
    levels_diff, ticks_diff, vmin_diff, vmax_diff       = make_aligned_levels(vmin_diff, vmax_diff)
    extend_main                                         = 'max' if plot_var == 'variability' else 'both'

    #SNR
    model_std     = model_stack.std(dim='model', skipna=True)
    snr_pcc       = abs(best_pcc_mmm) / model_std
    snr_rmse      = abs(best_rmse_mmm) / model_std
    snr_diff_pcc  = abs(diff_pcc) / model_std
    snr_diff_rmse = abs(diff_rmse) / model_std

    #statistics
    region_mask = create_region_mask(best_pcc_mmm, EXTENT)

    stats_obs       = calculate_weighted_stats(obs_data, region_mask)
    stats_pcc       = calculate_weighted_stats(best_pcc_mmm, region_mask, data_y=obs_data)
    stats_rmse      = calculate_weighted_stats(best_rmse_mmm, region_mask, data_y=obs_data)
    stats_diff_pcc  = calculate_weighted_stats(diff_pcc, region_mask)
    stats_diff_rmse = calculate_weighted_stats(diff_rmse, region_mask)

    #layout
    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])

    fig = plt.figure(figsize=(22, 11))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.1, wspace=0.1, left=0.05, right=0.95, bottom=0.12, top=0.88)

    ax_obs       = fig.add_subplot(gs[:, 0], projection=proj)
    ax_pcc       = fig.add_subplot(gs[0, 1], projection=proj)
    ax_rmse      = fig.add_subplot(gs[1, 1], projection=proj)
    ax_diff_pcc  = fig.add_subplot(gs[0, 2], projection=proj)
    ax_diff_rmse = fig.add_subplot(gs[1, 2], projection=proj)

    #a) observed
    add_map_features(ax_obs, EXTENT, is_left=True, is_bottom=True)
    mesh_obs = obs_data.plot.contourf(ax=ax_obs, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False, levels=levels_main, extend=extend_main)
    if obs_p_value is not None:
        ax_obs.contourf(obs_p_value.longitude, obs_p_value.latitude, obs_p_value < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
    ax_obs.set_title(f'a) Observed ODSL ({cfg["name"]})\nMean: {stats_obs["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_obs["std_x"]:.2f} {cfg["units"]}', fontsize=11)

    #b1) PCC multi-model mean
    add_map_features(ax_pcc, EXTENT, is_left=False, is_bottom=False)
    mesh_pcc = best_pcc_mmm.plot.contourf(ax=ax_pcc, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False, levels=levels_main, extend=extend_main)
    if best_pcc_p_value is not None:
        ax_pcc.contourf(best_pcc_p_value.longitude, best_pcc_p_value.latitude, best_pcc_p_value < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
    ax_pcc.contour(snr_pcc.longitude, snr_pcc.latitude, snr_pcc, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())
    ax_pcc.set_title(f'b1) Best PCC window multi-model mean ({cfg["name"]})\nMean: {stats_pcc["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_pcc["std_x"]:.2f} {cfg["units"]}', fontsize=11)

    #b2) RMSE multi-model mean
    add_map_features(ax_rmse, EXTENT, is_left=False, is_bottom=True)
    mesh_rmse = best_rmse_mmm.plot.contourf(ax=ax_rmse, transform=ccrs.PlateCarree(), cmap=cmap_unified, vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False, levels=levels_main, extend=extend_main)
    if best_rmse_p_value is not None:
        ax_rmse.contourf(best_rmse_p_value.longitude, best_rmse_p_value.latitude, best_rmse_p_value < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])
    ax_rmse.contour(snr_rmse.longitude, snr_rmse.latitude, snr_rmse, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())
    ax_rmse.set_title(f'b2) Best RMSE window multi-model mean ({cfg["name"]})\nMean: {stats_rmse["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_rmse["std_x"]:.2f} {cfg["units"]}', fontsize=11)

    #c1) difference (PCC - obs)
    add_map_features(ax_diff_pcc, EXTENT, is_left=False, is_bottom=False)
    mesh_d_pcc = diff_pcc.plot.contourf(ax=ax_diff_pcc, transform=ccrs.PlateCarree(), cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff, add_colorbar=False, levels=levels_diff, extend='both')
    ax_diff_pcc.contour(snr_diff_pcc.longitude, snr_diff_pcc.latitude, snr_diff_pcc, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())
    ax_diff_pcc.set_title(f'c1) Difference (b1 - a)\nMean: {stats_diff_pcc["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_diff_pcc["std_x"]:.2f} {cfg["units"]}', fontsize=11)

    #c2) difference (RMSE - obs)
    add_map_features(ax_diff_rmse, EXTENT, is_left=False, is_bottom=True)
    mesh_d_rmse = diff_rmse.plot.contourf(ax=ax_diff_rmse, transform=ccrs.PlateCarree(), cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff, add_colorbar=False, levels=levels_diff, extend='both')
    ax_diff_rmse.contour(snr_diff_rmse.longitude, snr_diff_rmse.latitude, snr_diff_rmse, levels=[1.0, 1.5, 2.0], colors=['black', 'gray', 'white'], linewidths=1.0, transform=ccrs.PlateCarree())
    ax_diff_rmse.set_title(f'c2) Difference (b2 - a)\nMean: {stats_diff_rmse["mean_x"]:.2f} {cfg["units"]} | RMS: {stats_diff_rmse["std_x"]:.2f} {cfg["units"]}', fontsize=11)

    #overall title
    title_str = (f'Observed vs. best window multi-model means ({START_YEAR}-{END_YEAR})\nb1) PCC = {stats_pcc["pcc"]:.2f}, RMSE = {stats_pcc["rmse"]:.2f} {cfg["units"]}')
    #sign agreement if not variability (variability \geq 0)
    if plot_var != 'variability':
        title_str += f', Sign agreement = {stats_pcc["sign_agreement"]:.0%}'
    title_str += f' | b2) PCC = {stats_rmse["pcc"]:.2f}, RMSE = {stats_rmse["rmse"]:.2f} {cfg["units"]}'
    if plot_var != 'variability':
        title_str += f', Sign agreement = {stats_rmse["sign_agreement"]:.0%}'
    fig.suptitle(title_str, fontsize=14, y=0.96, fontweight='bold')

    #colorbar
    pos_obs = ax_obs.get_position()
    pos_rmse = ax_rmse.get_position()
    pos_d_rmse = ax_diff_rmse.get_position()
    cbar_bottom = pos_rmse.y0 - 0.08
    cbar_height = 0.02

    #shared colorbar a), b1), b2)
    cbar_ax1 = fig.add_axes((pos_obs.x0, cbar_bottom, pos_rmse.x1 - pos_obs.x0, cbar_height))
    cbar1 = fig.colorbar(mesh_obs, cax=cbar_ax1, orientation='horizontal', extend=extend_main)
    cbar1.set_ticks(ticks_main.tolist())
    cbar1.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=12)

    #difference colorbar
    cbar_ax2 = fig.add_axes((pos_d_rmse.x0, cbar_bottom, pos_d_rmse.width, cbar_height))
    cbar2 = fig.colorbar(mesh_d_pcc, cax=cbar_ax2, orientation='horizontal', extend='both')
    cbar2.set_ticks(ticks_diff.tolist())
    cbar2.set_label(f'Difference ({cfg["units"]})', fontsize=12)

    plt.savefig(os.path.join(fig_dir, f'observed_vs_best_windows_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_all_models_overview_best_windows(cmip_results, sliding_results, fig_dir, plot_var=None, metric='pcc'):
    """Overview figure all models best PCC and RMSE windows."""

    if plot_var is None:
        plot_var = PLOT_VARIABLE
    cfg = PLOT_CONFIG[plot_var]

    metric_label = 'PCC' if metric == 'pcc' else 'RMSE'
    print(f"Plotting best {metric_label} window overview for {cfg['name']}...")

    #PCC/RMSE time series and sliding maps
    if plot_var == 'trend':
        pcc_ts = sliding_results['pcc']
        rmse_ts = sliding_results['rmse']
        sliding_maps = sliding_results['sliding_trends']
        obs_ref = sliding_results['odsl_mm_yr_regridded']
        cmap_unified = 'coolwarm'
    elif plot_var == 'variability':
        pcc_ts = sliding_results['pcc_variability']
        rmse_ts = sliding_results['rmse_variability']
        sliding_maps = sliding_results['sliding_variability']
        obs_ref = sliding_results['odsl_var_obs_regridded']
        cmap_unified = cfg['cmap']
    elif plot_var == 'ODSL':
        pcc_ts = sliding_results['pcc_mean_odsl']
        rmse_ts = sliding_results['rmse_mean_odsl']
        sliding_maps = sliding_results['sliding_odsl_mean']
        obs_ref = sliding_results['odsl_obs_regridded']
        cmap_unified = cfg['cmap']
    else:
        raise ValueError(f"Unknown PLOT_VARIABLE: {plot_var}")

    #best window years per model
    if metric == 'pcc':
        best_years = pcc_ts.idxmax('window_start_year')
    else:
        best_years = rmse_ts.idxmin('window_start_year')

    #best window patterns per model
    window_size = END_YEAR - START_YEAR + 1
    model_names = sliding_maps.model.values

    best_patterns = []
    best_p_values = []

    for model_name in model_names:
        best_year = best_years.sel(model=model_name).item()

        if plot_var == 'trend':
            full_ts = cmip_results['full_timeseries'].sel(model=model_name)
            window_data = full_ts.sel(time=slice(best_year, best_year + window_size - 1))
            sig = compute_field_significance(window_data, plot_var)
            pattern = sig['field']
        else:
            window_data = cmip_results['full_timeseries'].sel(model=model_name, time=slice(best_year, best_year + window_size - 1))
            pattern = sliding_maps.sel(model=model_name, window_start_year=best_year)
            sig = compute_field_significance(window_data, plot_var)

        best_patterns.append(pattern)
        best_p_values.append(sig.get('p_value', None))

    models_stack = xr.concat(best_patterns, dim=pd.Index(model_names, name='model'))
    n_models = len(model_names)

    #color limits
    if plot_var == 'trend' or plot_var == 'ODSL':
        vmax_val = abs(models_stack.quantile(0.98, skipna=True).item())
        vmin_val = -vmax_val
    else:
        vmax_val = models_stack.quantile(0.98, skipna=True).item()
        vmin_val = 0

    levels, major_ticks, vmin_val, vmax_val = make_aligned_levels(vmin_val, vmax_val, force_zero_min=(plot_var == 'variability'))
    extend                                  = 'max' if plot_var == 'variability' else 'both'

    #grid
    ncols = 5
    nrows = math.ceil(n_models / ncols)

    proj = ccrs.AlbersEqualArea(central_longitude=PROJECTION_PARAMS['central_longitude'], central_latitude=PROJECTION_PARAMS['central_latitude'], standard_parallels=PROJECTION_PARAMS['standard_parallels'])

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 22), subplot_kw={'projection': proj})
    axes = axes.flatten()

    region_mask = create_region_mask(models_stack.isel(model=0), EXTENT)

    plot_handle = None
    for i, ax in enumerate(axes):
        if i < n_models:
            model_name = model_names[i]
            data_model = models_stack.sel(model=model_name)
            best_year = best_years.sel(model=model_name).item()

            #stats against observed reference
            stats = calculate_weighted_stats(data_model, region_mask, data_y=obs_ref)
            pcc = stats['pcc']
            rmse = stats['rmse']
            sign_agreement = stats['sign_agreement']

            #axes
            row_idx = i // ncols
            col_idx = i % ncols
            is_left = (col_idx == 0)
            is_bottom = (row_idx == nrows - 1) or (i + ncols >= n_models)

            add_map_features(ax, EXTENT, is_left=is_left, is_bottom=is_bottom)
            plot_handle = data_model.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_unified, levels=levels, vmin=vmin_val, vmax=vmax_val, add_colorbar=False, extend=extend)
            p_val_map = best_p_values[i]
            if p_val_map is not None:
                x_coord = p_val_map.longitude if 'longitude' in p_val_map.coords else p_val_map.lon
                y_coord = p_val_map.latitude if 'latitude' in p_val_map.coords else p_val_map.lat
                ax.contourf(x_coord, y_coord, p_val_map < 0.05, levels=[0.5, 1.5], transform=ccrs.PlateCarree(), colors='none', hatches=['...'])

            #title
            end_year = best_year + window_size - 1
            if plot_var in ('trend', 'variability'):
                title = (fr"$\bf{{{model_name}}}\ ({best_year:.0f}-{end_year:.0f})$" + f"\nPCC: {pcc:.2f} | RMSE: {rmse:.2f} | Sign agreement: {sign_agreement:.0%}")
            else:
                title = fr"$\bf{{{model_name}}}\ ({best_year:.0f}-{end_year:.0f})$"

            ax.set_title(title, fontsize=8)

        else:
            ax.set_visible(False)

    #layout and colorbar
    fig.subplots_adjust(bottom=0.08, top=0.95, wspace=0.1, hspace=0)
    cbar_ax = fig.add_axes((0.15, 0.04, 0.7, 0.015))

    if plot_handle:
        cbar = fig.colorbar(plot_handle, cax=cbar_ax, orientation='horizontal', extend=extend)
        cbar.set_ticks(major_ticks.tolist())
        cbar.set_label(f'{cfg["name"]} ({cfg["units"]})', fontsize=14)

    eof_label = f'Best {metric_label} window'
    fig.suptitle(f"Individual model performance: {cfg['name']} (best {metric_label} window per model)", fontsize=20, fontweight='bold', y=0.95)

    filename = f'all_models_overview_best_{metric}_{cfg["name"]}_{START_YEAR}_{END_YEAR}.png'
    output_path = os.path.join(fig_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_spectral_analysis(spectral_results, fig_dir, n_modes=5):
    """Plot power spectra: left column altimetry era (1993-2024) and right CMIP model historical period (1850-2024)."""

    print("Plotting spectral analysis...")
    print(f"Available keys: {list(spectral_results.keys())}")

    has_obs        = 'obs_spectrum' in spectral_results
    has_obs_period = 'obs_period_model_spectra' in spectral_results
    model_spectra  = spectral_results['model_spectra']
    mmm_spectrum   = spectral_results['mmm_spectrum']

    print(f"has_obs_period: {has_obs_period}")

    available_modes = model_spectra.mode.values
    plot_modes      = available_modes[:n_modes]

    subplot_height = 3.5
    fig_height     = len(plot_modes) * subplot_height
    fig, axes = plt.subplots(nrows=len(plot_modes), ncols=2, figsize=(18, fig_height), gridspec_kw={'width_ratios': [2, 3], 'wspace': 0.02}, sharey='row')    
    if len(plot_modes) == 1:
        axes = axes.reshape(1, -1)

    for idx, mode in enumerate(plot_modes):
        periods_r = None
        periods = None

        ax_obs  = axes[idx, 0]
        ax_full = axes[idx, 1]
        subplot_letter_l = chr(97 + idx)              
        subplot_letter_r = chr(97 + len(plot_modes) + idx) 

        #left column altimetry era (1993-2024)
        if has_obs_period:
            obs_p_model = spectral_results['obs_period_model_spectra']
            obs_p_mmm   = spectral_results['obs_period_mmm_spectrum']

            obs_p_modes = obs_p_model.mode.values
            if idx < len(obs_p_modes):
                obs_mode = obs_p_modes[idx]
            else:
                ax_obs.set_visible(False)
                continue

            is_first = True
            for model_name in obs_p_model.model.values:
                spec_da   = obs_p_model.sel(model=model_name, mode=obs_mode).dropna('frequency_obs')
                freqs_r   = spec_da.frequency_obs.values
                mask_r    = freqs_r > 0
                periods_r = 1.0 / freqs_r[mask_r]
                power     = spec_da.values[mask_r]
                label     = 'Individual models' if is_first else ''
                ax_obs.loglog(periods_r, power, color='lightgrey', linewidth=0.8, label=label, zorder=1)
                is_first = False

            mmm_da      = obs_p_mmm.sel(mode=obs_mode).dropna('frequency_obs')
            freqs_mmm   = mmm_da.frequency_obs.values
            mask_mmm    = freqs_mmm > 0
            periods_mmm = 1.0 / freqs_mmm[mask_mmm]
            ax_obs.loglog(periods_mmm, mmm_da.values[mask_mmm], color='black', linewidth=2.5, label='Multi-model mean', zorder=3)

        if has_obs:
            obs_spectrum    = spectral_results['obs_spectrum']
            obs_modes_avail = obs_spectrum.mode.values
            if idx < len(obs_modes_avail):
                obs_da    = obs_spectrum.sel(mode=obs_modes_avail[idx]).dropna('frequency_obs')
                obs_freqs = obs_da.frequency_obs.values
                obs_mask  = obs_freqs > 0
                ax_obs.loglog(1.0 / obs_freqs[obs_mask], obs_da.values[obs_mask], color='green', linewidth=2, label='Observed', zorder=2)

        ax_obs.set_title(f'{subplot_letter_l}) EOF mode {idx + 1} ({START_YEAR}-{END_YEAR})', fontweight='bold', fontsize=11)
        ax_obs.set_ylabel('Power spectral density')
        ax_obs.grid(True, linestyle='--', alpha=0.4, which='both', axis='x')
        if periods_r is not None:
            ax_obs.set_xlim(periods_r.min(), periods_r.max())
        if idx == 0:
            ax_obs.legend(loc='lower right', fontsize=9)
        for spine in ax_obs.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2.0)

        #right column historical model period (1850-2024)
        full_modes = model_spectra.mode.values
        if idx < len(full_modes):
            full_mode = full_modes[idx]
        else:
            ax_full.set_visible(False)
            continue

        is_first = True
        for model_name in model_spectra.model.values:
            spec_da = model_spectra.sel(model=model_name, mode=full_mode).dropna('frequency')
            freqs   = spec_da.frequency.values
            mask    = freqs > 0
            periods = 1.0 / freqs[mask]
            label   = 'Individual models' if is_first else ''
            ax_full.loglog(periods, spec_da.values[mask], color='lightgrey', linewidth=0.8, label=label, zorder=1)
            is_first = False

        mmm_da    = mmm_spectrum.sel(mode=full_mode).dropna('frequency')
        freqs_mmm = mmm_da.frequency.values
        mask_mmm  = freqs_mmm > 0
        ax_full.loglog(1.0 / freqs_mmm[mask_mmm], mmm_da.values[mask_mmm], color='black', linewidth=2.5, label='Multi-model mean', zorder=3)

        ax_full.set_title(f'{subplot_letter_r}) EOF mode {idx + 1} (1850-{END_YEAR})', fontweight='bold', fontsize=11)
        ax_full.grid(True, linestyle='--', alpha=0.4, which='both', axis='x')
        ax_full.tick_params(axis='y', labelleft=False, length=4, which='both')
        ax_full.spines['left'].set_visible(True)
        if periods is not None:
            ax_full.set_xlim(periods.min(), periods.max())
        if idx == 0:
            ax_full.legend(loc='lower right', fontsize=9)

    axes[-1, 0].set_xlabel('Period (years)')
    axes[-1, 1].set_xlabel('Period (years)')

    eof_label = 'Rotated EOF' if USE_ROTATED_EOF else 'Regular EOF'
    fig.suptitle(f'Power spectra of leading {eof_label} modes', fontsize=16, fontweight='bold', y=0.92)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = os.path.join(fig_dir, 'eof_power_spectra.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

