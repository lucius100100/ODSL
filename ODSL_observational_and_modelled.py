#%%

# Import packages
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe
import os
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as ctk
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

#%%
### --- OBSERVATIONAL DATA ANALYSIS FOR ODSL --- ###

# Observational configuration
duacs_dir = r'C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\Altimetry\\'
budget_dir = r'C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\Budget\Frederikse\\'
gia_dir = r'C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\Budget\GIA\\'
fig_dir = './figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

#period following Richter et al. 2017
start_year = 1993
end_year = 2012

#function for rotating longitude
def rotate_longitude(ds, name_lon):
    """Convert longitude from 0-360 to -180-180 coordinate system."""
    ds_copy = ds.copy()
    new_lon_values = (((ds_copy[name_lon] + 180) % 360) - 180)
    ds_copy = ds_copy.assign_coords({name_lon: new_lon_values})
    return ds_copy.sortby(ds_copy[name_lon])

#%%

# Altimetry data
print("Loading altimetry data...")
duacs_ds = xr.open_mfdataset(f'{duacs_dir}cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_*.nc', combine='by_coords').load()
duacs_ds = rotate_longitude(duacs_ds, 'longitude')
duacs_ds['sla'] *= 100  #m to cm
duacs_yearly = duacs_ds.groupby('time.year').mean()

print(f"Altimetry range: {duacs_yearly.sla.min().item():.2f} to {duacs_yearly.sla.max().item():.2f} cm/yr")

#%%

# Frederikse budget
print("Loading Frederikse budget data...")
ds_frederikse = xr.open_dataset(f'{budget_dir}total.nc')

#geocentric sea level = RSL + RAD
rsl_component = ds_frederikse['total_rsl_mean']  #Relative Sea Level (RSL)          [mm]
vlm_component = ds_frederikse['total_rad_mean']  #RAD or Vertical Land Motion (VLM) [mm]
asl_frederikse = rsl_component + vlm_component   #geocentric sea level

#rotate and standardize coordinates
asl_frederikse = rotate_longitude(asl_frederikse, 'lon')
asl_frederikse = asl_frederikse.rename({'lon': 'longitude', 'lat': 'latitude', 'time': 'year'})

print(f"Geocentric sea level range: {asl_frederikse.min().item():.2f} to {asl_frederikse.max().item():.2f} cm/yr")

#%%

# GIA 
print("Loading GIA data...")
gia_rad = xr.open_dataset(f'{gia_dir}drad.1grid_O512.nc')
gia_sea = xr.open_dataset(f'{gia_dir}dsea.1grid_O512.nc')

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

#%%

# Calculations

#common years
year_slice = slice(start_year, end_year)
alt_years = duacs_yearly.sel(year=year_slice).year.values
fr_years = asl_frederikse.sel(year=year_slice).year.values
common_years = np.intersect1d(alt_years, fr_years)

print(f"Analysis period: {common_years.min()}-{common_years.max()} ({len(common_years)} years)")

#calculate trends over common period
trend_sla_alt = duacs_yearly.sla.sel(year=common_years).polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)    #MSL proxy   (cm/yr)
trend_asl_fr = asl_frederikse.sel(year=common_years).polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)       #geoid proxy (mm/yr)

#%%

# Regridding using xESMF
print("Creating regridders...")
regridder_frederikse = xe.Regridder(trend_asl_fr, trend_sla_alt, 'bilinear', periodic=True)
regridder_gia = xe.Regridder(gia_rad_da, trend_sla_alt, 'bilinear', periodic=True)

#regrid to altimetry grid
print("Regridding Frederikse budget trend...")
trend_asl_fr_regridded = regridder_frederikse(trend_asl_fr)
print("Regridding GIA data...")
gia_rad_regridded = regridder_gia(gia_rad_da)
gia_sea_regridded = regridder_gia(gia_sea_da)
gia_geoid_correction_mm_yr = gia_sea_regridded + gia_rad_regridded

#convert components to mm/yr
print("Converting all trends to mm/yr...")
#altimetry MSL (cm/yr -> mm/yr)
trend_sla_alt_mm_yr = trend_sla_alt * 10
#Frederikse geoid trend (is already mm/yr)
trend_asl_fr_regridded_mm_yr = trend_asl_fr_regridded
gia_regridded_mm_yr = gia_geoid_correction_mm_yr

#calculate ODSL (mm/yr)
print("Calculating ODSL...")
odsl_mm_yr = trend_sla_alt_mm_yr - trend_asl_fr_regridded_mm_yr - gia_regridded_mm_yr

print(f"ODSL range: {odsl_mm_yr.min().item():.1f} to {odsl_mm_yr.max().item():.1f} mm/yr")

#clean up regridder
print("Cleaning up regridder...")
try:
    regridder_frederikse.clean_weight_file()
    regridder_gia.clean_weight_file()
except AttributeError:
    #older version of xESMF, cleanup happens automatically
    pass

#%%

# Plotting ODSL results
print("Plotting ODSL results...")

#North Atlantic Ocean projection and extent following Richter et al. 2017 figure 2
proj = ccrs.AlbersEqualArea(central_longitude=-15.0, central_latitude=60.0, standard_parallels=(55.0, 75.0))
extent = [-65, 40, 50, 80]

#plotting
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(11, 10),
    subplot_kw={'projection': proj}
) 
ax1, ax2, ax3, ax4 = axes.flatten()

#mask for regionwide statistics
def create_region_mask(data_array, extent):
    """Create a mask for the North Atlantic region."""
    lon_min, lon_max, lat_min, lat_max = extent
    mask = ((data_array.longitude >= lon_min) & 
            (data_array.longitude <= lon_max) & 
            (data_array.latitude >= lat_min) & 
            (data_array.latitude <= lat_max))
    return mask

region_mask = create_region_mask(trend_sla_alt_mm_yr, extent)

#calculate statistics
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
        pcc = unbiased_factor * np.sum(w * ((x - mu_x) * (y - mu_y))) / (var_x * var_y)
        
        results.update({
            'mean_y': mu_y.item(), 'std_y': std_y.item(),
            'rmse': rmse.item(), 'pcc': pcc.item()
        })
        
    return results

stats_msl = calculate_weighted_stats(trend_sla_alt_mm_yr, region_mask)
stats_geoid = calculate_weighted_stats(trend_asl_fr_regridded_mm_yr, region_mask)
stats_gia = calculate_weighted_stats(gia_regridded_mm_yr, region_mask)
stats_odsl = calculate_weighted_stats(odsl_mm_yr, region_mask)

#colormap
vmax_components = max(abs(trend_sla_alt_mm_yr.quantile(0.02)), abs(trend_sla_alt_mm_yr.quantile(0.98)),
                    abs(trend_asl_fr_regridded_mm_yr.quantile(0.02)), abs(trend_asl_fr_regridded_mm_yr.quantile(0.98)))

#same map features for all subplots
def add_map_features(ax, is_left=False, is_bottom=False):
    """Add standard map features and set the conic boundary shape."""
    #boundary in lat/lon coordinates
    lon_min, lon_max, lat_min, lat_max = extent
    boundary_path = mpath.Path([
        [lon_min, lat_min], [lon_max, lat_min],
        [lon_max, lat_max], [lon_min, lat_max],
        [lon_min, lat_min]
    ]).interpolated(50)

    #AlbersEqualArea projection
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    boundary_in_proj_coords = proj_to_data.transform_path(boundary_path)
    ax.set_boundary(boundary_in_proj_coords)

    #x/y limits to the extents of the projected boundary
    ax.set_boundary(boundary_in_proj_coords)
    verts = boundary_in_proj_coords.vertices
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    
    #features
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)

    #gridlines with labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
    
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = is_left
    gl.bottom_labels =  is_bottom
    
#subplot 1: MSL (Altimetry SLA)
print("Plotting subplot 1...")
im1 = ax1.pcolormesh(trend_sla_alt_mm_yr.longitude, trend_sla_alt_mm_yr.latitude, 
                     trend_sla_alt_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax1, is_left=True, is_bottom=True)
ax1.set_title(f'a) MSL (Altimetry SLA)\nMean: {stats_msl['mean_x']:.1f} mm/yr, RMS: {stats_msl['std_x']:.1f} mm/yr', 
              fontsize=11, pad=10)

#subplot 2: Geoid (Frederikse ASL change)
print("Plotting subplot 2...")
im2 = ax2.pcolormesh(trend_asl_fr_regridded_mm_yr.longitude, trend_asl_fr_regridded_mm_yr.latitude,
                     trend_asl_fr_regridded_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax2, is_left=True, is_bottom=True)
ax2.set_title(f'b) Geoid (Frederikse budget ASL)\nMean: {stats_geoid['mean_x']:.1f} mm/yr, RMS: {stats_geoid['std_x']:.1f} mm/yr', 
              fontsize=11, pad=10)

#subplot 3: GIA regridded
print("Plotting subplot 3...")
im3 = ax3.pcolormesh(gia_regridded_mm_yr.longitude, gia_regridded_mm_yr.latitude,
                     gia_regridded_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax3, is_left=True, is_bottom=True)
ax3.set_title(f'c) GIA\nMean: {stats_gia['mean_x']:.1f} mm/yr, RMS: {stats_gia['std_x']:.1f} mm/yr', 
              fontsize=11, pad=10)

#subplot 4: ODSL result (with GIA correction)
print("Plotting subplot 4...")
im4 = ax4.pcolormesh(odsl_mm_yr.longitude, odsl_mm_yr.latitude,
                     odsl_mm_yr, transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax4, is_left=True, is_bottom=True)
ax4.set_title(f'd) ODSL (MSL - Geoid - GIA)\nMean: {stats_odsl['mean_x']:.1f} mm/yr, RMS: {stats_odsl['std_x']:.1f} mm/yr', 
              fontsize=11, pad=10)

#single colorbar for all subplots
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.025])
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', pad=0.02)
cbar.set_label('Sea level trend (mm/yr)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

#figure layout
plt.suptitle(f'Observed ODSL trend ({common_years.min()}-{common_years.max()})',
             fontsize=16, fontweight='bold', y=0.98)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.88, hspace=0.1, wspace=0.1)

#save figure
print("Saving figure...")
fig_path = os.path.join(fig_dir, f'ODSL_with_GIA_{common_years.min()}_{common_years.max()}.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Observational ODSL figure saved to: {fig_path}")

#%%

# Summary
"""
print("\n=== ODSL analysis summary ===")
print(f"Analysis period: {common_years.min()}-{common_years.max()}")
print(f"MSL proxy (altimetry SLA) range: {trend_sla_alt_mm_yr.min().item():.1f} to {trend_sla_alt_mm_yr.max().item():.1f} mm/yr")
print(f"Geoid proxy (Frederikse ASL) range: {trend_asl_fr_regridded_mm_yr.min().item():.1f} to {trend_asl_fr_regridded_mm_yr.max().item():.1f} mm/yr")
print(f"GIA radial component range: {gia_rad_regridded.min().item():.1f} to {gia_rad_regridded.max().item():.1f} mm/yr")
print(f"GIA sea component range: {gia_sea_regridded.min().item():.1f} to {gia_sea_regridded.max().item():.1f} mm/yr")
print(f"ODSL range: {odsl_mm_yr.min().item():.1f} to {odsl_mm_yr.max().item():.1f} mm/yr")
print(f"ODSL global mean: {odsl_mm_yr.mean().item():.1f} mm/yr")
print("\n=== Regional statistics (North Atlantic) ===")
print(f"MSL mean: {msl_mean:.1f} mm/yr, RMS: {msl_rms:.1f} mm/yr")
print(f"Geoid mean: {geoid_mean:.1f} mm/yr, RMS: {geoid_rms:.1f} mm/yr")
print(f"GIA mean: {gia_mean:.1f} mm/yr, RMS: {gia_rms:.1f} mm/yr")
print(f"ODSL mean: {odsl_mean:.1f} mm/yr, RMS: {odsl_rms:.1f} mm/yr")
"""

#%%

### --- CMIP DATA ANALYSIS FOR ODSL --- ###

# Load in CMIP data

#paths
CMIP_BASE_PATHS = {
    "CMIP5": r"C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\CMIP\CMIP5",
    "CMIP6": r"C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\CMIP\CMIP6"
}

def get_scenario_files(cmip_version, scenario, return_models=False):
    """Get all NetCDF files for a specific CMIP scenario."""
    base_path = CMIP_BASE_PATHS[cmip_version]
    
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

def extract_model_from_path(filepath):
    """Extract model name from a CMIP file path"""
    basename = os.path.basename(filepath)
    #remove .nc extension
    name_without_ext = basename.replace('.nc', '')
    #split by underscore
    parts = name_without_ext.split('_')
    
    if len(parts) >= 6:
        model_parts = parts[3:-2]
        return '_'.join(model_parts)
    else:
        return parts[3] if len(parts) > 3 else "unknown"

def calculate_zos_trend(filepath):
    """Load a CMIP NetCDF file, calculate the linear trend of zos, and convert units to mm/year."""
    try:
        with xr.open_dataset(filepath) as ds:
            #variable name
            zos_var_name = 'CorrectedReggrided_zos'
            
            #squeeze and rename
            data = ds[zos_var_name].squeeze('model', drop=True)
            if 'lon' in data.coords and 'lat' in data.coords:
                data = data.rename({'lon': 'longitude', 'lat': 'latitude'})

            #linear trend along the time dimension
            trend_coeffs = data.polyfit(dim='time', deg=1)
            slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
            
            #cm/year -> mm/year
            slope_mm_yr = slope * 10
            
            return slope_mm_yr
            
    except Exception as e:
        print(f"Could not process file {os.path.basename(filepath)}: {e}")
        return None

#define scenarios
cmip5_scenarios = ["historical", "rcp26", "rcp45", "rcp85"]
cmip6_scenarios = ["historical", "ssp126", "ssp245", "ssp585"]
output_pdf_path = "cmip5_model_odsl_trends.pdf"

#find all available files and unique models
print("Finding all available models and files...")
all_files = {}
all_models = set()
for scenario in cmip5_scenarios:
    files, models = get_scenario_files("CMIP5", scenario, return_models=True)
    #dictionary for easy lookup: {model_name: filepath}
    all_files[scenario] = {model: file for model, file in zip(models, files)}
    all_models.update(models)

#sort models
sorted_models = sorted(list(all_models))
print(f"Found {len(sorted_models)} unique models to process.")

#%%

# Plot all available models and scenarios

"""
#loop through each model, plot, and save to PDF
with PdfPages(output_pdf_path) as pdf:
    for i, model_name in enumerate(sorted_models):
        print(f"Processing model {i+1}/{len(sorted_models)}: {model_name}")

        #figure with 2x2 subplots for four scenarios
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 10),
            subplot_kw={'projection': proj}
        )
        axes_flat = axes.flatten()
        
        #store trends
        model_trends = {}

        #calculate trend for each scenario per model
        for scenario in cmip5_scenarios:
            if model_name in all_files[scenario]:
                filepath = all_files[scenario][model_name]
                trend_data = calculate_zos_trend(filepath)
                if trend_data is not None:
                    model_trends[scenario] = trend_data

        #if no data could be processed for model, skip
        if not model_trends:
            print(f"  -> Skipping model {model_name} due to no processable data.")
            plt.close(fig)
            continue
        
        #symmetric color range
        max_abs_val = 0
        for trend in model_trends.values():
            #quantiles to avoid extreme outliers skewing color scale
            v_max_q = abs(trend.quantile(0.99, skipna=True).item())
            v_min_q = abs(trend.quantile(0.01, skipna=True).item())
            max_abs_val = max(max_abs_val, v_max_q, v_min_q)
        
        vmax = max_abs_val
        vmin = -max_abs_val
        
        #plotting
        for ax, scenario in zip(axes_flat, cmip5_scenarios):
            if scenario in model_trends:
                data_to_plot = model_trends[scenario]
                
                #determine if axis labels should be drawn
                is_left = ax in [axes[0,0], axes[1,0]]
                is_bottom = ax in [axes[1,0], axes[1,1]]
                
                add_map_features(ax, is_left=is_left, is_bottom=is_bottom)

                mesh = data_to_plot.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap='coolwarm',
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False #single colorbar for each subplot per model
                )
                ax.set_title(f"Scenario: {scenario.upper()}")
            else:
                #if no data, hide axis
                ax.set_visible(False)
        
        #single colorbar for the whole figure
        fig.subplots_adjust(right=0.85, hspace=0.15, wspace=0.1)
        cbar_ax = fig.add_axes([0.88, 0.25, 0.03, 0.5])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        cbar.set_label('Sea level trend (mm/year)')

        #main title for the model
        fig.suptitle(f'CMIP5 ODSL trend for model: {model_name}', fontsize=16, y=0.96)
        
        #save to PDF
        pdf.savefig(fig)
        plt.close(fig)

print(f"Output saved to: {output_pdf_path}")
"""

# %%

# Time coverage CMIP scenarios

def print_scenario_time_coverage(cmip_version, scenarios):
    """Analyzes and prints the overall time range for each CMIP scenario."""
    print(f"\n{cmip_version} time coverage:")
    
    for scenario in scenarios:
        #all files for the current scenario
        files = get_scenario_files(cmip_version, scenario)
        
        if not files:
            print(f"  - {scenario.upper():<12}: No files found.")
            continue

        #initialize with extreme values
        overall_start_year = float('inf')
        overall_end_year = float('-inf')

        #loop through each model file in the scenario
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    start_year = int(ds.time.min().item())
                    end_year = int(ds.time.max().item())

                    #update overall time range
                    if start_year < overall_start_year:
                        overall_start_year = start_year
                    if end_year > overall_end_year:
                        overall_end_year = end_year
            except Exception as e:
                #warning if not processable
                print(f"[Warning] Could not process file {os.path.basename(f)}: {e}")

        #print result for scenario
        if overall_start_year != float('inf'):
            print(f"  - {scenario.upper():<12}: {overall_start_year} - {overall_end_year}")
        else:
            print(f"  - {scenario.upper():<12}: Could not read time data from any files.")

print_scenario_time_coverage("CMIP5", cmip5_scenarios)

# %%

# CMIP multimodel mean ODSL (historical + RCP4.5)

print("\nCalculating multi-model mean trend for 1993-2012")

#target models from supplementary material table 1 Richter et al. 2017
target_cmip5_models = [
    "ACCESS1-0",
    "bcc-csm1-1",
    "CanESM2",
    "CCSM4",
    "CMCC-CMS",
    "CNRM-CM5",
    "CSIRO-Mk3-6-0",
    "GFDL-ESM2G",
    "GFDL-ESM2M",
    "HadGEM2-CC",
    "HadGEM2-ES",
    "inmcm4",
    "IPSL-CM5A-MR",
    "MIROC-ESM",
    "MPI-ESM-LR",
    "MRI-CGCM3",
    "NorESM1-M",
    "NorESM1-ME"
]

all_files = {}
for scenario in ["historical", "rcp45"]:
    files, models = get_scenario_files("CMIP5", scenario, return_models=True)
    all_files[scenario] = {model: file for model, file in zip(models, files)}

#store the trend
model_trends_for_period = []
valid_models_count = 0

#loop
for i, model_name in enumerate(target_cmip5_models):
    #check
    hist_file = all_files['historical'].get(model_name)
    rcp45_file = all_files['rcp45'].get(model_name)

    if hist_file and rcp45_file:
        try:
            with xr.open_dataset(hist_file) as ds_hist, xr.open_dataset(rcp45_file) as ds_rcp:
                
                print(f"Processing target model: {model_name} ({i+1}/{len(target_cmip5_models)})")

                #squeeze and rename
                zos_hist = ds_hist['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
                zos_rcp45 = ds_rcp['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
                
                #combine historical and rcp45
                combined_zos = xr.concat([zos_hist, zos_rcp45], dim='time')
                
                #select time period
                period_data = combined_zos.sel(time=slice(start_year, end_year))

                #linear trend
                trend_coeffs = period_data.polyfit(dim='time', deg=1)
                slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
                
                #cm/year -> mm/year
                slope_mm_yr = slope * 10
                
                model_trends_for_period.append(slope_mm_yr)
                valid_models_count += 1
                print(f"Processed model: {model_name}")

        except Exception as e:
            print(f"Could not process model {model_name}: {e}")
    else:
        pass

print(f"\nFinished processing. Found and processed {valid_models_count} out of {len(target_cmip5_models)} target models.")

print("Calculating multi-model mean and plotting...")

#mean trend calculation and plotting
all_trends_da = xr.concat(model_trends_for_period, dim='model')

#mean across the model dimension
model_mean_trend = all_trends_da.mean(dim='model', skipna=True)

fig, ax = plt.subplots(
    figsize=(9, 8),
    subplot_kw={'projection': proj}
)

add_map_features(ax, is_left=True, is_bottom=True)

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
region_mask = create_region_mask(model_mean_trend, extent)
stats_model = calculate_weighted_stats(model_mean_trend, region_mask)

#set the main title
ax.set_title(
    f'CMIP5 multi-model mean ({valid_models_count} models (historical + RCP4.5))\nODSL trend ({start_year}-{end_year}) Mean: {stats_model['mean_x']:.1f} mm/yr, RMS: {stats_model['std_x']:.1f} mm/yr',
    fontsize=12, pad=15
)

plt.show()

#%%

### Observed vs. modeled ODSL (1993-2012) ###

print("\nComparing observed and modeled ODSL")

#regrid observed to modelled
print("Creating xesmf regridder...")
regridder = xe.Regridder(
    odsl_mm_yr,           
    model_mean_trend,     
    'bilinear',          
    periodic=True
)

print("Regridding observed data to match model grid...")
odsl_observed_regridded = regridder(odsl_mm_yr)

#difference (model - Observation)
difference = model_mean_trend - odsl_observed_regridded

#Pattern Correlation Coefficient (PCC)
print("Calculating area-weighted statistics...")

region_mask = create_region_mask(model_mean_trend, extent)
stats_comparison = calculate_weighted_stats(model_mean_trend, region_mask, data_y=odsl_observed_regridded)
stats_difference = calculate_weighted_stats(difference, region_mask)
pcc_w = stats_comparison['pcc']

print(f"Area-Weighted PCC in North Atlantic: {pcc_w:.2f}")

#%%

# Plotting observed vs. modeled trend

#three subplots
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3,
    figsize=(22, 7),
    subplot_kw={'projection': proj}
)

vmax_unified = max(
    abs(odsl_observed_regridded.quantile(0.98, skipna=True).item()),
    abs(model_mean_trend.quantile(0.98, skipna=True).item()),
    abs(difference.quantile(0.98, skipna=True).item())
)
vmin_unified = -vmax_unified

#subplot 1: observed ODSL
add_map_features(ax1, is_left=True, is_bottom=True)
mesh1 = odsl_observed_regridded.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(), cmap='coolwarm',
    vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False
)
ax1.set_title(f'a) Observed ODSL (Regridded)\nMean: {stats_comparison["mean_y"]:.1f} mm/yr, RMS: {stats_comparison["std_y"]:.1f} mm/yr', fontsize=11)

#subplot 2: modelled ODSL
add_map_features(ax2, is_left=False, is_bottom=True)
mesh2 = model_mean_trend.plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), cmap='coolwarm',
    vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False
)
ax2.set_title(f'b) CMIP5 mean ODSL\nMean: {stats_comparison["mean_x"]:.1f} mm/yr, RMS: {stats_comparison["std_x"]:.1f} mm/yr', fontsize=11)

#subplot 3: difference (model - observed)
add_map_features(ax3, is_left=False, is_bottom=True)
mesh3 = difference.plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), cmap='coolwarm',
    vmin=-vmax_unified, vmax=vmax_unified, add_colorbar=False
)
ax3.set_title(f'c) Difference (model - obs)\nMean: {stats_difference["mean_x"]:.1f} mm/yr, RMS: {stats_difference["std_x"]:.1f} mm/yr', fontsize=11)

#title and layout
fig.suptitle(
    f'Observed vs. modeled ODSL trend ({start_year}-{end_year})\n'
    f'North Atlantic PCC = {pcc_w:.2f}',
    fontsize=16, y=1.02
)

#shared colorbar
cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.03])
cbar = fig.colorbar(mesh1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Trend (mm/year)', fontsize=12)

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.85, wspace=0.15)
plt.show()

# %%
