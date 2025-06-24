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
#----------------------------------------------------------------------------------------------------------------------#
# --- Altimetry data ---
# DOI: 10.48670/moi-00148
# Link to data: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
#----------------------------------------------------------------------------------------------------------------------#
# --- Budget data ---
# Frederikse, T., Landerer, F., Caron, L., Adhikari, S., Parkes, D., Humphrey, V. W., ... & Wu, Y. H. (2020). The causes of sea-level rise since 1900. Nature, 584(7821), 393-397.
# Link to data: https://zenodo.org/records/3862995
#----------------------------------------------------------------------------------------------------------------------#
# --- GIA data ---
# Peltier, W. R., Argus, D. F., & Drummond, R. (2015). Space geodesy constrains ice age terminal deglaciation: The global ICE‐6G_C (VM5a) model. Journal of Geophysical Research: Solid Earth, 120(1), 450-487.
# Link to data: https://www.atmosp.physics.utoronto.ca/~peltier/data.php
#----------------------------------------------------------------------------------------------------------------------#

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
trend_sla_alt = duacs_yearly.sla.sel(year=common_years).polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1) #MSL proxy   (cm/yr)
trend_asl_fr = asl_frederikse.sel(year=common_years).polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1) #geoid proxy (mm/yr)

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

print(f"ODSL range: {odsl_mm_yr.min().item():.2f} to {odsl_mm_yr.max().item():.2f} mm/yr")

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
        pcc = unbiased_factor * np.sum(w * ((x - mu_x) * (y - mu_y))) / (std_x * std_y)
        
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
ax1.set_title(f'a) MSL (Altimetry SLA)\nMean: {stats_msl['mean_x']:.2f} mm/yr, RMS: {stats_msl['std_x']:.2f} mm/yr', 
              fontsize=11, pad=10)

#subplot 2: Geoid (Frederikse ASL change)
print("Plotting subplot 2...")
im2 = ax2.pcolormesh(trend_asl_fr_regridded_mm_yr.longitude, trend_asl_fr_regridded_mm_yr.latitude,
                     trend_asl_fr_regridded_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax2, is_left=True, is_bottom=True)
ax2.set_title(f'b) Geoid (Frederikse budget ASL)\nMean: {stats_geoid['mean_x']:.2f} mm/yr, RMS: {stats_geoid['std_x']:.2f} mm/yr', 
              fontsize=11, pad=10)

#subplot 3: GIA regridded
print("Plotting subplot 3...")
im3 = ax3.pcolormesh(gia_regridded_mm_yr.longitude, gia_regridded_mm_yr.latitude,
                     gia_regridded_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax3, is_left=True, is_bottom=True)
ax3.set_title(f'c) GIA\nMean: {stats_gia['mean_x']:.2f} mm/yr, RMS: {stats_gia['std_x']:.2f} mm/yr', 
              fontsize=11, pad=10)

#subplot 4: ODSL result (with GIA correction)
print("Plotting subplot 4...")
im4 = ax4.pcolormesh(odsl_mm_yr.longitude, odsl_mm_yr.latitude,
                     odsl_mm_yr, transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax4, is_left=True, is_bottom=True)
ax4.set_title(f'd) ODSL (MSL - Geoid - GIA)\nMean: {stats_odsl['mean_x']:.2f} mm/yr, RMS: {stats_odsl['std_x']:.2f} mm/yr', 
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

### --- CMIP DATA ANALYSIS FOR ODSL --- ###
#----------------------------------------------------------------------------------------------------------------------#
# Preprocessing done by Dewi Le Bars, KNMI
# https://github.com/dlebars/CMIP_SeaLevel/blob/master/code/PreparePlaneVariables.py
#----------------------------------------------------------------------------------------------------------------------#

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

#define scenarios
cmip5_scenarios = ["historical", "rcp26", "rcp45", "rcp85"]
cmip6_scenarios = ["historical", "ssp126", "ssp245", "ssp585"]

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

# %%

# CMIP5 multimodel ODSL (historical + RCP4.5) following Richter et al. 2017 model selection

print("\nCalculating multi-model for 1993-2012")

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

def process_cmip_model(model_name, hist_file, rcp_file, start_year, end_year, extent):
    """Process a single CMIP model and return results in dictionary."""
    try:
        with xr.open_dataset(hist_file) as ds_hist, xr.open_dataset(rcp_file) as ds_rcp:
            #squeeze and rename
            zos_hist = ds_hist['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
            zos_rcp = ds_rcp['CorrectedReggrided_zos'].squeeze('model', drop=True).rename({'lon': 'longitude', 'lat': 'latitude'})
            
            #combine historical and rcp
            combined_zos = xr.concat([zos_hist, zos_rcp], dim='time')
            
            #create region mask for this model's grid
            region_mask = create_region_mask(combined_zos.isel(time=0), extent)
            
            #build model data dictionary
            model_data = {
                'full_timeseries': combined_zos,
                'time_range': (int(combined_zos.time.min().item()), int(combined_zos.time.max().item())),
                'historical_years': (int(zos_hist.time.min().item()), int(zos_hist.time.max().item())),
                'rcp_years': (int(zos_rcp.time.min().item()), int(zos_rcp.time.max().item())),
                'grid_info': {
                    'lon': combined_zos.longitude.values,
                    'lat': combined_zos.latitude.values,
                    'lon_res': float(combined_zos.longitude.diff('longitude').mean().item()),
                    'lat_res': float(combined_zos.latitude.diff('latitude').mean().item())
                },
                'region_mask': region_mask  
            }
            
            #select time period for current analysis
            period_data = combined_zos.sel(time=slice(start_year, end_year))
            
            #linear trend
            trend_coeffs = period_data.polyfit(dim='time', deg=1)
            slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
            
            #cm/year -> mm/year
            slope_mm_yr = slope * 10
            
            #calculate period mean in mm
            period_mean_mm = period_data.mean(dim='time') * 10
            
            #calculate area-weighted statistics
            trend_stats = calculate_weighted_stats(slope_mm_yr, region_mask)
            mean_stats = calculate_weighted_stats(period_mean_mm, region_mask)
            
            #add period-specific results
            model_data['period_analysis'] = {
                'period': (start_year, end_year),
                'trend_mm_yr': slope_mm_yr,
                'trend_coeffs': trend_coeffs,
                'period_mean_mm': period_mean_mm,
                'trend_stats': trend_stats,  
                'mean_stats': mean_stats,     
                'n_years': len(period_data.time)
            }
            
            return model_data, slope_mm_yr
            
    except Exception as e:
        print(f"Could not process model {model_name}: {e}")
        return None, None

#store model data in dictionaries
model_data_dict = {}
valid_models_count = 0
model_trends_for_period = []

#loop
for i, model_name in enumerate(target_cmip5_models):
    #check
    hist_file = all_files['historical'].get(model_name)
    rcp45_file = all_files['rcp45'].get(model_name)

    if hist_file and rcp45_file:
        print(f"Processing target model: {model_name} ({i+1}/{len(target_cmip5_models)})")
        
        model_data, slope_mm_yr = process_cmip_model(
            model_name, hist_file, rcp45_file, 
            start_year, end_year, extent
        )
        
        if model_data is not None:
            model_data_dict[model_name] = model_data
            model_trends_for_period.append(slope_mm_yr)
            valid_models_count += 1
            print(f"Processed model: {model_name}")

print(f"\nFinished processing. Found and processed {valid_models_count} out of {len(target_cmip5_models)} target models.")

print(f"\n{'='*80}")
print(f"Processed models summary ({len(model_data_dict)} models)")
print(f"{'='*80}")
for model_name, data in model_data_dict.items():
    print(f"\n{model_name}:")
    print(f"Time coverage: {data['time_range'][0]}-{data['time_range'][1]} "
          f"(Historical: {data['historical_years'][0]}-{data['historical_years'][1]}, "
          f"RCP4.5: {data['rcp_years'][0]}-{data['rcp_years'][1]})")
    print(f"Grid resolution: {data['grid_info']['lon_res']:.2f}° x {data['grid_info']['lat_res']:.2f}°")
    print(f"Period analyzed: {data['period_analysis']['period'][0]}-{data['period_analysis']['period'][1]} "
          f"({data['period_analysis']['n_years']} years)")
    print(f"Regional trend: {data['period_analysis']['trend_stats']['mean_x']:.2f} ± "
          f"{data['period_analysis']['trend_stats']['std_x']:.2f} mm/yr")
    print(f"Regional mean: {data['period_analysis']['mean_stats']['mean_x']:.2f} ± "
          f"{data['period_analysis']['mean_stats']['std_x']:.2f} mm")
print(f"\n{'='*80}")

#%%

#  Multi-model mean trend (1993-2012) and plotting

print("\nCalculating multi-model mean trend for 1993-2012 and plotting...")

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
    f'CMIP5 multi-model mean ({valid_models_count} models (historical + RCP4.5))\nODSL trend ({start_year}-{end_year}) Mean: {stats_model['mean_x']:.2f} mm/yr, RMS: {stats_model['std_x']:.2f} mm/yr',
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

#removing global mean trend from observed ODSL (CMIP is anomaly field)
print("Removing global mean trend from observed ODSL...")

#contribution of each grid cell to the final average is proportional to its actual area on the earths surface
weights = np.cos(np.deg2rad(odsl_observed_regridded.latitude))
global_mean_obs_trend = odsl_observed_regridded.weighted(weights).mean(dim=("latitude", "longitude")).item()
print(f"Global mean trend in observations: {global_mean_obs_trend:.2f} mm/yr")
odsl_obs_dynamic_trend = odsl_observed_regridded - global_mean_obs_trend

#difference (model - Observation)
difference = model_mean_trend - odsl_obs_dynamic_trend

#Pattern Correlation Coefficient (PCC)
print("Calculating area-weighted statistics...")

region_mask = create_region_mask(model_mean_trend, extent)
stats_comparison = calculate_weighted_stats(model_mean_trend, region_mask, data_y=odsl_obs_dynamic_trend)
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
ax1.set_title(f'a) Observed ODSL (Regridded)\nMean: {stats_comparison["mean_y"]:.2f} mm/yr, RMS: {stats_comparison["std_y"]:.2f} mm/yr', fontsize=11)

#subplot 2: modelled ODSL
add_map_features(ax2, is_left=False, is_bottom=True)
mesh2 = model_mean_trend.plot.pcolormesh(
    ax=ax2, transform=ccrs.PlateCarree(), cmap='coolwarm',
    vmin=vmin_unified, vmax=vmax_unified, add_colorbar=False
)
ax2.set_title(f'b) CMIP5 mean ODSL\nMean: {stats_comparison["mean_x"]:.2f} mm/yr, RMS: {stats_comparison["std_x"]:.2f} mm/yr', fontsize=11)

#subplot 3: difference (model - observed)
add_map_features(ax3, is_left=False, is_bottom=True)
mesh3 = difference.plot.pcolormesh(
    ax=ax3, transform=ccrs.PlateCarree(), cmap='coolwarm',
    vmin=-vmax_unified, vmax=vmax_unified, add_colorbar=False
)
ax3.set_title(f'c) Difference (model - obs)\nMean: {stats_difference["mean_x"]:.2f} mm/yr, RMS: {stats_difference["std_x"]:.2f} mm/yr', fontsize=11)

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

# --- SLIDING WINDOW ANALYSIS --- # 
# Richter et al. 2017, Chapter 2. Data and methods, d. Methods

#regrid observations to model grid
print("Regridding observations to CMIP model grid...")
sample_model_data = list(model_data_dict.values())[0]
sample_model_grid = sample_model_data['period_analysis']['trend_mm_yr']

#regridder
regridder_obs_to_model = xe.Regridder(odsl_mm_yr, sample_model_grid, 'bilinear', periodic=True)
odsl_mm_yr_regridded = regridder_obs_to_model(odsl_mm_yr)

try:
    regridder_obs_to_model.clean_weight_file()
except AttributeError:
    pass

print(f"Original observation grid: {odsl_mm_yr.shape}")
print(f"Regridded observation grid: {odsl_mm_yr_regridded.shape}")
print(f"Model grid: {sample_model_grid.shape}")

def sliding_window_analysis_for_model(model_name, model_data, obs_odsl_regridded, window_size=20, 
                                    start_year=1850, end_year=2012):
    """Sliding window analysis on a single model."""
    
    print(f"  Processing sliding windows for {model_name}...")
    
    #full timeseries and region mask
    full_timeseries = model_data['full_timeseries']
    region_mask = model_data['region_mask']
    
    #initialize storage
    results = {
        'windows': [],
        'trends': [],
        'variability': [],
        'mdt': [],
        'pcc': [],
        'rmse': [],
        'rmse_total': []
    }
    
    #slide window
    n_windows = 0
    for window_start in range(start_year, end_year - window_size + 1):
        window_end = window_start + window_size - 1
        
        #data for current window
        window_data = full_timeseries.sel(time=slice(window_start, window_end))
        
        if len(window_data.time) < window_size * 0.75:
            continue
            
        #linear trend (cm/yr -> mm/yr)
        trend_coeffs = window_data.polyfit(dim='time', deg=1)
        trend_mm_yr = trend_coeffs.polyfit_coefficients.sel(degree=1) * 10
        
        #cmip odsl
        mdt = window_data.mean(dim='time') * 10
        
        #variability after detrending
        intercept = trend_coeffs.polyfit_coefficients.sel(degree=0)
        slope = trend_coeffs.polyfit_coefficients.sel(degree=1)
        time_values = window_data.time.values
        
        detrended = window_data.copy()
        for t_idx, t_val in enumerate(time_values):
            trend_value = intercept + slope * t_val
            detrended[t_idx] = window_data[t_idx] - trend_value
            
        variability = detrended.std(dim='time') * 10
        
        #results
        results['windows'].append((window_start, window_end))
        results['trends'].append(trend_mm_yr)
        results['variability'].append(variability)
        results['mdt'].append(mdt)
        
        #centered statistics
        stats = calculate_weighted_stats(trend_mm_yr, region_mask, data_y=obs_odsl_regridded)
        results['pcc'].append(stats['pcc'])
        results['rmse'].append(stats['rmse'])
        
        #total RMSE without removing regional mean
        weights = np.cos(np.deg2rad(trend_mm_yr.latitude))
        weights_normalized = weights / weights.sum()
        diff_squared = (trend_mm_yr - obs_odsl_regridded)**2
        masked_diff = diff_squared.where(region_mask)
        weighted_mse = (weights_normalized * masked_diff).sum()
        rmse_total = np.sqrt(weighted_mse)
        results['rmse_total'].append(float(rmse_total))
        
        n_windows += 1
        
    print(f"Processed {n_windows} windows")
    
    #best periods
    pcc_array = np.array(results['pcc'])
    rmse_array = np.array(results['rmse'])
    
    #check valid PCC/RMSE values
    valid_pcc = ~np.isnan(pcc_array)
    valid_rmse = ~np.isnan(rmse_array)
    
    if valid_pcc.any():
        best_pcc_idx = np.nanargmax(pcc_array)
        results['best_pcc'] = {
            'window': results['windows'][best_pcc_idx],
            'pcc': results['pcc'][best_pcc_idx],
            'trend': results['trends'][best_pcc_idx],
            'variability': results['variability'][best_pcc_idx],
            'mdt': results['mdt'][best_pcc_idx]
        }
        print(f"Best PCC: {results['best_pcc']['pcc']:.3f} ({results['best_pcc']['window'][0]}-{results['best_pcc']['window'][1]})")
    else:
        print(f"WARNING: No valid PCC values found for {model_name}")
        
    if valid_rmse.any():
        best_rmse_idx = np.nanargmin(rmse_array)
        results['best_rmse'] = {
            'window': results['windows'][best_rmse_idx],
            'rmse': results['rmse'][best_rmse_idx],
            'trend': results['trends'][best_rmse_idx],
            'variability': results['variability'][best_rmse_idx],
            'mdt': results['mdt'][best_rmse_idx]
        }
        print(f"Best RMSE: {results['best_rmse']['rmse']:.3f} ({results['best_rmse']['window'][0]}-{results['best_rmse']['window'][1]})")
    else:
        print(f"WARNING: No valid RMSE values found for {model_name}")
    
    return results

print("\nPerforming sliding window analysis...")
print("=" * 60)

#run sliding window analysis all models
sliding_results = {}
for model_name, model_data in model_data_dict.items():
    results = sliding_window_analysis_for_model(
        model_name, model_data, odsl_mm_yr_regridded,
        window_size=20, start_year=1850, end_year=2012
    )
    sliding_results[model_name] = results

#%%

#ensemble means following Richter et al. 2017
print("\nCreating ensemble means...")

#observational period (1993-2012) ensemble
obs_period_trends = []
for model_name, model_data in model_data_dict.items():
    obs_period_trends.append(model_data['period_analysis']['trend_mm_yr'])

#best PCC period ensemble
best_pcc_trends = []
for model_name, results in sliding_results.items():
    if 'best_pcc' in results:
        best_pcc_trends.append(results['best_pcc']['trend'])

#best RMSE period ensemble
best_rmse_trends = []
for model_name, results in sliding_results.items():
    if 'best_rmse' in results:
        best_rmse_trends.append(results['best_rmse']['trend'])

#ensemble means
obs_period_mean = xr.concat(obs_period_trends, dim='model').mean(dim='model')
best_pcc_mean = xr.concat(best_pcc_trends, dim='model').mean(dim='model') if best_pcc_trends else None
best_rmse_mean = xr.concat(best_rmse_trends, dim='model').mean(dim='model') if best_rmse_trends else None

#statistics for each ensemble
region_mask = create_region_mask(obs_period_mean, extent)

#%%

print("\nEnsemble mean statistics:")
print("-" * 40)

#remove global mean from regridded observations (models are anomalies)
weights = np.cos(np.deg2rad(odsl_mm_yr_regridded.latitude))
global_mean = odsl_mm_yr_regridded.weighted(weights).mean(dim=("latitude", "longitude")).item()
odsl_obs_dynamic = odsl_mm_yr_regridded - global_mean

#observation period ensemble
stats_obs = calculate_weighted_stats(obs_period_mean, region_mask, data_y=odsl_obs_dynamic)
print(f"Observational period (1993-2012) ensemble:")
print(f"  PCC: {stats_obs['pcc']:.3f}, RMSE: {stats_obs['rmse']:.3f} mm/yr")

#best PCC ensemble
if best_pcc_mean is not None:
    stats_pcc = calculate_weighted_stats(best_pcc_mean, region_mask, data_y=odsl_obs_dynamic)
    print(f"Best PCC period ensemble:")
    print(f"  PCC: {stats_pcc['pcc']:.3f}, RMSE: {stats_pcc['rmse']:.3f} mm/yr")

#best RMSE ensemble
if best_rmse_mean is not None:
    stats_rmse = calculate_weighted_stats(best_rmse_mean, region_mask, data_y=odsl_obs_dynamic)
    print(f"Best RMSE period ensemble:")
    print(f"  PCC: {stats_rmse['pcc']:.3f}, RMSE: {stats_rmse['rmse']:.3f} mm/yr")

#%%

#plot PCC time series
print("\nPlotting PCC time series...")
fig, ax = plt.subplots(figsize=(12, 6))

for model_name, results in sliding_results.items():
    if results['windows']:
        window_centers = [(w[0] + w[1]) / 2 for w in results['windows']]
        ax.plot(window_centers, results['pcc'], alpha=0.5, linewidth=1)

ax.axvline(2002.5, color='red', linestyle='--', alpha=0.7, label='Obs period center')
ax.set_xlabel('Window center year', fontsize=12)
ax.set_ylabel('PCC', fontsize=12)
ax.set_title('Model-observation PCC\n20-year sliding windows (1850-2012)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(-1, 1)
plt.tight_layout()
plt.show()

#summary best periods across models
print("\n" + "=" * 60)
print("SUMMARY: Best matching periods for each model")
print("=" * 60)
for model_name, results in sliding_results.items():
    if 'best_pcc' in results:
        print(f"{model_name:20s} | Best PCC: {results['best_pcc']['window'][0]}-{results['best_pcc']['window'][1]} "
              f"(PCC={results['best_pcc']['pcc']:.3f})")

# %%
