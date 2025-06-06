#%%

# Import packages
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as ctk
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

#%%
### OBSERVATIONAL DATA ANALYSIS FOR ODSL ###

# Configuration
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
    """Convert longitude from 0-360 to -180-180 coordinate system"""
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
    """Create a mask for the North Atlantic region"""
    lon_min, lon_max, lat_min, lat_max = extent
    mask = ((data_array.longitude >= lon_min) & 
            (data_array.longitude <= lon_max) & 
            (data_array.latitude >= lat_min) & 
            (data_array.latitude <= lat_max))
    return mask

region_mask = create_region_mask(trend_sla_alt_mm_yr, extent)

#calculate statistics
def calculate_regional_stats(data_array, mask):
    """Calculate mean and RMS for the masked region"""
    masked_data = data_array.where(mask)
    regional_mean = masked_data.mean(skipna=True).item()
    regional_rms = np.sqrt((masked_data**2).mean(skipna=True)).item()
    return regional_mean, regional_rms

msl_mean, msl_rms = calculate_regional_stats(trend_sla_alt_mm_yr, region_mask)
geoid_mean, geoid_rms = calculate_regional_stats(trend_asl_fr_regridded_mm_yr, region_mask)
gia_mean, gia_rms = calculate_regional_stats(gia_regridded_mm_yr, region_mask)
odsl_mean, odsl_rms = calculate_regional_stats(odsl_mm_yr, region_mask)

#colormap
vmax_components = max(abs(trend_sla_alt_mm_yr.quantile(0.02)), abs(trend_sla_alt_mm_yr.quantile(0.98)),
                    abs(trend_asl_fr_regridded_mm_yr.quantile(0.02)), abs(trend_asl_fr_regridded_mm_yr.quantile(0.98)))

#same map features for all subplots
def add_map_features(ax, is_left=False, is_bottom=False):
    """Add standard map features and set the conic boundary shape"""
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
ax1.set_title(f'a) MSL (Altimetry SLA)\nMean: {msl_mean:.1f} mm/yr, RMS: {msl_rms:.1f} mm/yr', 
              fontsize=11, pad=10)

#subplot 2: Geoid (Frederikse ASL change)
print("Plotting subplot 2...")
im2 = ax2.pcolormesh(trend_asl_fr_regridded_mm_yr.longitude, trend_asl_fr_regridded_mm_yr.latitude,
                     trend_asl_fr_regridded_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax2, is_left=True, is_bottom=True)
ax2.set_title(f'b) Geoid (Frederikse budget ASL)\nMean: {geoid_mean:.1f} mm/yr, RMS: {geoid_rms:.1f} mm/yr', 
              fontsize=11, pad=10)

#subplot 3: GIA regridded
print("Plotting subplot 3...")
im3 = ax3.pcolormesh(gia_regridded_mm_yr.longitude, gia_regridded_mm_yr.latitude,
                     gia_regridded_mm_yr, transform=ccrs.PlateCarree(), 
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax3, is_left=True, is_bottom=True)
ax3.set_title(f'c) GIA\nMean: {gia_mean:.1f} mm/yr, RMS: {gia_rms:.1f} mm/yr', 
              fontsize=11, pad=10)

#subplot 4: ODSL result (with GIA correction)
print("Plotting subplot 4...")
im4 = ax4.pcolormesh(odsl_mm_yr.longitude, odsl_mm_yr.latitude,
                     odsl_mm_yr, transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', vmin=-vmax_components, vmax=vmax_components, shading='auto')
add_map_features(ax4, is_left=True, is_bottom=True)
ax4.set_title(f'd) ODSL (MSL - Geoid - GIA)\nMean: {odsl_mean:.1f} mm/yr, RMS: {odsl_rms:.1f} mm/yr', 
              fontsize=11, pad=10)

#single colorbar for all subplots
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.025])
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', pad=0.02)
cbar.set_label('Sea level trend (mm/yr)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

#figure layout
plt.suptitle(f'Trends in ODSL ({common_years.min()}-{common_years.max()})',
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
