import pandas as pd
import xarray as xr
from datetime import datetime

# --- User settings ---
# Define the path to your source pickle file
pickle_file_path = r"C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\Budget\Carolina\OM_reconstructions_1993-2016.p"

# Define where you want to save the new NetCDF file
netcdf_output_path = r"C:\Users\luciu\OneDrive - Universiteit Utrecht\Overige documenten\Thesis_KNMI\Data\Budget\Carolina\OM_reconstructions_1993-2016.nc"
# --- End of user settings ---

print(f"--- Converting Pickle to NetCDF ---")
print(f"Input: {pickle_file_path}")

try:
    # 1. Load the DataFrame from the pickle file
    df = pd.read_pickle(pickle_file_path)
    print("Successfully loaded the pickle file into a pandas DataFrame.")

    # 2. Set 'lat' and 'lon' as the DataFrame index.
    # This is a key step that prepares the data for reshaping into a grid.
    df.set_index(['lat', 'lon'], inplace=True)
    print("Set 'lat' and 'lon' as a multi-index for gridding.")

    # 3. Convert the multi-indexed DataFrame to an xarray Dataset.
    # xarray automatically detects the multi-index and creates the 2D grid structure.
    ds = xr.Dataset.from_dataframe(df)
    print("Converted DataFrame to an xarray Dataset.")
    
    # 4. Add metadata (attributes) to make the NetCDF file self-describing.
    # This is crucial for making the data understandable and usable in the future.

    # -- Global attributes --
    ds.attrs['title'] = 'Regional Barystatic Sea-Level Change Reconstructions (1993-2016)'
    ds.attrs['description'] = (
        "Contains regional barystatic (ocean mass) sea-level change trends and uncertainties. "
        "Barystatic sea-level change is the sum of contributions from Antarctica (AIS), "
        "Greenland (GIS), Glaciers (GLA), and Land Water Storage (LWS)."
    )
    ds.attrs['source'] = 'Original data from Carolina Camargo (carolina.camargo@nioz.nl)'
    ds.attrs['history'] = f"Created on {datetime.utcnow().isoformat()}Z by converting the pickle file to NetCDF."
    
    # -- Variable attributes (units, long names) --
    # The README doesn't specify units, but mm/yr is the standard for sea level trends.
    trend_units = 'mm/yr'
    
    for var_name in ds.data_vars:
        if 'trend' in var_name:
            ds[var_name].attrs['units'] = trend_units
            ds[var_name].attrs['long_name'] = f"{var_name.replace('_', ' ')}"
            ds[var_name].attrs['description'] = f"Barystatic sea level trend for the {var_name.replace('_trend_tot', '')} reconstruction."
        elif 'unc' in var_name:
            ds[var_name].attrs['units'] = trend_units
            ds[var_name].attrs['long_name'] = f"{var_name.replace('_', ' ')}"
            ds[var_name].attrs['description'] = f"Uncertainty in the barystatic sea level trend for the {var_name.split('_unc')[0]} combination."

    # -- Coordinate attributes --
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lat'].attrs['long_name'] = 'Latitude'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lon'].attrs['long_name'] = 'Longitude'

    print("Added metadata (units, descriptions) to the Dataset.")

    # 5. Save the xarray Dataset to a NetCDF file.
    ds.to_netcdf(netcdf_output_path)
    
    print("\n--- Success! ---")
    print(f"NetCDF file saved successfully to:\n{netcdf_output_path}")

    # Optional: Print the dataset structure to see the result
    print("\n--- Final NetCDF Structure ---")
    print(ds)


except FileNotFoundError:
    print(f"Error: The file was not found at the specified path:\n{pickle_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")