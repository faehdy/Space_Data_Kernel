# %%
import netCDF4 as nc
# miscellaneous operating system interfaces
import os

# visualizes the data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

#processes the data
import numpy as np
import pandas as pd

# helps visualize the data
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature
from cartopy.feature import BORDERS, COASTLINE


from matplotlib.colors import TwoSlopeNorm

import geopandas as gpd
from shapely.geometry import Point

from sklearn.decomposition import PCA
from datetime import datetime, timedelta

# %%
# LOAD DATA
file_path = 'Grace/Project/Data/JPL_Mascons.nc' #<- Adjust this path if needed

# Check if the file exists
if os.path.exists(file_path):
    mascons = nc.Dataset(file_path)
else:
    # If not found in relative path, try an absolute path or common location
    # Example: Try looking in a 'Data' subdirectory from the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    alt_file_path = os.path.join(script_dir, 'Data', 'JPL_Mascons.nc')
    if os.path.exists(alt_file_path):
        file_path = alt_file_path
        mascons = nc.Dataset(file_path)
    else:
         raise FileNotFoundError(f"File not found: {file_path} or {alt_file_path}")

# %%
def Convert_2002_2_DMJ(days, unit='days since 2002-01-01 00:00:00', calendar='standard'):
    """
    Convert days since a reference date to datetime objects.

    Parameters:
    -----------
    days : array-like
        Days since the reference date specified in unit.
    unit : str
        Unit of the time, e.g., 'days since 2002-01-01 00:00:00'.
    calendar : str
        Calendar type used in the NetCDF file.

    Returns:
    --------
    date : array-like
        Array of standard Python datetime objects.
    """
    # Use netCDF4.num2date to handle conversion, specifying calendar
    # Set only_use_cftime_datetimes=False to attempt conversion to Python datetimes
    # If it fails (e.g., non-standard calendar not convertible), it will return cftime objects
    cftime_dates = nc.num2date(days, unit, calendar=calendar,
                               only_use_cftime_datetimes=False,
                               only_use_python_datetimes=True) # Force Python datetime

    # Ensure the output is always a list of Python datetime objects
    if isinstance(cftime_dates, np.ndarray):
         # Convert numpy array of objects (potentially mixed cftime/datetime) to list of Python datetimes
         py_dates = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in cftime_dates]
    elif isinstance(cftime_dates, (datetime, pd.Timestamp)): # Handle single date
        py_dates = datetime(cftime_dates.year, cftime_dates.month, cftime_dates.day,
                           cftime_dates.hour, cftime_dates.minute, cftime_dates.second)
    else: # Handle potential single cftime object if conversion failed for some reason
        try:
             py_dates = datetime(cftime_dates.year, cftime_dates.month, cftime_dates.day,
                                cftime_dates.hour, cftime_dates.minute, cftime_dates.second)
        except AttributeError:
             raise TypeError(f"Could not convert num2date output ({type(cftime_dates)}) to Python datetime.")

    return py_dates

# %%

# Extract necessary variables
time_data = mascons.variables['time'][:]  # Time in days since reference date
lwe_thickness = mascons.variables['lwe_thickness'][:]  # Shape (time, lat, lon)
land_mask = mascons.variables['land_mask'][:] # Shape (lat, lon)
lat = mascons.variables['lat'][:]
lon = mascons.variables['lon'][:]

# Apply land mask: set ocean LWE to NaN or a specific value if preferred
# Using NaN is often better for spatial analysis/plotting
lwe_land_only = np.where(land_mask[np.newaxis, :, :] == 1, lwe_thickness, np.nan)
# lwe_land_only = lwe_thickness * land_mask[np.newaxis, :, :] # Alternative: sets ocean to 0

print(f"LWE shape: {lwe_land_only.shape}") # Should be (time, lat, lon)

# Step 1: Convert time to datetime using the Convert_2002_2_DMJ function
time_units = mascons.variables['time'].units
# Determine calendar, default to 'standard' if not present
time_calendar = getattr(mascons.variables['time'], 'calendar', 'standard')
time_dates = Convert_2002_2_DMJ(time_data, unit=time_units, calendar=time_calendar)

# Step 2: Create a pandas DataFrame for easier time handling
time_df_orig = pd.DataFrame({'time': time_dates})
time_df_orig['year_month'] = time_df_orig['time'].dt.to_period('M')  # Convert to 'YYYY-MM' format

# %%
def plot_lwe_thickness_region(center_lat, center_lon, width, height, time_index=0):
    """
    Plots the LWE thickness for a specific region defined by center latitude, center longitude,
    width, and height in degrees, applying a landmask.

    Parameters:
        center_lat (float): Center latitude of the region.
        center_lon (float): Center longitude of the region.
        width (float): Width of the region in degrees.
        height (float): Height of the region in degrees.
        time_index (int): Index of the time step to plot.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
        float: Minimum LWE value in the plotted subset.
        float: Maximum LWE value in the plotted subset.
    """
    global lat, lon, lwe_land_only, time_dates, mascons # Access global variables

    # Adjust center_lon to match the dataset's range (0 to 360) if necessary
    center_lon_adj = center_lon + 360 if center_lon < 0 else center_lon

    # Calculate the bounds of the region
    # Adjusted bounds for Canada example
    lat_min_req = max(center_lat - height / 2, 45) # Lower bound constraint
    lat_max_req = min(center_lat + height / 2, 70) # Upper bound constraint
    lon_min_req = center_lon_adj - width / 2
    lon_max_req = center_lon_adj + width / 2

    # Find dataset indices corresponding to the requested bounds
    lat_indices = np.where((lat >= lat_min_req) & (lat <= lat_max_req))[0]
    # Handle longitude wrapping around 0/360 degrees
    if lon_min_req < 0 or lon_max_req > 360:
         # Case 1: Wraps around 0/360 (e.g., lon_min=350, lon_max=10 needs lon >= 350 OR lon <= 10)
         lon_indices = np.where((lon >= lon_min_req % 360) | (lon <= lon_max_req % 360))[0]
    else:
         # Case 2: Standard range
         lon_indices = np.where((lon >= lon_min_req) & (lon <= lon_max_req))[0]


    # Check if indices are valid
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        print("No data available for the specified region boundaries.")
        return None, None, None

    # Extract the subset of data and coordinates
    subset_lat = lat[lat_indices]
    subset_lon = lon[lon_indices]

    # Ensure subset_lon is monotonic for pcolormesh if it wrapped
    # If lon_indices represent a wrapped selection (e.g., [..., 718, 719, 0, 1, ...]), sort them.
    # A simple check: are the differences mostly positive? If not, it might be wrapped.
    if np.any(np.diff(lon_indices) < 0):
         # This handles the wrap-around by concatenating parts > lon_min and < lon_max
         lon_indices = np.concatenate([
             np.where(lon >= lon_min_req % 360)[0],
             np.where(lon <= lon_max_req % 360)[0]
         ])
         lon_indices = np.unique(lon_indices) # Keep unique indices
         subset_lon = lon[lon_indices]

    # Extract LWE data for the specific time index and spatial subset
    # Use np.ix_ for N-dimensional indexing
    subset_lwe_thickness = lwe_land_only[time_index, np.ix_(lat_indices, lon_indices)]

    # Check if subset_lwe_thickness is valid
    if subset_lwe_thickness.size == 0:
        print("No LWE thickness data available for the specified region subset.")
        return None, None, None

    # Calculate data range, ignoring NaNs
    data_min = np.nanmin(subset_lwe_thickness)
    data_max = np.nanmax(subset_lwe_thickness)

    if np.isnan(data_min) or np.isnan(data_max):
        print("LWE data in the subset is all NaN.")
        # Optionally plot with a default range or return
        # return None, None, None # Or handle as needed
        data_min, data_max = -1, 1 # Example default range

    # Create a figure and axis with a Cartopy projection
    fig = plt.figure(figsize=(12, 8)) # Adjusted size for better map view
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # Use PlateCarree if data is lat/lon grid

    # Define a normalization that centers the colorbar at 0
    # Handle cases where all data is positive or negative
    if data_min < 0 and data_max > 0:
         # Make symmetric around 0
         limit = max(abs(data_min), abs(data_max))
         norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
         cmap = 'seismic'
    elif data_max > 0: # All positive
         norm = plt.Normalize(vmin=0, vmax=data_max)
         cmap = 'viridis' # Or another sequential map
    else: # All negative or zero
         norm = plt.Normalize(vmin=data_min, vmax=0)
         cmap = 'plasma_r' # Or another reversed sequential map


    # Plot the lwe_thickness using pcolormesh
    # Need meshgrid for pcolormesh coordinates
    lon_mesh, lat_mesh = np.meshgrid(subset_lon, subset_lat)
    lwe_plot = ax.pcolormesh(lon_mesh, lat_mesh, subset_lwe_thickness,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0) # Add land background
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0) # Add ocean background

    # Set extent to match the requested region more closely
    ax.set_extent([lon_min_req, lon_max_req, lat_min_req, lat_max_req], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


    # Add a colorbar
    cbar = plt.colorbar(lwe_plot, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label('Liquid Water Equivalent Thickness Anomaly (cm)') # Assuming units are cm

    # Set the title
    plot_date = time_dates[time_index]
    ax.set_title(f'LWE Thickness Anomaly for {plot_date.strftime("%Y-%m-%d")} in Region')

    return fig, data_min, data_max



# %%
# Plotting the LWE thickness with landmask for Canada
# Set the right rectangle parameters for Canada
center_lat = 62.5 # Adjusted center
center_lon = -95  # Adjusted center
width = 70       # Width in degrees longitude
height = 35      # Height in degrees latitude
time_index_to_plot = 3 # Example time index

fig, plotted_min, plotted_max = plot_lwe_thickness_region(center_lat, center_lon, width, height, time_index_to_plot)

if fig:
    print(f"Data range for plot: min={plotted_min:.2f}, max={plotted_max:.2f}")
    # Show the plot
    plt.show()
else:
    print("Plotting failed.")

# %% [markdown]
# #### Starting with the EOF of Canada
#
# - **To do:** GAP handling (Implemented below)

# %%
# Define the region of interest (Canada)
center_lat = 62.5
center_lon = -95
width = 70
height = 35

# Adjust center_lon to match the dataset's range (0 to 360) if necessary
center_lon_adj = center_lon + 360 if center_lon < 0 else center_lon

# Calculate the bounds of the region
lat_min_req = max(center_lat - height / 2, 45)  # Ensure latitude is not below 45°N
lat_max_req = min(center_lat + height / 2, 70)  # Ensure latitude is not above 70°N
lon_min_req = center_lon_adj - width / 2
lon_max_req = center_lon_adj + width / 2


# Filter the latitude and longitude indices from the full dataset grid
lat_indices = np.where((lat >= lat_min_req) & (lat <= lat_max_req))[0]

# Handle longitude wrapping
if lon_min_req < 0 or lon_max_req > 360:
     lon_indices = np.where((lon >= lon_min_req % 360) | (lon <= lon_max_req % 360))[0]
else:
     lon_indices = np.where((lon >= lon_min_req) & (lon <= lon_max_req))[0]


# Check if indices are valid
if len(lat_indices) == 0 or len(lon_indices) == 0:
    raise ValueError("No grid cells found for the specified region.")

# Extract the subset of data using n-dimensional slicing
# np.ix_ creates index arrays suitable for multi-dimensional indexing
subset_indices = np.ix_(range(lwe_land_only.shape[0]), lat_indices, lon_indices)
subset_data = lwe_land_only[subset_indices]  # Shape: (time, lat_subset, lon_subset)

# Also extract corresponding coordinates for plotting/analysis
subset_lat = lat[lat_indices]
subset_lon = lon[lon_indices] # Use original lon values for extent

# Flatten spatial dimensions for PCA: (time, space)
num_time_steps = subset_data.shape[0]
num_lat = len(lat_indices)
num_lon = len(lon_indices)
reshape_data = subset_data.reshape(num_time_steps, num_lat * num_lon)

# --- Handling NaNs before PCA ---
# Option 1: Mask NaNs (if PCA implementation supports it) - sklearn PCA doesn't directly.
# Option 2: Fill NaNs (e.g., with mean of the spatial point over time, or spatial mean at that time)
# Option 3: Remove spatial points (columns) that contain ANY NaNs (simplest, but loses data)

# Let's use Option 3 for simplicity here, but be aware of data loss.
# Find columns (spatial points) that are not all NaN
valid_columns = ~np.all(np.isnan(reshape_data), axis=0)
reshape_data_nonan = reshape_data[:, valid_columns]

# Check if any data remains
if reshape_data_nonan.shape[1] == 0:
     raise ValueError("All spatial points in the selected region contain NaNs after masking/flattening.")

# --- Proceed with PCA on non-NaN data ---
# Remove time mean to compute anomalies (centering the data)
# Important: Calculate mean *after* handling NaNs
column_time_mean = np.nanmean(reshape_data_nonan, axis=0, keepdims=True)
data_centered = reshape_data_nonan - column_time_mean

# Perform PCA
n_components_pca = 10 # Calculate more components initially
pca = PCA(n_components=n_components_pca)
# Fit PCA on the data with valid columns and removed mean
pca.fit(data_centered)

# Principal Components (time series)
transformed_data = pca.transform(data_centered) # Shape: (time, n_components_pca)

# EOFs (spatial patterns)
# These correspond to the 'valid_columns' space
eofs_valid = pca.components_ # Shape: (n_components_pca, n_valid_columns)

explained_variance = pca.explained_variance_ratio_

# --- Map EOFs back to the original spatial grid ---
# Create a NaN-filled array with the original flattened spatial shape
eofs_full_space = np.full((n_components_pca, num_lat * num_lon), np.nan)
# Fill in the EOF values for the valid columns
eofs_full_space[:, valid_columns] = eofs_valid
# Reshape back to (n_components, lat, lon)
eofs_spatial = eofs_full_space.reshape(n_components_pca, num_lat, num_lon)


# --- Visualize the EOFs and their time series ---
n_plot = 5 # Number of EOFs/PCs to plot
fig, axes = plt.subplots(n_plot, 2, figsize=(15, 4 * n_plot))

lon_mesh_sub, lat_mesh_sub = np.meshgrid(subset_lon, subset_lat)

for i in range(n_plot):
    # --- EOF spatial pattern ---
    eof_pattern = eofs_spatial[i] # Already in (lat, lon) shape

    # Determine symmetric color limits around 0
    eof_abs_max = np.nanmax(np.abs(eof_pattern))
    eof_norm = TwoSlopeNorm(vmin=-eof_abs_max, vcenter=0, vmax=eof_abs_max)

    # Use pcolormesh on the axes subplot
    ax_eof = fig.add_subplot(n_plot, 2, 2*i + 1) # Create subplot for map
    # Note: Using imshow might be simpler if grid is regular, but pcolormesh is safer
    im = ax_eof.pcolormesh(lon_mesh_sub, lat_mesh_sub, eof_pattern, cmap='RdBu_r', norm=eof_norm, shading='auto')

    ax_eof.set_title(f'EOF {i+1} ({explained_variance[i]*100:.2f}% variance)')
    fig.colorbar(im, ax=ax_eof, label='EOF Loading')
    # Add basic map features if needed (can slow down plotting)
    # ax_eof.add_feature(cfeature.COASTLINE)
    # ax_eof.add_feature(cfeature.BORDERS, linestyle=':')

    # --- Principal component time series ---
    ax_pc = fig.add_subplot(n_plot, 2, 2*i + 2) # Create subplot for time series
    ax_pc.plot(time_dates, transformed_data[:, i], label=f'PC {i+1}', color='b', marker='.', linestyle='-')
    ax_pc.set_title(f'PC {i+1} Time Series')
    ax_pc.legend()
    ax_pc.grid(True)

    # Formatting the x-axis (time)
    ax_pc.xaxis.set_major_locator(mdates.YearLocator(2))  # Major ticks every 2 years
    ax_pc.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7])) # Minor ticks Jan, Jul
    ax_pc.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Show year

    # Rotate the tick labels for better readability
    plt.setp(ax_pc.get_xticklabels(), rotation=45, ha="right")


plt.suptitle("EOF Analysis of LWE Anomaly in Canada Region", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout
plt.show()

# %% [markdown]
# ### Handling GAPs - Minding the GAP & Interpolation

# %%
# --- Identify Gaps in the Original Time Series ---
time_df = pd.DataFrame({'time': time_dates})
time_df['month_diff'] = time_df['time'].diff().dt.days

# Identify gaps (e.g., months with more than ~45 days difference indicates > 1 month missing)
gap_threshold_days = 45
gaps = time_df[time_df['month_diff'] > gap_threshold_days].copy()
print(f"Found {len(gaps)} gaps in the data based on >{gap_threshold_days} day difference:")
gap_indices = [] # Store indices *before* each gap
for idx, row in gaps.iterrows():
    prev_date = time_df.iloc[idx-1]['time']
    curr_date = row['time']
    # Calculate months spanned by the gap (inclusive of start/end month representation)
    gap_months = (curr_date.year - prev_date.year) * 12 + (curr_date.month - prev_date.month)
    print(f"- Gap between {prev_date.strftime('%Y-%m')} and {curr_date.strftime('%Y-%m')} (approx. {gap_months -1 } missing months)")
    gap_indices.append(idx - 1) # Index of the data point *before* the gap starts

# --- Determine Number of Components for Reconstruction (>90% Variance) ---
cumulative_variance = np.cumsum(explained_variance)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\nNumber of components needed for >90% variance: {n_components_90}")
print(f"Variance explained by first {n_components_90} components: {cumulative_variance[n_components_90-1]*100:.2f}%")

# Select the PCs and EOFs needed for reconstruction
pcs_selected = transformed_data[:, :n_components_90]
eofs_selected = eofs_valid[:n_components_90, :] # Use the EOFs corresponding to valid columns

# --- Create Full Monthly Time Index ---
start_date = time_dates[0]
end_date = time_dates[-1]
# Create a monthly date range, ensuring start of the month for consistency
full_time_index = pd.date_range(start=start_date.replace(day=1),
                                end=end_date.replace(day=1), freq='MS') # 'MS' for Month Start frequency

print(f"\nOriginal time steps: {len(time_dates)}")
print(f"Full monthly time steps: {len(full_time_index)}")
print(f"Number of missing months to interpolate: {len(full_time_index) - len(time_dates)}")


# --- Interpolate Principal Components (PCs) ---
# Create a pandas DataFrame for the selected PCs with the original time index
pc_df = pd.DataFrame(pcs_selected, index=pd.to_datetime(time_dates))

# Reindex the PC DataFrame to the full monthly index. Missing times will be filled with NaN.
pc_df_reindexed = pc_df.reindex(full_time_index)

# Interpolate the NaNs using a time-based linear method
pc_df_interpolated = pc_df_reindexed.interpolate(method='time')

# Check if any NaNs remain (e.g., at the very beginning/end if outside original range)
if pc_df_interpolated.isnull().values.any():
    print("Warning: NaNs remain after interpolation. Applying forward/backward fill.")
    pc_df_interpolated = pc_df_interpolated.ffill().bfill() # Fill remaining NaNs

interpolated_pcs = pc_df_interpolated.values # Shape: (full_time, n_components_90)

# --- Reconstruct Data using Interpolated PCs ---
# Reconstruct the *anomaly* data (centered data) for the full time series
# reconstructed_anomalies_valid = interpolated_pcs @ eofs_selected
# Reconstructed shape: (full_time, n_valid_columns)
reconstructed_anomalies_valid = np.dot(interpolated_pcs, eofs_selected)


# --- Add Mean Back and Reshape to Spatial Grid ---
# Add the time mean (calculated earlier only for valid columns) back
reconstructed_data_valid = reconstructed_anomalies_valid + column_time_mean[:, valid_columns] # Ensure mean aligns correctly

# Create a NaN-filled array for the full spatial grid over the full time series
reconstructed_data_full_space = np.full((len(full_time_index), num_lat * num_lon), np.nan)

# Place the reconstructed data for valid columns into the full spatial array
reconstructed_data_full_space[:, valid_columns] = reconstructed_data_valid

# Reshape back to (time, lat, lon)
reconstructed_data_final = reconstructed_data_full_space.reshape(len(full_time_index), num_lat, num_lon)

print(f"\nShape of original subset data: {subset_data.shape}")
print(f"Shape of reconstructed data with interpolated gaps: {reconstructed_data_final.shape}")

# --- Verification/Plotting Example (Optional) ---
# Plot the time series of a specific grid point before and after interpolation

# Choose a sample grid point (adjust indices as needed)
sample_lat_idx = num_lat // 2
sample_lon_idx = num_lon // 2

# Find the corresponding column index in the flattened nonan array
# Need to map the (lat, lon) index back to the flattened index, then check if it was 'valid'
flat_idx = np.ravel_multi_index((sample_lat_idx, sample_lon_idx), (num_lat, num_lon))

# Check if this flat index was kept (is in valid_columns)
is_valid = valid_columns[flat_idx]

if is_valid:
    # Find the index within the 'valid_columns' space
    valid_col_idx = np.where(np.where(valid_columns)[0] == flat_idx)[0][0]

    original_ts = reshape_data_nonan[:, valid_col_idx] # Original data for this point (NaNs maybe present if skipped by PCA handling)
    reconstructed_ts = reconstructed_data_valid[:, valid_col_idx] # Reconstructed data for this point (valid columns)

    plt.figure(figsize=(15, 6))
    plt.plot(time_dates, original_ts, 'o-', label='Original Data (at valid points)', markersize=4, alpha=0.7)
    plt.plot(full_time_index, reconstructed_ts, '.-', label=f'Reconstructed Data (>90% Var, {n_components_90} modes)', markersize=3, alpha=0.8, color='red')

    # Highlight gap periods for visualization
    for start_idx in gap_indices:
        gap_start_date = time_dates[start_idx]
        gap_end_date = time_dates[start_idx + 1]
        plt.axvspan(gap_start_date, gap_end_date, color='gray', alpha=0.2, label='Gap Period' if start_idx == gap_indices[0] else "") # Label only once


    plt.title(f'Original vs Reconstructed LWE at approx. Lat={subset_lat[sample_lat_idx]:.2f}, Lon={subset_lon[sample_lon_idx]:.2f}')
    plt.xlabel('Time')
    plt.ylabel('LWE Anomaly (cm)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"Sample point Lat={subset_lat[sample_lat_idx]:.2f}, Lon={subset_lon[sample_lon_idx]:.2f} (flat index {flat_idx}) was removed due to NaNs. Cannot plot comparison.")


# The final interpolated data is in 'reconstructed_data_final'
# The corresponding time axis is 'full_time_index'
# The corresponding spatial coordinates are 'subset_lat' and 'subset_lon'
# %%
