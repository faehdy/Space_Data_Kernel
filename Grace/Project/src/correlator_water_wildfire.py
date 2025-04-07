import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Helper: Ensure Water Data is GeoDataFrame ---
def prepare_water_gdf(water_df, lon_col='lon', lat_col='lat', crs="EPSG:4326"):
    """Converts water DataFrame to GeoDataFrame."""
    if not isinstance(water_df, gpd.GeoDataFrame):
        geometry = [Point(xy) for xy in zip(water_df[lon_col], water_df[lat_col])]
        water_gdf = gpd.GeoDataFrame(water_df, geometry=geometry, crs=crs)
    else:
        water_gdf = water_df # Assume it's already a GeoDataFrame if passed as one
        if water_gdf.crs is None:
            print(f"Warning: Setting water_gdf CRS to {crs}")
            water_gdf.set_crs(crs, inplace=True)
        elif water_gdf.crs != crs:
            print(f"Warning: Reprojecting water_gdf CRS to {crs}")
            water_gdf = water_gdf.to_crs(crs) # Ensure consistent CRS
    return water_gdf

# --- Module 1: Find Closest Water Storage Point ---
def find_closest_water_location(target_lon, target_lat, water_gdf):
    """
    Finds the coordinates of the nearest point in water_gdf to a target location.

    Args:
        target_lon (float): Longitude of the target fire grid centroid.
        target_lat (float): Latitude of the target fire grid centroid.
        water_gdf (GeoDataFrame): GeoDataFrame of water storage points.

    Returns:
        tuple: (longitude, latitude) of the closest water storage point, or None if error.
    """
    try:
        target_point = Point(target_lon, target_lat)
        # Calculate distances - ensure consistent CRS first (handled in prepare_water_gdf)
        # Note: For lon/lat, distance calculation gives degrees, but argmin finds the correct point.
        # For more accurate *distance values*, project both to a suitable projected CRS first.
        distances = water_gdf.geometry.distance(target_point)
        closest_idx = distances.idxmin()
        closest_point_geom = water_gdf.loc[closest_idx, 'geometry']
        return (closest_point_geom.x, closest_point_geom.y)
    except Exception as e:
        print(f"Error finding closest water location: {e}")
        return None

# --- Module 2: Prepare and Align Monthly Time Series ---
def prepare_aligned_timeseries(grid_gdf, water_df, target_fire_lon, target_fire_lat, closest_water_lon, closest_water_lat):
    """
    Prepares and merges fire and water time series for a specific location, aligned monthly.

    Args:
        grid_gdf (GeoDataFrame): Aggregated monthly fire data.
        water_df (DataFrame): Raw water storage data (will be converted to GDF).
        target_fire_lon (float): Longitude of the target fire centroid.
        target_fire_lat (float): Latitude of the target fire centroid.
        closest_water_lon (float): Longitude of the corresponding water point.
        closest_water_lat (float): Latitude of the corresponding water point.

    Returns:
        pd.DataFrame: DataFrame with aligned monthly 'cumuarea' and 'waterstorage',
                      or None if data is insufficient.
    """
    try:
        # --- Filter and Prepare Fire Data ---
        fire_ts = grid_gdf[
            (np.isclose(grid_gdf.geometry.x, target_fire_lon)) &
            (np.isclose(grid_gdf.geometry.y, target_fire_lat))
        ].copy()

        if fire_ts.empty:
            print(f"No fire data found for location ({target_fire_lon}, {target_fire_lat})")
            return None

        # Create a proper monthly datetime index for fire data
        # Assuming 'year' and 'month' columns exist
        fire_ts['time'] = pd.to_datetime(fire_ts[['year', 'month']].assign(day=1))
        fire_ts = fire_ts.set_index('time')[['cumuarea']].sort_index()

        # --- Filter and Prepare Water Data ---
        # Filter for the specific closest water point
        water_point_ts = water_df[
            (np.isclose(water_df['lon'], closest_water_lon)) &
            (np.isclose(water_df['lat'], closest_water_lat))
        ].copy()

        if water_point_ts.empty:
            print(f"No water data found for location ({closest_water_lon}, {closest_water_lat})")
            return None

        water_point_ts['time'] = pd.to_datetime(water_point_ts['time'])
        water_point_ts = water_point_ts.set_index('time')[['waterstorage']].sort_index()

        # --- Resample Water Data to Monthly ---
        # Resample to monthly frequency, taking the mean value within each month.
        # 'MS' ensures the timestamp is the start of the month, matching fire data.
        water_ts_monthly = water_point_ts.resample('MS').mean()

        # --- Merge Data ---
        # Merge based on the monthly index. Use an inner join to keep only matching months.
        aligned_df = pd.merge(fire_ts, water_ts_monthly, left_index=True, right_index=True, how='inner')

        # Drop rows where either value might be NaN (e.g., if resample had no data)
        aligned_df.dropna(subset=['cumuarea', 'waterstorage'], inplace=True)

        if len(aligned_df) < 3: # Need at least 3 points for meaningful correlation
             print(f"Insufficient overlapping data points ({len(aligned_df)}) for correlation analysis.")
             return None

        return aligned_df

    except Exception as e:
        print(f"Error preparing aligned timeseries: {e}")
        return None


# --- Module 3: Calculate Standard Correlation ---
def calculate_correlation(aligned_df):
    """
    Calculates the Pearson correlation coefficient and p-value.

    Args:
        aligned_df (pd.DataFrame): DataFrame with 'cumuarea' and 'waterstorage' columns.

    Returns:
        tuple: (correlation_coefficient, p_value), or (None, None) if error.
    """
    if aligned_df is None or not all(col in aligned_df.columns for col in ['cumuarea', 'waterstorage']):
        print("Invalid DataFrame passed to calculate_correlation.")
        return None, None
    if aligned_df['cumuarea'].isnull().all() or aligned_df['waterstorage'].isnull().all():
        print("One or both series contain only NaNs.")
        return None, None
    if len(aligned_df) < 3:
        print("Not enough data points for correlation.")
        return None, None

    try:
        # Ensure data is numeric and finite
        area = pd.to_numeric(aligned_df['cumuarea'], errors='coerce')
        water = pd.to_numeric(aligned_df['waterstorage'], errors='coerce')
        valid_indices = area.notna() & water.notna() & np.isfinite(area) & np.isfinite(water)

        if valid_indices.sum() < 3:
            print("Not enough valid (non-NaN, finite) data points for correlation.")
            return None, None

        corr, p_value = pearsonr(area[valid_indices], water[valid_indices])
        return corr, p_value
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return None, None

# --- Module 4: Calculate Cross-Correlation ---
def calculate_cross_correlation(aligned_df, max_lag=12):
    """
    Calculates the cross-correlation function (CCF) between two time series.

    Args:
        aligned_df (pd.DataFrame): DataFrame with 'cumuarea' and 'waterstorage'.
        max_lag (int): Maximum lag (in months) to consider.

    Returns:
        tuple: (lags, ccf_values), or (None, None) if error.
            - lags: Array of lags from -max_lag to +max_lag.
            - ccf_values: Corresponding cross-correlation coefficients.
            Positive lag means water storage leads fire area.
            Negative lag means fire area leads water storage.
    """
    if aligned_df is None or not all(col in aligned_df.columns for col in ['cumuarea', 'waterstorage']):
         print("Invalid DataFrame passed to calculate_cross_correlation.")
         return None, None
    if len(aligned_df) <= max_lag: # Need enough points to calculate lags
         print(f"Not enough data points ({len(aligned_df)}) for max_lag={max_lag}.")
         return None, None

    try:
        # Ensure data is numeric and finite, handle potential NaNs
        area = pd.to_numeric(aligned_df['cumuarea'], errors='coerce').dropna()
        water = pd.to_numeric(aligned_df['waterstorage'], errors='coerce').dropna()

        # Re-align after dropping NaNs individually if necessary
        common_index = area.index.intersection(water.index)
        if len(common_index) <= max_lag:
            print(f"Not enough overlapping non-NaN points ({len(common_index)}) for max_lag={max_lag}.")
            return None, None

        area = area.loc[common_index]
        water = water.loc[common_index]

        # Statsmodels ccf requires subtraction of mean for interpretability
        ccf_vals = ccf(area - area.mean(), water - water.mean(), adjusted=False) # Calculate full ccf

        # Extract values around lag 0
        # ccf output: index 0 is lag 0, index 1 is lag 1 (x leads y), etc.
        # We need negative lags too. Statsmodels doesn't directly return neg lags easily.
        # Let's calculate manually for symmetrical view or use numpy.correlate as fallback
        # Using numpy.correlate for easier symmetrical lag handling:
        # Note: np.correlate(a, v) where a is fire, v is water
        #       Positive lag index means 'v' (water) is shifted left (leads fire)
        area_norm = (area - area.mean()) / (area.std() * len(area)) # Normalize for correlation scale
        water_norm = (water - water.mean()) / water.std()
        corr_np = np.correlate(area_norm, water_norm, mode='full') # mode='full' gives -N+1 to N-1 lags

        # Lags corresponding to np.correlate output: from -(N-1) to +(N-1)
        n = len(area)
        lags_np = np.arange(-(n - 1), n)

        # Limit to max_lag
        center_idx = np.where(lags_np == 0)[0][0]
        indices_to_keep = (lags_np >= -max_lag) & (lags_np <= max_lag)
        limited_lags = lags_np[indices_to_keep]
        limited_ccf_vals = corr_np[indices_to_keep]

        # return limited_lags, limited_ccf_vals # Use this if using numpy approach
        # Using statsmodels ccf and manually getting negative lags:
        ccf_pos_lags = ccf(area - area.mean(), water - water.mean(), adjusted=False)[ : max_lag + 1] # Lag 0 to max_lag
        ccf_neg_lags_rev = ccf(water - water.mean(), area - area.mean(), adjusted=False)[1 : max_lag + 1] # Lag 1 to max_lag (water vs PAST fire) -> reverse for (fire vs PAST water)
        
        ccf_symm = np.concatenate((ccf_neg_lags_rev[::-1], ccf_pos_lags))
        lags_symm = np.arange(-max_lag, max_lag + 1)
        
        return lags_symm, ccf_symm # Using statsmodels symmetric approach


    except Exception as e:
        print(f"Error calculating cross-correlation: {e}")
        return None, None


# --- Module 5: Plot Aligned Time Series ---
def plot_aligned_timeseries(aligned_df, target_fire_lon, target_fire_lat):
    """
    Plots the aligned cumuarea and waterstorage time series with dual y-axes.

    Args:
        aligned_df (pd.DataFrame): DataFrame with 'cumuarea' and 'waterstorage'.
        target_fire_lon (float): Longitude for plot title.
        target_fire_lat (float): Latitude for plot title.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot, or None.
    """
    if aligned_df is None or aligned_df.empty:
        print("Cannot plot empty or None DataFrame.")
        return None

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = 'tab:red'
    ax1.set_xlabel('Time (Monthly)')
    ax1.set_ylabel('Cumulative Area (cumuarea)', color=color1)
    ax1.plot(aligned_df.index, aligned_df['cumuarea'], color=color1, marker='o', linestyle='-', label='Cumu Area')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_major_locator(mdates.YearLocator()) # Tick per year
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format as YYYY-MM

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Water Storage', color=color2)
    ax2.plot(aligned_df.index, aligned_df['waterstorage'], color=color2, marker='x', linestyle='--', label='Water Storage')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add legends
    # Getting handles and labels from both axes for a single legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')


    plt.title(f'Monthly Fire Area vs Water Storage\nLocation: ({target_fire_lon:.3f}, {target_fire_lat:.3f})')
    fig.tight_layout() # Adjust layout to prevent overlapping labels
    plt.grid(True, which='major', axis='x', linestyle='--')
    fig.autofmt_xdate() # Rotate date labels for better readability
    # plt.show() # Don't show here, return the fig for potential further use/saving
    return fig

# --- Module 6: Plot Cross-Correlation ---
def plot_cross_correlation(lags, ccf_values, target_fire_lon, target_fire_lat):
    """
    Plots the cross-correlation function.

    Args:
        lags (np.array): Array of time lags.
        ccf_values (np.array): Array of cross-correlation coefficients.
        target_fire_lon (float): Longitude for plot title.
        target_fire_lat (float): Latitude for plot title.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot, or None.
    """
    if lags is None or ccf_values is None:
        print("Cannot plot None lags or ccf_values.")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stem(lags, ccf_values, basefmt=" ", use_line_collection=True) # Use stem plot for CCF
    ax.set_xlabel("Lag (Months)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f'Cross-Correlation: Fire Area vs Water Storage\nLocation: ({target_fire_lon:.3f}, {target_fire_lat:.3f})\n(Positive lag means Water Storage leads Fire)')
    ax.axhline(0, color='grey', lw=0.5) # Line at 0 correlation
    # Optional: Add confidence intervals (requires more stats)
    # n = len(aligned_df_used_for_ccf) # Need the N used
    # conf_level = 1.96 / np.sqrt(n)
    # ax.axhline(conf_level, color='red', linestyle='--', lw=0.8, label='95% Conf. Interval')
    # ax.axhline(-conf_level, color='red', linestyle='--', lw=0.8)
    # ax.legend()
    plt.grid(True, linestyle='--')
    # plt.show() # Don't show here
    return fig


# --- Orchestration Function (Example Usage) ---
def analyze_location_correlation(grid_gdf, water_df_raw, target_lon, target_lat):
    """
    Runs the full analysis workflow for a single target location.

    Args:
        grid_gdf (GeoDataFrame): Monthly aggregated fire data.
        water_df_raw (DataFrame): Raw water storage data.
        target_lon (float): Target fire grid centroid longitude.
        target_lat (float): Target fire grid centroid latitude.

    Returns:
        dict: Dictionary containing results (correlation, p-value, lags, ccf, plots).
    """
    print(f"\n--- Analyzing Location: ({target_lon}, {target_lat}) ---")
    results = {
        'target_lon': target_lon,
        'target_lat': target_lat,
        'closest_water_lon': None,
        'closest_water_lat': None,
        'correlation': None,
        'p_value': None,
        'lags': None,
        'ccf_values': None,
        'timeseries_plot': None,
        'ccf_plot': None,
        'aligned_data': None
    }

    # 1. Prepare water data (ensure GeoDataFrame)
    water_gdf = prepare_water_gdf(water_df_raw)
    print(f"Water data prepared with {len(water_gdf)} points.")
    if water_gdf is None: return results # Stop if error

    # 2. Find closest water location
    closest_coords = find_closest_water_location(target_lon, target_lat, water_gdf)
    print(f"Closest water location found at: {closest_coords}")
    if closest_coords is None: return results # Stop if error
    results['closest_water_lon'], results['closest_water_lat'] = closest_coords
    print(f"Closest water point found at: ({closest_coords[0]:.3f}, {closest_coords[1]:.3f})")

    # 3. Prepare aligned time series
    print("Preparing aligned time series...")
    aligned_df = prepare_aligned_timeseries(grid_gdf, water_df_raw, target_lon, target_lat, closest_coords[0], closest_coords[1])
    results['aligned_data'] = aligned_df # Store the aligned data
    if aligned_df is None: return results # Stop if data insufficient

    print(f"Found {len(aligned_df)} overlapping monthly data points.")

    # 4. Calculate standard correlation
    print("Calculating Pearson correlation...")
    corr, p_val = calculate_correlation(aligned_df)
    results['correlation'] = corr
    results['p_value'] = p_val
    if corr is not None:
        print(f"Pearson Correlation: {corr:.4f} (p-value: {p_val:.4f})")
    else:
        print("Could not calculate Pearson Correlation.")


    # 5. Calculate cross-correlation
    print("Calculating cross-correlation...")
    lags, ccf_vals = calculate_cross_correlation(aligned_df, max_lag=12) # Check up to 1 year lag
    results['lags'] = lags
    results['ccf_values'] = ccf_vals
    if lags is not None:
        max_corr_idx = np.argmax(np.abs(ccf_vals))
        print(f"Cross-correlation: Max abs correlation of {ccf_vals[max_corr_idx]:.4f} at lag {lags[max_corr_idx]} months.")
    else:
         print("Could not calculate Cross-Correlation.")

    # 6. Generate plots
    print("Generating plots...")
    results['timeseries_plot'] = plot_aligned_timeseries(aligned_df, target_lon, target_lat)
    results['ccf_plot'] = plot_cross_correlation(lags, ccf_vals, target_lon, target_lat)

    print(f"--- Analysis Complete for Location: ({target_lon}, {target_lat}) ---")
    return results

# --- Example Usage ---

wildfire_grid_gdf = gpd.read_file('Grace/Project/Data/aggregated_wildfire_grid.csv')

water_df_raw = gpd.read_file('Grace/Project/Data/data_GLDAS/gdf_compiled_canada_soil_moisture.csv')

# Example target location (longitude, latitude)
target_lon_example = -116.125
target_lat_example = 56.875



# Run the analysis
analysis_results = analyze_location_correlation(wildfire_grid_gdf, water_df_raw, target_lon_example, target_lat_example)

# Access results:
print(f"Correlation: {analysis_results['correlation']}")
if analysis_results['timeseries_plot']:
    analysis_results['timeseries_plot'].show()
if analysis_results['ccf_plot']:
    analysis_results['ccf_plot'].show()
print(analysis_results['aligned_data']) # Show the final data used

# To show plots if they were generated:
plt.show()