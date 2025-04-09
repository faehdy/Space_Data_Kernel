import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt # Comment out if plots aren't generated/saved
import matplotlib.dates as mdates # Comment out if plots aren't generated/saved
import os
import pickle
import gc
from shapely import wkt # Needed for the robust prepare_water_gdf

# --- Constants ---
COORD_PRECISION = 5
CACHE_DIR = 'cache'
# Cache filenames
LOCATION_MAP_CACHE = os.path.join(CACHE_DIR, 'location_map_exact.pkl') # New name for exact match
UNIQUE_WATER_COORDS_CACHE = os.path.join(CACHE_DIR, 'unique_water_coords.pkl')
WATER_DATA_CACHE = os.path.join(CACHE_DIR, 'monthly_water_data.pkl')
NUM_PLOTS_TO_GENERATE = 100 # Number of plots to generate
PLOT_OUTPUT_DIR = 'Grace/Project/output/corr_w_w_plots_5deg' # Directory for plots


# --- Helper Functions ---

# Keep your robust prepare_water_gdf function here
def prepare_water_gdf(water_df, lon_col='lon', lat_col='lat', geom_col='geometry', crs="EPSG:4326"):
    # ... (Use the robust version from the previous answer) ...
    if isinstance(water_df, gpd.GeoDataFrame):
        water_gdf = water_df
        if water_gdf.crs is None: water_gdf = water_gdf.set_crs(crs, allow_override=True)
        elif str(water_gdf.crs).upper() != str(crs).upper(): water_gdf = water_gdf.to_crs(crs)
        return water_gdf
    elif geom_col in water_df.columns:
        try:
            if pd.api.types.is_string_dtype(water_df[geom_col]): geometry_objects = water_df[geom_col].apply(wkt.loads)
            else: geometry_objects = water_df[geom_col]
            water_gdf = gpd.GeoDataFrame(water_df, geometry=geometry_objects, crs=crs)
            return water_gdf
        except Exception as e: print(f"Warning: Failed GDF from '{geom_col}': {e}. Trying lon/lat.")
    if lon_col in water_df.columns and lat_col in water_df.columns:
        try:
            water_df[lon_col] = pd.to_numeric(water_df[lon_col], errors='coerce')
            water_df[lat_col] = pd.to_numeric(water_df[lat_col], errors='coerce')
            original_len = len(water_df)
            water_df.dropna(subset=[lon_col, lat_col], inplace=True)
            if len(water_df) < original_len: print(f"  Dropped {original_len - len(water_df)} rows invalid lon/lat.")
            if not water_df.empty:
                geometry_objects = [Point(xy) for xy in zip(water_df[lon_col], water_df[lat_col])]
                water_gdf = gpd.GeoDataFrame(water_df, geometry=geometry_objects, crs=crs)
                return water_gdf
            else: raise ValueError("No valid lon/lat pairs remaining.")
        except Exception as e: raise ValueError(f"Failed GDF from lon/lat: {e}")
    raise ValueError("Cannot convert to GDF. Check columns.")


# Keep get_unique_locations_gdf (used for FIRE data only now)
def get_unique_locations_gdf(gdf, precision=6):
    if 'geometry' not in gdf.columns or not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame with a geometry column.")
    print(f"Finding unique locations in GDF with {len(gdf)} rows...")
    unique_coords = gdf.geometry.apply(
        lambda geom: (round(geom.x, precision), round(geom.y, precision))
    ).drop_duplicates()
    unique_indices = unique_coords.index
    # Use .iloc with unique indices derived from drop_duplicates - safer if original index has duplicates
    # Find integer positions of unique indices
    # Note: This assumes unique_indices are labels present in gdf.index
    # If index is not unique, .loc might return multiple rows.
    # Alternative: gdf.iloc[gdf.index.get_indexer(unique_indices)] if index is standard
    unique_gdf = gdf.loc[unique_indices].copy()
    print(f"Found {len(unique_gdf)} unique locations.")
    return unique_gdf[['geometry']] # Return only geometry

# NEW: Function to get unique water coords as a set (Memory Efficient)
def get_unique_water_coords_set(df, lon_col='lon', lat_col='lat', precision=6):
    """
    Extracts unique, rounded (lon, lat) coordinates as a set from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with longitude and latitude columns.
        lon_col (str): Name of the longitude column.
        lat_col (str): Name of the latitude column.
        precision (int): Decimal places for rounding.

    Returns:
        set: A set of unique (rounded_lon, rounded_lat) tuples.
    """
    print(f"Extracting unique coordinate pairs from {len(df)} rows...")
    if not {lon_col, lat_col}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns '{lon_col}' and '{lat_col}'")

    # Ensure numeric types, handle potential errors gracefully
    lon_series = pd.to_numeric(df[lon_col], errors='coerce')
    lat_series = pd.to_numeric(df[lat_col], errors='coerce')

    # Create tuples of rounded coordinates, dropping pairs where conversion failed
    coords = set(
        (round(lon, precision), round(lat, precision))
        for lon, lat in zip(lon_series, lat_series)
        if pd.notna(lon) and pd.notna(lat) # Check for NaN after coerce
    )
    print(f"Found {len(coords)} unique coordinate pairs.")
    return coords

# NEW: Mapping function using exact match
def map_fire_to_water_exact_match(unique_fire_gdf, unique_water_coords_set, precision=6):
    """
    Maps fire locations to water locations assuming exact (rounded) coordinate matches.

    Args:
        unique_fire_gdf (GeoDataFrame): GDF with unique fire centroid geometries.
        unique_water_coords_set (set): Set of unique (lon, lat) tuples from water data.
        precision (int): Coordinate rounding precision.

    Returns:
        dict: { (fire_lon, fire_lat) : (water_lon, water_lat) } where keys/values are identical if matched.
              Returns None for fire locations not found in the water set.
    """
    print("Mapping fire locations to water locations (Exact Match)...")
    location_map = {}
    not_found_count = 0
    for fire_index in unique_fire_gdf.index:
        fire_geom = unique_fire_gdf.loc[fire_index, 'geometry']
        fire_coord_key = (round(fire_geom.x, precision), round(fire_geom.y, precision))

        # Check if the exact rounded fire coordinate exists in the water set
        if fire_coord_key in unique_water_coords_set:
            location_map[fire_coord_key] = fire_coord_key # Map to itself
        else:
            # Optionally log or store unmatched coordinates
            # print(f"  Warning: Fire coordinate {fire_coord_key} not found in water data set.")
            location_map[fire_coord_key] = None # Mark as not found
            not_found_count += 1

    print(f"Mapping complete. {len(location_map) - not_found_count} fire locations matched.")
    if not_found_count > 0:
        print(f"  Warning: {not_found_count} unique fire locations were NOT found in the water data coordinates.")
    return location_map


# Keep optimized preprocess_all_water_timeseries_monthly
def preprocess_all_water_timeseries_monthly(water_df_input, lon_col='lon', lat_col='lat', time_col='time', value_col='waterstorage', precision=6):
    # ... (use the optimized version from the previous answer) ...
    # (Ensure it takes DataFrame, converts types, groups by rounded coords, resamples)
    print("Preprocessing water data (Optimized)...")
    water_df = water_df_input # Work directly? Or copy? Let's assume input can be modified or is a copy.
    required_cols = [lon_col, lat_col, time_col, value_col]
    if not all(col in water_df.columns for col in required_cols):
         raise ValueError(f"Preprocessing input DF missing required columns: {required_cols}")
    try:
        # print("  Converting data types...")
        water_df[time_col] = pd.to_datetime(water_df[time_col], errors='coerce')
        for col in [lon_col, lat_col, value_col]:
            if col in water_df.columns: water_df[col] = pd.to_numeric(water_df[col], errors='coerce')
        # print("  Dropping rows with NaNs...")
        initial_rows = len(water_df); water_df.dropna(subset=required_cols, inplace=True)
        rows_dropped = initial_rows - len(water_df)
        # if rows_dropped > 0: print(f"  Dropped {rows_dropped} rows.")
        if water_df.empty: print("Error: No valid water data after NaN drop."); return {}
        # print("  Setting time index & sorting...")
        if not isinstance(water_df.index, pd.DatetimeIndex): water_df = water_df.set_index(time_col)
        if not water_df.index.is_monotonic_increasing: water_df = water_df.sort_index()
        # print(f"  Grouping by rounded coordinates (Precision: {precision})...")
        grouped = water_df.groupby([water_df[lon_col].round(precision), water_df[lat_col].round(precision)])
        monthly_water_data = {}
        total_groups = len(grouped); count = 0
        # print(f"  Processing {total_groups} unique locations...")
        for name, group in grouped:
            if not group.empty and value_col in group.columns:
                 monthly_series = group[value_col].resample('MS').mean().dropna()
                 if not monthly_series.empty: monthly_water_data[name] = monthly_series
            count += 1
            if count % 1000 == 0 or count == total_groups: print(f"    Processed {count}/{total_groups} groups..."); gc.collect()
        print(f"Preprocessing complete. Found data for {len(monthly_water_data)} locations.")
        return monthly_water_data
    except Exception as e: print(f"Error during water preprocessing: {e}"); import traceback; traceback.print_exc(); return {}

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
    #Check if the wildfire series is constant
    if aligned_df['cumuarea'].nunique() == 1:
        print("Wildfire series is constant. No correlation possible.")
        return 0, 1
    
    if aligned_df['cumuarea'].isnull().all() or aligned_df['waterstorage'].isnull().all():
        print("One or both series contain only NaNs.")
        return None, None
    if len(aligned_df) < 3:
        print("Not enough data points for correlation.")
        return None, None

    try:
        # Ensure data is numeric and finite
        print("Calculating correlation...")
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
    
def prepare_aligned_timeseries_optimized(grid_gdf, monthly_water_series, target_fire_lon, target_fire_lat, precision=6):
    """
    Prepares and merges fire and pre-processed water time series using rounded coords.

    Args:
        grid_gdf (GeoDataFrame): Aggregated monthly fire data.
        monthly_water_series (pd.Series): Pre-calculated monthly water data for the *corresponding* water location.
        target_fire_lon (float): Longitude of the target fire centroid.
        target_fire_lat (float): Latitude of the target fire centroid.
        precision (int): Coordinate rounding precision used for matching.


    Returns:
        pd.DataFrame: DataFrame with aligned monthly 'cumuarea' and 'waterstorage', or None.
    """
    try:
        # --- Filter and Prepare Fire Data ---
        # Use rounded coordinates for filtering grid_gdf to match water data keys
        target_lon_r = round(target_fire_lon, precision)
        target_lat_r = round(target_fire_lat, precision)

        # Apply rounding to geometry for filtering
        fire_ts = grid_gdf[
             (grid_gdf.geometry.x.round(precision) == target_lon_r) &
             (grid_gdf.geometry.y.round(precision) == target_lat_r)
        ].copy()


        if fire_ts.empty:
            # This might happen if the original target_lon/lat wasn't *exactly* matching after rounding
            # Or simply no data exists for this rounded location in grid_gdf
            # print(f"Debug: No fire data found for rounded location ({target_lon_r}, {target_lat_r})")
            return None

        # Create datetime index if year/month columns exist
        if {'year', 'month'}.issubset(fire_ts.columns):
             fire_ts['time'] = pd.to_datetime(fire_ts[['year', 'month']].assign(day=1))
             fire_ts = fire_ts.set_index('time')
        elif isinstance(fire_ts.index, pd.DatetimeIndex):
             pass # Already has datetime index
        else:
             print(f"Warning: Fire data for ({target_fire_lon}, {target_fire_lat}) lacks year/month columns or DatetimeIndex.")
             return None

        fire_ts = fire_ts[['cumuarea']].sort_index()
        fire_ts['cumuarea'] = pd.to_numeric(fire_ts['cumuarea'], errors='coerce')


        # --- Water data is already processed ---
        if monthly_water_series is None or monthly_water_series.empty:
             return None # No corresponding water data

        water_ts_monthly = monthly_water_series.rename('waterstorage') # Ensure column name
        water_ts_monthly = pd.to_numeric(water_ts_monthly, errors='coerce')

        # --- Merge Data ---
        aligned_df = pd.merge(fire_ts, water_ts_monthly, left_index=True, right_index=True, how='inner')

        # Drop rows where either value might be NaN after merge/conversion
        aligned_df.dropna(subset=['cumuarea', 'waterstorage'], inplace=True)

        # Check for sufficient points *after* merging and dropping NaNs
        if len(aligned_df) < 3: # Need at least 3 points for meaningful correlation
             return None

        return aligned_df

    except Exception as e:
        print(f"Error preparing aligned timeseries for ({target_fire_lon}, {target_fire_lat}): {e}")
        return None


#* Keep analyze_location_correlation_optimized (needs slight change to handle None mapping)
def analyze_location_correlation_optimized(
    target_lon, target_lat, grid_gdf, location_map, all_monthly_water_data, precision=6, calculate_plots=False
):
    fire_coords_key = (round(target_lon, precision), round(target_lat, precision))
    results = {
        'target_lon': target_lon,
        'target_lat': target_lat,
        'fire_coords_key': fire_coords_key, # Store the key used
        'closest_water_lon': None,
        'closest_water_lat': None,
        'water_coords_key': None, # Store the key used
        'correlation': None,
        'p_value': None,
        'lags': None,
        'ccf_values': None,
        'n_points': 0, # Number of points used for correlation
        'timeseries_plot': None, # Avoid storing plots by default in batch
        'ccf_plot': None,
        'status': 'Pending' # Start with Pending status
    }

    water_coords_key = location_map.get(fire_coords_key)

  
    if water_coords_key is None: # Check if mapping returned None (no match found)
        results['status'] = 'Error: Fire location not found in water data'
        return results
  

    results['water_coords_key'] = water_coords_key
    results['closest_water_lon'] = water_coords_key[0]
    results['closest_water_lat'] = water_coords_key[1]

    monthly_water_series = all_monthly_water_data.get(water_coords_key)
    if monthly_water_series is None:
        results['status'] = f'Error: No preprocessed water data for key {water_coords_key}'
        return results

    aligned_df = prepare_aligned_timeseries_optimized(
        grid_gdf, monthly_water_series, target_lon, target_lat, precision=precision
    )
    if aligned_df is None:
        results['status'] = 'Error: Insufficient overlap or fire data invalid'
        return results
    results['n_points'] = len(aligned_df)

    corr, p_val = calculate_correlation(aligned_df)
    if corr is None: results['status'] = 'Error: Correlation failed'
    else: results['correlation'] = corr; results['p_value'] = p_val; results['status'] = 'OK'

    if results['status'] == 'OK':
        lags, ccf_vals = calculate_cross_correlation(aligned_df, max_lag=12)
        results['lags'] = lags; results['ccf_values'] = ccf_vals

    # Optional plots removed for brevity/memory

    return results

#* ---- Plotting Function ----
def plot_first_x_timeseries(
    results_list,
    num_plots,
    output_dir,
    # Required data sources for regenerating aligned_df:
    wildfire_grid_gdf,
    all_monthly_water_data,
    location_map,
    precision=6,
    fire_col='cumuarea',
    water_col='waterstorage',
    fire_label='Cumulative Area', # Default label
    water_label='Water Storage' # Default label
):
    """
    Generates and saves time series plots for the first 'num_plots'
    successfully analyzed locations found in the results_list.

    Args:
        results_list (list): List of dictionaries from analyze_location_correlation_optimized.
        num_plots (int): The maximum number of plots to generate.
        output_dir (str): Directory path to save the plot PNG files.
        wildfire_grid_gdf (GeoDataFrame): Original aggregated fire data.
        all_monthly_water_data (dict): Dictionary of preprocessed water time series.
        location_map (dict): The mapping from fire coords to water coords.
        precision (int): Coordinate precision used in analysis.
        fire_col (str): Column name for fire data in aligned_df.
        water_col (str): Column name for water data in aligned_df.
        fire_label (str): Label for fire data axis/legend.
        water_label (str): Label for water data axis/legend.
    """
    print(f"\nGenerating up to {num_plots} time series plots...")
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created plot output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating plot directory {output_dir}: {e}")
            return # Cannot proceed without output directory

    plots_generated = 0
    for result in results_list:
        if plots_generated >= num_plots:
            break # Stop once desired number of plots is reached

        if result.get('status') == 'OK':
            target_lon = result['target_lon']
            target_lat = result['target_lat']
            water_coords_key = result['water_coords_key']
            correlation = result.get('correlation', np.nan) # Get correlation if available

            print(f"  Plotting for: ({target_lon:.{precision}f}, {target_lat:.{precision}f})...")

            # Need to regenerate the aligned_df for plotting
            monthly_water_series = all_monthly_water_data.get(water_coords_key)
            if monthly_water_series is None:
                print(f"  Warning: Could not find water series for key {water_coords_key} needed for plotting. Skipping.")
                continue

            aligned_df = prepare_aligned_timeseries_optimized(
                wildfire_grid_gdf, monthly_water_series, target_lon, target_lat, precision=precision
            )

            if aligned_df is None or aligned_df.empty:
                print(f"  Warning: Could not regenerate valid aligned data for plotting location ({target_lon:.{precision}f}, {target_lat:.{precision}f}). Skipping.")
                continue

            # --- Generate Plot ---
            try:
                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Plot Fire Data (e.g., Cumulative Area)
                color1 = 'tab:red'
                ax1.set_xlabel('Time')
                ax1.set_ylabel(fire_label, color=color1)
                ax1.plot(aligned_df.index, aligned_df[fire_col], color=color1, marker='o', linestyle='-', markersize=4, label=fire_label)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

                # Create a second y-axis for the Water Storage data
                ax2 = ax1.twinx()
                color2 = 'tab:blue'
                ax2.set_ylabel(water_label, color=color2)
                ax2.plot(aligned_df.index, aligned_df[water_col], color=color2, marker='x', linestyle='--', markersize=4, label=water_label)
                ax2.tick_params(axis='y', labelcolor=color2)

                # Formatting
                plt.title(f'Fire ({fire_label}) vs. {water_label}\n'
                          f'Location: Lon={target_lon:.{precision}f}, Lat={target_lat:.{precision}f}\n'
                          f'Correlation: {correlation:.3f} (n={len(aligned_df)})')
                fig.autofmt_xdate() # Rotate date labels for better readability
                ax1.xaxis.set_major_locator(mdates.YearLocator()) # Major ticks yearly
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format ticks
                ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7])) # Minor ticks semi-annually
                plt.minorticks_on() # Show minor ticks

                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

                fig.tight_layout() # Adjust layout to prevent overlap

                # Save the plot
                plot_filename = f"timeseries_{target_lon:.{precision}f}_{target_lat:.{precision}f}.png"
                plot_filepath = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_filepath)
                plt.close(fig) # Close the figure to free memory
                # print(f"    Plot saved to: {plot_filepath}")
                plots_generated += 1

            except Exception as e:
                print(f"  Error generating plot for ({target_lon:.{precision}f}, {target_lat:.{precision}f}): {e}")
                plt.close('all') # Close any potentially open figures in case of error

    print(f"Finished generating plots. {plots_generated} plots saved to {output_dir}.")



# =======================================================================
# --- Main Execution Script (Optimized Workflow with Exact Match) ---
# =======================================================================
if __name__ == "__main__":

    # --- Configuration ---
    wildfire_file = 'Grace/Project/Data/aggregated_wildfire_grid_complete_1deg_w0.gpkg'
    water_file = 'Grace/Project/output/downsampled_1deg_water.parquet'
    results_output_file = 'Grace/Project/output/correlation_results_exact_match.csv'

    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)

    # --- Load Fire Data (Minimal Columns) ---
    print("Loading fire data...")
    try:
        wildfire_cols = ['year', 'month', 'cumuarea', 'geometry']
        wildfire_grid_gdf = gpd.read_file(wildfire_file, columns=wildfire_cols)
        if wildfire_grid_gdf.crs is None: wildfire_grid_gdf.crs = "EPSG:4326"
        print(f"Loaded {len(wildfire_grid_gdf)} wildfire rows.")
    except Exception as e: print(f"Error loading fire data: {e}"); exit()

    # --- Step 2: Get Unique Fire Locations ---
    print("\nStep 2: Getting unique fire locations...")
    unique_fire_gdf = get_unique_locations_gdf(wildfire_grid_gdf, precision=COORD_PRECISION)
    unique_fire_gdf.to_csv('Grace/Project/output/DEBUG_unique_fire_locations.csv', index=False)
    gc.collect() # Collect garbage after potential large GDF operation

    # --- Step 3: Get Unique Water Coords & Map Locations (with Caching) ---
    unique_water_coords_set = None
    location_map = None

    # Try loading map cache first
    if os.path.exists(LOCATION_MAP_CACHE):
         print(f"\nStep 3a: Loading location map from cache: {LOCATION_MAP_CACHE}")
         try:
              with open(LOCATION_MAP_CACHE, 'rb') as f:
                   location_map = pickle.load(f)
              print(f"  Loaded map for {len(location_map)} fire locations.")
         except Exception as cache_err:
              print(f"  Error loading location map cache: {cache_err}. Recomputing.")
              location_map = None # Ensure recomputation

    # If map not loaded from cache, compute it
    if location_map is None:
        # Try loading unique water coords cache
        if os.path.exists(UNIQUE_WATER_COORDS_CACHE):
            print(f"\nStep 3b: Loading unique water coordinates from cache: {UNIQUE_WATER_COORDS_CACHE}")
            try:
                 with open(UNIQUE_WATER_COORDS_CACHE, 'rb') as f:
                      unique_water_coords_set = pickle.load(f)
                 print(f"  Loaded set with {len(unique_water_coords_set)} unique water coordinates.")
            except Exception as cache_err:
                 print(f"  Error loading unique water coords cache: {cache_err}. Recomputing.")
                 unique_water_coords_set = None # Ensure recomputation
        
        # If unique water coords not loaded from cache, compute them
        if unique_water_coords_set is None:
            print("\nStep 3c: Calculating unique water coordinates (may take time)...")
            try:
                # Load ONLY lon/lat from the large water file for this step
                print(f"  Loading minimal columns (lon, lat) from {water_file}...")
                water_coords_df = pd.read_parquet(water_file, columns=['lon', 'lat'])
                # Alternative if geometry column is reliable and preferred:
                # water_coords_gdf = gpd.read_parquet(water_file, columns=['geometry'])
                # water_coords_df = pd.DataFrame({'lon': water_coords_gdf.geometry.x, 'lat': water_coords_gdf.geometry.y})
                print(f"  Loaded {len(water_coords_df)} rows for coordinate extraction.")
                unique_water_coords_set = get_unique_water_coords_set(water_coords_df, precision=COORD_PRECISION)
                # Save to cache
                print(f"  Saving unique water coordinates set to cache: {UNIQUE_WATER_COORDS_CACHE}")
                with open(UNIQUE_WATER_COORDS_CACHE, 'wb') as f:
                     pickle.dump(unique_water_coords_set, f)
                # Clean up temporary dataframe
                del water_coords_df
                gc.collect()
            except Exception as e:
                print(f"Error getting unique water coordinates: {e}"); exit()

        # Now compute the location map using the set
        print("\nStep 3d: Mapping fire locations (Exact Match)...")
        location_map = map_fire_to_water_exact_match(unique_fire_gdf, unique_water_coords_set, precision=COORD_PRECISION)
        # Save map to cache
        if location_map:
            print(f"  Saving location map to cache: {LOCATION_MAP_CACHE}")
            with open(LOCATION_MAP_CACHE, 'wb') as f:
                 pickle.dump(location_map, f)
        else:
            print("Error: Location mapping failed."); exit()
        # Clean up water coords set if no longer needed? Keep it for potential validation?
        # del unique_water_coords_set
        # gc.collect()


    # Clean up unique fire GDF
    print("Deleting unique fire GDF...")
    del unique_fire_gdf
    gc.collect()

    # --- Step 4: Preprocess Water Time Series (with Caching) ---
    print("\nStep 4: Preprocessing water time series...")
    all_monthly_water_data = None
    if os.path.exists(WATER_DATA_CACHE):
        print(f"  Loading preprocessed water data from cache: {WATER_DATA_CACHE}")
        try:
            with open(WATER_DATA_CACHE, 'rb') as f:
                all_monthly_water_data = pickle.load(f)
            print(f"  Loaded monthly data for {len(all_monthly_water_data)} water locations.")
        except Exception as cache_err:
            print(f"  Error loading water data cache: {cache_err}. Recomputing.")
            all_monthly_water_data = None # Ensure recomputation
    
    if all_monthly_water_data is None:
        print("  Cache not found or failed to load. Preprocessing all water time series...")
        try:
            # Load necessary columns for preprocessing
            water_cols_for_preprocess = ['lon', 'lat', 'time', 'waterstorage']
            print(f"  Loading water data for preprocessing (Cols: {water_cols_for_preprocess})...")
            water_df_for_preprocess = pd.read_parquet(water_file, columns=water_cols_for_preprocess)
            print(f"  Loaded {len(water_df_for_preprocess)} rows.")

            all_monthly_water_data = preprocess_all_water_timeseries_monthly(
                water_df_for_preprocess, lon_col='lon', lat_col='lat', time_col='time', value_col='waterstorage', precision=COORD_PRECISION
            )
            # Clean up dataframe used for preprocessing
            del water_df_for_preprocess
            gc.collect()

            if all_monthly_water_data:
                print(f"  Saving preprocessed water data to cache: {WATER_DATA_CACHE}")
                with open(WATER_DATA_CACHE, 'wb') as f: pickle.dump(all_monthly_water_data, f)
            else:
                print("Error: Water data preprocessing failed."); exit()
        except Exception as e:
             print(f"Error during water preprocessing: {e}"); exit()


    # --- Step 5: Run Analysis per Location ---
    print(f"\nStep 5: Running analysis for {len(location_map)} mapped fire locations...")
    all_results = []
    fire_locations_processed = 0
    total_fire_locations = len(location_map)

    # Filter out fire locations that weren't matched before iterating
    valid_fire_keys = {key for key, val in location_map.items() if val is not None}
    print(f"  Analyzing {len(valid_fire_keys)} successfully matched fire locations.")

    for fire_coords_key in valid_fire_keys:
        target_lon, target_lat = fire_coords_key
        fire_locations_processed += 1

        if fire_locations_processed % 100 == 0 or fire_locations_processed == len(valid_fire_keys):
            print(f"  Analyzing fire location {fire_locations_processed}/{len(valid_fire_keys)}: ({target_lon:.{COORD_PRECISION}f}, {target_lat:.{COORD_PRECISION}f})")

        result = analyze_location_correlation_optimized(
            target_lon, target_lat, wildfire_grid_gdf, location_map, all_monthly_water_data, precision=COORD_PRECISION, calculate_plots=False
        )
        all_results.append(result)
        # if fire_locations_processed % 500 == 0: gc.collect()

    
    # --- Step 5.5: Generate Plots (BEFORE deleting data) ---
    # Check if there are results and data needed for plotting still exists
    if all_results and wildfire_grid_gdf is not None and all_monthly_water_data is not None and location_map is not None:
            plot_first_x_timeseries(
                results_list=all_results,
                num_plots=NUM_PLOTS_TO_GENERATE, # Use constant defined at top
                output_dir=PLOT_OUTPUT_DIR,     # Use constant defined at top
                wildfire_grid_gdf=wildfire_grid_gdf,
                all_monthly_water_data=all_monthly_water_data,
                location_map=location_map,
                precision=COORD_PRECISION,
                fire_label='Cumulative Area [ha]', # Add units if known
                water_label='Water Storage [m3/m3]' # Add units if known
            )
    else:
            print("\nSkipping plot generation because input data is missing or no results were generated.")




    # Clean up large data objects after analysis
    print("Deleting large data objects after analysis...")
    del all_monthly_water_data, wildfire_grid_gdf, location_map
    gc.collect()

    # --- Step 6: Consolidate and Summarize Results ---
    # ... (Your existing Step 6 code) ...
    print("\nStep 6: Consolidating results...")
    if not all_results:
         print("No results were generated.")
    else:
         results_df = pd.DataFrame(all_results)
         print("\n--- Analysis Summary ---")
         print(results_df.head())
         print(f"\nTotal unique fire locations analyzed: {len(results_df)}")
         print(f"Successful analyses (Status OK): {len(results_df[results_df['status'] == 'OK'])}")
         print("\nStatus Counts:")
         print(results_df['status'].value_counts())
         # Save results
         try:
              if 'lags' in results_df.columns: results_df['lags'] = results_df['lags'].astype(str)
              if 'ccf_values' in results_df.columns: results_df['ccf_values'] = results_df['ccf_values'].astype(str)
              results_df.to_csv(results_output_file, index=False)
              print(f"\nResults saved to {results_output_file}")
         except Exception as save_err: print(f"\nError saving results to CSV: {save_err}")

    print("\n--- Workflow Complete ---")