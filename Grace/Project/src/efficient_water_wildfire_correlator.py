import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import cKDTree # For efficient nearest neighbor

# --- Existing Helper/Module Functions (Modules 3, 5, 6 remain mostly the same) ---
# calculate_correlation, plot_aligned_timeseries, plot_cross_correlation
# prepare_water_gdf (may need adjustment if loading from CSV without geometry)

# --- REVISED/NEW Helper Functions ---

def prepare_water_gdf(water_df, lon_col='lon', lat_col='lat', crs="EPSG:4326"):
    """Converts water DataFrame to GeoDataFrame if not already one."""
    # If reading from CSV, geometry needs to be created
    if 'geometry' not in water_df.columns and lon_col in water_df.columns and lat_col in water_df.columns:
         print("Creating geometry for water DataFrame...")
         geometry = [Point(xy) for xy in zip(water_df[lon_col], water_df[lat_col])]
         water_gdf = gpd.GeoDataFrame(water_df, geometry=geometry, crs=crs)
         print("Geometry creation complete.")
         return water_gdf
    elif isinstance(water_df, gpd.GeoDataFrame):
         water_gdf = water_df # Assume it's already a GeoDataFrame if passed as one
         if water_gdf.crs is None:
             print(f"Warning: Setting water_gdf CRS to {crs}")
             water_gdf.set_crs(crs, inplace=True)
         elif str(water_gdf.crs).upper() != str(crs).upper(): # Compare CRS robustly
             print(f"Warning: Reprojecting water_gdf CRS from {water_gdf.crs} to {crs}")
             water_gdf = water_gdf.to_crs(crs) # Ensure consistent CRS
         return water_gdf
    else:
         raise ValueError("Input water_df is not a GeoDataFrame and lacks lon/lat columns.")


def get_unique_locations_gdf(gdf, precision=6):
    """
    Extracts unique locations from a GeoDataFrame based on rounded geometry coordinates.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.
        precision (int): Number of decimal places to round coordinates for uniqueness check.

    Returns:
        GeoDataFrame: A GeoDataFrame containing only the unique locations.
    """
    if 'geometry' not in gdf.columns or not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame with a geometry column.")

    # Round coordinates to handle potential floating point inaccuracies
    unique_coords = gdf.geometry.apply(lambda geom: (round(geom.x, precision), round(geom.y, precision))).drop_duplicates()
    unique_indices = unique_coords.index
    unique_gdf = gdf.loc[unique_indices].copy()
    print(f"Found {len(unique_gdf)} unique locations.")
    # Return essential columns - geometry is key, keep index if useful
    return unique_gdf[['geometry']] # Or keep other columns if needed


def map_fire_to_water_locations_optimized(unique_fire_gdf, unique_water_gdf, precision=6):
    """
    Efficiently maps each fire location to the nearest water location using cKDTree.

    Args:
        unique_fire_gdf (GeoDataFrame): GDF with unique fire centroid geometries.
        unique_water_gdf (GeoDataFrame): GDF with unique water point geometries.
        precision (int): Coordinate rounding precision for dictionary keys.


    Returns:
        dict: { (fire_lon, fire_lat) : (water_lon, water_lat) }
    """
    print("Mapping fire locations to nearest water locations using cKDTree...")
    if unique_fire_gdf.crs != unique_water_gdf.crs:
        print(f"Warning: CRS mismatch! Fire: {unique_fire_gdf.crs}, Water: {unique_water_gdf.crs}. Ensure compatibility.")
        # Consider projecting both to a suitable projected CRS for accurate distances

    n_fire = len(unique_fire_gdf)
    n_water = len(unique_water_gdf)
    if n_fire == 0 or n_water == 0:
        print("Error: Cannot map locations, one or both input GDFs are empty.")
        return {}
    print(f"  Unique Fire locations: {n_fire}, Unique Water locations: {n_water}")

    # Prepare coordinate arrays for cKDTree (more robust extraction)
    fire_coords = np.array([(geom.x, geom.y) for geom in unique_fire_gdf.geometry])
    water_coords = np.array([(geom.x, geom.y) for geom in unique_water_gdf.geometry])

    # Build the KDTree on the water points (usually more numerous)
    tree = cKDTree(water_coords)

    # Query the tree for nearest neighbors for all fire points
    distances, indices = tree.query(fire_coords, k=1)

    location_map = {}
    # Use original index from unique_fire_gdf to retrieve geometry reliably
    for i, fire_index in enumerate(unique_fire_gdf.index):
        fire_geom = unique_fire_gdf.loc[fire_index, 'geometry']
        # Use rounded coordinates as the dictionary key
        fire_coord_key = (round(fire_geom.x, precision), round(fire_geom.y, precision))

        nearest_water_idx_in_unique = indices[i] # Index within unique_water_gdf
        # Get the original index if unique_water_gdf preserved it, or use iloc
        nearest_water_geom = unique_water_gdf.geometry.iloc[nearest_water_idx_in_unique]
        # Use rounded coordinates as the dictionary value
        water_coord_val = (round(nearest_water_geom.x, precision), round(nearest_water_geom.y, precision))
        location_map[fire_coord_key] = water_coord_val

    print(f"Mapping complete. {len(location_map)} fire locations mapped.")
    return location_map


def preprocess_all_water_timeseries_monthly(water_df_raw, lon_col='lon', lat_col='lat', time_col='time', value_col='waterstorage', precision=6):
    """
    Groups water data by location, processes time, and resamples to monthly means.
    Uses coordinate rounding for grouping.

    Args:
        water_df_raw (pd.DataFrame): The large raw water data.
        lon_col, lat_col, time_col, value_col (str): Column names.
        precision (int): Coordinate rounding precision for grouping.


    Returns:
        dict: { (lon, lat) : pd.Series(monthly_mean_values, index=monthly_timestamps) }
    """
    print("Preprocessing water data (grouping, time conversion, resampling)...")
    try:
        # --- Data Type Conversion ---
        print("  Converting data types...")
        # Convert time first, errors='coerce' will turn failures into NaT
        water_df_raw[time_col] = pd.to_datetime(water_df_raw[time_col], errors='coerce')
        # Convert numeric columns, failures become NaN
        water_df_raw[lon_col] = pd.to_numeric(water_df_raw[lon_col], errors='coerce')
        water_df_raw[lat_col] = pd.to_numeric(water_df_raw[lat_col], errors='coerce')
        water_df_raw[value_col] = pd.to_numeric(water_df_raw[value_col], errors='coerce')

        # --- Drop Rows with Missing Essential Info ---
        print("  Dropping rows with NaNs in essential columns...")
        initial_rows = len(water_df_raw)
        water_df_raw.dropna(subset=[lon_col, lat_col, time_col, value_col], inplace=True)
        print(f"  Dropped {initial_rows - len(water_df_raw)} rows.")
        if water_df_raw.empty:
            print("Error: No valid water data remaining after NaN drop.")
            return {}

        # --- Create Rounded Coordinate Columns for Grouping ---
        print("  Creating rounded coordinate columns for grouping...")
        water_df_raw['lon_rounded'] = water_df_raw[lon_col].round(precision)
        water_df_raw['lat_rounded'] = water_df_raw[lat_col].round(precision)

        # --- Set Index and Group ---
        print("  Setting time index...")
        water_df_raw = water_df_raw.set_index(time_col)
        print(f"  Grouping by rounded coordinates (Precision: {precision})...")
        # Group by the rounded coordinates
        grouped = water_df_raw.groupby(['lon_rounded', 'lat_rounded'])

        monthly_water_data = {}
        total_groups = len(grouped)
        count = 0
        print(f"  Processing {total_groups} unique locations...")
        # --- Iterate Through Groups and Resample ---
        for name, group in grouped:
            # name is a tuple (lon_rounded, lat_rounded)
            # group is a DataFrame for this location with datetime index
            # Resample each group to monthly start frequency ('MS'), calculate mean
            # Use only the value column for resampling
            monthly_series = group[value_col].resample('MS').mean()
            monthly_series = monthly_series.dropna() # Remove months resulting in NaN mean
            if not monthly_series.empty:
                monthly_water_data[name] = monthly_series # Use rounded coords as key

            # --- Progress Update ---
            count += 1
            if count % 500 == 0 or count == total_groups: # Print every 500 or on the last one
                 print(f"    Processed {count}/{total_groups} water locations...")

        print(f"Water data preprocessing complete. Found monthly data for {len(monthly_water_data)} locations.")
        return monthly_water_data

    except KeyError as e:
         print(f"Error: Column not found during water preprocessing: {e}. Check column names.")
         return {}
    except Exception as e:
         print(f"An unexpected error occurred during water preprocessing: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback for debugging
         return {}


# --- REVISED Module 2: Prepare Aligned Time Series (Optimized) ---
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



# --- REVISED Orchestration Function ---
def analyze_location_correlation_optimized(
    target_lon, target_lat, grid_gdf, location_map, all_monthly_water_data, precision=6, calculate_plots=False
):
    """
    Runs the optimized analysis workflow using pre-calculated mapping and water data.

    Args:
        target_lon (float): Target fire grid centroid longitude.
        target_lat (float): Target fire grid centroid latitude.
        grid_gdf (GeoDataFrame): Monthly aggregated fire data.
        location_map (dict): Map from fire coords (lon, lat) to closest water coords (lon, lat).
        all_monthly_water_data (dict): Dict mapping water coords (lon, lat) to monthly pd.Series.
        precision (int): Coordinate rounding precision.
        calculate_plots (bool): Whether to generate plot objects (can consume memory).


    Returns:
        dict: Dictionary containing results.
    """
    # Use rounded coordinates for lookups
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
        # 'timeseries_plot': None, # Avoid storing plots by default in batch
        # 'ccf_plot': None,
        'status': 'Pending'
    }

    # 1. Find corresponding water location from map
    water_coords_key = location_map.get(fire_coords_key)

    if water_coords_key is None:
        results['status'] = 'Error: Fire location key not in map'
        return results

    # Store actual (rounded) coordinates used for water data lookup
    results['water_coords_key'] = water_coords_key
    # Attempt to retrieve original (or at least representative) coords if needed,
    # but for processing, the rounded key is primary. We'll store the key itself.
    # You might need another lookup if exact original coords of the water point are required.
    results['closest_water_lon'] = water_coords_key[0] # Lon is first element
    results['closest_water_lat'] = water_coords_key[1] # Lat is second element


    # 2. Retrieve preprocessed water data using the rounded water coordinate key
    monthly_water_series = all_monthly_water_data.get(water_coords_key)
    if monthly_water_series is None:
        results['status'] = f'Error: No preprocessed water data found for water key {water_coords_key}'
        return results

    # 3. Prepare aligned time series (using optimized function)
    # Pass original target_lon/lat for filtering fire data initially if needed,
    # but the function now uses rounded coords primarily for internal matching logic
    aligned_df = prepare_aligned_timeseries_optimized(
        grid_gdf, monthly_water_series, target_lon, target_lat, precision=precision
    )

    if aligned_df is None:
        results['status'] = 'Error: Insufficient overlapping data or fire data missing/invalid'
        return results

    results['n_points'] = len(aligned_df) # Store number of points used

    # 4. Calculate standard correlation
    corr, p_val = calculate_correlation(aligned_df)
    if corr is None:
         results['status'] = 'Error: Correlation calculation failed'
         # Keep n_points, but mark corr/pval as None
    else:
         results['correlation'] = corr
         results['p_value'] = p_val
         results['status'] = 'OK' # Mark as successful if correlation calculated


    # 5. Calculate cross-correlation (only if correlation was successful)
    if results['status'] == 'OK':
        lags, ccf_vals = calculate_cross_correlation(aligned_df, max_lag=12)
        results['lags'] = lags # Store even if None
        results['ccf_values'] = ccf_vals # Store even if None
        if lags is None:
             # Optionally downgrade status or add a note
             # results['status'] = 'Warning: CCF calculation failed'
             pass


    # 6. Generate plots (Optional)
    # if calculate_plots and results['status'] == 'OK':
    #     try:
    #         # Pass representative coords for title
    #         plot_lon = results['target_lon']
    #         plot_lat = results['target_lat']
    #         results['timeseries_plot'] = plot_aligned_timeseries(aligned_df, plot_lon, plot_lat)
    #         if lags is not None: # Only plot ccf if available
    #             results['ccf_plot'] = plot_cross_correlation(lags, ccf_vals, plot_lon, plot_lat)
    #     except Exception as plot_err:
    #         print(f"Warning: Plot generation failed for ({target_lon}, {target_lat}): {plot_err}")


    return results


# =======================================================================
# --- Main Execution Script (Optimized Workflow) ---
# =======================================================================
if __name__ == "__main__": # Ensures code runs only when script is executed directly

    # --- Configuration ---
    # Define file paths (replace with your actual paths)
    wildfire_file = 'Grace/Project/Data/aggregated_wildfire_grid.gpkg' # Example: GeoPackage preferred
    # wildfire_file =  # Using dummy data path below for runnable example
    # water_file = 'Grace/Project/Data/data_GLDAS/gdf_compiled_canada_soil_moisture.parquet' # Example: Parquet preferred
    water_file = 'Grace/Project/Data/data_GLDAS/gdf_compiled_canada_soil_moisture.parquet' # Using dummy data path below for runnable example



    # Define coordinate precision for matching/grouping
    COORD_PRECISION = 5 # Adjust based on your data's precision

    # --- Load Data ---
    print("Loading data...")
    # try:
    #     # wildfire_grid_gdf = gpd.read_file(wildfire_file)
    #     # water_df_raw = pd.read_parquet(water_file) # Use read_csv for CSV
    #     # print(f"Loaded wildfire data with {len(wildfire_grid_gdf)} rows.")
    #     # print(f"Loaded water data with {len(water_df_raw)} rows.")

    #     # --- Create Dummy Data if files don't exist (for testing) ---
    #     print("Creating dummy data for demonstration as files not found...")
    #     grid_data = {
    #         'year': [2002, 2002, 2003, 2002, 2003, 2004],
    #         'month': [6, 7, 6, 6, 7, 8],
    #         'cumuarea': [100, 150, 80, 200, 120, 90],
    #         'geometry': [Point(-116.125, 56.875), Point(-116.125, 56.875), Point(-116.125, 56.875),
    #                      Point(-110.375, 60.125), Point(-110.375, 60.125), Point(-110.375, 60.125)]
    #     }
    wildfire_grid_gdf = gpd.read_file(wildfire_file) # Use existing file
    wildfire_grid_gdf.crs = "EPSG:4326" # Set CRS if not already set

    
    # read the parquet file
    water_gdf_raw = gpd.read_parquet(water_file) #  # Save dummy file
    water_gdf_raw.crs = "EPSG:4326" # Set CRS if not already set



    # --- Workflow Steps ---
    print("\n--- Starting Analysis Workflow ---")

    # Step 1: Prepare Water GDF (ensure geometry)
    print("Step 1: Preparing Water GeoDataFrame...")
    water_gdf_raw = prepare_water_gdf(water_gdf_raw, lon_col='lon', lat_col='lat')
    if water_gdf_raw is None: exit()

    # Step 2: Get Unique Locations
    print("\nStep 2: Getting unique locations...")
    unique_fire_gdf = get_unique_locations_gdf(wildfire_grid_gdf, precision=COORD_PRECISION)
    unique_water_gdf = get_unique_locations_gdf(water_gdf_raw, precision=COORD_PRECISION)

    # Step 3: Map Fire to Water Locations
    print("\nStep 3: Mapping fire locations to nearest water locations...")
    location_map = map_fire_to_water_locations_optimized(unique_fire_gdf, unique_water_gdf, precision=COORD_PRECISION)
    if not location_map:
        print("Error: Location mapping failed or produced no results.")
        exit()

    # Step 4: Preprocess Water Time Series (Heavy Lifting)
    print("\nStep 4: Preprocessing all water time series (this may take time)...")
    # Pass raw DataFrame (not GDF) for efficiency if lon/lat columns exist
    all_monthly_water_data = preprocess_all_water_timeseries_monthly(
        water_gdf_raw, lon_col='lon', lat_col='lat', time_col='time', value_col='waterstorage', precision=COORD_PRECISION
    )
    if not all_monthly_water_data:
        print("Error: Water data preprocessing failed or yielded no data.")
        exit()

    # Step 5: Run Analysis per Location
    print(f"\nStep 5: Running analysis for {len(unique_fire_gdf)} unique fire locations...")
    all_results = []
    # Iterate through unique fire locations using index for reliable geometry access
    for i, fire_index in enumerate(unique_fire_gdf.index):
        fire_geom = unique_fire_gdf.loc[fire_index, 'geometry']
        target_lon = fire_geom.x
        target_lat = fire_geom.y

        if (i + 1) % 100 == 0 or (i + 1) == len(unique_fire_gdf): # Progress update
           print(f"  Analyzing fire location {i+1}/{len(unique_fire_gdf)}: ({target_lon:.{COORD_PRECISION}f}, {target_lat:.{COORD_PRECISION}f})")

        result = analyze_location_correlation_optimized(
            target_lon, target_lat, wildfire_grid_gdf, location_map, all_monthly_water_data, precision=COORD_PRECISION, calculate_plots=False # Disable plots for batch speed
        )
        all_results.append(result)

    # Step 6: Consolidate and Summarize Results
    print("\nStep 6: Consolidating results...")
    results_df = pd.DataFrame(all_results)

    print("\n--- Analysis Summary ---")
    if not results_df.empty:
         print(results_df.head())
         print(f"\nTotal unique fire locations processed: {len(results_df)}")
         print(f"Successful analyses (Status OK): {len(results_df[results_df['status'] == 'OK'])}")
         print("\nStatus Counts:")
         print(results_df['status'].value_counts())

         # Save results
         results_output_file = 'correlation_results.csv'
         try:
              # Convert list columns (lags, ccf_values) to string for CSV compatibility if they exist
              if 'lags' in results_df.columns:
                   results_df['lags'] = results_df['lags'].astype(str)
              if 'ccf_values' in results_df.columns:
                   results_df['ccf_values'] = results_df['ccf_values'].astype(str)

              results_df.to_csv(results_output_file, index=False)
              print(f"\nResults saved to {results_output_file}")
         except Exception as save_err:
              print(f"\nError saving results to CSV: {save_err}")

    else:
         print("No results were generated.")

    print("\n--- Workflow Complete ---")