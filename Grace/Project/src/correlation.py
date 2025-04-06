import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.distance import cdist # For finding nearest points efficiently

# --- Helper Function (for Mascon time) ---
def Convert_2002_2_DMJ(days, unit='days since 2002-04-01 00:00:00.0'):
    """
    Convert days since 2002-04-01 to datetime objects.

    Parameters:
    -----------
    days : array-like
        Days since the reference date specified in unit.
    unit : str
        Unit string from the NetCDF file, e.g., 'days since 2002-04-01 00:00:00.0'

    Returns:
    --------
    date : list or datetime
        List of datetime objects or a single datetime object.
    """
    # Use cftime=False to get standard Python datetime objects if possible
    try:
        dates_cf = nc.num2date(days, unit, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    except ValueError:
         # Fallback if standard datetime is out of range (though unlikely for this range)
         print("Warning: Using cftime.datetime objects due to potential date range issues.")
         dates_cf = nc.num2date(days, unit, only_use_cftime_datetimes=True, only_use_python_datetimes=False)


    # Ensure output is standard datetime for consistency if possible
    if isinstance(dates_cf, np.ndarray):
        # Convert cftime objects (if any) to standard datetime
        # Handle potential mix of datetime and cftime objects from num2date
        py_dates = []
        for d in dates_cf.flat:
            try:
                # Attempt conversion for cftime objects
                 py_dates.append(datetime(d.year, d.month, d.day, getattr(d, 'hour', 0), getattr(d, 'minute', 0), getattr(d, 'second', 0)))
            except AttributeError:
                 # If it's already a standard datetime object
                 py_dates.append(d)
        # Reshape back if the original was multi-dimensional (though time is usually 1D)
        if dates_cf.ndim > 1:
             return np.array(py_dates).reshape(dates_cf.shape)
        else:
             return py_dates

    else: # Handle single date value
         try:
              return datetime(dates_cf.year, dates_cf.month, dates_cf.day, getattr(dates_cf, 'hour', 0), getattr(dates_cf, 'minute', 0), getattr(dates_cf, 'second', 0))
         except AttributeError:
              return dates_cf # Return as is if already standard datetime


# --- Main Function ---
def correlate_mascons_gldas(location, time_start, time_end, soil_depths, gldas_filepath, mascons_filepath):
    """
    Calculates the correlation between JPL Mascon LWE and GLDAS soil moisture
    for a given location and time period.

    Parameters:
    -----------
    location : tuple
        (Longitude, Latitude) tuple for the point of interest.
    time_start : str or datetime object
        Start date for the analysis (e.g., 'YYYY-MM-DD' or datetime object).
    time_end : str or datetime object
        End date for the analysis (e.g., 'YYYY-MM-DD' or datetime object).
    soil_depths : list of str
        List of soil depths to use from GLDAS.
        Expected values like '0-10cm', '10-40cm', '40-100cm', '100-200cm'.
    gldas_filepath : str
        Filepath to the GLDAS soil moisture CSV data.
    mascons_filepath : str
        Filepath to the JPL Mascons NetCDF data.

    Returns:
    --------
    correlation : float
        Pearson correlation coefficient between the two time series. Returns NaN if
        correlation cannot be calculated (e.g., insufficient overlapping data).
    fig : matplotlib.figure.Figure
        Matplotlib figure object containing the plot of the time series.
        Returns None if plotting fails.
    """

    print(f"--- Starting Analysis ---")
    print(f"Location: {location}")
    print(f"Time Range: {time_start} to {time_end}")
    print(f"Soil Depths: {soil_depths}")

    # --- 1. Load and Prepare Mascon Data ---
    try:
        print(f"Loading Mascons data from: {mascons_filepath}")
        with nc.Dataset(mascons_filepath, 'r') as ds_mascons:
            mascon_lwe = ds_mascons.variables['lwe_thickness'][:] # Load all LWE data
            mascon_lat = ds_mascons.variables['lat'][:]
            mascon_lon = ds_mascons.variables['lon'][:]
            mascon_time_raw = ds_mascons.variables['time'][:]
            mascon_time_units = ds_mascons.variables['time'].units

            # Convert Mascon time to datetime objects
            mascon_time = Convert_2002_2_DMJ(mascon_time_raw, mascon_time_units)
            print(f"Mascon time range: {mascon_time[0]} to {mascon_time[-1]}")

            # Find the nearest Mascon grid point
            # Create a grid of lon, lat pairs for Mascons
            lon_grid, lat_grid = np.meshgrid(mascon_lon, mascon_lat)
            mascon_coords = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T

            # Use cdist for efficient nearest neighbor search
            if location[0] < 0:
            # add 360 to lon to get the correct distance
                location_mascons = np.array(location) + np.array([360, 0]) # Adjust lon to match Mascon's 0-360 range
            dist = cdist([location_mascons], mascon_coords)
            nearest_idx = np.argmin(dist)
            mascon_lat_idx, mascon_lon_idx = np.unravel_index(nearest_idx, lon_grid.shape)

            nearest_mascon_lon = mascon_lon[mascon_lon_idx]
            nearest_mascon_lat = mascon_lat[mascon_lat_idx]
            print(f"Nearest Mascon point: Lon={nearest_mascon_lon -360:.3f}, Lat={nearest_mascon_lat:.3f}")

            # Extract the time series for the nearest point
            mascon_ts_data = mascon_lwe[:, mascon_lat_idx, mascon_lon_idx]

            # Create a pandas Series with datetime index
            mascon_series = pd.Series(mascon_ts_data, index=pd.to_datetime(mascon_time), name='Mascon_LWE_cm')
            # Remove potential mask fill values if they exist (often large negative numbers)
            if hasattr(mascon_lwe, '_FillValue'):
                 fill_value = mascon_lwe._FillValue
                 mascon_series[mascon_series == fill_value] = np.nan


    except FileNotFoundError:
        print(f"Error: Mascons file not found at {mascons_filepath}")
        return np.nan, None
    except KeyError as e:
        print(f"Error: Variable {e} not found in Mascons file.")
        return np.nan, None
    except Exception as e:
        print(f"An unexpected error occurred while processing Mascons data: {e}")
        return np.nan, None

    # --- 2. Load and Prepare GLDAS Data ---
    try:
        print(f"Loading GLDAS data from: {gldas_filepath}")
        gldas_df = pd.read_csv(gldas_filepath, parse_dates=['time'])

        # Find the nearest GLDAS grid point (assuming unique lat/lon pairs per timestep)
        gldas_coords = gldas_df[['lon', 'lat']].drop_duplicates().values
        dist_gldas = cdist([location], gldas_coords)
        nearest_gldas_idx = np.argmin(dist_gldas)
        nearest_gldas_lon, nearest_gldas_lat = gldas_coords[nearest_gldas_idx]

        print(f"Nearest GLDAS point: Lon={nearest_gldas_lon:.3f}, Lat={nearest_gldas_lat:.3f}")

        # Filter data for the nearest location
        gldas_location_df = gldas_df[
            (gldas_df['lon'] == nearest_gldas_lon) & (gldas_df['lat'] == nearest_gldas_lat)
        ].set_index('time').sort_index()

        if gldas_location_df.empty:
             print(f"Error: No GLDAS data found for the nearest coordinates.")
             return np.nan, None

        print(f"GLDAS time range for location: {gldas_location_df.index.min()} to {gldas_location_df.index.max()}")


        # Map requested soil depths to column names and sum them
        soil_col_map = {
            '0-10cm': 'SoilMoi0_10cm_inst',
            '10-40cm': 'SoilMoi10_40cm_inst',
            '40-100cm': 'SoilMoi40_100cm_inst',
            '100-200cm': 'SoilMoi100_200cm_inst'
        }
        selected_cols = []
        for depth in soil_depths:
            col_name = soil_col_map.get(depth)
            if col_name and col_name in gldas_location_df.columns:
                selected_cols.append(col_name)
            else:
                print(f"Warning: Soil depth '{depth}' or corresponding column '{col_name}' not found in GLDAS data. Skipping.")

        if not selected_cols:
            print("Error: No valid soil depth columns selected or found.")
            return np.nan, None

        # Sum the selected soil moisture columns (kg/m^2 is equivalent to mm)
        gldas_series = gldas_location_df[selected_cols].sum(axis=1)
        gldas_series.name = f"GLDAS_SoilMoisture_{'_'.join(soil_depths)}_mm"


    except FileNotFoundError:
        print(f"Error: GLDAS file not found at {gldas_filepath}")
        return np.nan, None
    except KeyError as e:
        print(f"Error: Column {e} not found in GLDAS file. Check CSV headers and soil_col_map.")
        return np.nan, None
    except Exception as e:
        print(f"An unexpected error occurred while processing GLDAS data: {e}")
        return np.nan, None

# --- 3. Align Time Series to Monthly Frequency ---
    print("Aligning time series to monthly start frequency...")
    # Convert time_start and time_end to datetime objects if they are strings
    time_start_dt = pd.to_datetime(time_start)
    time_end_dt = pd.to_datetime(time_end)

    # --- Prepare Mascon data ---
    # Ensure index is datetime
    mascon_series.index = pd.to_datetime(mascon_series.index)
    # Drop NaNs that might exist from loading/fill values *before* resampling
    mascon_series = mascon_series.dropna()
    if mascon_series.empty:
        print("Warning: Mascon series is empty after initial NaN drop, before resampling.")
        # Decide how to handle: return error or continue to see if GLDAS has data?
        # Let's continue for now, the join will handle it.
    # Resample Mascon data to the start of the month ('MS').
    # .mean() aggregates any Mascon data points falling within the same calendar month.
    # If only one point exists in a month, .mean() just returns that point's value.
    mascon_monthly = mascon_series.resample('MS').mean()
    mascon_monthly.name = mascon_series.name # Keep the original name


    # --- Prepare GLDAS data ---
    # Ensure index is datetime
    gldas_series.index = pd.to_datetime(gldas_series.index)
    # Drop NaNs *before* resampling (though likely already clean if summed)
    gldas_series = gldas_series.dropna()
    if gldas_series.empty:
         print("Warning: GLDAS series is empty after initial NaN drop, before resampling.")
    # Resample GLDAS data to the start of the month ('MS').
    # This standardizes the index and ensures it aligns perfectly,
    # even if the source CSV had slight inconsistencies (it shouldn't if format is YYYY-MM-01).
    # Using .mean() handles cases where data might somehow not be exactly on the 1st.
    gldas_monthly = gldas_series.resample('MS').mean()
    gldas_monthly.name = gldas_series.name # Keep the original name


    # --- Filter both MONTHLY series to the specified time range ---
    # Pandas slicing with datetimes is generally inclusive of the start and end points.
    mascon_monthly_filt = mascon_monthly[time_start_dt:time_end_dt]
    gldas_monthly_filt = gldas_monthly[time_start_dt:time_end_dt]

    # --- Combine the monthly series ---
    # Inner join ensures we only keep months where *both* datasets have a value after resampling.
    df_aligned = pd.concat([mascon_monthly_filt, gldas_monthly_filt], axis=1, join='inner')

    # Drop rows where *either* value became NaN during the resampling (.mean() of empty month)
    # Note: The inner join already handles months missing in one *entire* series.
    # This dropna handles cases where a month might exist in both indices post-resample,
    # but the .mean() resulted in NaN perhaps due to NaNs in the original higher-freq data.
    df_aligned.dropna(inplace=True)

    if df_aligned.empty or len(df_aligned) < 2: # Need at least 2 points to correlate
        print("Error: No overlapping data points found after monthly alignment and filtering.")
        # Provide more context for debugging:
        print(f"Time range requested: {time_start_dt.date()} to {time_end_dt.date()}")
        print(f"Mascon months available in range (before join): {len(mascon_monthly_filt.dropna())}")
        print(f"GLDAS months available in range (before join): {len(gldas_monthly_filt.dropna())}")
        if not mascon_monthly_filt.dropna().empty:
             print(f"  Mascon date range available: {mascon_monthly_filt.dropna().index.min().date()} to {mascon_monthly_filt.dropna().index.max().date()}")
        if not gldas_monthly_filt.dropna().empty:
             print(f"  GLDAS date range available: {gldas_monthly_filt.dropna().index.min().date()} to {gldas_monthly_filt.dropna().index.max().date()}")
        return np.nan, None

    print(f"Aligned monthly data points for correlation: {len(df_aligned)}")
    print(f"Aligned time range: {df_aligned.index.min().strftime('%Y-%m-%d')} to {df_aligned.index.max().strftime('%Y-%m-%d')}")

    # --- 4. Standardize Data (Calculate Z-scores) ---
    print("Standardizing data (Z-score)...")

    # Get the original column names dynamically
    col_mascon = df_aligned.columns[0] # Assumes Mascon is the first column
    col_gldas = df_aligned.columns[1]  # Assumes GLDAS is the second column

    # Calculate mean and std dev for each column
    mean_mascon = df_aligned[col_mascon].mean()
    std_mascon = df_aligned[col_mascon].std()
    mean_gldas = df_aligned[col_gldas].mean()
    std_gldas = df_aligned[col_gldas].std()

    # Create new standardized columns (Z-scores)
    # Check for zero standard deviation to avoid division by zero
    col_mascon_std_name = f'{col_mascon}_std'
    col_gldas_std_name = f'{col_gldas}_std'

    if std_mascon > 1e-9: # Use a small threshold instead of == 0 for float precision
        df_aligned[col_mascon_std_name] = (df_aligned[col_mascon] - mean_mascon) / std_mascon
        print(f"  Standardized {col_mascon} (mean={mean_mascon:.2f}, std={std_mascon:.2f})")
    else:
        print(f"Warning: Standard deviation for {col_mascon} is near zero. Standardization results in zeros.")
        df_aligned[col_mascon_std_name] = 0.0 # Assign 0 if no variance

    if std_gldas > 1e-9:
        df_aligned[col_gldas_std_name] = (df_aligned[col_gldas] - mean_gldas) / std_gldas
        print(f"  Standardized {col_gldas} (mean={mean_gldas:.2f}, std={std_gldas:.2f})")
    else:
        print(f"Warning: Standard deviation for {col_gldas} is near zero. Standardization results in zeros.")
        df_aligned[col_gldas_std_name] = 0.0 # Assign 0 if no variance



    # --- 4. Calculate Correlation ---
    try:
        correlation = df_aligned[col_mascon_std_name].corr(df_aligned[col_gldas_std_name])
        print(f"Calculated Pearson Correlation: {correlation:.4f}")
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        correlation = np.nan


    # --- 5. Generate Plot ---
    print("Generating plot...")
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel(mascon_series.name, color=color)
        ax1.plot(df_aligned.index, df_aligned[mascon_series.name], color=color, marker='o', linestyle='-', markersize=4, label=mascon_series.name)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.5)


        # Create a second y-axis for the GLDAS data
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(gldas_series.name, color=color)
        ax2.plot(df_aligned.index, df_aligned[gldas_series.name], color=color, marker='x', linestyle='--', markersize=4, label=gldas_series.name)
        ax2.tick_params(axis='y', labelcolor=color)

        # Improve formatting
        plt.title(f'JPL Mascon LWE vs GLDAS Soil Moisture ({", ".join(soil_depths)})\n'
                  f'Location: Lon={location[0]}, Lat={location[1]} (Nearest points used)\n'
                  f'Correlation: {correlation:.3f}')
        fig.autofmt_xdate() # Rotate date labels
        fig.tight_layout() # Adjust layout
        # Add legends together
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    except Exception as e:
        print(f"Error generating plot: {e}")
        fig = None

    # --- 6. Save Final Output with all the Input params---
    # Save parameters and correlation to a CSV file
    output_csv_path = '/home/faehdy/repos/Grace/Space_Data_Kernel/Grace/Project/output/correlation_results.csv'
    try:
        # Create a dictionary of parameters and results
        result_data = {
            'Location_Lon': [location[0]],
            'Location_Lat': [location[1]],
            'Time_Start': [time_start],
            'Time_End': [time_end],
            'Mascon_coordinates': [f'Lon={nearest_mascon_lon-360:.3f}, Lat={nearest_mascon_lat:.3f}'],
            'GLDAS_coordinates': [f'Lon={nearest_gldas_lon:.3f}, Lat={nearest_gldas_lat:.3f}'],
            'Soil_Depths': [', '.join(soil_depths)],
            'Correlation': [correlation]
        }
        # Convert to a DataFrame
        result_df = pd.DataFrame(result_data)

        # Check if the file exists
        if not pd.io.common.file_exists(output_csv_path):
            # If the file does not exist, write with header
            result_df.to_csv(output_csv_path, mode='w', index=False, header=True)
            print(f"Results saved to new file: {output_csv_path}")
        else:
            # Append to the CSV without overwriting existing data
            result_df.to_csv(output_csv_path, mode='a', index=False, header=False)
            print(f"Results appended to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    print("--- Analysis Complete ---")
    return correlation, fig

# --- Example Usage ---

# Define Inputs
LOCATION = (-80, 55.0)  
TIME_START = '2002-01-01'
TIME_END = '2017-12-31'
SOIL_DEPTHS = ['0-10cm', '10-40cm', '40-100cm'] # Combine top two layers
#SOIL_DEPTHS = ['0-10cm', '10-40cm', '40-100cm', '100-200cm'] # Use all layers
GLDAS_FILE = '/home/faehdy/repos/Grace/Space_Data_Kernel/Grace/Project/Data/data_GLDAS/compiled_canada_soil_moisture.csv' 
MASCONS_FILE = '/home/faehdy/repos/Grace/Space_Data_Kernel/Grace/Project/Data/JPL_Mascons.nc'

# --- Important Notes Before Running ---
# 1. REPLACE 'path/to/your/...' with the ACTUAL file paths on your system.
# 2. Ensure your GLDAS CSV has columns named 'time', 'lat', 'lon', and the soil moisture columns
#    (e.g., 'SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', etc.).
# 3. Ensure your Mascon NetCDF file has variables named 'lwe_thickness', 'lat', 'lon', 'time'.
# 4. The Mascon time units should be compatible with nc.num2date (like 'days since YYYY-MM-DD HH:MM:SS').
# 5. The performance for finding the nearest GLDAS point depends on the CSV size.
#    For very large CSVs, pre-processing or using a spatial index might be faster.

# Run the function (Make sure paths are correct before uncommenting!)
# correlation_value, plot_figure = correlate_mascons_gldas(
#     location=LOCATION,
#     time_start=TIME_START,
#     time_end=TIME_END,
#     soil_depths=SOIL_DEPTHS,
#     gldas_filepath=GLDAS_FILE,
#     mascons_filepath=MASCONS_FILE
# )

# if plot_figure:
#     # Save the plot to a file
#     plot_filename = f"/home/faehdy/repos/Grace/Space_Data_Kernel/Grace/Project/output/correlation_plots/mascon_gldas_correlation_{LOCATION[0]}_{LOCATION[1]}.png"
#     plot_figure.savefig(plot_filename)
#     print(f"Plot saved as: {plot_filename}")

#     # Show the plot
#     plt.show()
# else:
#     print("\nCorrelation calculation or plotting failed.")



### Iterate over entire canada with a grid of 10° lon/lat
# Define the grid of locations
lon_range = np.arange(-120, -90, 5)  # Example range for longitudes
lat_range = np.arange(50, 62, 5)    # Example range for latitudes
locations = [(lon, lat) for lon in lon_range for lat in lat_range]

# Iterate over each location
for loc in locations:
    print(f"Processing location: {loc}")
    correlation_value, plot_figure = correlate_mascons_gldas(
        location=loc,
        time_start=TIME_START,
        time_end=TIME_END,
        soil_depths=SOIL_DEPTHS,
        gldas_filepath=GLDAS_FILE,
        mascons_filepath=MASCONS_FILE
    )

    if plot_figure:
        # Save the plot to a file
        plot_filename = f"/home/faehdy/repos/Grace/Space_Data_Kernel/Grace/Project/output/correlation_plots/mascon_gldas_correlation_{loc[0]}_{loc[1]}.png"
        plot_figure.savefig(plot_filename)
        print(f"Plot saved as: {plot_filename}")

    else:
        print("\nCorrelation calculation or plotting failed.")
