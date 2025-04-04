import os
import requests
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta

# NASA Earthdata URL for the dataset
BASE_URL = "https://data.gesdisc.earthdata.nasa.gov/data/GLDAS/GLDAS_CLSM10_M.2.0"

# Authentication token
TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImZhZWhkeV9ldGgiLCJleHAiOjE3NDg5NDc3OTMsImlhdCI6MTc0Mzc2Mzc5MywiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.Bhlk4BgDTqKEX9-FRCWw5cbLKa97_TXbjos8ccsKWbcSGxy_Y7T9ZMrhzXw-mZslJ92WBzHLm52jEzh4xKwlqSuKsMcIZuNDsDTaP6DsBNeRji2L7GVUuhTS8-_3ldHr9aqBFbcp0-Q_7OQkDhZyP6E5v4qgtrA01i9ERPYqtO3VV2A56InbhFoDpxV0dwQufXyOgpJYa_1MkE2dl3prwxW_n5T2767E1RPn0dZ7dwmTaKKj4pL8PcglaSMSzcyAqhUMTL_FAeUxCJfYT1RS8chrYzCEWyk8Ega5_poJhXQNFbugbCO0FNX3yM_ru6krvYYb29FWZf04z633x7Ak4Q"

# Headers for authentication
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Start and end dates
start_date = datetime(2002, 1, 1)
end_date = datetime(2022, 1, 1)

# Folder to store downloaded files
download_folder = "../Data/data_GLDAS"
os.makedirs(download_folder, exist_ok=True)

# List to store datasets
datasets = []

# Loop through each month
current_date = start_date
while current_date <= end_date:
    # Format the filename
    file_name = f"GLDAS_CLSM10_M.A{current_date.strftime('%Y%m')}.020.nc4"
    file_url = f"{BASE_URL}/{current_date.year}/{file_name}"
    local_file_path = os.path.join(download_folder, file_name)

    # Check if the file is already downloaded
    if not os.path.exists(local_file_path):
        response = requests.get(file_url, headers=HEADERS)

        if response.status_code == 200:
            with open(local_file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Missing file: {file_name}")
            current_date += relativedelta(months=1)
            continue  # Skip to the next month

    # Load NetCDF file using xarray
    ds = xr.open_dataset(local_file_path)
    datasets.append(ds)

    # Move to the next month
    current_date += relativedelta(months=1)

# Merge all datasets into one NetCDF file
if datasets:
    combined_ds = xr.concat(datasets, dim="time")  # Merge along the time dimension
    combined_ds.to_netcdf("./Project/Hyd_Model.nc4")
    print("All available files have been merged into: Hyd_Model.nc4")
else:
    print("No files were available for merging.")
