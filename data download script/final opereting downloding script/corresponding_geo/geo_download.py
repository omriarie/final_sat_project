# # import GOES
# # import geopandas as gpd
# # import numpy as np
# # import pandas as pd
# # from datetime import datetime
# # from datetime import timedelta
# #
# # def time_delta(date_time_obj, time_detla):
# #     """This function gets a single date time object and a time delta and returns a new date time object.
# #
# #     Args:
# #         date_time_obj (datetime): a single date time object
# #         time_detla (int): a time delta in minutes
# #     """
# #     new_date_time_obj = date_time_obj + timedelta(minutes = time_detla) ## add the time delta to the date time object
# #     return new_date_time_obj
# #
# # def convert_timedate_to_string(date_time_obj):
# #     """This function gets a single date time object and returns a string of the date and time according to GOES library.
# #
# #     Args:
# #         date_time_obj (datetime): a single date time object
# #     """
# #     date_time_string = date_time_obj.strftime("%Y-%m-%d %H:%M") ## convert the date time object to a string
# #     date_time_split = date_time_string.split(" ") ## split the string by space
# #     YMD = date_time_split[0] ## get the date
# #     HM = date_time_split[1] ## get the time
# #     Year = YMD.split("-")[0] ## get the year
# #     Month = YMD.split("-")[1] ## get the month
# #     Day = YMD.split("-")[2] ## get the day
# #     Hour = HM.split(":")[0] ## get the hour
# #     Minute = HM.split(":")[1] ## get the minute
# #     Second = "00" ## set the second to 00
# #
# #     out_string = f"{Year}{Month}{Day}-{Hour}{Minute}{Second}" ## create the output string
# #     return out_string
# #
# # def download_goes_using_virs(virs_path,out_dir,sat,time_interval = 10,prod="ABI-L2-FDCF",band = [],rename = '%Y%m%d%H%M'):
# #     VIIRS_data = gpd.read_file(virs_path) ## read VIIRS point shapefile
# #     VIIRS_other_col_version = ["ACQ_DATE", "ACQ_TIME"] ## columns in the VIIRS data
# #     con = np.any(np.isin(VIIRS_other_col_version, VIIRS_data.columns)) ## check if the columns are in the VIIRS data
# #     if con == True:
# #         VIIRS_data = VIIRS_data.rename(columns={"ACQ_DATE": "DATE", "ACQ_TIME": "TIME"}) ## rename the columns to date and time
# #     VIIRS_data["string_date_time"] = VIIRS_data["DATE"] + " " + VIIRS_data["TIME"] ## combine date and time columns to one column
# #     VIIRS_data["string_date_time"][0] ## check the first row of the new column
# #     unique_date_time_string = np.unique(VIIRS_data["string_date_time"]) ## get unique date and time strings
# #     unique_date_time_string[:5] ## check the first 5 unique date and time strings
# #     datetime_obj_list = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"), unique_date_time_string))
# #     datetime_obj_list[:5]
# #
# #     for time in datetime_obj_list:
# #         start_date = time_delta(time, -time_interval) ## get the start date
# #         end_date = time_delta(time, time_interval) ## get the end date
# #         start_date_string = convert_timedate_to_string(start_date) ## convert the start date to the GOES format
# #         end_date_string = convert_timedate_to_string(end_date) ## convert the end date to the GOES format
# #         GOES.download(sat, prod,
# #                   DateTimeIni = start_date_string, DateTimeFin = end_date_string,
# #                   channel = band, rename_fmt = rename, path_out=out_dir) ## download the GOES data
# #         print(f"{time} downloaded") ## print the time that was downloaded
# # VIIRS_file_path =r'C:\Users\Admin\Desktop\geo_cuples\viirs\VNP02IMG.A2022335.1048.002.2022335184920.nc'
# # out_dir = r'C:\Users\Admin\Desktop\geo_cuples'
# # download_goes_using_virs(VIIRS_file_path,out_dir ,sat="goes16")
# # download_goes_using_virs(VIIRS_file_path,out_dir ,sat="goes16")
# import xarray as xr
# import pandas as pd
# from datetime import datetime, timedelta
# import GOES
# import os
# from datetime import datetime, timedelta
# from goes2go.data import goes_nearesttime
# import rioxarray
# import matplotlib.pyplot as plt
# import numpy as np
# import netCDF4
# import leafmap  # Add this import
#
# # def time_delta(date_time_obj, time_delta_minutes):
# #     """Return a new datetime object offset by a time delta in minutes."""
# #     return date_time_obj + timedelta(minutes=time_delta_minutes)
# #
# # def convert_timedate_to_string(date_time_obj):
# #     """Convert datetime object to GOES string format."""
# #     return date_time_obj.strftime("%Y%m%d-%H%M00")
# #
# # def read_viirs_netcdf(netcdf_path):
# #     """Read NetCDF file and extract the required VIIRS data."""
# #     # Open the NetCDF file
# #     dataset = xr.open_dataset(netcdf_path)
# #
# #     # Check for available variables
# #     print("Available Variables: ", dataset.attrs)
# #
# #     # Extract the required variables (adjust based on your data structure)
# #     if 'RangeBeginningDate' in dataset.attrs and 'RangeBeginningTime' in dataset.attrs:
# #         # Extract the date and time values from the variables
# #         dates = pd.to_datetime(dataset['RangeBeginningDate'].values)  # Ensure this is the correct format
# #         times = pd.to_timedelta(dataset['RangeBeginningTime'].values, unit='m')  # Ensure this is in minutes
# #
# #         # Combine dates and times into datetime objects
# #         datetime_list = [date + time for date, time in zip(dates, times)]
# #         return datetime_list
# #     else:
# #         raise ValueError("Variables 'RangeBeginningDate' and/or 'RangeBeginningTime' not found in the dataset.")
#
# # def download_goes_using_virs(netcdf_path, out_dir, sat, time_interval=10, prod="ABI-L2-FDCF", band=[], rename='%Y%m%d%H%M'):
# #     """Download GOES data using VIIRS data."""
# #     try:
# #         datetime_obj_list = read_viirs_netcdf(netcdf_path)
# #     except Exception as e:
# #         print(f"Error reading NetCDF file: {e}")
# #         return
# #
# #     for time in datetime_obj_list:
# #         try:
# #             start_date = time_delta(time, -time_interval)
# #             end_date = time_delta(time, time_interval)
# #             start_date_string = convert_timedate_to_string(start_date)
# #             end_date_string = convert_timedate_to_string(end_date)
# #             GOES.download(sat, prod,
# #                           DateTimeIni=start_date_string,
# #                           DateTimeFin=end_date_string,
# #                           channel=band, rename_fmt=rename, path_out=out_dir)
# #             print(f"{time} downloaded")
# #         except Exception as e:
# #             print(f"Error downloading for time {time}: {e}")
#
#
# def extract_datetime_from_filename(filename):
#     """Extract the date and time from the filename."""
#     try:
#         # Remove the prefix (e.g., "VNP02IMG.") and split by the first dot to get the date part
#         parts = filename.split('.')
#         if len(parts) < 4:
#             raise ValueError(f"Invalid filename format: {filename}")
#
#         # Extract the date part encoded as AYYYYDDD (e.g., A2022335 for December 1, 2022)
#         date_part = parts[1][1:]  # Remove 'A' prefix from 'AYYYYDDD'
#         year = int(date_part[:4])
#         day_of_year = int(date_part[4:])
#         extracted_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
#
#         # Extract the time part from the filename (e.g., 1048 in VNP02IMG.AYYYYDDD.1048.002...)
#         time_part = parts[2]
#         hours = int(time_part[:2])
#         minutes = int(time_part[2:])
#         extracted_datetime = extracted_date.replace(hour=hours, minute=minutes)
#
#         return extracted_datetime
#     except Exception as e:
#         print(f"Error extracting date and time from filename '{filename}': {e}")
#         return None
#
# def download_goes_data(start_time,out_dir):
#     # Specify the satellite and product
#     satellite = "goes16"
#     product = "ABI-L2-MCMIPC"  # Use the specified product
#     # out_dir = r"GOES-16\\"  # Your specified output directory
#
#     os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist
#
#     # Download data for the nearest observation at the current time
#     try:
#         print(f"Attempting to find data nearest to: {start_time}...")
#         g = goes_nearesttime(
#             attime=start_time.strftime("%Y-%m-%d %H:%M"),  # Format the time as a string
#             satellite=satellite,
#             product=product,
#             return_as="filelist",
#             download=True,  # Set to True to download the file immediately
#             save_dir=out_dir,  # Specify the output directory
#             overwrite=False  # Set to False to avoid overwriting existing files
#         )
#
#         # Print the structure of g to understand its contents
#         print(f"Downloaded data structure: {g}")
#
#         # Collect the downloaded file paths
#         for index, row in g.iterrows():  # Iterate over DataFrame rows
#             print(f"Downloaded file: {row['file']}")  # Print each downloaded file
#
#     except Exception as e:
#         print(f"An error occurred while finding data: {e}")
#
# # File paths
# VIIRS_file_path = r'C:\Users\Admin\Desktop\geo_cuples\viirs\VNP02IMG.A2022335.1048.002.2022335184920.nc'
# date_time = extract_datetime_from_filename(VIIRS_file_path)
# out_dir = r'C:\Users\Admin\Desktop\geo_cuples'
# download_goes_data(date_time,out_dir)
#
# # Run the download process
# # download_goes_using_virs(VIIRS_file_path, out_dir, sat="goes16")


import rioxarray
import numpy as np


def show_tif_value_range(tif_path):
    """
    Display the range of values in a GeoTIFF file.

    Parameters:
        tif_path (str): Path to the GeoTIFF file.

    Returns:
        None
    """
    try:
        # Open the GeoTIFF file
        tif_data = rioxarray.open_rasterio(tif_path)

        # Convert to numpy array and compute min and max
        values = tif_data.values
        min_value = np.nanmin(values)  # Use nanmin to ignore NaN values
        max_value = np.nanmax(values)  # Use nanmax to ignore NaN values

        print(f"Value range for {tif_path}:")
        print(f"Minimum value: {min_value}")
        print(f"Maximum value: {max_value}")
    except Exception as e:
        print(f"Error processing the GeoTIFF file: {e}")


# Example usage
tif_file_path = r"C:\Users\Admin\Desktop\viirs_cupels\2022-12-01_11-18\I04.tif"  # Replace with the path to your TIF file
show_tif_value_range(tif_file_path)


