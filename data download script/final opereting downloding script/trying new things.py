from __future__ import (division, print_function, absolute_import, unicode_literals)
import os

from pandas import Timedelta
from satpy import Scene
from glob import glob
import rioxarray
import sqlite3
import numpy as np
import sys
import os
import os.path
import shutil
import time
import json
import os
from datetime import datetime, timedelta
from goes2go.data import goes_nearesttime
import rioxarray
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import netCDF4
import leafmap  # Add this import
from io import StringIO
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')
from datetime import datetime, timedelta

def combine_viirs_images(input_dir, output_dir):
    """
    Combine two VIIRS images using Satpy, slicing them, and save the output.

    Parameters:
        input_dir (str): Directory containing VIIRS image files.
        output_dir (str): Directory where the combined image will be saved.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Locate VIIRS files
    viirs_files = glob(os.path.join(input_dir, "*.nc"))
    if len(viirs_files) < 2:
        raise ValueError("At least two VIIRS files are required for combining.")

    # Load files into a Satpy Scene
    scene = Scene(reader="viirs_l1b", filenames=viirs_files[:2])

    # Load all available datasets
    scene.load(scene.available_dataset_names())
    print(scene.available_dataset_names())

    target_area = "goes_east_abi_c_500m"
    combined_scene = scene.resample(target_area, resampler="nearest")

    # Save datasets as GeoTIFFs
    my_bands = ["I04", "I05"]
    band_data_list = []
    for band in my_bands:
        band_data = combined_scene[band].values  # Extract values
        band_data_list.append(band_data)

    # Combine the bands into a single 3D array (bands, y, x)
    combined_bands = np.stack(band_data_list, axis=0)

    # Wrap the combined data into an xarray DataArray with rioxarray extension
    combined_data_rio = xr.DataArray(
        combined_bands,
        dims=["band", "y", "x"],
        coords={
            "band": [1, 2],  # Band indices
            "y": combined_scene["I04"].y,
            "x": combined_scene["I04"].x
        },
        attrs=combined_scene["I04"].attrs,  # Copy CRS and attributes
    )
    combined_data_rio.rio.write_crs(combined_scene["I04"].attrs["area"].crs.to_string(), inplace=True)

    # Save the combined GeoTIFF
    output_tif = os.path.join(output_dir, "combined_bands.tif")
    combined_data_rio.rio.to_raster(output_tif)
    print(f"Combined file with both bands saved at {output_tif}")


    # # Modify and save GeoTIFF for I04 band
    # I04_band = rioxarray.open_rasterio(os.path.join(output_dir, "I04.tif"))
    # new_I04_band = I04_band.copy()  # Copy the rioxarray object
    # new_I04_band = new_I04_band.astype(np.float32)  # Change the datatype to float32 to include NaN values
    # new_I04_band.values[0] = combined_scene["I04"].values  # Assign the Satpy scene values to the rioxarray object
    # new_I04_band.rio.to_raster(os.path.join(output_dir, "new_I04.tif"))  # Save the rioxarray object to a new GeoTIFF file
    # print(f"Saved at {os.path.join(output_dir, 'new_I04.tif')}")
    #
    # I05_band = rioxarray.open_rasterio(os.path.join(output_dir, "I05.tif"))
    # new_I05_band = I05_band.copy()  # Copy the rioxarray object
    # new_I05_band = new_I05_band.astype(np.float32)  # Change the datatype to float32 to include NaN values
    # new_I05_band.values[0] = combined_scene["I05"].values  # Assign the Satpy scene values to the rioxarray object
    # new_I05_band.rio.to_raster(os.path.join(output_dir, "new_I05.tif"))  # Save the rioxarray object to a new GeoTIFF file
    # print(f"Saved at {os.path.join(output_dir, 'new_I05.tif')}")


# this is the choice of last resort, when other attempts have failed
def getcURL(url, headers=None, out=None):
    # OS X Python 2 and 3 don't support tlsv1.1+ therefore... cURL
    import subprocess
    try:
        print('trying cURL', file=sys.stderr)
        args = ['curl', '--fail', '-sS', '-L', '-b session', '--get', url]
        for (k, v) in headers.items():
            args.extend(['-H', ': '.join([k, v])])
        if out is None:
            # python3's subprocess.check_output returns stdout as a byte string
            result = subprocess.check_output(args)
            return result.decode('utf-8') if isinstance(result, bytes) else result
        else:
            subprocess.call(args, stdout=out)
    except subprocess.CalledProcessError as e:
        print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
    return None


# read the specified URL and output to a file
def geturl(url, token=None, out=None):
    headers = {'user-agent': USERAGENT}
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read()
                else:
                    shutil.copyfileobj(fh, out)
            except urllib2.HTTPError as e:
                print('TLSv1_2 sys 2 : HTTP GET error code: %d' % e.code, file=sys.stderr)
                return getcURL(url, headers, out)
            except urllib2.URLError as e:
                print('TLSv1_2 sys 2 : Failed to make request: %s, RETRYING' % e.reason, file=sys.stderr)
                return getcURL(url, headers, out)
            return None

        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            try:
                fh = urlopen(Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read().decode('utf-8')
                else:
                    shutil.copyfileobj(fh, out)
            except HTTPError as e:
                print('TLSv1_2 : HTTP GET error code: %d' % e.code, file=sys.stderr)
                return getcURL(url, headers, out)
            except URLError as e:
                print('TLSv1_2 : Failed to make request: %s' % e.reason, file=sys.stderr)
                return getcURL(url, headers, out)
            return None

    except AttributeError:
        return getcURL(url, headers, out)


################################################################################


DESC = "This script will recursively download all files if they don't exist from a LAADS URL and will store them to the specified path"


def sync(src, dest, tok):
    '''synchronize src url with dest directory'''
    try:
        import csv
        files = {}
        files['content'] = [f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True)]
    except ImportError:
        files = json.loads(geturl(src + '.json', tok))

    # Use os.path since python 2/3 both support it while pathlib is 3.4+
    for f in files['content']:
        # Debugging: Print the content of 'f' to understand its structure
        print(f)  # This will show the actual structure of each 'f'

        # Safely access 'name' and 'size'
        name = f.get('name', 'default_name')  # Default to 'default_name' if 'name' is missing
        filesize = int(f.get('size', 0))  # Default to 0 if 'size' is missing
        path = os.path.join(dest, name)
        url = src + '/' + name

        if filesize == 0:  # size FROM RESPONSE
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + name, path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path) or os.path.getsize(path) == 0:  # filesize FROM OS
                    print('\ndownloading: ', path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
    return 0



from datetime import datetime, timedelta

base_url = "https://example.com"  # Replace with your actual base URL
def generate_file_paths(start_date, end_date):
    """Generate file paths for each date in the given range"""
    file_paths = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.strftime('%Y')
        day_of_year = current_date.strftime('%j')  # Day of the year
        filename = f"VNP02IMG.A{year}{day_of_year}.0000.002.{year}{day_of_year}100726.nc"
        url = f"{base_url}/{year}/{day_of_year}/{filename}"
        file_paths.append((url, filename))
        current_date += timedelta(days=1)
    return file_paths


from datetime import datetime, timedelta

base_url = "https://example.com"  # Replace with your actual base URL
def generate_file_paths(start_date, end_date):
    """Generate file paths for each date in the given range"""
    file_paths = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.strftime('%Y')
        day_of_year = current_date.strftime('%j')  # Day of the year
        filename = f"VNP02IMG.A{year}{day_of_year}.1048.002.{year}{day_of_year}184920.nc"
        url = f"{base_url}/{year}/{day_of_year}/{filename}"
        file_paths.append((url, filename))
        current_date += timedelta(days=1)
    return file_paths


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

def extract_datetime_from_filename(filepath):
    """Extract the date and time from the file path by working from the end."""
    try:
        # Extract the filename from the path
        filename = os.path.basename(filepath)

        # Split the filename by dots
        parts = filename.split('.')

        # Find the part that starts with 'A' and has the correct length
        for part in reversed(parts):
            if part.startswith('A') and len(part) >= 8:
                date_part = part[1:]  # Remove 'A' prefix
                break
        else:
            raise ValueError(f"No valid date part found in filename: {filename}")

        # Find the first part that looks like a time (HHMM format)
        for part in reversed(parts):
            if len(part) == 4 and part.isdigit():
                time_part = part
                break
        else:
            raise ValueError(f"No valid time part found in filename: {filename}")

        # Extract year and day of year
        year = int(date_part[:4])
        day_of_year = int(date_part[4:])
        extracted_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        # Extract hours and minutes
        hours = int(time_part[:2])
        minutes = int(time_part[2:])
        extracted_datetime = extracted_date.replace(hour=hours, minute=minutes)

        return extracted_datetime
    except Exception as e:
        print(f"Error extracting date and time from filepath '{filepath}': {e}")
        return None


from goes2go.data import goes_nearesttime
from pandas import Timedelta
import os
from glob import glob


def download_goes_data(start_time, out_dir, satellite):
    """ Tries to find a GOES image for a given satellite by adjusting timestamps in ±5-minute steps. """
    product = "ABI-L2-MCMIPC"
    os.makedirs(out_dir, exist_ok=True)

    # Generate search timestamps in ±5-minute increments up to ±1 hour
    search_offsets = [i for i in range(0, 121, 5)]  # 0, 5, 10, ..., 60

    for offset in search_offsets:
        for sign in [-1, 1]:  # Try both negative (-offset) and positive (+offset)
            adjusted_time = start_time + Timedelta(minutes=offset * sign)

            try:
                print(f"🔍 Searching {satellite} image at {adjusted_time}...")
                g = goes_nearesttime(
                    attime=adjusted_time.strftime("%Y-%m-%d %H:%M"),
                    satellite=satellite,
                    product=product,
                    domain="C",
                    return_as="filelist",
                    download=True,
                    save_dir=out_dir,
                    overwrite=False  # Avoid unnecessary re-downloads
                )

                if g is not None and not g.empty:
                    print(f"✅ Found {satellite} image at {adjusted_time}.")
                    return g  # Return first found image and stop searching
            except Exception as e:
                print(f"⚠️ No image found at {adjusted_time}: {e}")

    print(f"❌ No image found for {satellite} within ±1 hour of {start_time}.")
    return None


def ensure_two_goes_images(start_time, out_dir):
    """ Ensures we get at least one GOES-17 image and one from either GOES-16 or GOES-18. """

    # Step 1: Ensure we get a GOES-17 image
    goes_17_image = download_goes_data(start_time, out_dir, "goes17")

    # Step 2: Try to get GOES-16 first
    goes_16_image = download_goes_data(start_time, out_dir, "goes16")

    # Step 3: If GOES-16 is missing, try GOES-18
    if goes_16_image is None:
        goes_16_image = download_goes_data(start_time, out_dir, "goes18")

    # Step 4: Check if we got 2 images
    if goes_17_image is None and goes_16_image is None:
        print(f"🚨 WARNING: Could not find **ANY** GOES images for {start_time}!")
    elif goes_17_image is None or goes_16_image is None:
        print(f"⚠️ WARNING: Only found **one** GOES image for {start_time}. Missing a second!")

    return goes_17_image, goes_16_image  # Return both (even if one is None)


import os
import shutil

def move_files_to_parent(root_dir, dest):
    """
    Recursively navigates through subdirectories of root_dir to find files
    and moves them to the destination directory (dest).
    """
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)

        # If item_path is a directory, navigate into it
        if os.path.isdir(item_path):
            move_files_to_parent(item_path, dest)  # Recursively process subdirectories

        # If item_path is a file, move it to the destination directory
        elif os.path.isfile(item_path):
            try:
                # Check if the file already exists in the destination directory
                dest_file = os.path.join(dest, os.path.basename(item_path))
                if os.path.exists(dest_file):
                    # Generate a unique name for the file
                    base, ext = os.path.splitext(os.path.basename(item_path))
                    counter = 1
                    while os.path.exists(dest_file):
                        dest_file = os.path.join(dest, f"{base}_{counter}{ext}")
                        counter += 1

                shutil.move(item_path, dest_file)  # Move the file to the destination
                print(f"Moved {item_path} to {dest_file}")
            except Exception as e:
                print(f"Failed to move {item_path} to {dest}: {e}")

    # After processing, remove the empty directory
    if root_dir != dest:  # Prevent deleting the destination directory
        try:
            os.rmdir(root_dir)  # Only removes the folder if it's empty
            print(f"Removed empty folder: {root_dir}")
        except OSError:
            pass  # Skip if the folder is not empty or can't be removed

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

if __name__ == '__main__':
    source = "https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502339804/"
    # destination = "C:/Users/Admin/Desktop/viirs_cupels/temp"
    destination = r"W:\Sat_Project\6.3.25\temp"
    # destination = r"C:\Users\97254\Desktop/viirs_cupels/temp"
    # token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImxpZWxsYSIsImV4cCI6MTczOTAyMTc5MywiaWF0IjoxNzMzODM3NzkzLCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhc3N1cmFuY2VfbGV2ZWwiOjJ9.kpJGLKA5q1npFqpHioj36HsPYq8Bf7A6Mx1NbgMMqfyiWSMyatZY4kwi9ha5MSoZQKFoXiMrR-psP3K57-AtH7U3hAN14C3WwNe0zZm6V9zBjEcYSnHV0fUzUHHplOYG2OY5hGCmuCLm-LHIRSSJe0CjW4zgQaP2jpsIebfbwbfaiZKNpLli504hqil4XCG_4Ys4_-A79q3YDITfOROQ3rjutjS5G-CkqT9wTjHlpeuAEJx47mAzaOqkXzIVlVY60mfafWpsUGJ_bXEKvgl439fs0ZP0EEYhz06m7MMfXKBElxWqxTDQl-80rw4sHx1rSap0d-XNHzCry6lbRWPKMw"
    token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im9tcmlhcmllIiwiZXhwIjoxNzQ2NDU2MjY3LCJpYXQiOjE3NDEyNzIyNjcsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.vt_PXHbnoLP9p-fiFGugKAH5mjdLNflUmE8yVYdDZBENBf4ajkXMjuBstzmHlBlzV4Uz5HB_NGws8u81FBCjwK4MqTvLSjPMyZvWUD9S9_gYAxd8aNNOn-7yNBOC9num9J7B_vu1ODZgfLyqi03HdjMt4JG-PaguzZq-C6mZdDLe7Tr-Wwd0uwluEYLCK4D1dO7Fx5w0lZZZl-m-6fhQ6qGLdO3u-1Z-9xuaIF8IVm5CToCjqoxx4QuYuMMg7OSFlQXcbhlJd3K3I9dm-Ii0meSOOkqf289hTo8Aykil8lFBawnlM18nifJCBlIdqVd0UlPiH3H_gJstqwDxVjlrqQ"
    # Ensure the destination path exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    # # destination_without_temp = "C:/Users/Admin/Desktop/viirs_cupels"
    destination_without_temp = r"W:\Sat_Project\6.3.25"
    # # destination_without_temp = r"C:\Users\97254\Desktop/viirs_cupels"


    # # Call the sync function to start the download process
    # sync(source, destination, token)
    #
    # nc_files = glob(os.path.join(destination, "*.nc"))
    # if not nc_files:
    #     print("No .nc files found in the directory.")
    #     sys.exit(1)
    #
    # # Process each .nc file
    # for nc_file in nc_files:
    #     filename = os.path.basename(nc_file)
    #     extracted_datetime = extract_datetime_from_filename(filename)
    #
    #     if not extracted_datetime:
    #         print(f"Skipping file due to failed datetime extraction: {filename}")
    #         continue
    #
    #     # Create a subdirectory based on the extracted datetime (e.g., "2022-12-01_10-48")
    #     subdir_name = extracted_datetime.strftime('%Y-%m-%d_%H-%M')
    #     subdir = os.path.join(destination_without_temp, subdir_name)
    #
    #     # Ensure the subdirectory exists
    #     if not os.path.exists(subdir):
    #         os.makedirs(subdir)
    #
    #     # Move the file to the subdirectory
    #     try:
    #         shutil.move(nc_file, subdir)
    #         print(f"Moved {filename} to {subdir}")
    #     except Exception as e:
    #         print(f"Failed to move {filename} to {subdir}: {e}")
    #
    # try:
    #     shutil.rmtree(destination)
    #     print(f"Temp folder '{destination}' removed successfully.")
    # except Exception as e:
    #     print(f"Failed to remove temp folder '{destination}': {e}")

    # Iterate through subdirectories in the parent folder and run combine_viirs_images
    for folder in os.listdir(destination_without_temp):
        folder_path = os.path.join(destination_without_temp, folder)
        if os.path.isdir(folder_path):
            # combine_viirs_images(folder_path, folder_path)
            nc_files = glob(os.path.join(folder_path, "*.nc"))

            if not nc_files:
                print(f"⚠️ No VIIRS files found in {folder_path}, skipping.")
                continue

            VIIRS_file_path = nc_files[0]
            date_time = extract_datetime_from_filename(VIIRS_file_path)

            # Ensure we get 1 GOES-17 and 1 from either GOES-16 or GOES-18
            goes_17_img, goes_other_img = ensure_two_goes_images(date_time, folder_path)

            # Move downloaded files to the main folder
            for sat in ["noaa-goes16", "noaa-goes17", "noaa-goes18"]:
                sat_folder = os.path.join(folder_path, sat)
                if os.path.isdir(sat_folder):
                    move_files_to_parent(sat_folder, folder_path)

    print("✅ File relocation complete.")





