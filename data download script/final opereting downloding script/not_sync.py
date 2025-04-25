from __future__ import (division, print_function, absolute_import, unicode_literals)

import gc
import csv
import geopandas as gpd
import shapely
import sys

from satpy import Scene
from glob import glob

import sys

import os.path

import json

from goes2go.data import goes_nearesttime
import rioxarray

import numpy as np
import re
import xarray as xr
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




from io import StringIO
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')
from datetime import datetime, timedelta

def combine_viirs_images(input_dir, output_dir, AOI):
    """
    Combine two VIIRS images using Satpy, slice them, clip to AOI, and save the final clipped version.

    Parameters:
        input_dir (str): Directory containing VIIRS image files.
        output_dir (str): Directory where the final clipped image will be saved.
        AOI (GeoDataFrame): Area of Interest (AOI) polygon for clipping.

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
    # print(scene.available_dataset_names())

    # Resample to target area
    target_area = "goes_east_abi_c_500m"
    combined_scene = scene.resample(target_area, resampler="nearest")

    # Extract band data and combine them
    my_bands = ["I04", "I05"]
    band_data_list = [combined_scene[band].values for band in my_bands]
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

    # ✅ Clip the image using AOI
    AOI_reprojected = AOI.to_crs(combined_data_rio.rio.crs)
    clipped_img = combined_data_rio.rio.clip(AOI_reprojected.geometry)

    # ✅ Save only the final clipped version
    output_tif = os.path.join(output_dir, "combined_clip.tif")
    clipped_img.rio.to_raster(output_tif)

    print(f"✅ Final clipped VIIRS image saved at {output_tif}")





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
    '''Synchronize src URL with dest directory (downloads files).'''
    try:
        import csv
        files = {}
        files['content'] = [f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True)]
    except ImportError:
        files = json.loads(geturl(src + '.json', tok))

    # Use os.path since python 2/3 both support it while pathlib is 3.4+
    for f in files['content']:
        print(f)  # Debugging: Print the content of 'f' to understand its structure

        # Safely access 'name' and 'size'
        name = f.get('name', 'default_name')  # Default to 'default_name' if 'name' is missing
        filesize = int(f.get('size', 0))  # Default to 0 if 'size' is missing
        path = os.path.join(dest, name)
        url = src + '/' + name

        if filesize == 0:  # size FROM RESPONSE
            try:
                print('creating dir:', path)
                os.makedirs(path, exist_ok=True)  # ✅ Use `makedirs` instead of `mkdir` to avoid errors
                sync(src + '/' + name, path, tok)  # ✅ Recursively sync subdirectories
            except Exception as e:  # ✅ Catch all exceptions but don't stop
                print(f"⚠ Warning: Failed to create directory `{path}`: {e}", file=sys.stderr)
                continue  # ✅ Skip to the next file instead of stopping

        else:
            try:
                if not os.path.exists(path) or os.path.getsize(path) == 0:  # filesize FROM OS
                    print('\n📥 Downloading:', path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('✅ Skipping (already exists):', path)
            except Exception as e:
                print(f"❌ Error: Failed to download `{name}`: {e}", file=sys.stderr)
                continue  # ✅ Skip to the next file instead of stopping

    print("✅ Sync completed!")  # ✅ Indicate that sync finished
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


def download_goes_data(start_time,out_dir,satellite):
    # Specify the satellite and product
    product = "ABI-L2-MCMIPC"  # Use the specified product
    os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Download data for the nearest observation at the current time
    try:
        print(f"Attempting to find data nearest to: {start_time}...")
        g = goes_nearesttime(
            attime=start_time.strftime("%Y-%m-%d %H:%M"),  # Format the time as a string
            satellite=satellite,
            product=product,
            return_as="filelist",
            download=True,  # Set to True to download the file immediately
            save_dir=out_dir,  # Specify the output directory
            overwrite=False  # Set to False to avoid overwriting existing files
        )

        # Print the structure of g to understand its contents
        print(f"Downloaded data structure: {g}")

        # Collect the downloaded file paths
        for index, row in g.iterrows():  # Iterate over DataFrame rows
            print(f"Downloaded file: {row['file']}")  # Print each downloaded file

    except Exception as e:
        print(f"An error occurred while finding data: {e}")


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
            # print(f"Removed empty folder: {root_dir}")
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


def get_VIIRS_bounds_polygon(VIIRS_fire_path):
    """
    Get a VIIRS fire product path return a polygon of VIIRS image bounds

    :VIIRS_fire_path: VIIRS fire product for example "C:\\Asaf\\VIIRS\\VNP14IMG.A2020251.2042.001.2020258064138"

    :return: geodataframe of bounds in lat/lon WGS84 projection
    """
    VIIRS = xr.open_dataset(VIIRS_fire_path)  ## Open file
    ## Now for the bbox bounds
    lat_bounds = VIIRS.attrs['GRingPointLatitude']
    lon_bounds = VIIRS.attrs['GRingPointLongitude']

    pol_bounds = []  ## Empty bounds list
    for i in range(len(lat_bounds)):  ## for all bounds
        coord = (lon_bounds[i], lat_bounds[i])  ## Take lon/lat
        pol_bounds.append(coord)  ## append
    pol_bounds.append((lon_bounds[0], lat_bounds[0]))  ## append firt coords to "close" the polygon

    pol = shapely.Polygon(pol_bounds)  ## Create polygon
    gdf_pol = gpd.GeoDataFrame({'geometry': [pol]}, crs=4326)  ## Create as gdf

    return (gdf_pol)



def log_invalid_order(order_folder: str, error_msg: str):
    """
    Append a row to W:\\Sat_Project\\log.csv indicating the order folder that failed
    and the exception message describing the failure reason.
    """
    log_csv = r"W:\Sat_Project\log.csv"
    # Open CSV in append mode so each run adds new lines
    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        # Columns: [current system time, order folder, error message]
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), order_folder, error_msg])


import os
import shutil
import re
import gc
import rioxarray
import geopandas as gpd
from glob import glob
from datetime import datetime


# -------------------------------------------------------------------------
# 1. Helper function to parse folder names like "2024-11-01_14-30"
#    and convert to a datetime object.
# -------------------------------------------------------------------------
def parse_folder_datetime(folder_name):
    """
    Expects folder_name in format: YYYY-MM-DD_HH-MM
    Returns a datetime object or None if parsing fails.
    """
    try:
        return datetime.strptime(folder_name, "%Y-%m-%d_%H-%M")
    except ValueError:
        return None




import os
import shutil

# If you have a function like this in your code, reuse it.
# Otherwise, implement it or replace with a simple `pass`.
def log_invalid_order(order_folder, reason):
    """
    Example stub to show how you'd log invalid orders.
    Adjust to match your actual logging mechanism if needed.
    """
    log_path = r"W:\Sat_Project\log.csv"
    with open(log_path, 'a', encoding='utf-8') as f:
        # Write 'order_folder,reason' or your preferred format
        f.write(f"{order_folder},{reason}\n")


# if __name__ == "__main__":
#     downloads_dir = r"W:\Sat_Project\DOWNLOADS"  # Same as in your main script
#     processed_dir = r"W:\Sat_Project\Processed"  # Same as in your main script
#
#     valid_dir = os.path.join(processed_dir, "VALID")
#     invalid_dir = os.path.join(processed_dir, "INVALID")
#
#     # Ensure these directories exist
#     os.makedirs(valid_dir, exist_ok=True)
#     os.makedirs(invalid_dir, exist_ok=True)
#
#     print("✅ Starting classification-only script for leftover folders...")
#
#     # Go through everything in DOWNLOADS
#     for order_folder in os.listdir(downloads_dir):
#         order_path = os.path.join(downloads_dir, order_folder)
#         # Skip if not a directory
#         if not os.path.isdir(order_path):
#             continue
#
#         # --------------------- CHECK FINAL FILES ---------------------
#         combined_ok = os.path.exists(os.path.join(order_path, "combined_clip.tif"))
#
#         # Must have at least two out of clipped_geo16, clipped_geo17, clipped_geo18
#         possible_goes = ["clipped_geo16.tif", "clipped_geo17.tif", "clipped_geo18.tif"]
#         goes_found = sum(os.path.exists(os.path.join(order_path, gf)) for gf in possible_goes)
#
#         # Build year_month subfolder based on folder name if it starts like YYYY-MM-DD
#         # We'll try safely splitting:
#         # e.g., "2024-11-01_12-00" -> "2024-11"
#         try:
#             year_month = order_folder.split('_')[0][:7]  # "2024-11"
#         except IndexError:
#             year_month = "UNKNOWN"
#
#         valid_subdir = os.path.join(valid_dir, year_month)
#         invalid_subdir = os.path.join(invalid_dir, year_month)
#         os.makedirs(valid_subdir, exist_ok=True)
#         os.makedirs(invalid_subdir, exist_ok=True)
#
#         # --------------------- VALID vs. INVALID ---------------------
#         if combined_ok and goes_found >= 2:
#             target_dir = valid_subdir
#             status = "VALID"
#         else:
#             target_dir = invalid_subdir
#             status = "INVALID"
#             # Log reason
#             log_invalid_order(order_folder, "Missing final required files (combined_clip or enough GOES images)")
#
#         # --------------------- MOVE FOLDER ---------------------
#         dest_path = os.path.join(target_dir, order_folder)
#         if os.path.exists(dest_path):
#             print(f"✅ Folder '{order_folder}' already in {status}, skipping.")
#         else:
#             try:
#                 shutil.move(order_path, dest_path)
#                 print(f"✅ Moved '{order_folder}' -> {status}")
#             except Exception as e:
#                 print(f"❌ Error moving '{order_folder}' to {target_dir}: {e}")
#
#     print("✅ Classification-only run complete. All leftover folders should be moved.")


import os
import shutil
import sys
import re
import gc
import geopandas as gpd
import rioxarray
from glob import glob


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python process_orders.py <downloads_folder_suffix>")
        sys.exit(1)

    num = sys.argv[1]

    # Set up directories based on provided num
    downloads_dir = fr"W:\Sat_Project\DOWNLOADS_{num}"
    processed_dir = r"W:\Sat_Project\Processed"
    valid_dir = os.path.join(processed_dir, "VALID")
    invalid_dir = os.path.join(processed_dir, "INVALID")

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    print(f"✅ Starting processing for DOWNLOADS_{num} folder...")

    # Load AOI shapefile
    AOI_PATH = r"shape_files/boundary_gdf.shp"
    AOI = gpd.read_file(AOI_PATH)

    order_results = {}

    for order_folder in os.listdir(downloads_dir):
        order_path = os.path.join(downloads_dir, order_folder)
        if not os.path.isdir(order_path):
            continue

        print(f"\n✅ Processing order folder: {order_folder}")
        success = True

        try:
            # ------------------ VIIRS PROCESSING ------------------
            viirs_files = glob(os.path.join(order_path, "*.nc"))
            if len(viirs_files) < 2:
                raise ValueError(f"❌ Not enough VIIRS files in {order_folder}")

            viirs_bound_poly = get_VIIRS_bounds_polygon(viirs_files[0])
            if not (AOI.crs == viirs_bound_poly.crs and viirs_bound_poly.contains(AOI).any()):
                raise ValueError(f"❌ AOI not contained within VIIRS data for {order_folder}")

            # Combine & clip VIIRS
            combine_viirs_images(order_path, order_path, AOI)

            # Get date/time from VIIRS for GOES downloads
            date_time = extract_datetime_from_filename(viirs_files[0])
            for sat in ["goes16", "goes17", "goes18"]:
                download_goes_data(date_time, order_path, sat)

            # Move newly created GOES subfolders
            for sat_folder in ["noaa-goes16", "noaa-goes17", "noaa-goes18"]:
                sat_path = os.path.join(order_path, sat_folder)
                if os.path.isdir(sat_path):
                    move_files_to_parent(sat_path, order_path)

            # ------------------ GOES PROCESSING ------------------
            goes_files = [f for f in os.listdir(order_path) if f.startswith("OR") and f.endswith(".nc")]
            if len(goes_files) < 2:
                raise ValueError("❌ Not enough GOES files found.")

            goes_files.sort(key=lambda x: "G16" not in x)
            reference_raster = None
            reference_AOI_geo = None

            for file in goes_files:
                file_path = os.path.join(order_path, file)
                print(f"📂 Processing GOES image: {file}")

                attempt_success = False
                for attempt in range(2):
                    try:
                        with rioxarray.open_rasterio(file_path, masked=True) as ds:
                            img = ds.squeeze()

                            if "G16" in file:
                                reference_raster = img.copy()
                                reference_AOI_geo = AOI.to_crs(img.rio.crs)

                            elif reference_raster is not None:
                                img = img.rio.reproject_match(reference_raster)

                            geo_clip = img.rio.clip(reference_AOI_geo.geometry, all_touched=True)

                            num_match = re.search(r"G(\d{2})", file)
                            num_goes = num_match.group(1) if num_match else "unknown"
                            output_tif = os.path.join(order_path, f"clipped_geo{num_goes}.tif")
                            geo_clip.rio.to_raster(output_tif)
                            print(f"✅ Clipped GOES {num_goes} saved.")

                        attempt_success = True
                        break

                    except Exception as read_err:
                        print(f"❌ Attempt {attempt + 1} failed for {file}: {read_err}")
                        if attempt == 0:
                            os.remove(file_path)
                            sat_name = "goes16" if "G16" in file else "goes17" if "G17" in file else "goes18"
                            download_goes_data(date_time, order_path, sat_name)
                        gc.collect()

                if not attempt_success:
                    success = False
                    print(f"❌ Failed processing {file} after re-download.")
                    break  # or continue, based on your preference

        except Exception as e:
            success = False
            print(f"❌ ERROR processing {order_folder}: {e}")

        order_results[order_folder] = success

    # ------------------ FINAL CLASSIFICATION ------------------
    print("\n✅ Classifying and moving order folders...")

    for order_folder, success in order_results.items():
        order_path = os.path.join(downloads_dir, order_folder)
        if not os.path.isdir(order_path):
            continue

        year_month = order_folder.split('_')[0][:7]
        valid_subdir = os.path.join(valid_dir, year_month)
        invalid_subdir = os.path.join(invalid_dir, year_month)
        os.makedirs(valid_subdir, exist_ok=True)
        os.makedirs(invalid_subdir, exist_ok=True)

        if success:
            combined_ok = os.path.exists(os.path.join(order_path, "combined_clip.tif"))
            possible_goes = ["clipped_geo16.tif", "clipped_geo17.tif", "clipped_geo18.tif"]
            goes_found = sum(os.path.exists(os.path.join(order_path, gf)) for gf in possible_goes)

            if combined_ok and goes_found >= 2:
                target_dir, status = valid_subdir, "VALID"
            else:
                target_dir, status = invalid_subdir, "INVALID"
                log_invalid_order(order_folder, "Missing final files (combined_clip or enough GOES images)")

        else:
            target_dir, status = invalid_subdir, "INVALID"
            log_invalid_order(order_folder, "Order marked as 'failed' during processing")

        dest_path = os.path.join(target_dir, order_folder)
        if os.path.exists(dest_path):
            print(f"⚠ {order_folder} already in {status}, skipping move.")
        else:
            shutil.move(order_path, dest_path)
            print(f"✅ Moved {order_folder} → {status}")

    print("✅ Processing complete. All folders classified and moved.")




