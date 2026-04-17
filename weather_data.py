"""
weather_data.py
---------------
Handles Google Earth Engine initialization and satellite data retrieval.
Uses GPM (Global Precipitation Measurement) for precipitation data
and SRTM for elevation data.
"""

import ee
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MAX_PRECIPITATION = 30  # mm/h — clamp precipitation values to this range
MAX_ELEVATION = 6000     # meters — clamp elevation values to this range
SCALE = 5000             # meters per pixel for sampling


def initialize_gee():
    """
    Initializes Google Earth Engine.
    Expects GEE_PROJECT_ID in .env or environment.
    """
    project_id = os.getenv("GEE_PROJECT_ID")
    try:
        if project_id:
            print(f"Initializing Earth Engine with project: {project_id}")
            ee.Initialize(project=project_id)
        else:
            print("Initializing Earth Engine (no project ID)...")
            ee.Initialize()
        print("Earth Engine successfully initialized.")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Tip: Run './venv/bin/earthengine authenticate' in your terminal.")
        raise


def get_gpm(date):
    """
    Retrieves GPM (Global Precipitation Measurement) half-hourly data
    for a specific date.

    Args:
        date: ee.Date object or string in 'YYYY-MM-DD' format.

    Returns:
        ee.ImageCollection of GPM precipitation data for that date.
    """
    if isinstance(date, str):
        date = ee.Date(date)

    start = date
    end = date.advance(1, 'day')

    gpm = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
           .filterDate(start, end))
    return gpm


def get_elevation():
    """
    Retrieves SRTM elevation data.

    Returns:
        ee.Image of elevation in meters.
    """
    return ee.Image('USGS/SRTMGL1_003').select('elevation')


def get_goes_data(date):
    """
    Retrieves GOES-16 satellite imagery bands for a specific date.
    GOES provides the multi-spectral bands used as model inputs.

    Args:
        date: ee.Date object or string in 'YYYY-MM-DD' format.

    Returns:
        ee.ImageCollection of GOES satellite data.
    """
    if isinstance(date, str):
        date = ee.Date(date)

    start = date
    end = date.advance(1, 'day')

    goes = (ee.ImageCollection('NOAA/GOES/16/MCMIPC')
            .filterDate(start, end))
    return goes


def get_precipitation_bins(data, num_bins):
    """
    Bins the precipitation values from GPM data.

    Precipitation is a continuous value; we clamp it to [0, MAX_PRECIPITATION],
    divide by MAX_PRECIPITATION, multiply by num_bins, and convert to integer
    to create discrete bin labels.

    Args:
        data: ee.Image with precipitation band.
        num_bins: int, number of bins.

    Returns:
        ee.Image with precipitation bin values (uint8).
    """
    precipitation_bins = (
        data.select('precipitation')
        .clamp(0, MAX_PRECIPITATION)
        .divide(MAX_PRECIPITATION)
        .multiply(num_bins - 1)
        .uint8()
    )
    return precipitation_bins


def get_elevation_bins(num_bins):
    """
    Bins the elevation values from SRTM.

    Elevation is clamped to [0, MAX_ELEVATION], divided, multiplied
    by num_bins, and converted to integer.

    Args:
        num_bins: int, number of bins.

    Returns:
        ee.Image with elevation bin values (uint8).
    """
    elevation_bins = (
        get_elevation()
        .clamp(0, MAX_ELEVATION)
        .divide(MAX_ELEVATION)
        .multiply(num_bins - 1)
        .uint8()
    )
    return elevation_bins


if __name__ == "__main__":
    initialize_gee()

    # Quick test: fetch GPM data for a sample date
    test_date = '2023-06-15'
    gpm_data = get_gpm(test_date)
    count = gpm_data.size().getInfo()
    print(f"GPM images found for {test_date}: {count}")

    # Test elevation
    elev = get_elevation()
    print(f"Elevation image bands: {elev.bandNames().getInfo()}")
