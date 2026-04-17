"""
create_dataset.py
-----------------
Creates training datasets from Google Earth Engine satellite data.
Implements the sample_points function for balanced coordinate selection
and get_training_example for retrieving (input, label) pairs.

Based on the professor's project specification:
- Inputs: Multi-band satellite imagery (5x5 pixel patches, 52 bands)
- Labels: Precipitation values at 2h and 6h in the future
"""

import ee
import numpy as np
import os
import datetime
import json
from weather_data import (
    initialize_gee,
    get_gpm,
    get_elevation,
    get_goes_data,
    get_precipitation_bins,
    get_elevation_bins,
    MAX_PRECIPITATION,
    MAX_ELEVATION,
    SCALE,
)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
NUM_BINS = 10           # Number of bins for stratified sampling
PATCH_SIZE = 5          # 5x5 pixel patches as training examples
NUM_POINTS = 500        # Number of sample points per date


def sample_points(date, num_bins=NUM_BINS):
    """
    Generates a balanced selection of (longitude, latitude) coordinates
    for a given date.

    This function:
    1. Gets precipitation bins for the date
    2. Gets elevation bins
    3. Combines them into a single "unique" bin value
    4. Uses ee.Image.stratifiedSample to select about the same
       number of spots for each combined bin class

    Args:
        date: datetime object or string 'YYYY-MM-DD'.
        num_bins: int, number of bins for both precipitation and elevation.

    Returns:
        Iterator of (date, point) tuples where point is a GEE geometry.
    """
    from weather_data import get_gpm

    if isinstance(date, str):
        date = ee.Date(date)

    # Get a representative GPM image for this date (use the mean composite)
    data = get_gpm(date).mean()

    # Create precipitation bins
    precipitation_bins = get_precipitation_bins(data, num_bins)

    # Create elevation bins
    elevation_bins = get_elevation_bins(num_bins)

    # Combine precipitation and elevation bins into a unique bin:
    # unique_bin = elevation_bins * num_bins + precipitation_bins
    # This gives us num_bins^2 possible classes
    unique_bins = (
        elevation_bins
        .multiply(num_bins)
        .add(precipitation_bins)
    )

    # Calculate how many points per class
    num_points_per_class = max(1, NUM_POINTS // (num_bins * num_bins))

    # Define a region of interest (global land areas)
    # Using a broad region; you can narrow this to your area of interest
    region = ee.Geometry.Polygon(
        [[[-180, -60], [-180, 60], [180, 60], [180, -60], [-180, -60]]]
    )

    # Stratified sampling to get balanced points across all bin classes
    points = unique_bins.stratifiedSample(
        numPoints=num_points_per_class,
        classBand='elevation',  # The band name after binning
        region=region,
        scale=SCALE,
        geometries=True
    )

    # Convert to a list and yield (date, point) tuples
    points_list = points.toList(points.size())
    size = points_list.size().getInfo()

    for i in range(size):
        point = ee.Feature(points_list.get(i)).geometry()
        yield (date, point)


def get_training_example(date, point):
    """
    Retrieves a pair of (inputs, labels) for the specified
    date and (longitude, latitude) coordinate.

    - Inputs: 5x5 pixel patch with all available satellite bands
    - Labels: Precipitation at +2h and +6h from the example's time

    Args:
        date: ee.Date object.
        point: ee.Geometry.Point.

    Returns:
        tuple: (inputs_array, labels_array) as numpy arrays,
               or None if data is unavailable.
    """
    try:
        # Get GOES satellite data as input (multi-spectral bands)
        goes_collection = get_goes_data(date)
        if goes_collection.size().getInfo() == 0:
            return None

        # Use the first available GOES image closest to the time
        goes_image = ee.Image(goes_collection.first())

        # Define a small region around the point for the 5x5 patch
        region = point.buffer(SCALE * PATCH_SIZE / 2).bounds()

        # Sample the input bands as a 5x5 patch
        input_data = goes_image.sampleRectangle(
            region=region,
            defaultValue=0
        )

        # Get all band arrays
        band_names = goes_image.bandNames().getInfo()
        input_arrays = []
        for band in band_names:
            arr = np.array(input_data.get(band).getInfo())
            input_arrays.append(arr)

        if len(input_arrays) == 0:
            return None

        # Stack bands: shape becomes (H, W, num_bands)
        inputs = np.stack(input_arrays, axis=-1).astype(np.float32)

        # --- Labels ---
        # Get precipitation at +2 hours
        date_plus_2h = date.advance(2, 'hour')
        gpm_2h = get_gpm(date_plus_2h).mean()
        precip_2h = (gpm_2h.select('precipitation')
                     .clamp(0, MAX_PRECIPITATION)
                     .sampleRectangle(region=region, defaultValue=0))

        # Get precipitation at +6 hours
        date_plus_6h = date.advance(6, 'hour')
        gpm_6h = get_gpm(date_plus_6h).mean()
        precip_6h = (gpm_6h.select('precipitation')
                     .clamp(0, MAX_PRECIPITATION)
                     .sampleRectangle(region=region, defaultValue=0))

        label_2h = np.array(precip_2h.get('precipitation').getInfo()).astype(np.float32)
        label_6h = np.array(precip_6h.get('precipitation').getInfo()).astype(np.float32)

        # Stack labels: shape (H, W, 2) — precipitation at 2h and 6h
        labels = np.stack([label_2h, label_6h], axis=-1)

        return inputs, labels

    except Exception as e:
        print(f"Error fetching training example: {e}")
        return None


def create_dataset(dates, output_dir='./data', max_points_per_date=50):
    """
    Creates a dataset by sampling points across multiple dates
    and saving the (input, label) pairs as .npz files.

    Args:
        dates: List of date strings in 'YYYY-MM-DD' format.
        output_dir: Directory to save the dataset files.
        max_points_per_date: Maximum number of points to sample per date.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_inputs = []
    all_labels = []
    total_examples = 0

    for date_str in dates:
        print(f"\n--- Processing date: {date_str} ---")
        date = ee.Date(date_str)

        count = 0
        for date_obj, point in sample_points(date_str):
            if count >= max_points_per_date:
                break

            result = get_training_example(date_obj, point)
            if result is not None:
                inputs, labels = result
                all_inputs.append(inputs)
                all_labels.append(labels)
                total_examples += 1
                count += 1
                print(f"  Collected example {count} | "
                      f"Input shape: {inputs.shape}, Label shape: {labels.shape}")

        print(f"  Total from {date_str}: {count} examples")

    if total_examples > 0:
        # Save as numpy arrays
        inputs_array = np.array(all_inputs)
        labels_array = np.array(all_labels)

        save_path = os.path.join(output_dir, 'weather_dataset.npz')
        np.savez(save_path, inputs=inputs_array, labels=labels_array)
        print(f"\nDataset saved to {save_path}")
        print(f"Total examples: {total_examples}")
        print(f"Inputs shape: {inputs_array.shape}")
        print(f"Labels shape: {labels_array.shape}")
    else:
        print("\nNo examples were collected. Check your GEE authentication and data availability.")


if __name__ == "__main__":
    initialize_gee()

    # Sample dates for dataset creation
    # Use dates where GPM and GOES data are available (2018 onwards for GOES-16)
    sample_dates = [
        '2023-06-01',
        '2023-06-15',
        '2023-07-01',
    ]

    print("Starting dataset creation...")
    print(f"Dates: {sample_dates}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Bins: {NUM_BINS}")

    create_dataset(
        dates=sample_dates,
        output_dir='./data',
        max_points_per_date=20  # Start small for testing
    )
