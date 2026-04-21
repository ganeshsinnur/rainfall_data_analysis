"""
create_dataset.py
-----------------
Creates training datasets from Google Earth Engine satellite data.
Implements the sample_points function for balanced coordinate selection
and get_training_example for retrieving (input, label) pairs.

Based on the professor's project specification:
- Inputs: Multi-band satellite imagery (5x5 pixel patches, 7 MODIS bands)
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
    get_satellite_data,
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

    # Combine bins into a unique class band.
    # Rename explicitly so classBand is always known regardless of GEE's
    # internal band-name propagation rules after arithmetic.
    unique_bins = (
        elevation_bins
        .multiply(num_bins)
        .add(precipitation_bins)
        .rename('class_bin')
        .uint8()
    )

    # Calculate how many points per class
    num_points_per_class = max(1, NUM_POINTS // (num_bins * num_bins))

    # Define a region of interest covering India
    # Longitude: ~68°E to ~97°E, Latitude: ~6°N to ~37°N
    region = ee.Geometry.Rectangle([68, 6, 97, 37])

    # Stratified sampling to get balanced points across all bin classes
    points = unique_bins.stratifiedSample(
        numPoints=num_points_per_class,
        classBand='class_bin',
        region=region,
        scale=SCALE,
        geometries=True
    )

    # Guard: if stratified sampling returned nothing, fall back to random points
    size = points.size().getInfo()
    print(f"  Stratified sample returned {size} points.")

    if size == 0:
        print("  Warning: stratified sample empty, falling back to random sampling.")
        points = unique_bins.sample(
            region=region,
            scale=SCALE,
            numPixels=NUM_POINTS,
            geometries=True
        )
        size = points.size().getInfo()
        print(f"  Random sample returned {size} points.")

    if size == 0:
        print("  No points found for this date, skipping.")
        return

    # Convert to a list and yield (date, point) tuples
    points_list = points.toList(size)

    for i in range(size):
        point = ee.Feature(points_list.get(i)).geometry()
        yield (date, point)


def get_training_example(date, point):
    """
    Retrieves a pair of (inputs, labels) for the specified
    date and (longitude, latitude) coordinate.

    - Inputs: 5x5 pixel patch with all available satellite bands
    - Labels: Precipitation at +2h and +6h from the example's time
    """
    try:
        # Get MODIS satellite data as input
        satellite_collection = get_satellite_data(date)
        if satellite_collection.size().getInfo() == 0:
            return None

        satellite_image = ee.Image(satellite_collection.first())
        band_names = satellite_image.bandNames().getInfo()

        # Prepare future labels
        date_plus_2h = date.advance(2, 'hour')
        gpm_2h = get_gpm(date_plus_2h).mean().select(['precipitation'], ['precip_2h']).clamp(0, MAX_PRECIPITATION)
        
        date_plus_6h = date.advance(6, 'hour')
        gpm_6h = get_gpm(date_plus_6h).mean().select(['precipitation'], ['precip_6h']).clamp(0, MAX_PRECIPITATION)

        # Combine into one image to fetch EVERYTHING in one single network request
        combined_image = satellite_image.addBands(gpm_2h).addBands(gpm_6h)

        # Define region for the 5x5 patch
        region = point.buffer(SCALE * PATCH_SIZE / 2).bounds()

        # Force the image to the correct scale (5000m) before sampling
        # This ensures we get exactly 5x5 pixels for our region
        reprojected_image = combined_image.reproject(
            crs='EPSG:4326', 
            scale=SCALE
        )

        # ONE network request for all bands and labels
        info = reprojected_image.sampleRectangle(region=region, defaultValue=0).getInfo()

        if not info:
            return None

        # Extract input bands
        input_arrays = []
        for band in band_names:
            arr = np.array(info.get(band))
            if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                return None
            input_arrays.append(arr)
        
        inputs = np.stack(input_arrays, axis=-1).astype(np.float32)

        # Extract labels
        label_2h = np.array(info.get('precip_2h')).astype(np.float32)
        label_6h = np.array(info.get('precip_6h')).astype(np.float32)
        
        if label_2h.shape != (PATCH_SIZE, PATCH_SIZE) or label_6h.shape != (PATCH_SIZE, PATCH_SIZE):
            return None

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

            print(f"  Collecting example {count + 1}/{max_points_per_date}...", end='\r')
            result = get_training_example(date_obj, point)
            if result is not None:
                inputs, labels = result
                all_inputs.append(inputs)
                all_labels.append(labels)
                total_examples += 1
                count += 1
                # print(f"  Collected example {count} | "
                #       f"Input shape: {inputs.shape}, Label shape: {labels.shape}")

        print(f"\n  Total from {date_str}: {count} examples")

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

    # Sample dates during the Indian monsoon season (Jun–Sep)
    # MODIS data is available from 2000 onwards, GPM from 2000 onwards
    sample_dates = [
        '2023-07-01',   # Start of peak monsoon
        '2023-07-15',   # Mid-July heavy rains
        '2023-08-01',   # Peak monsoon month
        '2023-08-15',   # Continued heavy rains
        '2023-09-01',   # Late monsoon / retreating monsoon
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
