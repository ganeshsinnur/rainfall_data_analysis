import ee
import os
from dotenv import load_dotenv

load_dotenv()

def debug():
    project_id = os.getenv("GEE_PROJECT_ID")
    ee.Initialize(project=project_id)
    
    date_str = '2023-06-01'
    date = ee.Date(date_str)
    start = date
    end = date.advance(1, 'day')
    
    # 1. Check GPM
    gpm = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
           .filterDate(start, end)
           .select('precipitation'))
    
    gpm_count = gpm.size().getInfo()
    print(f"GPM images for {date_str}: {gpm_count}")
    
    if gpm_count > 0:
        gpm_image = gpm.mean()
        # Check if it has any non-masked pixels in a small sample
        sample = gpm_image.sample(region=ee.Geometry.Point([-70, -10]), scale=5000, numPixels=1).size().getInfo()
        print(f"GPM sample point exists: {sample > 0}")

    # 2. Check SRTM
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
    print(f"SRTM bands: {srtm.bandNames().getInfo()}")
    
    # 3. Check GOES
    goes = (ee.ImageCollection('NOAA/GOES/16/MCMIPC')
            .filterDate(start, end))
    goes_count = goes.size().getInfo()
    print(f"GOES images for {date_str}: {goes_count}")

    # 4. Check sampling region - try a smaller one (South America)
    region = ee.Geometry.Rectangle([-100, -50, -30, 10])
    
    # Check if a simple sample works
    try:
        test_sample = srtm.sample(region=region, scale=5000, numPixels=10, geometries=True)
        print(f"SRTM small region sample success: {test_sample.size().getInfo()} points")
    except Exception as e:
        print(f"SRTM sample failed: {e}")

    # Let's also check the GPM image values in this region
    gpm_val = gpm.mean().reduceRegion(ee.Reducer.mean(), region, 10000).getInfo()
    print(f"GPM mean in region: {gpm_val}")

if __name__ == "__main__":
    debug()
