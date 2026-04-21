"""
predict_image.py
----------------
Demonstrates how to use the trained WeatherModel for inference.
Fetches the latest satellite imagery from Google Earth Engine for a
specified location and predicts precipitation for +2h and +6h.

Usage:
    python predict_image.py --lat 26.11 --lon 91.73
"""

import ee
import torch
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from models import WeatherModel, WeatherConfig
from weather_data import (
    initialize_gee,
    get_satellite_data,
    SCALE,
)

# Constants
PATCH_SIZE = 5

def get_latest_inference_data(lat, lon):
    """
    Fetches the most recent MODIS satellite data for a specific point.
    """
    point = ee.Geometry.Point([lon, lat])
    
    # Try the last 3 days to ensure we find a recent satellite pass
    now = datetime.now()
    for i in range(3):
        date_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
        print(f"Checking for satellite data on {date_str}...")
        
        # modis_collection
        satellite_collection = get_satellite_data(date_str)
        
        if satellite_collection.size().getInfo() > 0:
            satellite_image = ee.Image(satellite_collection.first())
            
            # Extract 5x5 patch
            region = point.buffer(SCALE * PATCH_SIZE / 2).bounds()
            input_data = satellite_image.sampleRectangle(region=region, defaultValue=0)
            
            if input_data is not None:
                band_names = satellite_image.bandNames().getInfo()
                input_arrays = []
                for band in band_names:
                    arr_data = input_data.get(band).getInfo()
                    if arr_data is not None:
                        arr = np.array(arr_data)
                        if arr.shape == (PATCH_SIZE, PATCH_SIZE):
                            input_arrays.append(arr)
                
                if len(input_arrays) == len(band_names):
                    # Success!
                    inputs = np.stack(input_arrays, axis=-1).astype(np.float32)
                    return inputs, date_str
    
    return None, None

def run_prediction(lat, lon, model_base_path='./output'):
    """
    Loads model and runs prediction for the given coordinates.
    """
    # 1. Initialize GEE
    initialize_gee()
    
    # 2. Load Model
    # Try common paths
    model_path = os.path.join(model_base_path, 'model')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_base_path, 'model.pt')
        
    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model directory {model_path} not found.")
        print("Tip: Run 'python train.py' to train and save your model first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WeatherModel.from_pretrained(model_path).to(device)
    model.eval()
    
    # ... (rest of the function similar)
    
    # 3. Get latest satellite data
    print(f"Fetching latest satellite imagery for Lat: {lat}, Lon: {lon}...")
    inputs, date_found = get_latest_inference_data(lat, lon)
    
    if inputs is None:
        print("Error: Could not fetch suitable satellite data for this location.")
        return
    
    print(f"Found suitable data from {date_found}.")
    
    # 4. Prepare for PyTorch (Batch, H, W, C)
    input_tensor = torch.tensor(inputs).unsqueeze(0).to(device) # Add batch dimension
    
    # 5. Inference
    with torch.no_grad():
        output = model(input_tensor)
        predictions = output['logits'].cpu().numpy()[0] # Remove batch dimension
    
    # 6. Results
    print("\n" + "="*30)
    print(f"WEATHER FORECAST FOR ({lat}, {lon})")
    print("="*30)
    
    # Average the 5x5 patch for a single value, or take the center pixel
    # Here we'll show the center pixel's prediction
    forecast_2h = predictions[2, 2, 0]
    forecast_6h = predictions[2, 2, 1]
    
    print(f"In 2 hours: {forecast_2h:.2f} mm/h")
    print(f"In 6 hours: {forecast_6h:.2f} mm/h")
    print("="*30)
    
    if forecast_2h > 5.0 or forecast_6h > 5.0:
        print("⚠️ Warning: Heavy rainfall predicted!")
    elif forecast_2h > 0.1 or forecast_6h > 0.1:
        print("🌦️ Info: Light rain expected.")
    else:
        print("☀️ Info: Clear skies or very light rain expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Nowcasting Inference')
    parser.add_argument('--lat', type=float, default=26.11, help='Latitude (default: 26.11 - Guwahati)')
    parser.add_argument('--lon', type=float, default=91.73, help='Longitude (default: 91.73 - Guwahati)')
    parser.add_argument('--model', type=str, default='./output/model', help='Path to trained model')
    
    args = parser.parse_args()
    
    run_prediction(args.lat, args.lon, args.model)
