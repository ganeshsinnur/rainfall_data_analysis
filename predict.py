"""
predict.py
----------
Inference script for the trained WeatherModel.

Takes a user-provided image (PNG/JPG), converts it into the model's
expected input format, runs prediction, and outputs the forecasted
precipitation at +2h and +6h as both numbers and a visualization.

Usage:
    python predict.py --image path/to/image.png
    python predict.py --image path/to/image.png --model-path ./output/model
"""

import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from models import WeatherConfig, WeatherModel


PATCH_SIZE = 5
MODEL_PATH = './output/model.pt'


def load_image(image_path, num_bands):
    """
    Loads an image file and converts it to the model's expected input format.

    The image is:
    1. Resized to PATCH_SIZE x PATCH_SIZE
    2. Converted to float values scaled to satellite-like range [180-320]
    3. Expanded to match the number of input bands the model expects

    Args:
        image_path: Path to the input image file.
        num_bands: Number of spectral bands the model expects.

    Returns:
        np.ndarray of shape (PATCH_SIZE, PATCH_SIZE, num_bands).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)  # Shape: (5, 5, 3)

    # Scale RGB [0-255] to satellite brightness temperature range [180-320]
    # This maps dark pixels (clouds) to lower temps and bright pixels to higher temps
    img_scaled = 180 + (img_array / 255.0) * 140  # Range: [180, 320]

    # Expand 3 RGB channels to num_bands by repeating and adding slight variations
    bands = []
    for b in range(num_bands):
        # Cycle through RGB channels with slight noise to create band variation
        base_channel = img_scaled[:, :, b % 3]
        noise = np.random.normal(0, 2, base_channel.shape).astype(np.float32)
        bands.append(base_channel + noise)

    return np.stack(bands, axis=-1).astype(np.float32)


def predict(model, image_array, device):
    """
    Runs the model on a single image array.

    Args:
        model: Trained WeatherModel.
        image_array: np.ndarray of shape (H, W, C).
        device: torch device.

    Returns:
        np.ndarray of predictions, shape (H, W, 2).
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # (1, H, W, C)
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        prediction = output['logits'].cpu().numpy()[0]  # (H, W, 2)
    return prediction


def visualize_result(image_path, prediction, output_path):
    """
    Creates a visualization showing the input image alongside
    the predicted precipitation heatmaps.
    """
    img = Image.open(image_path).convert('RGB')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original input image
    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Predicted precipitation at +2h
    im1 = axes[1].imshow(prediction[:, :, 0], cmap='Blues', vmin=0, vmax=30,
                          interpolation='nearest')
    axes[1].set_title('Predicted Rainfall (+2h)\nmm/h', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted precipitation at +6h
    im2 = axes[2].imshow(prediction[:, :, 1], cmap='Blues', vmin=0, vmax=30,
                          interpolation='nearest')
    axes[2].set_title('Predicted Rainfall (+6h)\nmm/h', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle('Rainfall Nowcasting Prediction', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict rainfall from satellite imagery using trained WeatherModel'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image (PNG/JPG)')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help=f'Path to trained model directory (default: {MODEL_PATH})')
    parser.add_argument('--output', type=str, default='./output/prediction_result.png',
                        help='Path to save the visualization')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image not found at '{args.image}'")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'")
        print("Tip: Run 'python train.py' first to train the model.")
        return

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = WeatherModel.from_pretrained(args.model_path)
    model = model.to(device)
    num_bands = model.config.num_inputs
    print(f"  Model expects {num_bands} input bands")

    # Load and process image
    print(f"Processing image: {args.image}")
    image_array = load_image(args.image, num_bands)
    print(f"  Input tensor shape: {image_array.shape}")

    # Run prediction
    print("Running prediction...")
    prediction = predict(model, image_array, device)

    # Print results
    precip_2h_mean = prediction[:, :, 0].mean()
    precip_6h_mean = prediction[:, :, 1].mean()
    precip_2h_max = prediction[:, :, 0].max()
    precip_6h_max = prediction[:, :, 1].max()

    print(f"\n{'='*50}")
    print(f"  RAINFALL PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"  +2h Forecast:")
    print(f"    Average: {precip_2h_mean:.2f} mm/h")
    print(f"    Peak:    {precip_2h_max:.2f} mm/h")
    print(f"  +6h Forecast:")
    print(f"    Average: {precip_6h_mean:.2f} mm/h")
    print(f"    Peak:    {precip_6h_max:.2f} mm/h")
    print(f"{'='*50}")

    # Interpretation
    max_precip = max(precip_2h_mean, precip_6h_mean)
    if max_precip < 1:
        print("  Outlook: ☀️  Clear / No significant rainfall expected")
    elif max_precip < 5:
        print("  Outlook: 🌦️  Light rain expected")
    elif max_precip < 15:
        print("  Outlook: 🌧️  Moderate to heavy rain expected")
    else:
        print("  Outlook: ⛈️  Heavy rainfall / Storm conditions")

    # Save visualization
    visualize_result(args.image, prediction, args.output)

    print("\n✅ Prediction complete!")


if __name__ == "__main__":
    main()
