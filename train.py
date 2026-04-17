"""
train.py
--------
Training script for the WeatherModel.

Uses the dataset created by create_dataset.py to train the CNN
for rainfall nowcasting. Implements:
- Dataset loading from .npz files
- Train/test split
- Training loop with SmoothL1Loss
- Model saving via HuggingFace's save_pretrained
- Basic evaluation and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from models import WeatherConfig, WeatherModel

load_dotenv()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8       # 80% train, 20% test
DATA_PATH = os.getenv('DATA_DIR', './data')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', './output/model')


class WeatherDataset(Dataset):
    """
    PyTorch Dataset for weather nowcasting.
    Loads (inputs, labels) from a .npz file created by create_dataset.py.
    """
    def __init__(self, data_path):
        npz_file = os.path.join(data_path, 'weather_dataset.npz')
        if not os.path.exists(npz_file):
            raise FileNotFoundError(
                f"Dataset file not found at {npz_file}. "
                f"Run create_dataset.py first."
            )

        data = np.load(npz_file)
        self.inputs = torch.tensor(data['inputs'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)

        print(f"Loaded dataset:")
        print(f"  Inputs shape: {self.inputs.shape}")
        print(f"  Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def compute_mean_std(dataset):
    """
    Computes the per-band mean and standard deviation of the input data.
    These are needed for the Normalization layer in the model.

    Args:
        dataset: WeatherDataset instance.

    Returns:
        tuple: (mean_list, std_list) — lists of floats, one per input band.
    """
    all_inputs = dataset.inputs  # Shape: (N, H, W, C)
    # Compute mean and std across all samples and spatial dimensions
    mean = all_inputs.mean(dim=(0, 1, 2))  # Shape: (C,)
    std = all_inputs.std(dim=(0, 1, 2))    # Shape: (C,)

    # Prevent division by zero
    std = torch.where(std == 0, torch.ones_like(std), std)

    return mean.tolist(), std.tolist()


def train(model, train_loader, optimizer, device, epoch):
    """
    Runs one training epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(inputs, labels=labels)
        loss = output['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch:3d} | Train Loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, test_loader, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs, labels=labels)
            total_loss += output['loss'].item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def predict_batch(model, inputs_batch, device):
    """
    Runs prediction on a batch of inputs.

    Args:
        model: Trained WeatherModel.
        inputs_batch: Tensor of shape (B, H, W, C).
        device: torch device.

    Returns:
        np.ndarray of predictions.
    """
    model.eval()
    with torch.no_grad():
        inputs_batch = inputs_batch.to(device)
        outputs = model(inputs_batch)
        predictions = outputs['logits'].cpu().numpy()
    return predictions


def visualize_predictions(model, test_loader, device, num_samples=3):
    """
    Visualizes model predictions vs ground truth labels.
    Shows the predicted and actual precipitation maps side by side.
    """
    model.eval()
    inputs, labels = next(iter(test_loader))
    inputs = inputs[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs['logits'].cpu().numpy()

    labels = labels.numpy()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Our output (2h prediction)
        axes[i, 0].imshow(predictions[i, :, :, 0], cmap='Blues', vmin=0, vmax=30)
        axes[i, 0].set_title(f'Predicted (2h)')
        axes[i, 0].axis('off')

        # Ground truth (2h)
        axes[i, 1].imshow(labels[i, :, :, 0], cmap='Blues', vmin=0, vmax=30)
        axes[i, 1].set_title(f'Ground Truth (2h)')
        axes[i, 1].axis('off')

        # Our output (6h prediction)
        axes[i, 2].imshow(predictions[i, :, :, 1], cmap='Blues', vmin=0, vmax=30)
        axes[i, 2].set_title(f'Predicted (6h)')
        axes[i, 2].axis('off')

        # Ground truth (6h)
        axes[i, 3].imshow(labels[i, :, :, 1], cmap='Blues', vmin=0, vmax=30)
        axes[i, 3].set_title(f'Ground Truth (6h)')
        axes[i, 3].axis('off')

    plt.tight_layout()

    os.makedirs('./output', exist_ok=True)
    save_path = './output/predictions.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Predictions visualization saved to {save_path}")


def main():
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = WeatherDataset(DATA_PATH)

    # Compute normalization parameters from training data
    print("Computing normalization parameters...")
    mean, std = compute_mean_std(dataset)
    num_inputs = dataset.inputs.shape[-1]  # Number of bands
    print(f"  Number of input bands: {num_inputs}")

    # Train/test split
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"  Train: {train_size} examples, Test: {test_size} examples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model with computed mean/std
    config = WeatherConfig(
        num_inputs=num_inputs,
        mean=mean,
        std=std,
    )
    model = WeatherModel(config).to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("-" * 50)

    train_losses = []
    test_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        test_loss = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Test Loss:  {test_loss:.6f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or '.', exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='#2196F3')
    plt.plot(test_losses, label='Test Loss', color='#FF5722')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('./output', exist_ok=True)
    plt.savefig('./output/training_curve.png', dpi=150)
    plt.close()
    print("Training curve saved to ./output/training_curve.png")

    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, device)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
