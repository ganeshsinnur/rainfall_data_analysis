import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

class Normalization(nn.Module):
    """
    Applies Z-Score normalization to the input.
    """
    def __init__(self, mean, std):
        super().__init__()
        # Ensure mean and std are tensors
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
            
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class MoveDim(nn.Module):
    """
    Utility layer to move dimensions, e.g., for converting between
    channels-last and channels-first formats.
    """
    def __init__(self, source, destination):
        super().__init__()
        self.source = source
        self.destination = destination

    def forward(self, x):
        return x.movedim(self.source, self.destination)

class WeatherConfig(PretrainedConfig):
    model_type = "weather"

    def __init__(
        self,
        num_inputs=52,
        num_hidden1=64,
        num_hidden2=128,
        num_outputs=2,
        kernel_size=(3, 3),
        mean=[0.0] * 52,  # Placeholder mean
        std=[1.0] * 52,   # Placeholder std
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.mean = mean
        self.std = std

class WeatherModel(PreTrainedModel):
    config_class = WeatherConfig
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        
        self.layers = nn.Sequential(
            Normalization(config.mean, config.std),
            # Convert from channels-last (B, H, W, C) to channels-first (B, C, H, W)
            MoveDim(-1, 1),
            nn.Conv2d(
                in_channels=config.num_inputs,
                out_channels=config.num_hidden1,
                kernel_size=config.kernel_size,
                padding='same' # Padding same to keep dimensions if needed, 
                               # though image showed kernel size (3,3) without explicit padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.num_hidden1,
                out_channels=config.num_hidden2,
                kernel_size=config.kernel_size,
                padding='same'
            ),
            nn.ReLU(),
            # Convert back to channels-last (B, H, W, C) for the Linear layer
            MoveDim(1, -1),
            nn.Linear(config.num_hidden2, config.num_outputs),
            nn.ReLU()  # Precipitation cannot be negative
        )
        
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, inputs, labels=None):
        predictions = self.layers(inputs)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(predictions, labels)
            
        return {"loss": loss, "logits": predictions}

if __name__ == "__main__":
    # Quick verification
    config = WeatherConfig()
    model = WeatherModel(config)
    print("Model Architecture:")
    print(model)
    
    # Dummy forward pass (Batch, Height, Width, Channels)
    # Note: image text mentioned 5x5 window
    dummy_input = torch.randn(1, 5, 5, 52)
    output = model(dummy_input)
    print(f"\nForward pass successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output['logits'].shape}")
