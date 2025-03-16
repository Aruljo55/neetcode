import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        
        # Define the architecture
        # Input layer is 28x28 = 784 features (flattened image)
        # First hidden layer has 512 neurons with ReLU activation
        # Dropout layer with p=0.2
        # Output layer has 10 neurons (one for each digit) with Sigmoid activation
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        
        # Reshape the input images to flatten them
        # If multiple images: [batch_size, 28*28]
        # If single image: [1, 28*28]
        if len(images.shape) == 1:
            # Single image case
            x = images.view(1, -1)
        else:
            # Multiple images case
            x = images.view(images.shape[0], -1)
        
        # Forward pass through the model
        output = self.model(x)
        
        # Round to 4 decimal places
        return torch.round(output * 10000) / 10000