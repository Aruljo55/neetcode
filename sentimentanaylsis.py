import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Create embedding layer of size 16
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=16,
            padding_idx=0  # Use 0 as padding
        )
        
        # Create the final linear layer (16 -> 1)
        self.classifier = nn.Linear(16, 1)
        
        # Sigmoid activation for output between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        
        # Create a mask for padding (0 values)
        mask = (x != 0).float().unsqueeze(-1)  # Shape: B, T, 1
        
        # Apply embedding layer
        embedded = self.embedding(x)  # Shape: B, T, 16
        
        # Apply mask to handle padding correctly during averaging
        masked_embedded = embedded * mask
        
        # Sum all word embeddings and divide by number of non-padding tokens
        # Small epsilon to avoid division by zero
        eps = 1e-9
        sum_embeddings = masked_embedded.sum(dim=1)  # Shape: B, 16
        count_tokens = mask.sum(dim=1)  # Shape: B, 1
        avg_embeddings = sum_embeddings / (count_tokens + eps)  # Shape: B, 16
        
        # Apply linear layer
        logits = self.classifier(avg_embeddings)  # Shape: B, 1
        
        # Apply sigmoid activation
        predictions = self.sigmoid(logits)  # Shape: B, 1
        
        # Round to 4 decimal places
        return torch.round(predictions * 10000) / 10000