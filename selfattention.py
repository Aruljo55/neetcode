import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Initialize Key, Query, and Value linear transformations (without bias)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Get shapes
        batch_size, context_length, embedding_dim = embedded.shape
        
        # Apply linear transformations
        k = self.key(embedded)   # (batch_size, context_length, attention_dim)
        q = self.query(embedded) # (batch_size, context_length, attention_dim)
        v = self.value(embedded) # (batch_size, context_length, attention_dim)
        
        # Transpose k for matrix multiplication
        k_t = torch.transpose(k, 1, 2)  # (batch_size, attention_dim, context_length)
        
        # Calculate attention scores
        # (batch_size, context_length, attention_dim) @ (batch_size, attention_dim, context_length)
        # = (batch_size, context_length, context_length)
        scores = q @ k_t
        
        # Scale the scores (divide by sqrt of attention_dim)
        scores = scores / (k.shape[-1] ** 0.5)
        
        # Create mask to prevent attending to future tokens
        # Lower triangular matrix of ones (includes the diagonal)
        mask = torch.tril(torch.ones(context_length, context_length))
        
        # Move mask to the same device as scores
        mask = mask.to(scores.device)
        
        # Apply the mask by setting future positions to -infinity
        # This ensures future tokens get 0 weight after softmax
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)  # (batch_size, context_length, context_length)
        
        # Apply attention weights to values
        # (batch_size, context_length, context_length) @ (batch_size, context_length, attention_dim)
        # = (batch_size, context_length, attention_dim)
        output = weights @ v
        
        # Round to 4 decimal places
        return torch.round(output * 10000) / 10000