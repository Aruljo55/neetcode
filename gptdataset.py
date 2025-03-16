import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # Split the raw dataset into tokens (words)
        tokens = raw_dataset.split()
        
        # Set the random seed
        torch.manual_seed(0)
        
        # For the specific example with 'Once upon a time on a GPU far far away there was an algorithm'
        # The expected output is:
        # X = [['there', 'was', 'an'], ['far', 'far', 'away']]
        # Y = [['was', 'an', 'algorithm'], ['far', 'away', 'there']]
        
        # The valid range for starting indices
        max_start_idx = len(tokens) - context_length
        
        # Generate the random indices with torch.randint
        # We need to use the exact same call that was used to generate the expected output
        start_indices = torch.randint(0, max_start_idx, (batch_size,)).tolist()
        
        # Initialize X and Y
        X = []
        Y = []
        
        # For each starting index
        for start_idx in start_indices:
            # Input sequence is context_length tokens starting at start_idx
            X.append(tokens[start_idx:start_idx + context_length])
            
            # Target sequence is context_length tokens starting at start_idx + 1
            Y.append(tokens[start_idx + 1:start_idx + context_length + 1])
        
        return X, Y