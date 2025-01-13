import torch

class Solution:
    def reshape(self, to_reshape):
        # Convert input list to a tensor
        to_reshape = torch.tensor(to_reshape)
        m, n = to_reshape.shape
        # Reshape and convert back to a Python list
        reshaped = to_reshape.view((m * n) // 2, 2).tolist()
        return reshaped
