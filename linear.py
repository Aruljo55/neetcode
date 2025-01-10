import numpy as np

class Solution:
    def get_model_prediction(self, X, weights):
        # Convert X and weights to numpy arrays for easier computation
        X = np.array(X)
        weights = np.array(weights)
        # Compute predictions using matrix multiplication
        predictions = X @ weights
        # Round the predictions to 5 decimal places
        return [round(p, 5) for p in predictions]

    def get_error(self, model_prediction, ground_truth):
        # Convert inputs to numpy arrays
        model_prediction = np.array(model_prediction)
        ground_truth = np.array(ground_truth)
        # Compute mean squared error
        error = np.mean((model_prediction - ground_truth) ** 2)
        # Round the error to 5 decimal places
        return round(error, 5)

# Example usage:
solution = Solution()

# Example inputs for get_model_prediction
X = [[0.3745401188473625, 0.9507143064099162, 0.7319939418114051]]
weights = [1.0, 2.0, 3.0]
model_prediction = solution.get_model_prediction(X, weights)
print(model_prediction)  # Expected: [4.47195]

# Example inputs for get_error
model_prediction = [0.37454012, 0.95071431, 0.73199394]
ground_truth = [0.59865848, 0.15601864, 0.15599452]
error = solution.get_error(model_prediction, ground_truth)
print(error)  # Expected: 0.33785
