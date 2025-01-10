import numpy as np

class Solution:
    def get_model_prediction(self, X, weights):
        X = np.array(X)
        weights = np.array(weights)
        predictions = X @ weights
        return predictions.tolist()

    def train_model(self, X, Y, num_iterations, initial_weights, learning_rate=0.01):
        weights = np.array(initial_weights, dtype=float)
        X = np.array(X)
        Y = np.array(Y)

        for _ in range(num_iterations):
            # Get predictions
            predictions = self.get_model_prediction(X, weights)

            # Get derivatives (this function is assumed to be provided)
            derivatives = self.get_derivative(predictions, Y, X)

            # Update weights using gradient descent
            weights -= learning_rate * derivatives

        # Round weights to 5 decimal places
        return np.round(weights, 5)

    def get_derivative(self, predictions, Y, X):
        """
        Placeholder for the get_derivative function that calculates the gradient
        of the loss function with respect to the weights.
        """
        predictions = np.array(predictions)
        Y = np.array(Y)
        X = np.array(X)
        n = len(Y)
        derivatives = -2 / n * X.T @ (Y - predictions)
        return derivatives

# Example usage:
solution = Solution()

# Input example
X = [[1, 2, 3], [1, 1, 1]]
Y = [6, 3]
num_iterations = 10
initial_weights = [0.2, 0.1, 0.6]

# Train model
final_weights = solution.train_model(X, Y, num_iterations, initial_weights, learning_rate=0.01)
print(final_weights)  # Expected: [0.50678, 0.59057, 1.27435]
