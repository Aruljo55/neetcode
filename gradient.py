class Solution:
    def get_minimizer(self, iterations, learning_rate, init):
        x = init
        for _ in range(iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient
        return round(x, 5)

# Example usage:
solution = Solution()

# Example 1
print(solution.get_minimizer(0, 0.01, 5))  # Output: 5

# Example 2
print(solution.get_minimizer(10, 0.01, 5))  # Output: 4.08536
