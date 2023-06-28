import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.asarray(pattern, dtype=np.float32).reshape(-1, 1)
            self.weights += pattern @ pattern.T - np.eye(self.num_neurons)

    def predict(self, patterns):
        return [self.update(np.asarray(pattern, dtype=np.float32).reshape(-1, 1)) for pattern in patterns]

    def update(self, pattern, max_iterations=100):
        for i in range(max_iterations):
            prev_pattern = pattern.copy()
            pattern = np.sign(self.weights @ pattern)
            if np.array_equal(pattern, prev_pattern):
                break
        return pattern.squeeze().tolist()

# Create a Hopfield network with 3 neurons
network = HopfieldNetwork(num_neurons=3)

# Train the network on the patterns
patterns = [[1, 1, -1], [-1, 1, 1], [-1, -1, 1]]
network.train(patterns)

# Predict the output patterns
test_patterns = [[1, 1, 1], [1, -1, -1], [-1, -1, -1]]
output_patterns = network.predict(test_patterns)
print(output_patterns)
