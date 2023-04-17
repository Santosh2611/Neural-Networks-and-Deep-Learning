import numpy as np

# Define input features and labels for the NOT logic gate
not_features = np.array([[0,1]])
not_labels = np.array([[1,0]])

# Define step function to be used in perceptron algorithm
def step_function(sums):
    # Return 1 if the input is greater than or equal to 0, otherwise return 0
    return np.where(sums >= 0, 1, 0)

# Initialize weights and bias values for perceptron algorithm
w = -1
bias = 0.5

# Print initialized weights and bias values
print("\nWeights:", w)
print("Bias:", bias)
print("\n")

# Implement perceptron algorithm using vectorized operations
def perceptron(features, labels, weights, bias):
    
    # Calculate the weighted sum of inputs and bias
    sums = np.dot(features, weights) + bias
    
    # Apply step function to each element of the sum
    predicted = step_function(sums)
    
    # Return the predicted output
    return predicted

# Use perceptron algorithm to calculate predicted output for given features and labels
predicted_output = perceptron(not_features, not_labels, w, bias)

# Print the input features, actual output, and predicted output
print("Input Features:", not_features)
print("Actual Output:", not_labels)
print("Predicted Output:", predicted_output)
