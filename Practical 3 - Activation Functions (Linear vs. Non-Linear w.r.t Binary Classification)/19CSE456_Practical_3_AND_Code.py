import numpy as np

# Define the input features and labels for the AND logic gate
and_features = np.array([[0,0],[0,1],[1,0],[1,1]])
and_labels = np.array([0,0,0,1])

# Define the step function which returns 1 if the input is greater than or equal to 0, otherwise 0
def step_fun(sum):
    return np.where(sum >= 0, 1, 0)

# Initialize the weights and bias for the perceptron algorithm
w = np.array([1,1])
bias = -1.5

# Print the initialized weights and bias
print("\nWeight: ", w)
print("Bias: ", bias)
print("\n")

# Implement the perceptron algorithm using vectorized operations (instead of a nested loop) 
# Calculate the predicted values for all input features at once
def perceptron(features):
    sum = np.dot(features, w) + bias   # Find the weighted sum of inputs and bias
    predicted = step_fun(sum)          # Apply step function to the sum
    return predicted

# Use the perceptron function to find the predicted outputs for AND logic gate
predicted_and = perceptron(and_features)

# Print the actual and predicted outputs along with the corresponding weights for each input feature
for f, l, p in zip(and_features, and_labels, predicted_and):
    print(f[0], " and ", f[1], " -> Actual: ", l, "; Predicted: ", p, " ( w1: ", w[0], "& w2: ", w[1], ")")
