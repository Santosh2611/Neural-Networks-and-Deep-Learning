import numpy as np

# Define the input features and labels for the logic gate
or_features = np.array([[0,0],[0,1],[1,0],[1,1]])
or_labels = np.array([0,1,1,1])

# Define the step function that returns 1 if the input is greater than or equal to 0, else 0
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Implement the perceptron algorithm using vectorized operations (instead of a nested loop) - Calculate the predicted values for all input features at once
def perceptron(features, labels, weights, bias_value, operator):

    # Calculate the weighted sum of inputs and apply the bias
    z = np.dot(features, weights) + bias_value

    # Apply the step function to get the predicted values
    predicted = step_function(z)

    # Print the results for each input pair
    for x_input, y_output, prediction in zip(features, labels, predicted):
        print(f"{x_input[0]} {operator} {x_input[1]} -> Actual: {y_output}; Predicted: {prediction} (Weights: {weights[0]} & {weights[1]}, Bias: {bias_value})")

# Initialize the weights and bias for the perceptron algorithm
weights = np.array([1, 1])
bias_value = -0.5

# Print the weight and bias values
print("Weights: ", weights)
print("Bias: ", bias_value)

# Call the perceptron function with input features, labels, weights, bias, and operator type
perceptron(or_features, or_labels, weights, bias_value, "or")
