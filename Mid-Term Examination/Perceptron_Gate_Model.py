import numpy as np

# Define the AND gate inputs and expected outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])

# Define the initial weights and bias
weights = np.array([0.7, 0.6])
bias = 1

# Define the learning rate and number of epochs
learning_rate = 0.1
num_epochs = 2

# Define the activation function (step function)
def step_function(x):
    return np.where(x > 0, 1, 0)

# Define the forward pass function
def forward_pass(inputs, weights, bias):
    
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Apply the step function to the weighted sum to get the output
    output = step_function(weighted_sum)
    
    return output

# Define the backpropagation function
def backpropagation(inputs, weights, bias, outputs, learning_rate):
    
    # Calculate the predictions using the forward pass function
    predictions = forward_pass(inputs, weights, bias)
    
    # Calculate the error between predicted and actual outputs
    error = np.sum((predictions - outputs) ** 2)
    
    # Calculate the delta values for updating weights and bias
    delta = 2 * (outputs - predictions) * step_function(predictions)
    
    # Update the weights and bias based on delta values and learning rate
    weights += learning_rate * np.dot(inputs.T, delta)
    bias += learning_rate * np.sum(delta)
    
    return weights, bias, error

# Perform the forward pass with initial weights and bias
print("\nInitial predictions:")
for i in range(len(inputs)):
    output = forward_pass(inputs[i], weights, bias)
    print(inputs[i], "->", output)

# Perform the backpropagation for two epochs and update the weights and bias
for epoch in range(num_epochs):
    weights, bias, error = backpropagation(inputs, weights, bias, outputs, learning_rate)
    print("\nEpoch:", epoch + 1, "Error:", error)

# Perform the forward pass with updated weights and bias
print("\nUpdated predictions:")
for i in range(len(inputs)):
    output = forward_pass(inputs[i], weights, bias)
    print(inputs[i], "->", output)
