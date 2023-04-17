# Import numpy library
import numpy as np

# Define the sigmoid activation function using numpy's built-in exp() function
def sigmoid(x):
    """
    Calculate the sigmoid function given an input x.
    Args:
        x: Input value(s) to feed into the sigmoid function.
    Returns:
        Result of the sigmoid function applied to the input x.
    """
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sigmoid_derivative(x):
    """
    Calculate the derivative of the sigmoid function given an input x.
    Args:
        x: Input value(s) to feed into the derivative of sigmoid function.
    Returns:
        Result of the derivative of the sigmoid function applied to the input x.
    """
    return sigmoid(x) * (1 - sigmoid(x))

# Define the learning rate
learning_rate = 0.1

# Define the input data
X = np.array([[-0.3, -1.5], [-1.7, 0.7]])

# Define the output data
Y = np.array([[0.1], [-0.3]])

# Define the weight and bias vectors for each layer of the neural network using a dictionary
params = {
    'W1': np.array([[-0.2, -1.4, 0.6], [-0.1, 0.9, 1.2]]),
    'B1': np.array([[-1.4], [-0.8], [0.2]]),
    'W2': np.array([[0.2], [-1.1], [0.5]]),
    'B2': np.array([[-0.6]])
}

# Define batch size for batch processing
batch_size = 1

# Perform the feed-forward pass to get predictions
Z1 = np.dot(X, params['W1']) + params['B1'].T
A1 = sigmoid(Z1)
Z2 = np.dot(A1, params['W2']) + params['B2']
Y_hat = sigmoid(Z2)

# Calculate the mean squared error loss function
error = Y - Y_hat
mse_loss = np.mean(error ** 2)

# Perform backpropagation and weight-bias adjustments to optimize the neural network using batch processing
num_samples = X.shape[0]
for i in range(0, num_samples, batch_size):
    
    # Compute gradients
    dZ2 = error[i:i+batch_size] * sigmoid_derivative(Z2[i:i+batch_size])
    dW2 = np.dot(A1[i:i+batch_size].T, dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, params['W2'].T) * sigmoid_derivative(Z1[i:i+batch_size])
    dW1 = np.dot(X[i:i+batch_size].T, dZ1)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Update weights and biases
    params['W2'] -= learning_rate * dW2
    params['B2'] -= learning_rate * dB2
    params['W1'] -= learning_rate * dW1
    params['B1'] -= learning_rate * dB1.T

# Report the value of the loss function and the adjusted values of weight and bias vectors
print("\nMean Square Error Loss: ", mse_loss)
print("\nAdjusted W1: \n", params['W1'])
print("\nAdjusted B1: \n", params['B1'])
print("\nAdjusted W2: \n", params['W2'])
print("\nAdjusted B2: \n", params['B2'])
