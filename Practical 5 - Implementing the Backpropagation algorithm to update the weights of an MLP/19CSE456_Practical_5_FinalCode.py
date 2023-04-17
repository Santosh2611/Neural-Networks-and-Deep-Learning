import numpy as np

# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# Define propagation functions with activation parameter
def forward_propagation(X, W1, b1, W2, b2, activation=sigmoid):
    Z1 = np.dot(X, W1) + b1
    A1 = activation(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = activation(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def backward_propagation(X, Y, cache, W1, b1, W2, b2, learning_rate=0.1, activation_derivative=sigmoid_derivative):
    m = X.shape[0]

    # Retrieve cached data from forward propagation
    A1 = cache['A1']
    A2 = cache['A2']

    # Compute gradients
    dZ2 = (A2 - Y) * activation_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = np.dot(dZ2, W2.T) * activation_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update weights and biases using the learning rate and gradients
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2

    return W1_new, b1_new, W2_new, b2_new

# Define the training function with hidden_size parameter
def train(X, Y, hidden_size=4, iterations=10000, print_loss=False, activation=sigmoid, activation_derivative=sigmoid_derivative):
    input_size = X.shape[1] # Number of input nodes
    output_size = Y.shape[1] # Number of output nodes

    # Initialize the weights and biases
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    for i in range(iterations):
        # Forward propagation
        output, cache = forward_propagation(X, W1, b1, W2, b2, activation)

        # Backward propagation
        W1_new, b1_new, W2_new, b2_new = backward_propagation(X, Y, cache, W1, b1, W2, b2, activation_derivative=activation_derivative)

        # Print loss every 1000 iterations
        if print_loss and i % 1000 == 0:
            print("Loss after iteration ", i, ": ", np.mean(np.abs(output - Y)))

        # Update weights and biases
        W1, b1, W2, b2 = W1_new, b1_new, W2_new, b2_new

    return W1, b1, W2, b2

# Test the model with example data
X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

Y = np.array([[0, 1],
             [1, 0],
             [1, 0],
             [0, 1]])

W1, b1, W2, b2 = train(X, Y, iterations=10000, print_loss=True)

# Predict the output using trained weights and biases
output, _ = forward_propagation(X, W1, b1, W2, b2)
print("\nOutput: \n", output)
