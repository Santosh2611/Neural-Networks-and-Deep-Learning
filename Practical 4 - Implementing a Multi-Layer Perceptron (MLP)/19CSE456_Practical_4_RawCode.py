import numpy as np
from sklearn.model_selection import train_test_split

# Define the input features and labels
features = np.array([[1], [0.7], [1.2]])
labels = np.array([1, 0, 1]).reshape(-1, 1)

# Define the learning rate
learning_rate = 0.05

# Define the number of iterations
num_iterations = 2000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Implement the perceptron algorithm using vectorized operations
def perceptron(w, features, labels):

    # Compute the weighted sum and activation function output for all features at once
    sum = np.dot(features, w.T)
    predicted = sigmoid(sum)

    delta = (predicted - labels) * predicted * (1 - predicted)

    return predicted, delta


# Train the perceptron using online learning
def train_perceptron(features, labels):
    # Initialize the weights randomly
    w = np.random.randn(1, features.shape[1])

    for i in range(num_iterations):
        # Randomly select one feature from the input array
        idx = np.random.choice(features.shape[0], size=1)
        x, y = features[idx], labels[idx]

        # Implement the perceptron algorithm using stochastic gradient descent
        predicted, delta = perceptron(w, x, y)
        grad = np.dot(x.T, delta)
        w -= learning_rate * grad

    return w


# Define the MLP architecture
input_layer_size = 1
hidden_layer_size = 3
output_layer_size = 1


# Implement the MLP algorithm using backpropagation
def mlp(w_1, w_2, features, labels):

    # Add a bias term to the input layer
    x = np.hstack((np.ones((features.shape[0], 1)), features))

    # Forward pass - calculate the predicted values for all input features at once
    hidden_layer_sum = np.dot(x, w_1)
    hidden_layer_output = sigmoid(hidden_layer_sum)

    output_layer_input = np.hstack((np.ones((hidden_layer_output.shape[0], 1)), hidden_layer_output))
    output_layer_sum = np.dot(output_layer_input, w_2)
    output_layer_output = sigmoid(output_layer_sum)

    error = np.mean(0.5 * (output_layer_output - labels) ** 2)

    # Backward pass - adjust the weights and biases
    delta_output = (output_layer_output - labels) * output_layer_output * (1 - output_layer_output)
    delta_hidden = (hidden_layer_output * (1 - hidden_layer_output)) * np.dot(delta_output, w_2.T[:, 1:])

    grad_output = np.dot(output_layer_input.T, delta_output)
    grad_hidden = np.dot(x.T, delta_hidden)

    w_2 -= learning_rate * grad_output
    w_1 -= learning_rate * grad_hidden

    return output_layer_output, error, w_1, w_2


# Train the MLP using backpropagation on the training set
def train_mlp(X_train, y_train):
    # Initialize the weights randomly
    w_1 = np.random.randn(input_layer_size + 1, hidden_layer_size)
    w_2 = np.random.randn(hidden_layer_size + 1, output_layer_size)

    for i in range(num_iterations):
        # Forward pass - calculate the predicted values for all input features at once
        output_layer_output, error, w_1, w_2 = mlp(w_1, w_2, X_train, y_train)

    return w_1, w_2


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the perceptron and MLP
w_p = train_perceptron(X_train, y_train)
w_1, w_2 = train_mlp(X_train, y_train)

# Evaluate the MLP on the testing set
output_layer_output, error, _, _ = mlp(w_1, w_2, X_test, y_test)
print("\nTest Set Predictions: ", output_layer_output)
print("Test Set Error: ", error)
