import numpy as np

# Define the mean squared error cost function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Define batch gradient descent function
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = X.shape[0] # Number of training examples
    n = X.shape[1] # Number of features
    theta = np.zeros(n) # Initialize parameters

    for i in range(iterations):
        # Calculate predicted values using current parameters
        y_pred = np.dot(X, theta)
        
        # Calculate gradients and update parameters
        gradient = np.dot((y_pred-y), X) / m
        theta -= learning_rate * gradient

    return theta

# Define mini-batch gradient descent function
def minibatch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, iterations=1000):
    m = X.shape[0] # Number of training examples
    n = X.shape[1] # Number of features
    theta = np.zeros(n) # Initialize parameters

    indices = np.arange(m)

    for i in range(iterations):
        # Shuffle the data to create the mini-batches
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Split the shuffled data into batches of size batch_size
        batches_X = np.array_split(X_shuffled, m // batch_size)
        batches_y = np.array_split(y_shuffled, m // batch_size)

        # Loop through the mini-batches
        for X_batch, y_batch in zip(batches_X, batches_y):
            # Calculate predicted values
            y_pred = np.dot(X_batch, theta)

            # Calculate gradients and update parameters
            gradient = np.dot((y_pred-y_batch), X_batch) / batch_size
            theta -= learning_rate * gradient

    return theta

# Generate example data
m = 10000
n = 10
X = np.random.randn(m, n)
theta_true = np.random.randn(n)
y = np.dot(X, theta_true)

# Run batch gradient descent and calculate error
theta_batch = batch_gradient_descent(X, y)
y_pred_batch = np.dot(X, theta_batch)
mse_batch = mse(y, y_pred_batch)
print("MSE with batch gradient descent: ", mse_batch)

# Run mini-batch gradient descent and calculate error
theta_minibatch = minibatch_gradient_descent(X, y)
y_pred_minibatch = np.dot(X, theta_minibatch)
mse_minibatch = mse(y, y_pred_minibatch)
print("MSE with mini-batch gradient descent: ", mse_minibatch)
