import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        Class constructor that initializes the MLP object with input_size, hidden_size, output_size, and learning_rate parameters.
        It also initializes weights and biases for both the hidden and output layers.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.h_weights = np.random.randn(input_size, hidden_size)
        self.h_bias = np.random.randn(hidden_size)
        self.o_weights = np.random.randn(hidden_size, output_size)
        self.o_bias = np.random.randn(output_size)

    def linear(self, x):
        """
        Activation function that returns the input without any transformation.
        """
        return x
    
    def step(self, x):
        """
        Activation function that returns 1 if input is greater or equal to 0, else returns 0.
        """
        return np.where(x>=0, 1, 0)
    
    def sigmoid(self, x):
        """
        Activation function that returns the sigmoid of the input.
        """
        return 1/(1+np.exp(-x))
    
    def tanh(self, x):
        """
        Activation function that returns the hyperbolic tangent of the input.
        """
        return np.tanh(x)
    
    def relu(self, x):
        """
        Activation function that returns the ReLU (Rectified Linear Unit) of the input.
        """
        return np.maximum(0, x)
    
    def elu(self, x, alpha=1.0):
        """
        Activation function that returns the ELU (Exponential Linear Unit) of the input.
        """
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    def softmax(self, x):
        """
        Activation function that returns the normalized exponential of the input.
        """
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)
    
    def softplus(self, x):
        """
        Activation function that returns the log of 1 plus the exponential of the input.
        """
        return np.log(1 + np.exp(x))

    def forward(self, X):
        """
        Forward propagation method that computes and returns the predicted output given an input.
        """
        h_output = np.dot(X, self.h_weights) + self.h_bias
        h_activation = self.sigmoid(h_output)
        o_output = np.dot(h_activation, self.o_weights) + self.o_bias
        o_activation = self.sigmoid(o_output)
        return o_activation, h_activation

    def backward(self, X, y, output, h_activation):
        """
        Backward propagation method that calculates the gradient and adjusts the weights and biases accordingly.
        """
        error = (y - output) # Calculate the error between the predicted output and the actual output
        o_delta = error * self.sigmoid_derivative(output) # Calculate the delta for the output layer
        hidden_error = np.dot(o_delta, self.o_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(h_activation) # Calculate the delta for the hidden layer
        self.o_weights += self.learning_rate * np.dot(h_activation.T, o_delta) # Adjust the output layer weights
        self.o_bias += self.learning_rate * np.sum(o_delta, axis=0) # Adjust the output layer biases
        self.h_weights += self.learning_rate * np.dot(X.T, hidden_delta) # Adjust the hidden layer weights
        self.h_bias += self.learning_rate * np.sum(hidden_delta, axis=0) # Adjust the hidden layer biases

    def sigmoid_derivative(self, x):
        """
        Method that calculates the derivative of sigmoid activation function.
        """
        return x * (1 - x)

    def train(self, X, y, epochs):
        """
        Main method to train the MLP model for given number of epochs using forward and backward propagation.
        """
        for epoch in range(epochs):
            output, hidden_layer = self.forward(X)
            self.backward(X, y, output, hidden_layer)

    def predict(self, X):
        """
        Method that predicts the output of the trained MLP model given an input.
        """
        return self.forward(X)

# Initialize the input features and labels
X = np.random.randn(1000, 4)
y = np.random.randn(1000, 1)

# Initialize the MLP model
mlp = MLP(input_size=4, hidden_size=5, output_size=1, learning_rate=0.1)

# Train the MLP model
mlp.train(X, y, epochs=100)

# Test the MLP model
print(mlp.predict(X))

import matplotlib.pyplot as plt

# Initialize the input values
x = np.arange(-10, 10, 0.1)

# Create a figure to hold the plots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Plot the linear activation function
axs[0, 0].plot(x, mlp.linear(x))
axs[0, 0].set_title('Linear')

# Plot the step activation function
axs[0, 1].plot(x, mlp.step(x))
axs[0, 1].set_title('Step')

# Plot the sigmoid activation function
axs[0, 2].plot(x, mlp.sigmoid(x))
axs[0, 2].set_title('Sigmoid')

# Plot the tanh activation function
axs[1, 0].plot(x, mlp.tanh(x))
axs[1, 0].set_title('Tanh')

# Plot the ReLU activation function
axs[1, 1].plot(x, mlp.relu(x))
axs[1, 1].set_title('ReLU')

# Plot the ELU activation function
axs[1, 2].plot(x, mlp.elu(x))
axs[1, 2].set_title('ELU')

# Show the plots
plt.show()
