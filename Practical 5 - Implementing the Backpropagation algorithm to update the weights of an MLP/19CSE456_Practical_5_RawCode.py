# Import necessary libraries
import numpy as np

# Define the MLP class and its methods
class MLP:
    # Constructor function to initialize model parameters
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Randomly initialize weights and biases for hidden and output layers
        self.h_weights = np.random.randn(input_size, hidden_size)
        self.h_bias = np.random.randn(hidden_size)
        self.o_weights = np.random.randn(hidden_size, output_size)
        self.o_bias = np.random.randn(output_size)

    # Activation functions
    def linear(self, x):
        return x
    
    def step(self, x):
        return np.where(x>=0, 1, 0)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def elu(self, x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    def softplus(self, x):
        return np.log(1 + np.exp(x))

    # Forward propagation function
    def forward(self, X):
        h_output = np.dot(X, self.h_weights) + self.h_bias
        h_activation = self.sigmoid(h_output) # apply activation function to hidden layer output
        o_output = np.dot(h_activation, self.o_weights) + self.o_bias
        o_activation = self.sigmoid(o_output) # apply activation function to output layer output

        return o_activation, h_activation

    # Backward propagation function
    def backward(self, X, y, output, h_activation):
        error = (y - output)
        o_delta = error * self.sigmoid_derivative(output) # calculate output delta
        hidden_error = np.dot(o_delta, self.o_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(h_activation) # calculate hidden layer delta
        # update weights and biases for output and hidden layers
        self.o_weights += self.learning_rate * np.dot(h_activation.T, o_delta)
        self.o_bias += self.learning_rate * np.sum(o_delta, axis=0)
        self.h_weights += self.learning_rate * np.dot(X.T, hidden_delta)
        self.h_bias += self.learning_rate * np.sum(hidden_delta, axis=0)

    # Derivative of sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Training function
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output, hidden_layer = self.forward(X)
            self.backward(X, y, output, hidden_layer)

    # Prediction function
    def predict(self, X):
        return self.forward(X)

# Generate random input features and labels
X = np.random.randn(1000, 4)
y = np.random.randn(1000, 1)

# Create MLP model object with specified parameters
mlp = MLP(input_size=4, hidden_size=5, output_size=1, learning_rate=0.1)

# Train MLP model on generated data
mlp.train(X, y, epochs=100)

# Test the trained MLP model by predicting output given input features
print(mlp.predict(X))
