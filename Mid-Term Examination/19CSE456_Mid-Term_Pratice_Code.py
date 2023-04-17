# Import numpy library
import numpy as np

import warnings
warnings.filterwarnings('ignore') # Never print matching warnings

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, batch_size):
        
        """
        Class constructor that initializes the MLP object with input_size, hidden_size, output_size, and learning_rate parameters.
        It also initializes weights and biases for both the hidden and output layers.
        """

        # Assign inputs to the current instance of MLP
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
            # Define the weight and bias vectors for each layer of the neural network using a dictionary
        self.params = {
            
            # 'W1': np.array([[-0.2, -1.4, 0.6], [-0.1, 0.9, 1.2]]),
            # 'B1': np.array([[-1.4], [-0.8], [0.2]]),
            # 'W2': np.array([[0.2], [-1.1], [0.5]]),
            # 'B2': np.array([[-0.6]])
            
            'W1': np.random.randn(input_size, hidden_size),
            'B1': np.random.randn(hidden_size, output_size),
            'W2': np.random.randn(hidden_size, output_size),
            'B2': np.random.randn(output_size, output_size)
        }

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
    
    # Define the sigmoid activation function using numpy's built-in exp() function
    def sigmoid(self, x):
        """
        Calculate the sigmoid function given an input x.
        Args:
            x: Input value(s) to feed into the sigmoid function.
        Returns:
            Result of the sigmoid function applied to the input x.
        """
        return 1 / (1 + np.exp(-x))
    
    # Define the derivative of the sigmoid activation function
    def sigmoid_derivative(self, x):
        """
        Calculate the derivative of the sigmoid function given an input x.
        Args:
            x: Input value(s) to feed into the derivative of sigmoid function.
        Returns:
            Result of the derivative of the sigmoid function applied to the input x.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
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
    
    def forward_backward(self, X, Y, loss_func='mse'):
        """
        Forward propagation method that computes and returns the predicted output given an input.
        """
        
        num_epochs = 1000 # change to a bigger number
        
        for epoch in range(num_epochs): # loop over epochs
            
            # Perform the feed-forward pass to get predictions
            Z1 = np.dot(X, self.params['W1']) + self.params['B1'].T
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.params['W2']) + self.params['B2']
            Y_hat = self.sigmoid(Z2)
    
            # Calculate the loss function
            if loss_func == 'mse':
                # Mean Squared Error (MSE) Loss Function
                error = Y - Y_hat
                loss = np.mean(error ** 2)
                dZ2 = error * self.sigmoid_derivative(Z2)
    
            elif loss_func == 'mae':
                # Mean Absolute Error (MAE) Loss Function
                error = Y - Y_hat
                loss = np.mean(np.abs(error))
                dZ2 = np.where(error >= 0, 1, -1) * self.sigmoid_derivative(Z2)
    
            elif loss_func == 'huber':
                # Huber Loss Function
                delta = 1.0
                error = Y - Y_hat
                abs_error = np.abs(error)
                quadratic = np.minimum(abs_error, delta)
                linear = abs_error - quadratic
                loss = (quadratic**2 + 2*delta*linear) / (2*X.shape[0])
                dZ2 = np.where(abs_error <= delta, error, delta*np.where(error>=0,1,-1)) * self.sigmoid_derivative(Z2)
    
            elif loss_func == 'exponential':
                # Exponential Loss Function
                error = Y - Y_hat
                loss = np.mean(np.exp(-error))
                dZ2 = np.exp(-error) * (-1) * self.sigmoid_derivative(Z2)
    
            elif loss_func == 'log':
                # Logarithmic Loss (Log Loss) Function
                loss = -(1/X.shape[0])*np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))
                dZ2 = (Y_hat - Y) * self.sigmoid_derivative(Z2)
    
            elif loss_func == 'kld':
                # Kullback-Leibler Divergence (KLD) Loss Function
                eps = 1e-8
                loss = np.sum(Y * np.log((Y+eps)/(Y_hat+eps))) / X.shape[0]
                dZ2 = (-Y/(Y_hat+eps)) * self.sigmoid_derivative(Z2)
    
            elif loss_func == 'hinge':
                # Hinge Loss Function
                margin = 1.0
                error = Y - Y_hat
                hinge = np.maximum(0, margin-error)
                loss = np.mean(hinge)
                dZ2 = np.where(error <= margin, -Y, 0) * self.sigmoid_derivative(Z2)
                
            if epoch % 100 == 0:
                print("\nEPOCH: ", epoch+1)
                print(loss_func.upper()+" Loss: ", loss)
    
            # Perform backpropagation and weight-bias adjustments to optimize the neural network using batch processing
            num_samples = X.shape[0]
            for i in range(0, num_samples, self.batch_size):
    
                # Compute gradients
                dW2 = np.dot(A1[i:i+self.batch_size].T, dZ2[i:i+self.batch_size])
                dB2 = np.sum(dZ2[i:i+self.batch_size], axis=0, keepdims=True)
                dZ1 = np.dot(dZ2[i:i+self.batch_size], self.params['W2'].T) * self.sigmoid_derivative(Z1[i:i+self.batch_size])
                dW1 = np.dot(X[i:i+self.batch_size].T, dZ1)
                dB1 = np.sum(dZ1, axis=0, keepdims=True)
    
                # Update weights and biases
                self.params['W2'] -= self.learning_rate * dW2
                self.params['B2'] -= self.learning_rate * dB2
                self.params['W1'] -= self.learning_rate * dW1
                self.params['B1'] -= self.learning_rate * dB1.T
    
        # Report the value of the loss function and the adjusted values of weight and bias vectors
        print("\n"+loss_func.upper()+" Loss: ", loss)
        print("\nAdjusted W1: \n", self.params['W1'])
        print("\nAdjusted B1: \n", self.params['B1'])
        print("\nAdjusted W2: \n", self.params['W2'])
        print("\nAdjusted B2: \n", self.params['B2'])

# Define the input data
# X = np.random.randn(1000, 2)
X = np.array([[-0.3, -1.5], [-1.7, 0.7]])

# Define the output data
# Y = np.random.randn(1000, 1)
Y = np.array([[0.1], [-0.3]])

# Initialize the MLP model
mlp = MLP(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1, batch_size=1)

# Train the model using different loss functions
loss_funcs = ['mse', 'mae', 'huber', 'exponential', 'log', 'kld', 'hinge']
for loss_func in loss_funcs:
    heading = "_" * 35
    print("\n" + heading + " " + loss_func.upper() + " LOSS " + heading)
    mlp.forward_backward(X, Y, loss_func=loss_func)

import matplotlib.pyplot as plt

# Initialize the input values
x = np.arange(-10, 10, 0.1)

# Create a figure to hold the plots with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Define the activation functions that will be plotted
activation_funcs = [('Linear', mlp.linear), ('Step', mlp.step), ('Sigmoid', mlp.sigmoid), 
                    ('Tanh', mlp.tanh), ('ReLU', mlp.relu), ('ELU', mlp.elu)]

# Loop through the activation functions and plot each one in a separate subplot
for i, (title, func) in enumerate(activation_funcs):
    
    # Compute the row and column indices for the current subplot
    row, col = i // 3, i % 3
    
    # Plot the activation function on the current subplot
    axs[row, col].plot(x, func(x))
    axs[row, col].set_title(title)

# Show the plots
plt.show()
