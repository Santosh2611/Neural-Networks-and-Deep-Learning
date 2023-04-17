# This code generates data for binary classification and applies different activation functions on it to visualize their outputs

import matplotlib.pyplot as plt
import numpy as np

# Define activation functions
def sigmoid(x):
    """
    Returns the output of the sigmoid function for a given input x.
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Returns the output of the ReLU function for a given input x.
    """
    return np.maximum(0, x)

def linear(x):
    """
    Returns the output of the Linear function for a given input x.
    """
    return x

# Generate data for binary classification
input_data = np.random.randn(1000, 2) # Creating a 1000x2 array of random numbers from standard normal distribution
output_labels = (input_data.sum(axis=1) > 0).astype(int) # Creating a binary class label based on sum of each row of input_data

# Plot input data
plt.scatter(input_data[:, 0], input_data[:, 1], c=output_labels) # Plotting scatter plot with first column of input_data as x-axis and second column as y-axis. c=output_labels colors each point according to its corresponding binary class label
plt.title("Input Data") # Setting the title of the plot

plt.xlabel("Feature 1") # Setting the x-label
plt.ylabel("Feature 2") # Setting the y-label
plt.colorbar(label="Class Label") # Adding a colorbar and setting its label
plt.show() # Displaying the plot

# Apply activation functions to input data and store them in a dictionary
activation_functions = {'linear': linear(input_data.sum(axis=1)), 'sigmoid': sigmoid(input_data.sum(axis=1)), 'relu': relu(input_data.sum(axis=1))} # Storing the outputs of all activation functions evaluated at sum of rows of input_data in a dictionary

# Plot output of activation functions
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4)) # Creating a figure with one row and three columns of subplots, sharing y-axis and setting the size of the figure

for ax, activation_function_name in zip(axs, activation_functions.keys()): # Iterating through each subplot and its corresponding activation function name
    
    scatter = ax.scatter(input_data[:, 0], input_data[:, 1], c=activation_functions[activation_function_name]) # Plotting the scatter plot with first column of input_data as x-axis and second column as y-axis. c = activation_functions[activation_function_name] colors each point according to the output of respective activation function

    if scatter.get_offsets().size > 0: # Checking if any scatter points are present in the current subplot
        
        ax.set_title(f"{activation_function_name.capitalize()} Activation") # Setting the title of the subplot with the capitalized name of respective activation function
        
        ax.set_xlabel("Feature 1") # Setting the x-label for each subplot

fig.tight_layout() # Automatically adjust the subplot parameters to give specified padding between plots.

# Creating a global colorbar only if scatter points are present in any of the subplots
if scatter.get_offsets().size > 0:
    plt.colorbar(scatter, ax=axs.ravel().tolist(), label="Activation Output") # Adding a global colorbar and setting its label

plt.show() # Displaying the plot
