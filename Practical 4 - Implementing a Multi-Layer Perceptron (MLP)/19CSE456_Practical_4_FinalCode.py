# Importing the required libraries
import tensorflow as tf # importing tensorflow library for machine learning
from sklearn.datasets import load_iris #importing iris dataset from sklearn datasets
from sklearn.model_selection import train_test_split #importing train test split from sklearn model selection 
from sklearn.linear_model import Perceptron #importing perceptron algorithm from sklearn linear model

# Set random seed for consistency
random_state = 0 #assigning a fixed value to random seed for consistency

# Loading the iris dataset
iris = load_iris() #loading iris dataset into the 'iris' variable

# Defining the input and target variables
X = iris.data #dividing iris data into features
y = iris.target #defining target or dependent variable of iris dataset

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state) #splitting the dataset into training and testing with ratio of 80:20 respectively 

# Creating the MLP model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), #creating the neural network with first layer containing 10 nodes, using RELU activation function and with input dimensions (4,)
    tf.keras.layers.Dense(3, activation=tf.nn.softmax) #second layer containing 3 nodes using Softmax activation function
])

# Compiling the model
model.compile(optimizer='adam', #optimizing the neural network using adam optimizer
              loss='sparse_categorical_crossentropy', #loss function used is sparse categorical cross-entropy
              metrics=['accuracy']) #training accuracy is taken as our metric of interest

# Fitting the data to the model
model.fit(X_train, y_train, epochs=100) #fitting the neural network model on our training dataset for 100 epochs

# Evaluating the model on the test set
_, accuracy = model.evaluate(X_test, y_test) #evaluating the trained neural network on our testing dataset

# Printing the accuracy score of the model
print("MLP Accuracy:", accuracy) #printing the accuracy of our trained neural network

# Creating a perceptron object and fitting it to the data
perceptron = Perceptron(random_state=random_state) #creating a neuron named 'perceptron' with predefined random state
perceptron.fit(X_train, y_train) #fitting our perceptron model to our training dataset

# Printing the accuracy score of the perceptron
print("Perceptron Accuracy:", perceptron.score(X_test, y_test)) #printing the accuracy of our trained perceptron model

# Saving the trained MLP model for future use
model.save("iris_model.h5") #saving our neural network model for future reference
