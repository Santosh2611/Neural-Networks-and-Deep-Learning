import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

# Load data
digits = load_digits()
X = digits.data  # The input pixel values for each image
y = digits.target  # The target (label) values for each image

# Normalize pixel values using MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the RBM model using the Keras API
rbm = Sequential([
    Dense(128, input_shape=(64,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))
])

# Compile the RBM model
rbm.compile(optimizer='adam', loss='mean_squared_error')

# Train the RBM model
rbm.fit(X_train, X_train, epochs=5, batch_size=32)

# Extract features from RBM by computing the hidden layer activations for both training and testing data
hidden_train = rbm.predict(X_train)  # Hidden activations for training data
hidden_test = rbm.predict(X_test)  # Hidden activations for test data

# Define the logistic regression classifier and hyperparameter search space
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)  # Classifier to use for classification
params = {'C': np.logspace(-2, 2, 10)}  # Range of hyperparameters to try

# Use grid search to find optimal hyperparameters and evaluate the performance of the model using cross-validation
clf = GridSearchCV(logreg, params, cv=5)  # Perform a grid search over hyperparameters using 5-fold cross-validation
clf.fit(hidden_train, y_train)  # Fit the classifier to training data

# Evaluate the performance of the classifier on test data
y_pred = clf.predict(hidden_test)  # Predict labels for test data using the trained classifier
score = accuracy_score(y_test, y_pred)  # Calculate the accuracy of predicted labels compared to true labels
print('Accuracy:', score)  # Print the accuracy score
