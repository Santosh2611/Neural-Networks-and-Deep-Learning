import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_sample_image
from sklearn.model_selection import train_test_split

# Load sample image data
china = load_sample_image('china.jpg')
X = china / 255.0  # scale pixel values to range [0, 1]
n_samples, height, width = X.shape
X_train_val = X.reshape(n_samples, height * width)
y_train_val = np.arange(n_samples)

# Visualize a sample from the original dataset
plt.imshow(X[0])
plt.show()

# Splitting into train, validation, and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Scaling training, validation, and testing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Creating an MLP regressor with one hidden layer containing 5 neurons
mlp = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='adam', max_iter=1000)

# Fitting training data to the model
mlp.fit(X_train, y_train)

# Predicting with trained model and calculating accuracy
train_predictions = mlp.predict(X_train)
val_predictions = mlp.predict(X_val)
test_predictions = mlp.predict(X_test)

train_mse = mean_squared_error(y_train, train_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
val_r2 = r2_score(y_val, val_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Evaluating precision, recall, and f1 score
train_precision = precision_score(y_train, np.round(train_predictions), average='weighted')
train_recall = recall_score(y_train, np.round(train_predictions), average='weighted')
train_f1 = f1_score(y_train, np.round(train_predictions), average='weighted')

val_precision = precision_score(y_val, np.round(val_predictions), average='weighted')
val_recall = recall_score(y_val, np.round(val_predictions), average='weighted')
val_f1 = f1_score(y_val, np.round(val_predictions), average='weighted')

test_precision = precision_score(y_test, np.round(test_predictions), average='weighted')
test_recall = recall_score(y_test, np.round(test_predictions), average='weighted')
test_f1 = f1_score(y_test, np.round(test_predictions), average='weighted')

# Plotting the errors
fig, ax = plt.subplots()
ax.plot(y_train, label='True')
ax.plot(train_predictions, label='Predicted')
ax.set_title('Training Set Predictions')
ax.legend()

fig, ax = plt.subplots()
ax.plot(y_val, label='True')
ax.plot(val_predictions, label='Predicted')
ax.set_title('Validation Set Predictions')
ax.legend()

fig, ax = plt.subplots()
ax.plot(y_test, label='True')
ax.plot(test_predictions, label='Predicted')
ax.set_title('Test Set Predictions')
ax.legend()

# Printing the results
print(f'\nTrain Mean Squared Error: {train_mse:.2f}')
print(f'Validation Mean Squared Error: {val_mse:.2f}')
print(f'Test Mean Squared Error: {test_mse:.2f}\n')

print(f'Train R-Squared score: {train_r2:.2f}')
print(f'Validation R-Squared score: {val_r2:.2f}')
print(f'Test R-Squared score: {test_r2:.2f}')

print(f'\nTrain Precision: {train_precision:.2f}')
print(f'Train Recall: {train_recall:.2f}')
print(f'Train F1 Score: {train_f1:.2f}')

print(f'\nValidation Precision: {val_precision:.2f}')
print(f'Validation Recall: {val_recall:.2f}')
print(f'Validation F1 Score: {val_f1:.2f}')

print(f'\nTest Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')
print(f'Test F1 Score: {test_f1:.2f}')
