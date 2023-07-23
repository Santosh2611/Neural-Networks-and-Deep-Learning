# Import all necessary dependencies
from tensorflow.keras.datasets import cifar10
import warnings
warnings.filterwarnings('ignore')

# Load the cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Perform necessary preprocessing on the images
import numpy as np
warnings.filterwarnings('ignore')
train_images = np.reshape(train_images, (-1, 32 * 32 * 3))
test_images = np.reshape(test_images, (-1, 32 * 32 * 3))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Show the shapes of train and test sets
print("\nShape of Training Set: ", np.shape(train_images))
print("Shape of Testing Set: ", np.shape(test_images))

# Select a sample set of 16 images from the dataset and visualize in a 4X4 grid
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
images = test_images[:16]
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.ravel()
for i in np.arange(0, len(images)):
    axes[i].imshow(images[i].reshape((32, 32, 3)))
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.5)
plt.show()

from tensorflow.keras.utils import to_categorical
warnings.filterwarnings('ignore')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Construct a MLP network architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
warnings.filterwarnings('ignore')

MLP = Sequential()
MLP.add(InputLayer(input_shape=(32 * 32 * 3, ))) # input layer
MLP.add(Dense(256, activation='relu')) # hidden layer 1
MLP.add(Dense(128, activation='relu')) # hidden layer 2
MLP.add(Dense(64, activation='relu')) # hidden layer 3
MLP.add(Dense(10, activation='softmax')) # output layer

# Present the summary of the network
MLP.summary()

# Compile the model
# MLP.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
MLP.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the constructed MLP
# history = MLP.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
history = MLP.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# Performance Evaluation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy for Question 2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for Question 2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

test_loss, test_acc = MLP.evaluate(test_images, test_labels, batch_size=64, verbose=0)
print("\nTest Loss for Question 2:", test_loss)
print("Test Accuracy for Question 2:", test_acc)

# history = MLP.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)
history = MLP.fit(train_images, train_labels, epochs=15, batch_size=128, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy for Question 3')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss for Question 3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

test_loss, test_acc = MLP.evaluate(test_images, test_labels, batch_size=128, verbose=0)
print("\nTest Loss for Question 3:", test_loss)
print("Test Accuracy for Question 3:", test_acc)
