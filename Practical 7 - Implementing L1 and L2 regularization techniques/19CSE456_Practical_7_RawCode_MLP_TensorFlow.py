# acquire MNIST data
from tensorflow.keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')

# load and split the MNIST dataset into train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape pixel values of images to a 1D array for MLP's input layer
import numpy as np
train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))

# normalize the pixel values between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# convert labels to one-hot encoded matrix
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# create a Sequential model and add layers to it
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

MLP = Sequential()
MLP.add(InputLayer(input_shape=(784, ))) # input layer
MLP.add(Dense(256, activation='relu')) # hidden layer 1
MLP.add(Dense(256, activation='relu')) # hidden layer 2
MLP.add(Dense(10, activation='softmax')) # output layer

# print summary of the model architecture
MLP.summary()

# compile the model with appropriate loss and optimizer
MLP.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# use fit() method to train the model on training set and validate on test set
import matplotlib.pyplot as plt
history = MLP.fit(train_images, train_labels, epochs=20, batch_size=128, validation_data=(test_images, test_labels))

# plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# evaluate the performance of the trained model on test set
test_loss, test_acc = MLP.evaluate(test_images, test_labels, batch_size=128, verbose=0)
print("\nTest loss:", test_loss)
print("Test accuracy:", test_acc)

warnings.filterwarnings('ignore')

# make a prediction using a sample image from test set
digit = test_images[4]
digit_reshaped = digit.reshape(28, 28)
plt.imshow(digit_reshaped, cmap="binary")
plt.show()

# reshape the image data to match the input shape of our MLP
digit = np.reshape(digit, (-1, 784))
digit = digit.astype('float32') / 255

# use predict() method to generate output probabilities for the digit image 
prediction = MLP.predict(digit, verbose=0)
print("Prediction:", np.argmax(prediction)) # index of the class with maximum probability 
