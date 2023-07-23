# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Load CIFAR-10 dataset and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
dropout_rate = 0.3 # Flexible dropout rate
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Set up early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)

# Train model on training set
model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          verbose=2, # Less verbose output
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

# Evaluate model on test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
