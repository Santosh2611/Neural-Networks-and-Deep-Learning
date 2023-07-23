import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing import sequence

# Set random seed for reproducibility
np.random.seed(42)

# Set the maximum number of words to be used (vocabulary size)
max_words = 5000

# Load the IMDB dataset and pad sequences to a fixed length
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
max_length = 100
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# Create and compile the RNN model
model = Sequential([
    Embedding(max_words, 32, input_length=max_length),
    LSTM(100),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
