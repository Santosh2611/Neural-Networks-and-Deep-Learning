# Import necessary libraries
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

# Load MNIST dataset
(x_train, _), _ = mnist.load_data() # We only need the training set for now

# Define input shape and size
input_dim = 784 # For MNIST dataset
input_shape = (input_dim,)

# Reshape the data to fit the model
x_train = x_train.reshape(x_train.shape[0], input_dim).astype('float32') / 255.

# Define the encoder architecture
encoding_dim = 32
input_layer = Input(shape=input_shape)
encoder_layer1 = Dense(128, activation='relu', name='encoder_1')(input_layer)
encoder_layer2 = Dense(64, activation='relu', name='encoder_2')(encoder_layer1)
encoder_layer3 = Dense(encoding_dim, activation='relu', name='encoder_output')(encoder_layer2)
encoder_model = Model(inputs=input_layer, outputs=encoder_layer3)

# Define the decoder architecture with mirrored layers
encoded_input = Input(shape=(encoding_dim,))
decoded_layer1 = Dense(64, activation='relu', name='decoder_1')(encoded_input)
decoded_layer2 = Dense(128, activation='relu', name='decoder_2')(decoded_layer1)
decoded_layer3 = Dense(input_dim, activation='sigmoid', name='decoder_output')(decoded_layer2)
decoder_model = Model(inputs=encoded_input, outputs=decoded_layer3)

# Define the full autoencoder model including both encoder and decoder
autoencoder_model = Model(inputs=input_layer, outputs=decoder_model(encoder_model(input_layer)))

# Compile the model with optimization functions and an evaluation metric
autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

# Set up TensorBoard for visualization of the model architecture and training progress
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0)

# Fit the autoencoder to the data
autoencoder_model.fit(x_train, x_train,
                      epochs=20,
                      batch_size=128,
                      shuffle=True,
                      validation_split=0.1,
                      callbacks=[tensorboard_callback])
