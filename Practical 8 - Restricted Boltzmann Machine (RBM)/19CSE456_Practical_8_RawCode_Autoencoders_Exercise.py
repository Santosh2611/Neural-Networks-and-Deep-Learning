import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize data
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

# Add random noise to the training images
noise_factor = 0.5
train_noisy = train_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
train_noisy = np.clip(train_noisy, 0., 1.)

# Add random noise to the test images
test_noisy = test_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)
test_noisy = np.clip(test_noisy, 0., 1.)

# Reshape data for autoencoder model
train_images = train_images.reshape(train_images.shape[0], 784)
test_images = test_images.reshape(test_images.shape[0], 784)

# Define the autoencoder model
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train the model
history = autoencoder.fit(
    train_noisy.reshape(train_noisy.shape[0], 784),
    train_images.reshape(train_images.shape[0], 784),
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(
        test_noisy.reshape(test_noisy.shape[0], 784),
        test_images.reshape(test_images.shape[0], 784)
    ),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Load the best-saved model
autoencoder = tf.keras.models.load_model('best_model.h5')

# Evaluate the model on test data and display the loss
decoded_imgs = autoencoder.predict(test_noisy.reshape(test_noisy.shape[0], 784))
loss = autoencoder.evaluate(test_images.reshape(-1, 784), decoded_imgs)
print('Test loss:', loss)

# Display the original and reconstructed images
n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
