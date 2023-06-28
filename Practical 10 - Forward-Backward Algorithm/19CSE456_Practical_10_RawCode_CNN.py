import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset and split it into training, validation and test sets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, val_images = train_images[:45000], train_images[45000:]
train_labels, val_labels = train_labels[:45000], train_labels[45000:]

# Normalize pixel values to be between 0 and 1
train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

# Define class names for visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize sample images with labels
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("Visualize Sample Images with Labels")
for i, ax in enumerate(axs.flat):
    ax.imshow(train_images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(class_names[train_labels[i][0]])
plt.tight_layout()
plt.show()

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.summary()

# Compile the model with appropriate loss and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model using the training and validation sets
history = model.fit(train_images, train_labels, batch_size=64,
                    epochs=5, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Plot the accuracy and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.title('Accuracy Over Epochs')
plt.legend(loc='lower right')
plt.show()

# Define a new model that gives activations at each layer
layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Visualize activations/features at each layer for a sample image
sample_image = train_images[0]
activations = activation_model.predict(sample_image.reshape(1, 32, 32, 3))

plt.figure(figsize=(20, 20))
fig.suptitle("Visualize Activations/Feature Map")
for i, activation in enumerate(activations):
    n_features = activation.shape[-1]
    # Plot the activations/features in a grid
    plt.subplot(8, 8, i+1)
    plt.title(f'Layer {i+1}')
    plt.xticks([])
    plt.yticks([])
    
    if len(activation.shape) == 4: # Check if activation is from a convolutional layer
        for j in range(n_features):            
            # Reshape the 2D activation array to a 3D array with one color channel
            feature_map = activation[0, :, :, j].reshape((activation.shape[1], activation.shape[2], 1))
            plt.imshow(feature_map, cmap='gray')
    else:
        # For other layers (e.g. Dense), plot a histogram of the activations instead of feature maps
        plt.hist(tf.reshape(activation, [-1]), bins=100)
plt.tight_layout()
plt.show()
