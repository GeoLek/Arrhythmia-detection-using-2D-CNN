import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time
import os
import matplotlib.pyplot as plt

data_dir = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/Data_GAN/Q'
output_dir = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/Data_GAN/output'


# Function to load dataset
def load_dataset_from_directory(data_dir, img_height=224, img_width=224, batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode=None,  # For GANs, labels are not needed
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
    )
    # Normalize images to [-1, 1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)
    dataset = dataset.map(lambda x: normalization_layer(x))
    return dataset

# Generator Model
def make_generator_model(input_dim=100):
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(input_dim,)),  # Adjusted for a starting size of 7x7
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 128)),  # Start with 7x7 feature maps

        layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
        # Maintains 7x7 size
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 14x14
        layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 28x28
        layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 56x56
        layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 112x112
        layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Final upscale to 224x224
        layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        # Output layer for 224x224x1 image
    ])
    return model

# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential([
        # Input shape is 224x224x1
        layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(224, 224, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Downscale to 112x112
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Downscale to 56x56
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Downscale to 28x28
        layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Flatten and output decision
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

generator = make_generator_model()
discriminator = make_discriminator_model()

@tf.function
def train_step(images, batch_size=64, noise_dim=100):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        # Diagnose the shape of real and generated images
        print("Shape of real images:", images.shape)  # Expected to be (batch_size, 224, 224, 1)
        print("Shape of generated images:", generated_images.shape)  # Expected to be (batch_size, 224, 224, 1)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs=50, noise_dim=100, num_examples_to_generate=16, batches_per_epoch=None):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Calculate batches per epoch dynamically if not provided
    if not batches_per_epoch:
        batches_per_epoch = sum(1 for _ in dataset)

    for epoch in range(epochs):
        start_time = time.time()

        for batch_num, image_batch in enumerate(dataset):
            train_step(image_batch)
            if batch_num % 10 == 0:  # Update every 10 batches.
                print(f'Epoch {epoch + 1}, Batch {batch_num}/{batches_per_epoch}')

        # After the epoch ends, generate and save images
        generate_and_save_images(generator, epoch + 1, seed)

        # Print epoch duration and optionally losses
        print(f'Time for epoch {epoch + 1} is {time.time() - start_time} sec')

    # Save the generator model after training is complete
    generator.save('path/to/save/the/generator_model')
    print("Generator model saved.")

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

train_dataset = load_dataset_from_directory(data_dir, img_height=224, img_width=224, batch_size=64)
EPOCHS = 50
train(train_dataset, EPOCHS)