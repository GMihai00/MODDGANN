
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import math

from data_processing_helpers import IS_RGB

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 400
# change this to 1 when you want to just generate images for your model
num_examples_to_generate = 1

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
def generate_and_save_images(model, epoch, test_input, dir='./images'):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    ratio = int(math.sqrt(num_examples_to_generate))
    
    fig = plt.figure(figsize=(ratio, ratio))
    # enable when generating just the one photo
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for i in range(predictions.shape[0]):
        plt.subplot(ratio, ratio, i+1)
        if IS_RGB:
            img_rgb = tf.clip_by_value(predictions[i] * 127.5 + 127.5, 0, 255)
            img_rgb = tf.cast(img_rgb, tf.uint8)
            plt.imshow(img_rgb, aspect='auto')
        else:
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray', aspect='auto')
        plt.axis('off')
    
    
    plt.savefig(dir + '/image_at_epoch_{:04d}.png'.format(epoch))
def write_number_to_file(filename, number):
    try:
        with open(filename, 'w') as file:
            file.write(str(number))  # Write the number as a string
        print(f"Number {number} has been written to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
        
def read_number_from_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read().strip()  # Read entire file content and strip any extra whitespace
            number = int(content)  # Convert content to an integer
        return number
    except FileNotFoundError:
        print(f"File {filename} does not exist. Returning 0.")
        return 0
    except IOError as e:
        print(f"Error reading from file {filename}: {e}")
        return None
    except ValueError as e:
        print(f"Error: File {filename} does not contain a valid number.")
        return 

stddev=0.0002

gaussian_noise_layer = tf.keras.layers.GaussianNoise(stddev=stddev)
    
def add_gaussian_noise(images):
    rez = []
    for i in range (0, images.shape[0]):
        noisy_image_tensor = gaussian_noise_layer(images[i], training=True)
        noisy_image_tensor = tf.cast((noisy_image_tensor * 255.0 - 127.5) / 127.5, tf.float32)
        rez.append(noisy_image_tensor)
        
    return tf.convert_to_tensor(rez, dtype=tf.float32)

