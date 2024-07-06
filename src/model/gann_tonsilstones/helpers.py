
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import math
from PIL import Image

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
    
    # ratio = int(math.sqrt(num_examples_to_generate))
    
    for i in range(predictions.shape[0]):
        image_data = predictions[i].numpy() * 127.5 + 127.5
        image_data = image_data.astype(np.uint8)
        if IS_RGB:
            
            image = Image.fromarray(image_data, 'RGB')
            
            image.save(dir + '/image{:04d}_at_epoch_{:04d}.png'.format(i, epoch))
        else:
            image = Image.fromarray(image_data, 'L')
            
            image.save(dir + '/image{:04d}_at_epoch_{:04d}.png'.format(i, epoch))
            
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

