
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from data_processing_helpers import IS_RGB

PREVIOUS_EPOCHS = 0

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 100
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
    
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(1, 1))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for i in range(predictions.shape[0]):
        plt.subplot(1, 1, i+1)
        if IS_RGB:
            img_rgb = tf.clip_by_value(predictions[i] * 127.5 + 127.5, 0, 255)
            img_rgb = tf.cast(img_rgb, tf.uint8)
            plt.imshow(img_rgb, aspect='auto')
        else:
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray', aspect='auto')
        plt.axis('off')
    
    
    
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch + PREVIOUS_EPOCHS))
#   plt.show()