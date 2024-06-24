import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import discriminators
import generators

import tensorflow as tf
from IPython import display

from helpers import *

EPOCHS_FILE_NAME = "epochs.txt"

from data_processing_helpers import EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB, read_data

def make_generator_model(model_name):
    output_shape=(None, EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    return getattr(generators, model_name)(output_shape)

def make_discriminator_model(model_name):
    input_shape = [EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1]
    return getattr(discriminators, model_name)(input_shape, )

def cleanup_checkpoints(checkpoint_dir):
    
    original_dir =  os.getcwd()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.chdir(checkpoint_dir)
    file_list = filter(os.path.isfile, os.listdir('.'))

    # Sort the list of files based on their modification time.
    sorted_files = sorted(file_list, key=os.path.getmtime, reverse=True)
    
    for file in sorted_files[5:]:
        os.remove(file)
        
    os.chdir(original_dir)

def get_checkpoint(generator, discriminator, model_name):
    checkpoint_dir = './training_checkpoints/' + model_name
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
                                
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Latest checkpoint restored from {latest_checkpoint}")
    else:
        print("Checkpoint not found. Training from scratch.")
    
    cleanup_checkpoints(checkpoint_dir)
    
    return checkpoint, checkpoint_prefix

@tf.function
def train_step(generator, discriminator, batch_size, images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return disc_loss, gen_loss

def train(generator, discriminator, dataset, epochs, batch_size, checkpoint, checkpoint_prefix):
    for epoch in range(epochs):
        start = time.time()
        
        d_loss = 0.0
        g_loss = 0.0
        for image_batch in dataset:
            d_batch_loss, g_batch_loss = train_step(generator, discriminator, batch_size, image_batch)
            d_loss += d_batch_loss
            g_loss += g_batch_loss
        
        d_loss = d_loss/len(dataset)
        g_loss = g_loss/len(dataset)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)
    
        # Save the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
    
        print(f"[{epoch+1:1.0f}/{epochs}] {(time.time()-start):1.4f}s - d_loss: {d_loss:1.4f} - g_loss: {g_loss:1.4f}")
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            seed)
                                
def main(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="sample")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--buffer_size", type=int, help="Buffer size", default=6000)
    
    args = parser.parse_args()
    
    input_data = args.input_data
    train_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model_name
    buffer_size = args.buffer_size
    
    try:
        os.chdir("./src/model/gann")
    except:
        pass
        
    generator = make_generator_model(model_name)
    discriminator = make_discriminator_model(model_name)
    
    checkpoint, checkpoint_prefix = get_checkpoint(generator, discriminator, model_name)
        
    previous_epochs = read_number_from_file(EPOCHS_FILE_NAME)
        
    train_images = read_data(input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
    
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
    
    os.makedirs('./images', exist_ok=True)
    
    train(generator, discriminator, train_dataset, train_epochs, batch_size, checkpoint, checkpoint_prefix)
    
    write_number_to_file(EPOCHS_FILE_NAME, previous_epochs + train_epochs)

if __name__ == "__main__":
    main()