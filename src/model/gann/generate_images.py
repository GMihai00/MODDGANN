import argparse
import os

from data_processing_helpers import EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from helpers import generate_and_save_images, noise_dim
import generators
import discriminators
import tensorflow as tf
from helpers import generator_optimizer, discriminator_optimizer

def make_generator_model(model_name):
    output_shape=(None, EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    return getattr(generators, model_name)(output_shape)

def make_discriminator_model(model_name):
    input_shape = [EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1]
    return getattr(discriminators, model_name)(input_shape, )
    
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
    
    
    return checkpoint, checkpoint_prefix

        
def main():     
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, help="Model name", default="sample")
    parser.add_argument("--nr_samples", type=int, help="Nr samples", default=100)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    nr_samples = args.nr_samples
    
    try:
        os.chdir("./src/model/gann")
    except:
        pass
        
    generator = make_generator_model(model_name)
    discriminator = make_discriminator_model(model_name)
    
    checkpoint, checkpoint_prefix = get_checkpoint(generator, discriminator, model_name)
    
    os.makedirs('./generated_images', exist_ok=True)
    
    for i in range(0, nr_samples):
        seed = tf.random.normal([1, noise_dim])
        generate_and_save_images(generator,
                            i,
                            seed,
                            dir ='./generated_images')
                            
if __name__ == "__main__":
    main()