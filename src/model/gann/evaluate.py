import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D, Flatten, InputLayer, concatenate, GlobalAveragePooling2D, BatchNormalization, Input,  Resizing
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
import argparse
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_processing_helpers import EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB, read_data

EXPECTED_FID_SIZE = 299 # https://www.researchgate.net/profile/Yu-Yu-120/publication/354269184_Frechet_Inception_Distance_FID_for_Evaluating_GANs/links/612f01912b40ec7d8bd87953/Frechet-Inception-Distance-FID-for-Evaluating-GANs.pdf

def preprocess_images(images, is_rgb):
    images = tf.image.resize(images, (EXPECTED_FID_SIZE, EXPECTED_FID_SIZE))
    if not is_rgb:
        images = tf.image.grayscale_to_rgb(images)
    return tf.keras.applications.inception_v3.preprocess_input(images)

def get_activations(model, images, batch_size=16):
    activations = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        # Get activations
        act = model(batch, training=False).numpy()
        activations.append(act)
    
    # Concatenate activations
    return np.concatenate(activations, axis=0)

def calculate_activation_statistics(model, images, batch_size=16):
    act = get_activations(model, images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_fid_score(real_images, generated_images, model, is_rgb=True):

    real_images = preprocess_images(real_images, is_rgb)
    generated_images = preprocess_images(generated_images, is_rgb)

    mu1, sigma1 = calculate_activation_statistics(model, real_images)
    mu2, sigma2 = calculate_activation_statistics(model, generated_images)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

    
def build_InceptionV3_model(input_shape):

    model = tf.keras.applications.InceptionV3(
        input_shape=(EXPECTED_FID_SIZE, EXPECTED_FID_SIZE, 3), 
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    return model


def calculate_fid_score_real_vs_real(model, data, is_rgb):
    num_samples = 15 # Number of random samples to take
    fid_scores = []
    
    for _ in range(num_samples):
        random.shuffle(data)
        split_index = len(data) // 2
        real_images_sample = data[:split_index]
        fake_images_sample = data[split_index:]
        
        fid_score = calculate_fid_score(real_images_sample, fake_images_sample, model, is_rgb)
        print(f"FID score random samples real images, equal: {fid_score}")
        fid_scores.append(fid_score)

    average_fid_score = sum(fid_scores) / len(fid_scores)
    print("Average FID score random samples real images vs real images, equal:", average_fid_score)

def calculate_fid_score_real_vs_fake(model, real_images, fake_images, is_rgb):
    num_samples = 15 
    fid_scores = []

    for _ in range(num_samples):
        # Shuffle real and fake images
        random.shuffle(real_images)
        random.shuffle(fake_images)
        split_index = len(real_images) // 2
        
        # Ensure the number of fake images matches the number of real images
        fake_images_sample = random.sample(fake_images.tolist(), len(real_images))[split_index:]
        real_images_sample = real_images[split_index:]
        # Calculate FID score for the sampled real and fake images
        fid_score = calculate_fid_score(real_images_sample, fake_images_sample, model, is_rgb)
        print(f"FID score random samples real vs fake images: {fid_score}")
        fid_scores.append(fid_score)
    
    average_fid_score = sum(fid_scores) / len(fid_scores)
    print("Average FID score random samples real vs fake images:", average_fid_score)


def main(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file", required=True)
    
    args = parser.parse_args()
    
    input_data = args.input_data
    
    #  need to read data beforehand due to portability issue
    try:
        os.chdir("./src/model")
    except:
        pass
        
        
    # real_images = read_data(input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
    # fake_images = read_data("../dataset_helpers/data_copy.csv", EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
        
    # try:
    #     os.chdir("./gann")
    # except:
    #     pass
        
    # input_shape = (EXPECTED_FID_SIZE, EXPECTED_FID_SIZE, 3 if IS_RGB else 1)
    
    # model = build_InceptionV3_model(input_shape)
    
    # fid_score = calculate_fid_score(real_images, fake_images, model, IS_RGB)
    
    # print(f"FID score real vs fake images: {fid_score}")
    
    # # 341.98 fake against real for representative images
    
    real_images = read_data(input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
    fake_images = read_data(input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB, generated=True)
    
    data =  real_images
    
    try:
        os.chdir("./gann")
    except:
        pass
        
    input_shape = (EXPECTED_FID_SIZE, EXPECTED_FID_SIZE, 3 if IS_RGB else 1)
    
    model = build_InceptionV3_model(input_shape)
    
    # https://academic.oup.com/biomethods/article/9/1/bpae062/7739892 take "Advanced image generation for cancer using diffusion models" paper for ref
        
    calculate_fid_score_real_vs_real(model, data, IS_RGB)
    
    calculate_fid_score_real_vs_fake(model, real_images, fake_images, IS_RGB)
    
    
    # # 77.87 real against real (should be around 0 normally)
    # # 341.39 fake against real
    
if __name__ == "__main__":
    main()
