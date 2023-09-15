import subprocess
import tensorflow as tf
import os
import numpy as np
import csv
import argparse

from PIL import Image

EXPECTED_PHOTO_WIDTH = 320
EXPECTED_PHOTO_HEIGHT = 320
IS_RGB = False
NR_DISEASES = 0

def convert_image_to_bytes(image_path):
    image = Image.open(image_path)
    
    image = image.convert("L")
    
    width, height = image.size
    
    if height != EXPECTED_PHOTO_HEIGHT or width != EXPECTED_PHOTO_WIDTH:
        #resize to fit the model size
        print(f"Warning resizing image {image_path} to fit the model requirements")
        print(f"Initial size: {width}:{height}, target size: {EXPECTED_PHOTO_WIDTH}:{EXPECTED_PHOTO_HEIGHT}")
        image = image.resize((EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT), Image.ANTIALIAS)
    
    image_array = np.array(image)
    
    return image_array.reshape(EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 1)


def read_data(file_path):
    
    x_data = []
    y_data = []
        
    # Open the CSV file for reading
    with open(file_path, mode='r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        
        # Skip the header row (first row)
        next(reader, None)

        # Iterate through rows
        for row in reader:
            if len(row) == 2:  # Ensure there are two columns in the row
                image_path, disease = row
                x_data.append(convert_image_to_bytes(image_path))
                y_data.append(disease)

    y_data = tf.keras.utils.to_categorical(y_data)
    
    return (x_data, y_data)

def define_model():
    model = tf.keras.models.Sequential([
    
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
        ),
    
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
        # Flatten units
        tf.keras.layers.Flatten(),
    
        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
    
        # Add an output layer for each disease type
        tf.keras.layers.Dense(NR_DISEASES, activation="softmax")
    ])
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def save_model(model, model_name):
    if not os.path.exists("./saved_model"):
        subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    model.save("saved_model/" + model_name)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_input", type=str, help="CSV train input file")
    parser.add_argument("--test_input", type=str, help="CSV test input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="my_model")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=10)

    args = parser.parse_args()
    
    train_input = args.train_input
    test_input = args.test_input
    model_name = args.model_name
    train_epochs = args.epochs
    
    if train_input == None or test_input == None:
        print("Missing train or test input, quiting")
        exit(5)
    
    x_train, y_train = read_data(train_input)
    x_test, y_test = read_data(test_input)
    
    model = define_model()
    
    model.fit(x_train, y_train, epochs=train_epochs)
    
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    save_model(model, model_name)


if __name__ == "__main__":
    main()