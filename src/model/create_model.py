import os
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten, InputLayer
from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf

from sklearn.model_selection import train_test_split
import numpy as np
import csv
import argparse

from PIL import Image

EXPECTED_PHOTO_WIDTH = 320
EXPECTED_PHOTO_HEIGHT = 320

IS_RGB = False

WEIGHTS_BACKUP = 'my_model_weights.h5'


DISEASE_TO_CATEGORY = {
    "pharyngitis": 0,
    "tonsillitis": 1,
    "gastric reflux": 2,
    "tonsil stones": 3,
    "healthy" : 4
}
NR_DISEASES = len(DISEASE_TO_CATEGORY)

def convert_path(path):
    if os.name == 'posix':
        return path.replace('\\', '/')
    else:
        return path

def convert_image_to_bytes(image_path):
    
    try:
    
        image = Image.open(image_path)
        
        image = image.convert("L")
        
        width, height = image.size
        
        if height != EXPECTED_PHOTO_HEIGHT or width != EXPECTED_PHOTO_WIDTH:
            #resize to fit the model size
            print(f"Warning resizing image {image_path} to fit the model requirements")
            print(f"Initial size: {width}:{height}, target size: {EXPECTED_PHOTO_WIDTH}:{EXPECTED_PHOTO_HEIGHT}")
            image = image.resize((EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT), Image.Resampling.LANCZOS)
        
        image_array = np.array(image)
        
        return image_array.reshape(EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 1)
        
    except Exception as e:
        print(f"Image {image_path} not found")
        return []


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
                image_bytes = convert_image_to_bytes(convert_path(image_path))
                
                if len(image_bytes) != 0:
                    x_data.append(image_bytes)
                    y_data.append(DISEASE_TO_CATEGORY[disease])
                        

    return (x_data, y_data)

def define_model():

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NR_DISEASES, activation="softmax"))
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    if os.path.isfile(WEIGHTS_BACKUP):
        model.load_weights(WEIGHTS_BACKUP)
    
    return model

def save_model_weights(model):
    model.save_weights(WEIGHTS_BACKUP)

def save_model(model, model_name):
    if not os.path.exists("./saved_model"):
        subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    model.save("saved_model/" + model_name)

def main():
    print(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="my_model")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=10)
    parser.add_argument("--test_train_split", type=int, help="Data Split", default=0.3)

    args = parser.parse_args()
    
    input_data = args.input_data
    model_name = args.model_name
    train_epochs = args.epochs
    
    if input_data == None:
        print("Missing train and test input, quitting")
        exit(5)
    
    x_data, y_data = read_data(input_data)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=args.test_train_split, random_state=42)
    
    model = define_model()
    
    model.summary()
    
    # problems here to see what is going on rn, maybe input dimensions are just wrong
    model.fit(x_train, y_train, epochs=train_epochs)
    
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    save_model_weights(model)


if __name__ == "__main__":
    main()