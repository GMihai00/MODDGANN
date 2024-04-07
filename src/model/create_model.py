
import argparse
import os
import subprocess
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D, Flatten, InputLayer

from sklearn.model_selection import train_test_split

from data_processing_helpers import *
from training_callbacks import ImagePredictionLogger

EXPECTED_PHOTO_WIDTH = 320
EXPECTED_PHOTO_HEIGHT = 320

IS_RGB = False

WEIGHTS_BACKUP = 'my_model.weights.h5'

def define_model():

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="gelu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="gelu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="gelu"))
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
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=45)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--test_train_split", type=int, help="Data Split", default=0.2)

    args = parser.parse_args()
    
    input_data = args.input_data
    train_epochs = args.epochs
    batch_size = args.batch_size
    
    if input_data == None:
        print("Missing train and test input, quitting")
        exit(5)
    
    # WORKAROUND FOR NOW
    try:
        os.chdir("./src/model")
    except:
        pass

    x_data, y_data = read_data(input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
    
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=args.test_train_split, random_state=42)
    
    # create callbacks with unprocessed images
        
    train_callbacks = []
    train_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True))
    train_callbacks.append(ImagePredictionLogger((x_test, y_test), log_dir + "/prediction", train_epochs, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB))
        
    # normalize data
    
    x_train = x_train / 255
    x_test = x_test / 255
    
    model = define_model()
    
    model.summary()

    model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test,  y_test), callbacks=train_callbacks)
    
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    save_model_weights(model)


if __name__ == "__main__":
    main()