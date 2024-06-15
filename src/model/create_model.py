
import argparse
import os
import subprocess
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# to disable cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from training_callbacks import ImagePredictionLogger

from data_processing_helpers import *
import models

EXPECTED_PHOTO_WIDTH = 160
EXPECTED_PHOTO_HEIGHT = 160

IS_RGB = True

def define_model(model_name):

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    
    model = getattr(models, model_name)(input_shape, NR_DISEASES)
        
    WEIGHTS_BACKUP  = model_name + ".weights.h5"
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    if os.path.isfile(WEIGHTS_BACKUP):
        model.load_weights(WEIGHTS_BACKUP)
    
    return model

def save_model_weights(model, model_name):
    WEIGHTS_BACKUP  = model_name + ".weights.h5"
    model.save_weights(WEIGHTS_BACKUP)

def save_model(model, model_name):
    if not os.path.exists("./saved_model"):
        subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    model.save("saved_model/" + model_name)

def main():
    print(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="VGG16")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--test_train_split", type=int, help="Data Split", default=0.2)

    args = parser.parse_args()
    
    input_data = args.input_data
    train_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model_name
    test_train_split = args.test_train_split
    
    if input_data == None:
        print("Missing train and test input, quitting")
        exit(5)
    
    # WORKAROUND FOR NOW
    try:
        os.chdir("./src/model")
    except:
        pass
    
    data = read_data(input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
    
    x_train, y_train, x_split, y_split = balanced_data_split(data, test_train_split)
    
    x_valid, x_test, y_valid, y_test = train_test_split(x_split, y_split, test_size=0.5, random_state=42)
    
    print(f"Data size:{len(x_train) + len(x_test) + len(x_valid)}")

    log_dir = "logs/" + model_name + datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    
    # create callbacks with unprocessed images
        
    x_train = x_train / 255
    x_test = x_test / 255
    x_valid = x_valid / 255
    
    train_callbacks = []
    train_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True))
    train_callbacks.append(ImagePredictionLogger((x_test, y_test), log_dir + "/prediction", train_epochs, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB))

    model = define_model(model_name)
    
    model.summary()
    
    model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_valid,  y_valid), callbacks=train_callbacks)
    
    save_model_weights(model, model_name)
    

    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()