import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of the log messages
)

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

def get_log_dir(model_name):
    return "logs/" + model_name + datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")

def define_model(model_type, model_name):

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    
    model = getattr(models, model_name)(input_shape, get_nr_diseases(model_type))
        
    WEIGHTS_BACKUP  = model_name + ".weights.h5"
    
    model.compile(
        optimizer='adam',
        # # for bigger jumps, in case of stuck in plato zone for to long, applied to last model
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
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
    logging.debug(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="VGG16")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--train_rest_split", type=int, help="Data Split", default=0.2)
    parser.add_argument("--model_type", type=str, help="Type of model to use. Options: \"healthy-unhealthy\"; \"pharyngitis-tonsil_disease\"; \"tonsillitis-mononucleosis\"; \"ensemble\"")
    
    args = parser.parse_args()
    
    input_data = args.input_data
    train_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model_name
    train_rest_split = args.train_rest_split
    model_type = args.model_type
    
    if input_data == None:
        logging.error("Missing train and test input, quitting")
        exit(5)
    
    # WORKAROUND FOR NOW
    try:
        os.chdir("./src/model")
    except:
        pass
    
    data = read_data(model_type, input_data, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)
    
    normalize_data(data)
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = stratified_data_split(model_type, data, train_rest_split)
    
    logging.info(f"Train: {len(x_train)} Test: {len(x_test)} Validate: {len(x_valid)}")

    log_dir = get_log_dir(model_name)
    
    train_callbacks = []
    train_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True))
    train_callbacks.append(ImagePredictionLogger(model_type, (x_test, y_test), log_dir + "/prediction", train_epochs, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB))

    model = define_model(model_type, model_name)
    
    # model.summary()
    
    model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_valid,  y_valid), callbacks=train_callbacks)
    
    save_model_weights(model, model_name)
    
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()