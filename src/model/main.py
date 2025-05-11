import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of the log messages
)

import argparse
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# to disable cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf

from data_processing_helpers import *

from model_helpers import * 

from training_helpers import *

    
def main():
    start_time = time.time()
    logging.debug(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="VGG16")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--train_rest_split", type=int, help="Data Split", default=0.2)
    parser.add_argument("--model_type", type=str, help="Type of model to use. Options: \"healthy-unhealthy\"; \"pharyngitis-tonsil_disease\"; \"tonsillitis-mononucleosis\"; \"ensemble\", \"unbalanced\"")
    parser.add_argument("--training_sample", type=int, help="Number of train-test iterations", default=1)
    parser.add_argument("--number_folds", type=int, help="Number of folds for cross-validation", required=False)
    parser.add_argument("--learning_rate", type=float, help="Optimizer learning rate", default=0.001)
    
    args = parser.parse_args()
    
    input_data = args.input_data
    train_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model_name
    train_rest_split = args.train_rest_split
    model_type = args.model_type
    training_sample = args.training_sample
    learning_rate = args.learning_rate
    
    number_folds = None
    if hasattr(args, 'number_folds'):
        number_folds = args.number_folds
    
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

    
    for _ in range (0, training_sample):
        x_train, y_train, x_valid, y_valid, x_test, y_test = stratified_data_split(model_type, data, train_rest_split)
        
        logging.info(f"Train: {len(x_train)} Test: {len(x_test)} Validate: {len(x_valid)}")  
                    
        if model_type == "ensemble" and number_folds != None:
            K_fold_train_ensemble_model(number_folds, train_epochs, batch_size, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test)
        elif number_folds != None:
            K_fold_train_model(number_folds, model_type, model_name, train_epochs, batch_size, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test)
        else: 
            train_model(model_type, model_name, train_epochs, batch_size, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test)
        
    display_training_results(TRAINING_RESULTS)
    
    # Stop timer
    end_time = time.time()
    elapsed_time = end_time - start_time  # Time in seconds

    # Convert seconds to minutes
    elapsed_minutes = elapsed_time / 60
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    
if __name__ == "__main__":
    main()