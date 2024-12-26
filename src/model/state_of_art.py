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
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# to disable cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import accuracy_score, precision_score, recall_score

import tensorflow as tf

from training_callbacks import ImagePredictionLogger

from data_processing_helpers import *

import models

from sklearn.model_selection import StratifiedKFold

TRAINING_RESULTS = []

def display_training_results(training_results):
    
    if len(training_results) == 0:
        return
    
    if any(metrics[0] is None for metrics in training_results):
        filtered_results = [metrics[1:] for metrics in training_results]
        for i, metrics in enumerate(filtered_results):
            logging.info(f"Iteration {i + 1}: Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}")
        
        metrics_array = np.array(filtered_results)
        mean_metrics = np.mean(metrics_array, axis=0)
    
        logging.info(f"Mean Accuracy: {mean_metrics[0]:.4f} Mean Precision: {mean_metrics[1]:.4f} Mean Recall: {mean_metrics[2]:.4f}")
        
    else: 
        for i, metrics in enumerate(training_results):
            logging.info(f"Iteration {i + 1}: Loss: {metrics[0]:.4f}, Accuracy: {metrics[1]:.4f}, Precision: {metrics[2]:.4f}, Recall: {metrics[3]:.4f}")
        
        metrics_array = np.array(training_results)
        mean_metrics = np.mean(metrics_array, axis=0)
    
        logging.info(f"Mean Loss: {mean_metrics[0]:.4f} Mean Accuracy: {mean_metrics[1]:.4f} Mean Precision: {mean_metrics[2]:.4f} Mean Recall: {mean_metrics[3]:.4f}")
        
def get_log_dir(model_type, model_name, fold=None):
    if not hasattr(get_log_dir, 'call_count'):
        get_log_dir.call_count = {}
        
    if model_type not in get_log_dir.call_count:
        get_log_dir.call_count[model_type] = 0
        
    get_log_dir.call_count[model_type] += 1
    
    log_dir = os.path.join("logs", model_type)

    os.makedirs(log_dir, exist_ok=True)
    
    if fold != None:
        log_dir = os.path.join(log_dir, f"fold_{fold}")
        os.makedirs(log_dir, exist_ok=True)
    
    iteration = get_log_dir.call_count[model_type]
    
    return os.path.join(log_dir, f"{model_name}: Iteration_{iteration}: Time {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")

def define_model(model_type, model_name, load_weights=True, learning_rate=0.001):

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    
    model = getattr(models, model_name)(input_shape, get_nr_diseases(model_type))
        
    WEIGHTS_BACKUP  = model_name + ".weights.h5"
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    if load_weights and os.path.isfile(WEIGHTS_BACKUP):
        model.load_weights(WEIGHTS_BACKUP)
    
    return model
    
def K_fold_train_model(k, model_type, model_name, train_epochs, batch_size, learning_rate, _x_train, _y_train, _x_valid, _y_valid, x_test, y_test, random_state=42):
    
    training_results_k_fold = []
    
    x_train = np.concatenate((_x_train, _x_valid), axis=0)
    y_train = np.concatenate((_y_train, _y_valid), axis=0)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    k_fold_models = []
    
    nr_fold = 0
    
    for train_index, val_index in skf.split(x_train, np.argmax(y_train, axis=1)): 
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        
        nr_fold += 1
        log_dir = get_log_dir(model_type, model_name, nr_fold)
    
        train_callbacks = []
        train_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True))
        train_callbacks.append(ImagePredictionLogger(model_type, (x_test, y_test), log_dir + "/prediction", train_epochs, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB))
        
        model = define_model(model_type, model_name, False, learning_rate)
                
        model.fit(x_fold_train, y_fold_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_fold_val,  y_fold_val), callbacks=train_callbacks)
    
        results = model.evaluate(x_test,  y_test, verbose=0)
        training_results_k_fold.append(results)
        
        k_fold_models.append(model)
        
    display_training_results(training_results_k_fold)
    
    metrics_array = np.array(training_results_k_fold)
    mean_metrics = np.mean(metrics_array, axis=0)
    
    TRAINING_RESULTS.append(mean_metrics.tolist())
    
    return k_fold_models 

def main():
    start_time = time.time()
    logging.debug(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="VGG16")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--train_rest_split", type=int, help="Data Split", default=0.2)
    parser.add_argument("--model_type", type=str, help="Type of model to use. Options: \"healthy-pharyngitis\"")
    parser.add_argument("--number_folds", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--learning_rate", type=float, help="Optimizer learning rate", default=0.001)
    
    args = parser.parse_args()
    
    model_name = "ResNet50"
    input_data = args.input_data
    train_epochs = args.epochs
    batch_size = args.batch_size
    train_rest_split = args.train_rest_split
    model_type = "healthy-pharyngitis"
    learning_rate = args.learning_rate
    
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
    
    # reduce to have same number of samples for all classes
    print("AFTER REGULARIZATION")
    min_size = min(len(value) for value in data.values())
    for key, value in data.items():
        data[key] = np.array(value)[:min_size]  # Slice to min_size
        logging.info(f"{key}: {len(data[key])}")

    
    normalize_data(data)
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = stratified_data_split(model_type, data, train_rest_split)
        
    logging.info(f"Train: {len(x_train)} Test: {len(x_test)} Validate: {len(x_valid)}")  
        
    K_fold_train_model(number_folds, model_type, model_name, train_epochs, batch_size, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test)
    
    # Stop timer
    end_time = time.time()
    elapsed_time = end_time - start_time  # Time in seconds

    # Convert seconds to minutes
    elapsed_minutes = elapsed_time / 60
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    
if __name__ == "__main__":
    main()