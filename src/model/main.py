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

def save_model_weights(model, model_name):
    WEIGHTS_BACKUP  = model_name + ".weights.h5"
    model.save_weights(WEIGHTS_BACKUP)

def save_model(model, model_name):
    if not os.path.exists("./saved_model"):
        subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    model.save("saved_model/" + model_name)

TRAINING_RESULTS = []

MODEL_LAYERS = [
    {
        "model_type": "healthy-unhealthy",
        "model_name": "InceptionV3"
    },
    {
        "model_type":  "pharyngitis-tonsil_disease",
        "model_name": "AlteredInceptionV3",
        "batch_size": "16",
        "learning_rate": "0.00005"
    },
    {
        "model_type": "tonsillitis-mononucleosis",
        "model_name": "VGG16"
    }
]

def evaluate_ensemble_accuracy(models, x_test, y_test):
    y_pred_labels = []
    
    for input in x_test:
    
        if input.ndim == 3:
            input = np.expand_dims(input, axis=0)
            print(f"Expanded input shape: {input.shape}")
    
        if input.shape[1:] != (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1):
            raise ValueError(f"Input shape mismatch, got {input.shape[1:]}")
                
        it = 0
        prediction = -1
        
        while prediction == -1:
        
            output_prediction = models[it].predict(input)
            disease_index = np.argmax(output_prediction, axis=0)
            
            output_prediction = output_prediction.flatten()
            
            disease_index = np.argmax(output_prediction)
            
            if disease_index == 0:
                prediction = it
                
                # it = 0 healthy-unhealthy, 0 means healthy
                # it = 1 pharyngitis_tonsil_disease, 1 means pharyngitis
                # it = 2 tonsillitis_mononucleosis, 2 mean tonsillitis
                
            elif it == len(models) - 1:
                prediction = it + 1   # it = 2 tonsillitis_mononucleosis, 3 means mononucleosis
        
                
            it+=1
            
        y_pred_labels.append(prediction)
    
    y_labels = np.argmax(y_test, axis=1) 
    
    accuracy = accuracy_score(y_labels, y_pred_labels)
    precision = precision_score(y_labels, y_pred_labels, average='weighted')  
    recall = recall_score(y_labels, y_pred_labels, average='weighted')
    
    
    TRAINING_RESULTS.append([None, accuracy, precision, recall])


def match_model_labels(model_type, x_train, y_train, x_valid, y_valid, x_test, y_test):
        
    y_train_specific = []
    y_valid_specific = []
    y_test_specific = []
    x_train_specific = []
    x_valid_specific = []
    x_test_specific = []
    
    for x, y in zip(x_train, y_train):
        new_y = convert_model_distribution("ensemble", model_type, y)
        if new_y != None:
            y_train_specific.append(new_y)
            x_train_specific.append(x)
        
    for x, y in zip(x_valid, y_valid):
        new_y = convert_model_distribution("ensemble", model_type, y)
        if new_y != None:
            y_valid_specific.append(new_y)
            x_valid_specific.append(x)
        
    for x, y in zip(x_test, y_test):
        new_y = convert_model_distribution("ensemble", model_type, y)
        if new_y != None:
            y_test_specific.append(new_y)
            x_test_specific.append(x)
    
    return np.array(x_train_specific), np.array(y_train_specific), np.array(x_valid_specific), np.array(y_valid_specific), np.array(x_test_specific), np.array(y_test_specific)

def get_data_diseases(model_type, y_train, y_valid, y_test):

    entries = np.concatenate((y_train, y_valid, y_test), axis=0)
    
    data = {
    }
    
    for disease in get_disease_to_category(model_type).keys():
        data[disease] = 0

    for entry in entries:
        data[distribution_to_label(model_type, entry)] += 1
    
    return data

def K_fold_train_ensemble_model(k, train_epochs, batch_size, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test):
    
    k_fold_models = []
    
    for entry in MODEL_LAYERS:
        model_type = entry["model_type"]
        model_name = entry["model_name"]
        model_train_epochs = entry.get("epochs", train_epochs)
        model_batch_size = entry.get("batch_size", batch_size)
        model_learning_rate = entry.get("learning_rate", learning_rate)
        print(f"Training model: {model_type}")
        
        x_train_specific, y_train_specific, x_valid_specific, y_valid_specific, x_test_specific, y_test_specific = match_model_labels(model_type, x_train, y_train, x_valid, y_valid, x_test, y_test)
        
        logging.info(f"Train: {len(x_train_specific)} Test: {len(x_test_specific)} Validate: {len(x_valid_specific)}") 
        
        data = get_data_diseases(model_type, y_train_specific, y_test_specific, y_valid_specific)
        
        for key, value in data.items():
            logging.info(f"{key}: {value}")
        
        k_fold_sub_models = K_fold_train_model(k, model_type, model_name, model_train_epochs, model_batch_size, model_learning_rate, x_train_specific, y_train_specific, x_valid_specific, y_valid_specific, x_test_specific, y_test_specific)
        k_fold_models.append(k_fold_sub_models)
        
        display_training_results(TRAINING_RESULTS)
        
        print("....................................................................................................\n\n")
        TRAINING_RESULTS.clear()
    
    # group models from the same fold together to form the ensemble for each fold
    k_fold_model_groups = list(zip(*k_fold_models))
    
    for model_group in k_fold_model_groups:
        evaluate_ensemble_accuracy(model_group, x_test, y_test)
    

def train_model(model_type, model_name, train_epochs, batch_size, learning_rate, x_train, y_train, x_valid, y_valid, x_test, y_test):
    
    log_dir = get_log_dir(model_type, model_name)
    
    train_callbacks = []
    train_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True))
    train_callbacks.append(ImagePredictionLogger(model_type, (x_test, y_test), log_dir + "/prediction", train_epochs, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB))

    model = define_model(model_type, model_name, False, learning_rate)
    
    model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_valid,  y_valid), callbacks=train_callbacks)
    
    results = model.evaluate(x_test,  y_test, verbose=0)
    
    TRAINING_RESULTS.append(results)
    
    return model
    # save_model_weights(model, model_name)

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

def main():
    start_time = time.time()
    logging.debug(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="VGG16")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=20)
    parser.add_argument("--train_rest_split", type=int, help="Data Split", default=0.2)
    parser.add_argument("--model_type", type=str, help="Type of model to use. Options: \"healthy-unhealthy\"; \"pharyngitis-tonsil_disease\"; \"tonsillitis-mononucleosis\"; \"ensemble\"")
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