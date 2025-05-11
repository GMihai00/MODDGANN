from data_processing_helpers import *
from model_helpers import *
from evaluation import *

import datetime

import tensorflow as tf

from training_callbacks import ImagePredictionLogger


from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def get_log_dir(model_type, model_name, fold=None):
    if not hasattr(get_log_dir, 'call_count'):
        get_log_dir.call_count = {}
        
    if not hasattr(get_log_dir, "last_fold_nr"):
        get_log_dir.last_fold_nr = {}
        
    if model_type not in get_log_dir.call_count:
        get_log_dir.call_count[model_type] = 0
        
    log_dir = os.path.join("logs", model_type)

    os.makedirs(log_dir, exist_ok=True)
    
    if fold != None:
        log_dir = os.path.join(log_dir, f"fold_{fold}")
        os.makedirs(log_dir, exist_ok=True)
        
        if model_type not in get_log_dir.last_fold_nr:
            get_log_dir.last_fold_nr[model_type] = fold
        elif fold != get_log_dir.last_fold_nr[model_type]:
            get_log_dir.last_fold_nr[model_type] = fold
            get_log_dir.call_count[model_type] = 0
        
    get_log_dir.call_count[model_type] += 1
        
    
    iteration = get_log_dir.call_count[model_type]
    
    return os.path.join(log_dir, f"{model_name}: Iteration_{iteration}: Time {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")


TRAINING_RESULTS = []

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
            
        y_pred_one_hot = np.zeros(4)
        y_pred_one_hot[prediction] = 1
        
        y_pred_labels.append(y_pred_one_hot)

    
    # y_labels = np.argmax(y_test, axis=1) 
    
    y_pred_labels = np.array(y_pred_labels)
    print(y_pred_labels.shape)
    
    TRAINING_RESULTS.append(ClassificationPerformanceMetrics(y_test, y_pred_labels))

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
    train_callbacks.append(EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))
    train_callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1))
    # train_callbacks.append(ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True))

    model = define_model(model_type, model_name, learning_rate)
    
    model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_valid,  y_valid), callbacks=train_callbacks)
    
    y_labels = model.predict(x_test)
    
    performance_metrics = ClassificationPerformanceMetrics(y_test, y_labels)
    
    TRAINING_RESULTS.append(performance_metrics)
    
    save_model_weights(model, model_name, model_type, performance_metrics)
        
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
        train_callbacks.append(EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))
        train_callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1))
        # train_callbacks.append(ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True))
        model = define_model(model_type, model_name, learning_rate)
                
        model.fit(x_fold_train, y_fold_train, epochs=train_epochs, batch_size=batch_size, shuffle=True, validation_data=(x_fold_val,  y_fold_val), callbacks=train_callbacks)
        
        y_pred = model.predict(x_test)
        
        performance_metrics = ClassificationPerformanceMetrics(y_test, y_pred)
        
        save_model_weights(model, model_name, model_type, performance_metrics, nr_fold)
        
        training_results_k_fold.append(performance_metrics)
        
        k_fold_models.append(model)
        
    display_training_results(training_results_k_fold)
    
    mean_metric = calculate_metrics_average(training_results_k_fold)
    
    TRAINING_RESULTS.append(mean_metric)
    
    return k_fold_models