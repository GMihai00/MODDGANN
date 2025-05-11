from data_processing_helpers import *

import models
import datetime
import subprocess

from evaluation import *
import tensorflow as tf

MODEL_LAYERS = [
    {
        "model_type": "healthy-unhealthy",
        "model_name": "AlteredInceptionV3",
        "epochs": 40,
        "batch_size": 16,
        "learning_rate": 0.0008
    },
    {
        "model_type":  "pharyngitis-tonsil_disease",
        "model_name": "AlteredInceptionV3",
        "batch_size": 16,
        "learning_rate": 0.00005
    },
    {
        "model_type": "tonsillitis-mononucleosis",
        "model_name": "AlteredInceptionV3",
        "learning_rate": 0.0008,
        "batch_size": 16,
        "epochs": 30
    }
]

def define_model(model_type, model_name, learning_rate=0.001, weights_file=None):

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    
    model = getattr(models, model_name)(input_shape, get_nr_diseases(model_type))
        
    
    if not hasattr(define_model, "cnt"):
        model.summary()
        define_model.cnt = 0
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    if weights_file:
        if os.path.isfile(weights_file):
            model.load_weights(weights_file)
        elif weights_file == "BEST":
            
            best_model_file = os.path.join(os.curdir, "model", model_type, "best_model.txt")

            if os.path.isfile(best_model_file):
                with open(best_model_file, "r") as file:
                    weights_file_path = os.path.join(os.curdir, file.readline())
                    if os.path.isfile(weights_file_path):
                        model.load_weights(weights_file_path)
                    else:
                        raise FileNotFoundError(f"Weights file not found: {weights_file_path}")
            else:
                raise FileNotFoundError(f"Best model file not found: {best_model_file}")
    
    return model


def save_model_weights(model, model_name, model_type, performance_metrics, fold=None):

    if not hasattr(save_model_weights, 'call_count'):
        save_model_weights.call_count = {}
        
    if not hasattr(save_model_weights, 'best_performing_model'):
        save_model_weights.best_performing_model = {}
        
    if not hasattr(save_model_weights, "last_fold_nr"):
        save_model_weights.last_fold_nr = {}
        
    if model_type not in save_model_weights.call_count:
        save_model_weights.call_count[model_type] = 0
    
    model_backup_dir = os.path.join("model", model_type)

    main_model_backup_dir = os.path.join("model", model_type)

    os.makedirs(model_backup_dir, exist_ok=True)
    
    if fold != None:
        model_backup_dir = os.path.join(model_backup_dir, f"fold_{fold}")
        os.makedirs(model_backup_dir, exist_ok=True)
        if model_type not in save_model_weights.last_fold_nr:
            save_model_weights.last_fold_nr[model_type] = fold
        elif fold != save_model_weights.last_fold_nr[model_type]:
            save_model_weights.last_fold_nr[model_type] = fold
            save_model_weights.call_count[model_type] = 0
    
    save_model_weights.call_count[model_type] += 1
    
    iteration = save_model_weights.call_count[model_type]
    
    model_file = os.path.join(model_backup_dir, f"{model_name}:_Iteration_{iteration}:_Time_{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")

    WEIGHTS_BACKUP  = model_file + ".weights.h5"
    
    weights_info_file = os.path.join(main_model_backup_dir, "best_model.txt")
            
    if model_type not in save_model_weights.best_performing_model:
        save_model_weights.best_performing_model[model_type] = (performance_metrics, WEIGHTS_BACKUP)
        with open(weights_info_file, "w") as file:
            file.write(WEIGHTS_BACKUP)
            file.write("\n")
            file.write(str(performance_metrics))
    elif performance_metrics > save_model_weights.best_performing_model[model_type][0]:
        save_model_weights.best_performing_model[model_type] = (performance_metrics, WEIGHTS_BACKUP)
        with open(weights_info_file, "w") as file:
            file.write(WEIGHTS_BACKUP)
            file.write("\n")
            file.write(str(performance_metrics))
            
    if os.path.exists(WEIGHTS_BACKUP):
        os.remove(WEIGHTS_BACKUP)
    
    model.save_weights(WEIGHTS_BACKUP)

def save_model(model, model_name):
    if not os.path.exists("./saved_model"):
        subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    model.save("saved_model/" + model_name)