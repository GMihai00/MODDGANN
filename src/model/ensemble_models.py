# DEPRECATED

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
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np

def load_model(model_name, weights_file=None):

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    
    model = getattr(models, model_name)(input_shape, 2)
        
    if weights_file == None:    
        weights_file  = model_name + ".weights.h5"
    
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
    
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    
    return model
    
MODEL_LAYERS = [
    {
        "name": "healthy-unhealthy",
        "type": "InceptionV3",
        "weightsFile": "./final_saved_models/92acc_healthy_unhealthy_InceptionV3.weights.h5"
    },
    {
        "name": "pharyngitis_tonsil_disease",
        "type": "InceptionV3",
        "weightsFile": "./final_saved_models/90acc_pharyngitis_tonsil_disease_InceptionV3.weights.h5"
    },
    {
        "name": "tonsillitis_mononucleosis",
        "type": "VGG16",
        "weightsFile": "./final_saved_models/90acc_tonsillitis_mononucleosis_VGG16.weights.h5"
    }
]

def main():
    print(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--test_train_split", type=int, help="Data Split", default=0.2)

    args = parser.parse_args()
    
    input_data = args.input_data
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
    
    x_train = x_train / 255
    x_test = x_test / 255
    x_valid = x_valid / 255
    
    x = np.concatenate([x_train, x_test, x_valid], axis=0)
    y = np.concatenate([y_train, y_test, y_valid], axis=0)
    
    models = []
    
    for model_entry in MODEL_LAYERS:
        model = load_model(model_entry["type"], model_entry["weightsFile"])
        model.summary()
        models.append(model)
    
    # y_pred_distribution = []
    y_pred_labels = []
    
    for input in x:
    
        # If the input is a single image, expand its dimensions to create a batch
        if input.ndim == 3:
            input = np.expand_dims(input, axis=0)
            print(f"Expanded input shape: {input.shape}")
    
        # Ensure input data shape matches the model's expected input shape
        if input.shape[1:] != (160, 160, 3):
            raise ValueError(f"Input shape mismatch. Expected (160, 160, 3), got {input.shape[1:]}")
                
        it = 0
        prediction = -1
        # extended_prediction = np.zeros(4)
        
        # need to extend distribution
        while prediction == -1:
        
            output_prediction = models[it].predict(input)
            disease_index = np.argmax(output_prediction, axis=0)
            
            output_prediction = output_prediction.flatten()
            
            disease_index = np.argmax(output_prediction)
            
            if disease_index == 0:
                prediction = it
                # extended_prediction[it:it+2] = output_prediction
                
                # it = 0 healthy-unhealthy, 0 means healthy
                # it = 1 pharyngitis_tonsil_disease, 1 means pharyngitis
                # it = 2 tonsillitis_mononucleosis, 2 mean tonsillitis
            elif it == len(models) - 1:
                prediction = it + 1   # it = 2 tonsillitis_mononucleosis, 3 means mononucleosis
                # extended_prediction[it:it+2] = output_prediction
        
                
            it+=1
            
        # y_pred_distribution.append(extended_prediction)
        y_pred_labels.append(prediction)

        # print(f"value: {distribution_to_label("ensemble", output)} prediction: {get_diseases("ensemble")[prediction]}")
    
    # y_pred = np.array(y_pred_distribution)     
    
    
    y_labels = np.argmax(y, axis=1) 
    
    accuracy = accuracy_score(y_labels, y_pred_labels)
    precision = precision_score(y_labels, y_pred_labels, average='weighted')  # Adjust `average` parameter based on your needs
    recall = recall_score(y_labels, y_pred_labels, average='weighted')  # Adjust `average` parameter based on your needs
    
    # Accuracy: 0.9303
    # Precision: 0.9322
    # Recall: 0.9303
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    main()