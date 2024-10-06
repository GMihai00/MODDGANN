import os
import numpy as np
import csv
from sklearn.utils import shuffle
import logging

from PIL import Image

from sklearn.model_selection import train_test_split

EXPECTED_PHOTO_WIDTH = 160
EXPECTED_PHOTO_HEIGHT = 160

IS_RGB = True

def get_disease_to_category(model_type) -> dict:
    if model_type == 'healthy-unhealthy':
        return {
            "healthy": 0,
            "unhealthy": 1
        }
    elif model_type == 'pharyngitis-tonsil_disease':
        return {
            "pharyngitis": 0,
            "tonsil_disease": 1
        }
    elif model_type == 'tonsillitis-mononucleosis':
        return {
            "tonsillitis": 0,
            "mononucleosis": 1
        }
    elif model_type == 'ensemble':
        return {
            "healthy": 0,
            "pharyngitis": 1,
            "tonsillitis": 2,
            "mononucleosis": 3
        }
    else:
        raise ValueError(f"Unknown training type: {model_type}")

def get_diseases(model_type) -> list:
    return list(get_disease_to_category(model_type).keys())

def get_nr_diseases(model_type):
    return len(get_disease_to_category(model_type))

def convert_path(path):
    if os.name == 'posix':
        return path.replace('\\', '/')
    else:
        return path
        
def convert_image_to_bytes(image_path, expected_photo_height, expected_photo_width, rgb):
    try:
    
        image = Image.open(image_path)
        
        if not rgb:
            image = image.convert("L")
        
        width, height = image.size
        
        if height != expected_photo_height or width != expected_photo_width:
            #resize to fit the model size
            logging.debug(f"Resizing image {image_path} to fit the model requirements.\nInitial size: {width}:{height}, target size: {expected_photo_width}:{expected_photo_height}")
            image = image.resize((expected_photo_width, expected_photo_height), Image.Resampling.LANCZOS)
        
        # normalize data
        image_array = np.array(image)
        
        image.close()
        
        logging.debug(f"Image array shape before reshape: {image_array.shape}")

        return image_array.reshape(expected_photo_width, expected_photo_height, 3 if rgb else 1)
        
    except Exception as e:
        logging.error(f"Image {image_path} not found, {repr(e)}")
        return []
        
def disease_classification_distribution(model_type, disease_type):
    
    arr = []
    
    for i in range(0, get_nr_diseases(model_type)):
        if i == disease_type:
            arr.append(1)
        else:
            arr.append(0)
            
    return arr

def distribution_to_label(model_type, distribution):
    
    disease_index = np.argmax(distribution, axis=0)
    
    return get_diseases(model_type)[disease_index]
    
def read_data(model_type, file_path, expected_photo_height, expected_photo_width, rgb):
    
    data = {
    }
    
    for disease in get_disease_to_category(model_type).keys():
        data[disease] = []
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        
        next(reader, None)
        
        for row in reader:
            if len(row) == 2:  # Ensure there are two columns in the row
                image_path, disease = row
                image_bytes = convert_image_to_bytes(convert_path(image_path), expected_photo_height, expected_photo_width, rgb)
                
                if model_type == 'healthy-unhealthy':
                    if disease != 'healthy':
                        disease = 'unhealthy'
                elif model_type == 'pharyngitis-tonsil_disease':
                    if disease != "pharyngitis" and disease != "healthy":
                        disease = "tonsil_disease"
                
                if len(image_bytes) == 0:
                    if not os.path.exists(f'errs.txt'):
                        with open(f'errs.txt', 'w') as file:
                            file.write(f"{image_path}\n")
                    else:
                        with open('errs.txt', 'a') as file:
                            file.write(f"{image_path}\n")
                
                if len(image_bytes) != 0 and disease in data.keys():
                    data[disease].append(image_bytes)
                    
    for key, value in data.items():
        data[key] = np.array(value)
        logging.info(f"{key}: {len(data[key])}")

    return data
    
def normalize_data(data: dict):
    for key, value in data.items():
        data[key] = value / 255

def balanced_data_split(model_type, data, test_train_split, random_state=42):
    
    x_train_final = []
    y_train_final = []
    x_test_final = []
    y_test_final = []
    
    disease_to_category = get_disease_to_category(model_type)
    for key, value in data.items():
        x_data = value
        y_data = []
        
        data = disease_classification_distribution(model_type, disease_to_category[key])
        
        for _ in range(0, len(x_data)):
            y_data.append(data)
    
        y_data = np.array(y_data)
        
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_train_split, random_state=random_state)
        
        x_test_final.extend(x_test)
        y_test_final.extend(y_test)
        x_train_final.extend(x_train)
        y_train_final.extend(y_train)
        
        x_test_final, y_test_final = shuffle(x_test_final, y_test_final, random_state=random_state)
        x_train_final, y_train_final = shuffle(x_train_final, y_train_final, random_state=random_state)
        
    return np.array(x_train_final), np.array(y_train_final), np.array(x_test_final), np.array(y_test_final)
    
def stratified_data_split(model_type, data: dict, train_rest_split, test_validate_split=0.5):
    x_train, y_train, x_split, y_split = balanced_data_split(model_type, data, train_rest_split)
    
    x_valid, x_test, y_valid, y_test = train_test_split(x_split, y_split, test_size=test_validate_split, random_state=42)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test