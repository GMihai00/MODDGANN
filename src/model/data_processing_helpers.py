import os
import numpy as np
import csv

from PIL import Image

DISEASE_TO_CATEGORY = {
    "pharyngitis": 0,
    "tonsillitis": 1,
    "gastric reflux": 2,
    "tonsil stones": 3,
    "healthy" : 4
}

DISEASES = list(DISEASE_TO_CATEGORY.keys())

CLASS_NAMES = list(DISEASE_TO_CATEGORY.keys())

NR_DISEASES = len(DISEASE_TO_CATEGORY)

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
            print(f"Warning resizing image {image_path} to fit the model requirements")
            print(f"Initial size: {width}:{height}, target size: {expected_photo_width}:{expected_photo_height}")
            image = image.resize((expected_photo_width, expected_photo_height), Image.Resampling.LANCZOS)
        
        # normalize data
        image_array = np.array(image)
        
        return image_array.reshape(expected_photo_width, expected_photo_height, 3 if rgb else 1)
        
    except Exception as e:
        print(f"Image {image_path} not found")
        return []
        
def disease_classification_distribution(disease_type):
    
    arr = []
    
    for i in range(0, NR_DISEASES):
        if i == disease_type:
            arr.append(1)
        else:
            arr.append(0)
            
    return arr

def distribution_to_label(distribution):
    
    disease_index = np.argmax(distribution, axis=0)
    
    return DISEASES[disease_index]
    
def read_data(file_path, expected_photo_height, expected_photo_width, rgb):
    
    x_data = []
    y_data = []
    
    # Open the CSV file for reading
    with open(file_path, mode='r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        
        # Skip the header row (first row)
        next(reader, None)
        
        # Iterate through rows
        for row in reader:
            if len(row) == 2:  # Ensure there are two columns in the row
                image_path, disease = row
                image_bytes = convert_image_to_bytes(convert_path(image_path), expected_photo_height, expected_photo_width, rgb)
                
                if len(image_bytes) != 0:
                    x_data.append(image_bytes)
                    y_data.append(disease_classification_distribution(DISEASE_TO_CATEGORY[disease]))
                        

    return (np.array(x_data), np.array(y_data))