import itertools
import os
import subprocess
import datetime
import matplotlib.pyplot as plt
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten, InputLayer
from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf

from sklearn.model_selection import train_test_split
import numpy as np
import csv
import argparse

from PIL import Image

EXPECTED_PHOTO_WIDTH = 320
EXPECTED_PHOTO_HEIGHT = 320

IS_RGB = True

WEIGHTS_BACKUP = 'my_model.weights.h5'


DISEASE_TO_CATEGORY = {
    "pharyngitis": 0,
    "tonsillitis": 1,
    "gastric reflux": 2,
    "tonsil stones": 3,
    "healthy" : 4
}

CLASS_NAMES = list(DISEASE_TO_CATEGORY.keys())

NR_DISEASES = len(DISEASE_TO_CATEGORY)

def convert_path(path):
    if os.name == 'posix':
        return path.replace('\\', '/')
    else:
        return path

def convert_image_to_bytes(image_path):
    
    try:
    
        image = Image.open(image_path)
        
        if not IS_RGB:
            image = image.convert("L")
        
        
        width, height = image.size
        
        if height != EXPECTED_PHOTO_HEIGHT or width != EXPECTED_PHOTO_WIDTH:
            #resize to fit the model size
            print(f"Warning resizing image {image_path} to fit the model requirements")
            print(f"Initial size: {width}:{height}, target size: {EXPECTED_PHOTO_WIDTH}:{EXPECTED_PHOTO_HEIGHT}")
            image = image.resize((EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT), Image.Resampling.LANCZOS)
        
        image_array = np.array(image)
        
        return image_array.reshape(EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
        
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

def read_data(file_path):
    
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
                image_bytes = convert_image_to_bytes(convert_path(image_path))
                
                if len(image_bytes) != 0:
                    x_data.append(image_bytes)
                    y_data.append(disease_classification_distribution(DISEASE_TO_CATEGORY[disease]))
                        

    return (np.array(x_data), np.array(y_data))

def define_model():

    input_shape = (EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1)
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NR_DISEASES, activation="softmax"))
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    if os.path.isfile(WEIGHTS_BACKUP):
        model.load_weights(WEIGHTS_BACKUP)
    
    return model

def save_model_weights(model):
    model.save_weights(WEIGHTS_BACKUP)

def save_model(model, model_name):
    if not os.path.exists("./saved_model"):
        subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    model.save("saved_model/" + model_name)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def main():
    print(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, help="CSV train input file")
    parser.add_argument("--model_name", type=str, help="Model name", default="my_model")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=10)
    parser.add_argument("--test_train_split", type=int, help="Data Split", default=0.2)

    args = parser.parse_args()
    
    input_data = args.input_data
    model_name = args.model_name
    train_epochs = args.epochs
    
    if input_data == None:
        print("Missing train and test input, quitting")
        exit(5)
    
    try:
        os.chdir("./src/model")
    except:
        pass
        

    x_data, y_data = read_data(input_data)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=args.test_train_split, random_state=42)
    
    model = define_model()
    
    model.summary()
    
    print(f"Input data shape: {x_train.shape} : {y_train.shape}")
    
    log_dir = "logs/image/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    
    def log_confusion_matrix(epoch, logs):
      # Use the model to predict the values from the validation dataset.
      test_pred_raw = model.predict(x_test)
      test_pred = np.argmax(test_pred_raw, axis=1)
    
      # Calculate the confusion matrix.
      cm = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1), test_pred)
      # Log the confusion matrix as an image summary.
      figure = plot_confusion_matrix(cm, class_names=CLASS_NAMES)
      cm_image = plot_to_image(figure)
    
      # Log the confusion matrix as an image summary.
      with file_writer_cm.as_default():
        tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)
    
    # Define the per-epoch callback.
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    # problems here to see what is going on rn, maybe input dimensions are just wrong
    model.fit(x_train, y_train, epochs=train_epochs, shuffle=True, validation_data=(x_test,  y_test), callbacks=[tensorboard_callback, cm_callback])
    
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    save_model_weights(model)


if __name__ == "__main__":
    main()