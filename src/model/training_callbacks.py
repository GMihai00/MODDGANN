import io
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from data_processing_helpers import distribution_to_label, CLASS_NAMES

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

class ConfusionMatrixLogger(Callback):
    def __init__(self, validation_data, log_dir, nr_epochs):
        super(ConfusionMatrixLogger, self).__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.nr_epochs = nr_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.nr_epochs - 1:
        
            x_test, y_test = self.validation_data
            # Use the model to predict the values from the validation dataset.
            test_pred_raw = self.model.predict(x_test)
            test_pred = np.argmax(test_pred_raw, axis=1)
        
            # Calculate the confusion matrix.
            cm = confusion_matrix(np.argmax(y_test, axis=1), test_pred)
            # Log the confusion matrix as an image summary.
            figure = plot_confusion_matrix(cm, class_names=CLASS_NAMES)
            cm_image = plot_to_image(figure)
        
            # Log the confusion matrix as an image summary.
            with tf.summary.create_file_writer(self.log_dir).as_default():
                tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)
                    
class ImagePredictionLogger(Callback):
    def __init__(self, validation_data, log_dir, nr_epochs, expected_photo_height, expected_photo_width, rgb):
        super(ImagePredictionLogger, self).__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.nr_epochs = nr_epochs
        self.expected_photo_height = expected_photo_height
        self.expected_photo_width = expected_photo_width
        self.rgb = rgb

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.nr_epochs - 1:
            
            images, labels = self.validation_data
            
            with tf.summary.create_file_writer(self.log_dir).as_default():
                predictions = self.model.predict(images)
                
                for i in range(0, len(predictions)):
                    # Log images and predictions to TensorBoard
                    
                    display_image = np.reshape(images[i], (-1, self.expected_photo_width , self.expected_photo_height, 3 if self.rgb else 1))
                    
                    tf.summary.image(f"{distribution_to_label(labels[i])}_{distribution_to_label(predictions[i])}_{i}", display_image, step=epoch, description=f"label: {distribution_to_label(labels[i])}\n prediction: {distribution_to_label(predictions[i])}")