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

import tensorflow as tf

from model_helpers import *

import numpy as np
import matplotlib.pyplot as plt
import cv2

from data_processing_helpers import convert_image_to_bytes, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB

def make_cam_heatmap(img_array, model, last_conv_layer_name):
    # Get the output of the last convolutional layer
    base_model = model.get_layer("inception_v3")
    conv_outputs = base_model.get_layer(last_conv_layer_name).output

    # Get the weights of the final dense layer
    final_dense_layer = model.layers[-1]  # Assuming the last layer is the dense layer
    weights = final_dense_layer.get_weights()[0]  # Shape: (num_features, num_classes)

    # Get the predicted class
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])

    # Get the weights for the predicted class
    class_weights = weights[:, pred_index]

    img_array = tf.image.resize(img_array, (224, 224)) 
    # Compute the weighted sum of the feature maps
    conv_outputs = tf.keras.Model(inputs=base_model.input, outputs=conv_outputs).predict(img_array)
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    
    
    heatmap = np.dot(conv_outputs, class_weights)  # Shape: scalar

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    return heatmap

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    
    base_model = model.get_layer("inception_v3")
    
    for layer in base_model.layers:
        layer.trainable = True
        
    # Feed input into base model
    base_input = base_model.input
    base_output = base_model(base_input)

    # Pass base model output into the rest of the model manually
    x = base_output
    for layer in model.layers:
        if layer.name == "inception_v3":
            continue  # already handled
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue  # skip input layer
        try:
            x = layer(x)
        except TypeError:
            print(f"Skipping layer: {layer.name} ({type(layer)}) — not callable with a single input")
            continue
    final_output = x

    # Build grad model from new computation graph
    grad_model = tf.keras.Model(
        inputs=base_input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, final_output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

        class_channel = tf.reduce_sum(class_channel)
        
        print(f"conv_outputs shape: {conv_outputs.shape}")
        print(f"predictions shape: {predictions.shape}")
        print(f"class_channel shape: {class_channel.shape}")
        
        # Gradients of the class output value w.r.t. feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise ValueError("Gradients could not be computed. Ensure the tensors are connected to the computation graph.")
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the convolution outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    
        # Normalize between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded. Check the file path.")
    

    # Resize the heatmap to match the image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255

    # Apply a colormap to the heatmap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)

    # Save and display the result
    cv2.imwrite(cam_path, superimposed_img)
    plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=True)
    

def main():
    logging.debug(tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_path", type=str, help="CSV train input file", default="../../src/model/gann/generated_images/398.png")
    parser.add_argument("--model_type", type=str, help="Type of model to use. Options: \"healthy-unhealthy\"; \"pharyngitis-tonsil_disease\"; \"tonsillitis-mononucleosis\"; \"ensemble\", \"unbalanced\"", required=True)
    
    args = parser.parse_args()
    
    img_path = args.img_path
    model_type = args.model_type
    
    model = None
    
    # WORKAROUND FOR NOW
    try:
        os.chdir("./src/model")
    except:
        pass
    
    
    for data in MODEL_LAYERS:
        if data["model_type"] == model_type:
            model_name = data["model_name"]
            learning_rate = data["learning_rate"]
            model = define_model(model_type, model_name, learning_rate=learning_rate, weights_file="BEST")
            break
    
    if model is None:
        logging.error(f"Model type {model_type} not found.")
        return
    
    # inception_v3_model = model.get_layer("inception_v3")
    # inception_v3_model.summary()  #
    
    print(model.inputs)
    
    img_array = convert_image_to_bytes(img_path, EXPECTED_PHOTO_HEIGHT, EXPECTED_PHOTO_WIDTH, IS_RGB)

    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # img_array = tf.image.resize(img_array, (224, 224)) # resize to match inceptionv3 layers
    
    heatmap = make_cam_heatmap(img_array, model, last_conv_layer_name="mixed10")
    
    save_and_display_gradcam(img_path, heatmap, cam_path = model_type + "_cam.jpg")

if __name__ == "__main__":
    main()
