import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D, Flatten, InputLayer

def VGG16(input_shape, output_shape):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(output_shape, activation='softmax')(x) 
        
    return Model(inputs=base_model.input, outputs=predictions)

# only works with RGB images
def EfficientNetB0(input_shape, output_shape):
    base_model = tf.keras.applications.EfficientNetB0(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model

def EfficientNetB7(input_shape, output_shape):
    base_model = tf.keras.applications.EfficientNetB7(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model

def InceptionV3(input_shape, output_shape):
    base_model = tf.keras.applications.InceptionV3(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model

def ResNet50(input_shape, output_shape):
    base_model = tf.keras.applications.ResNet50(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model
    
    
def ResNet50V2(input_shape, output_shape):
    base_model = tf.keras.applications.ResNet50V2(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model