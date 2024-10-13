
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D, Flatten, InputLayer, concatenate, GlobalAveragePooling2D, BatchNormalization


def NASNetMobile(input_shape, output_shape):
    base_model = tf.keras.applications.NASNetMobile(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model
def Xception(input_shape, output_shape):
    base_model = tf.keras.applications.Xception(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model
    
# overfitting. Need something in between VGG16 and InceptionV3
def AlteredInceptionV3(input_shape, output_shape):
    # Load InceptionV3 without the top classification layer
    base_model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Add new layers on top of the base model
    x = base_model.output  # Use output of the base model
    
    x = GlobalAveragePooling2D()(x)
    
    # Optional dense layers after global pooling
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output layer
    x = Dense(output_shape, activation='softmax')(x)
    # Create the new model
    model = Model(inputs=base_model.input, outputs=x)

    return model

def VGG16(input_shape, output_shape):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(output_shape, activation='softmax')(x) 
        
    return Model(inputs=base_model.input, outputs=predictions)

def AlteredEfficientNetB0(input_shape, output_shape):
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global pooling layer
    x = Dropout(0.5)(x)              # Add a dropout layer to prevent overfitting
    x = Dense(128, activation='relu')(x)  # Dense layer with ReLU activation
    x = BatchNormalization()(x)
    x = Dense(output_shape, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)

    return model

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
    
    
def EfficientNetV2M(input_shape, output_shape):
    base_model = tf.keras.applications.EfficientNetV2M(input_shape = input_shape, include_top = False, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    # x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    
    # x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    
    # x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    predictions = tf.keras.layers.Dense(output_shape, activation='softmax')(x) 
    
    return Model(inputs=base_model.input, outputs=predictions)
    
def VGG19(input_shape, output_shape):
    base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    # for layer in base_model.layers:
    #     layer.trainable = False
        
    x = base_model.output  # Use output of the base model
    
    x = GlobalAveragePooling2D()(x)
    
    # Optional dense layers after global pooling
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output layer
    x = Dense(output_shape, activation='softmax')(x)
        
    return Model(inputs=base_model.input, outputs=x)