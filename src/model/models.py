
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D, Flatten, InputLayer, concatenate, GlobalAveragePooling2D, BatchNormalization, Input,  Resizing
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU


def NASNetMobile(input_shape, output_shape):
    base_model = tf.keras.applications.NASNetMobile(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model
def Xception(input_shape, output_shape):
    base_model = tf.keras.applications.Xception(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model
    
def AlteredTLInceptionV3(input_shape, output_shape, freeze_layers=True):
    # Load InceptionV3 without the top classification layer
    input_layer = Input(shape=input_shape)
    
    upsampled_input = Resizing(299, 299)(input_layer)
    
    base_model = tf.keras.applications.InceptionV3(
        input_shape=upsampled_input.shape[1:],
        include_top=False,
        weights="imagenet"
    )
    
    # Optionally freeze layers in the base model
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers in the base model
    
    # for layer in base_model.layers[-5:]:  # Unfreeze the last 20 layers
    #     layer.trainable = True
    
    # Add new layers on top of the base model
    x = base_model(upsampled_input)  # Use output of the base model
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation=LeakyReLU(alpha=0.1))(x)
    
    # Output layer for classification
    output = Dense(output_shape, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=input_layer, outputs=output)
    
    return model
    
def AlteredInceptionV3(input_shape, output_shape):
    # Load InceptionV3 without the top classification layer
    
    input_layer = Input(shape=input_shape)
    
    upsampled_input = Resizing(224, 224)(input_layer)
    
    base_model = tf.keras.applications.InceptionV3(
        input_shape=upsampled_input.shape[1:],
        include_top=False,
        weights="imagenet"
    )
    for layer in base_model.layers:
        layer.trainable = False

    # Add new layers on top of the base model
    x = base_model(upsampled_input)   # Use output of the base model
    
    x = GlobalAveragePooling2D()(x)
    
    # Optional dense layers after global pooling
    x = Dense(1024, activation=LeakyReLU(alpha=0.1))(x)
    
    # Output layer
    x = Dense(output_shape, activation='softmax')(x)
    # Create the new model
    model = Model(inputs=input_layer, outputs=x)

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

def AlteredEfficientNetB7(input_shape, output_shape):
    input_layer = Input(shape=input_shape)
    
    upsampled_input = Resizing(224, 224)(input_layer)
    
    base_model = tf.keras.applications.EfficientNetB7(
        input_shape=upsampled_input.shape[1:],
        include_top=False,
        weights="imagenet"
    )
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-5:]: #
        layer.trainable = True
        
    # Add new layers on top of the base model
    x = base_model(upsampled_input)   # Use output of the base model
    
    x = GlobalAveragePooling2D()(x)
    
    # Optional dense layers after global pooling
    x = Dense(1024, activation=LeakyReLU(alpha=0.1))(x)
    
    # Output layer
    x = Dense(output_shape, activation='softmax')(x)
    # Create the new model
    model = Model(inputs=input_layer, outputs=x)

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


def ResNet50(input_shape, output_shape, freeze_layers=True):
    input_layer = Input(shape=input_shape)
    
    upsampled_input = Resizing(224, 224)(input_layer)
    
    base_model = tf.keras.applications.ResNet50(
        input_shape=upsampled_input.shape[1:],
        include_top=False,
        weights="imagenet"
    )
    
    # Optionally freeze layers in the base model
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers in the base model
    
    for layer in base_model.layers[-5:]: #
        layer.trainable = True
    
    # Add new layers on top of the base model
    x = base_model(upsampled_input)  # Use output of the base model
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.1))(x)
    # x = Dropout(0.1)(x)
    # x = Dense(512, activation=LeakyReLU(alpha=0.1))(x)
    
    # Output layer for classification
    output = Dense(output_shape, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

def ResNet50V2(input_shape, output_shape, freeze_layers=True):
    input_layer = Input(shape=input_shape)
    
    upsampled_input = Resizing(224, 224)(input_layer)
    
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=upsampled_input.shape[1:],
        include_top=False,
        weights="imagenet"
    )
    
    # Optionally freeze layers in the base model
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers in the base model
    
    for layer in base_model.layers[-5:]: #
        layer.trainable = True
    
    # Add new layers on top of the base model
    x = base_model(upsampled_input)  # Use output of the base model
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.1))(x)
    # x = Dropout(0.1)(x)
    # x = Dense(512, activation=LeakyReLU(alpha=0.1))(x)
    
    # Output layer for classification
    output = Dense(output_shape, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=input_layer, outputs=output)
    
    return model
    
    
def EfficientNetV2M(input_shape, output_shape, freeze_layers=True):
    
    input_layer = Input(shape=input_shape)
    
    upsampled_input = Resizing(224, 224)(input_layer)
    
    base_model = tf.keras.applications.EfficientNetV2M(
        input_shape=upsampled_input.shape[1:],
        include_top=False,
        weights="imagenet"
    )
    # Optionally freeze layers in the base model
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False  # Freeze all layers in the base model
    
    for layer in base_model.layers[-5:]: #
        layer.trainable = True
    
    # Add new layers on top of the base model
    x = base_model(upsampled_input)  # Use output of the base model
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.1))(x)
    # x = Dropout(0.1)(x)
    # x = Dense(512, activation=LeakyReLU(alpha=0.1))(x)
    
    # Output layer for classification
    output = Dense(output_shape, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=input_layer, outputs=output)
    
    return model
    
def VGG19(input_shape, output_shape):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(output_shape, activation='softmax')(x) 
    
    return Model(inputs=base_model.input, outputs=predictions)