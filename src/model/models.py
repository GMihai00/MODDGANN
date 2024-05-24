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
    base_model = tf.keras.applications.EfficientNetB0(input_shape = input_shape, include_top = False, weights = 'imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_shape, activation="softmax")(x)
    
    return Model(inputs = base_model.input, outputs = predictions)

def EfficientNetB7(input_shape, output_shape):
    base_model = tf.keras.applications.EfficientNetB7(input_shape = input_shape, include_top = False, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_shape, activation="softmax")(x)
    
    return Model(inputs = base_model.input, outputs = predictions)

def InceptionV3(input_shape, output_shape):
    base_model = tf.keras.applications.InceptionV3(input_shape = input_shape, include_top = True, weights = None, classes = output_shape, classifier_activation='softmax')
    
    return base_model

def ResNet50(input_shape, output_shape):
    base_model = Sequential()
        
    base_model.add(tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max'))
    base_model.add(Dense(output_shape, activation="softmax"))
    
    return base_model