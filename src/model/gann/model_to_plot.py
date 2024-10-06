import keras

import generators, discriminators

from data_processing_helpers import *

input_shape = [EXPECTED_PHOTO_WIDTH, EXPECTED_PHOTO_HEIGHT, 3 if IS_RGB else 1]

model = discriminators.sample(input_shape)
 
keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=False
)