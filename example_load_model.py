import tensorflow as tf

PATH_TO_SAVED_MODEL = ".\\saved_model\\my_model"

new_model = tf.keras.models.load_model(PATH_TO_SAVED_MODEL)

new_model.summary()