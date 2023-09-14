#EXAMPLE TAKEN FROM HARDWARD CLASS

import subprocess
import tensorflow as tf

IMAGE_DIM = (28, 28)
KERNEL_SIZE = (3, 3)
FILTERS_NUMBER = 32
RGB = False
POOL_SIZE = (2, 2)
NR_HIDDEN_NODES = 128
NR_OUTPURS = 10
# Use MNIST handwriting dataset
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(
        FILTERS_NUMBER, KERNEL_SIZE, activation="relu", input_shape=(IMAGE_DIM, 3 if RGB else 1)
    ),

    tf.keras.layers.MaxPooling2D(pool_size=POOL_SIZE),

    # Flatten units
    tf.keras.layers.Flatten(),

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(NR_HIDDEN_NODES, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(NR_OUTPURS, activation="softmax")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

# Evaluate neural network performance
model.evaluate(x_test,  y_test, verbose=2)

result = subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
model.save('saved_model/my_model')