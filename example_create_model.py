import subprocess
import tensorflow as tf


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
                                        
# X_training, X_testing, y_training, y_testing = train_test_split(
#     evidence, labels, test_size=0.4
# )

# model.fit(X_training, y_training, epochs=20)

# # Evaluate how well model performs
# model.evaluate(X_testing, y_testing, verbose=2)


result = subprocess.run(["mkdir", "-p", "saved_model"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
model.save('saved_model/my_model')