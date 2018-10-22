import tensorflow as tf
import pandas as pd

train_fp = "Data/fashion-mnist_train.csv"
test_fp = "Data/fashion-mnist_test.csv",
img_width = 28
img_height = 28
number_of_classes = 10

train_df = pd.read_csv(train_fp).values
no_train = train_df.shape[0]
train_data = train_df[:no_train, 1:].reshape((no_train, img_height, img_width, 1))
train_label = train_df[:no_train, 0]

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = train_data
y_train = train_label
x_train = x_train / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Conv2D(64, (5, 5), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

lr = .001
model_dir = "C:/Development/repos/python_projects/tensorflow/fashion_mnist/"
log_dir   = "%s/logs/%s" % (model_dir, lr)

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cb = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)]
# cb = []

print("Number of Train Observations: {0}".format(x_train.shape[0]))

model.fit(x_train, y_train,
          epochs=10,
          callbacks=cb,
          verbose=True)
print("Model: {0}".format(model.summary()))
