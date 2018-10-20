# for update inspirations:
# https://github.com/cs230-stanford/cs230-code-examples
# shuffle repeat batch nicely visualized: https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Params import Params
from tensorflow import keras

# TODO: the training loop should be split
# TODO: saving model state on improvement
# TODO: this should be a high level file calling the whole model building/training - this is the training file

def show_row(m, width=28, height=28):
    plt.imshow(X = m.reshape((height, width)), cmap='gray')
    plt.show()


class TFFashionMNIST(tf.keras.Model):
    def __init__(self,
                 learning_rate=.003,
                 epochs    =10,
                 batch_size=10,
                 train_fp="Data/fashion-mnist_train.csv",
                 test_fp="Data/fashion-mnist_test.csv",
                 verbose=False):
        super(TFFashionMNIST, self).__init__(name='TFFashionMNIST')

        self.training_fp = train_fp
        self.testing_fp = test_fp
        self.img_width = 28
        self.img_height = 28
        self.number_of_classes = 10
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.no_epochs = epochs
        # image shape = batch, height, width, color channels
        if self.verbose:
            print("Learning Rate: {0} - Batch Size: {1} - Number of Epochs: {2}".format(self.learning_rate,
                                                                                         self.batch_size,
                                                                                         self.no_epochs))

        self.__create_model()
        self.__create_loss()
        self.__create_optimizer()

    def __global_preprocess(self, data, label):
        data = tf.cast(data, dtype=tf.float32)
        data = data / 255.
        label = tf.cast(label, dtype=tf.int64)
        return (data, label)

    def __train_preprocess(self, data, label):
        return (data, label)

    def __test_preprocess(self, data, label):
        # data = tf.reshape(data, (-1, self.img_height, self.img_width, 1))
        return (data, label)

    def import_data(self, import_test = False, verbose=False):
        if verbose:
            print("Reading Training Data: %s" % self.training_fp)

        self.train_df = pd.read_csv(self.training_fp).values
        self.no_train   = self.train_df.shape[0]
        self.train_data = self.train_df[:self.no_train, 1:].reshape((self.no_train, self.img_height, self.img_width, 1))
        self.train_data = self.train_data / 255.
        self.train_label= self.train_df[:self.no_train, 0]

        if verbose:
            print("Training Set Size: {0}".format(self.no_train))

        if import_test:
            if verbose:
                print("Reading Testing Data: %s" % self.testing_fp)
            self.test_df = pd.read_csv(self.training_fp).values
            self.no_test = self.train_df.shape[0]
            self.test_data = self.test_df[:self.no_test, 1:].reshape((self.no_test, self.img_height, self.img_width, 1))
            self.test_data = self.test_data / 255.
            self.test_label = self.test_df[:self.no_test, 0]
            if verbose:
                print("Testing Set Size: {0}".format(self.no_test))


    def __create_model(self):
        self.model = keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), padding="same"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    def __str__(self):
        self.model.summary()

    def __create_loss(self):
        # softmax cross entropy loss for multi-class classification
        with tf.name_scope('loss'):
            self.loss_fn = tf.keras.losses.sparse_categorical_crossentropy
            self.accuracy_fn = tf.keras.metrics.get('sparse_categorical_accuracy')

    def __create_optimizer(self):
        self.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                     loss = self.loss_fn,
                     metrics=[self.accuracy_fn])

    def train(self, log_dir=None):
        callbacks = []
        if log_dir:
            if self.verbose:
                print("Enabling Keras Tensorboard Callback: %s" % log_dir)
            kcb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
            callbacks.append(kcb)

        self.fit(x=self.train_data,
                 y=self.train_label,
                 epochs=self.no_epochs,
                 verbose=True,
                 callbacks=callbacks
                 )

    def call(self, inputs, training=None, mask=None):
        x = inputs
        return self.model(x)

# x_train shape: (60000, 28, 28) y_train shape: (60000,)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

tf.set_random_seed(1234)

parms = Params.create(json_file_path="experiments/base_model/params.json")

lr = parms.params["learning_rate"]
no_epochs = parms.params["no_epochs"]
batch_size = parms.params["batch_size"]

model_dir = "C:/Development/repos/python_projects/tensorflow/fashion_mnist"
log_dir   = "%s/logs/%s" % (model_dir, lr)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model = TFFashionMNIST(learning_rate=lr, batch_size=batch_size, epochs=no_epochs, verbose=True)
model.import_data(verbose=True)
print("x_train shape:", model.train_data.shape, "y_train shape:", model.train_label.shape)
model.train(log_dir=log_dir)
print(model.model.summary())
#
