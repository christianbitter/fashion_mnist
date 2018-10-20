# for update inspirations:
# https://github.com/cs230-stanford/cs230-code-examples
# shuffle repeat batch nicely visualized: https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Params import Params

# TODO: the training loop should be split
# TODO: saving model state on improvement
# TODO: this should be a high level file calling the whole model building/training - this is the training file

def show_row(m, width=28, height=28):
    plt.imshow(X = m.reshape((height, width)), cmap='gray')
    plt.show()


class TFFashionMNIST(object):
    def __init__(self,
                 learning_rate=.003,
                 epochs    =10,
                 batch_size=10,
                 train_fp="Data/fashion-mnist_train.csv",
                 test_fp="Data/fashion-mnist_test.csv",
                 verbose=False):
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

    def import_data(self, verbose=False):
        if verbose:
            print("Reading Training Data: %s" % self.training_fp)

        self.train_df = pd.read_csv(self.training_fp).values
        self.no_train   = self.train_df.shape[0]
        self.train_data = self.train_df[:self.no_train, 1:].reshape((self.no_train, self.img_height, self.img_width, 1))
        self.train_data = self.train_data / 255.
        self.train_label= self.train_df[:self.no_train, 0]
        if verbose:
            print("Training Set Size: {0}".format(self.no_train))

        if verbose:
            print("Reading Testing Data: %s" % self.testing_fp)
        self.test_df = pd.read_csv(self.training_fp).values
        self.no_test = self.train_df.shape[0]
        self.test_data = self.test_df[:self.no_test, 1:].reshape((self.no_test, self.img_height, self.img_width, 1))
        self.test_data = self.test_data / 255.
        self.test_label = self.test_df[:self.no_test, 0]
        if verbose:
            print("Testing Set Size: {0}".format(self.no_test))


    def create_model(self):
        # convnet layer 1 - convolution, max pooling, convolution max pooling, flatten, dense, dense
        self.model = tf.keras.Sequential()

        with tf.name_scope('conv1'):
            self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            # self.model.add(tf.keras.layers.Dropout(rate=.2))

        with tf.name_scope('conv2'):
            self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            # self.model.add(tf.keras.layers.Dropout(rate=.2))

        # with tf.name_scope('conv3'):
        #     self.model.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), activation='relu'))
        #     self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        #     self.model.add(tf.keras.layers.Dropout(rate=.2))

        with tf.name_scope('output'):
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))
            self.model.add(tf.keras.layers.Dropout(rate=.5))
            self.model.add(tf.keras.layers.Dense(units=self.number_of_classes))

    def create_loss(self):
        # softmax cross entropy loss for multi-class classification

        with tf.name_scope('loss'):
            self.loss_fn = tf.keras.losses.categorical_crossentropy
            self.accuracy_fn = tf.keras.metrics.categorical_accuracy

    def create_optimizer(self):
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                           loss=self.loss_fn,
                           metrics=[self.accuracy_fn])

    def train(self, log_dir=None):
        callbacks = []
        if log_dir:
            if self.verbose:
                print("Enabling Keras Tensorboard Callback: %s" % log_dir)
            kcb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
            callbacks.append(kcb)

        self.model.fit(x=self.train_data,
                       y=self.train_label,
                       batch_size=self.batch_size,
                       # steps_per_epoch=self.no_training_steps,
                       epochs=self.no_epochs,
                       validation_data= (self.test_data, self.test_label),
                       # validation_steps=self.no_testing_steps,
                       verbose=True,
                       callbacks=callbacks
                       )

    def __train_single_iteration(self):
        pass

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
#
model = TFFashionMNIST(learning_rate=lr, batch_size=batch_size, epochs=no_epochs, verbose=True)
model.import_data(verbose=True)
print("x_train shape:", model.train_data.shape, "y_train shape:", model.train_label.shape)
model.create_model()
model.create_loss()
model.create_optimizer()
model.train(log_dir=log_dir)


#loss: 14.5063 - sparse_categorical_crossentropy: 14.5063 - val_loss: 14.5063 - val_sparse_categorical_crossentropy: 14.5063