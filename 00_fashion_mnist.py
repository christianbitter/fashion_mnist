# for update inspirations:
# https://github.com/cs230-stanford/cs230-code-examples
# shuffle repeat batch nicely visualized: https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
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
                 train_fp="C:/Data/Kaggle_-_Fashion_MNIST/fashion-mnist_train.csv",
                 test_fp="C:/Data/Kaggle_-_Fashion_MNIST/fashion-mnist_test.csv",
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
        with tf.name_scope('input'):
            self.data = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1), name='data')
            self.label= tf.placeholder(dtype=tf.int32, shape=None, name='label')

        if self.verbose:
            print("Learning Rate: {0} - Batch Size: {1} - Number of Epochs: {2}".format(self.learning_rate,
                                                                                         self.batch_size,
                                                                                         self.no_epochs))

    def __global_preprocess(self, data, label):
        data = tf.cast(data, dtype=tf.float32)
        data = data / 255.
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
        self.train_label= self.train_df[:self.no_train, 0]
        if verbose:
            print("Training Set Size: {0}".format(self.no_train))

        if verbose:
            print("Reading Testing Data: %s" % self.testing_fp)
        self.test_df = pd.read_csv(self.training_fp).values
        self.no_test = self.train_df.shape[0]
        self.test_data = self.test_df[:self.no_test, 1:].reshape((self.no_test, self.img_height, self.img_width, 1))
        self.test_label= self.test_df[:self.no_test, 0]
        if verbose:
            print("Testing Set Size: {0}".format(self.no_test))

        if verbose:
            print("Performing Data Transformation ...")

        with tf.device('/cpu:0'):
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_label))
            self.train_dataset = self.train_dataset.shuffle(buffer_size=self.train_df.shape[0], reshuffle_each_iteration=True)
            self.train_dataset = self.train_dataset.repeat(count=self.no_epochs)
            self.train_dataset = self.train_dataset.map(self.__global_preprocess, num_parallel_calls=4)
            self.train_dataset = self.train_dataset.map(self.__train_preprocess, num_parallel_calls=4)
            self.train_dataset = self.train_dataset.batch(batch_size=self.batch_size)
            self.train_dataset = self.train_dataset.prefetch(buffer_size=self.batch_size)

            self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_data, self.test_label))
            self.test_dataset = self.test_dataset.map(self.__global_preprocess, num_parallel_calls=4)
            self.test_dataset = self.test_dataset.map(self.__test_preprocess, num_parallel_calls=4)
            self.test_dataset = self.test_dataset.batch(batch_size=self.batch_size)
            self.test_dataset = self.test_dataset.prefetch(buffer_size=self.batch_size)

            self.train_iter = self.train_dataset.make_initializable_iterator()
            self.test_iter = self.test_dataset.make_initializable_iterator()

    def create_model(self):
        # convnet layer 1 - convolution, max pooling, convolution max pooling, flatten, dense, dense
        with tf.name_scope('conv1'):
            self.conv1 = tf.layers.conv2d(inputs=self.data, kernel_size=(5, 5), filters=30, padding='same')
            self.mpool1= tf.layers.max_pooling2d(inputs=self.conv1, pool_size=(2, 2), strides=(2, 2))
            # potential dropout

        with tf.name_scope('conv2'):
            self.conv2 = tf.layers.conv2d(inputs=self.mpool1, kernel_size=(5, 5), filters=80, padding='same')
            self.mpool2= tf.layers.max_pooling2d(inputs=self.conv2, pool_size=(2, 2), strides=(2, 2))

        with tf.name_scope('conv3'):
            self.conv3 = tf.layers.conv2d(inputs=self.mpool2, kernel_size=(3, 3), filters=120, padding='same')
            self.mpool3= tf.layers.max_pooling2d(inputs=self.conv3, pool_size=(2, 2), strides=(2, 2))

        final_layer = self.mpool3

        with tf.name_scope('output'):
            self.flatten = tf.layers.flatten(inputs=final_layer)
            self.dense1  = tf.layers.dense(inputs=self.flatten, units=500)
            self.logits  = tf.layers.dense(inputs=self.dense1, units=self.number_of_classes)

    def create_loss(self):
        # softmax cross entropy loss for multi-class classification

        with tf.name_scope('loss'):
            self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            # run the logits through the softmax layer
            # predictions = tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)
            # self.accuracy_fn = tf.reduce_mean(tf.cast(tf.equal(x=self.label, y=predictions), dtype=tf.float32))

        tf.summary.scalar(name='Loss', tensor=self.loss_fn)
        # tf.summary.scalar(name='Accuracy', tensor=self.accuracy_fn)

    def create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        with tf.name_scope('Optimizer'):
            self.train_fn = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss_fn,
                                                                                              global_step=self.global_step)

    def __train_single_iteration(self):
        pass


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
model.create_model()
model.create_loss()
model.create_optimizer()

next_data, next_label = model.train_iter.get_next()
next_v_data, next_v_label = model.test_iter.get_next()
train_init_op = model.train_iter.initializer
test_init_op = model.test_iter.initializer
merged_summary = tf.summary.merge_all()

# now the training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train_summaries'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval_summaries'), sess.graph)

    validation_loss = 0.
    validation_accuracy = 0.
    i_batch = 0
    for epoch in range(no_epochs):
        print("Starting-Training-Epoch: %s" % epoch)

        sess.run([train_init_op, test_init_op])
        try:
            while True:
                train_data, train_label = sess.run([next_data, next_label])
                # print("%s" % (train_data.shape, ))
                _, accuracy_val, loss_val, summary_val = sess.run([model.train_fn, model.accuracy_fn, model.loss_fn, merged_summary],
                                          feed_dict={
                                              model.data: train_data,
                                              model.label: train_label
                                          })
                train_writer.add_summary(summary_val)
                # print("Batch Training Loss/ Accuracy: %s - %s" % (loss_val, accuracy_val))
        except tf.errors.OutOfRangeError:
            pass

        # evaluate on validation data
        print("Performing Validation: %s" % epoch)
        try:
            while True:
                test_data, test_label = sess.run([next_v_data, next_v_label])
                validation_loss_val, validation_accuracy_val = sess.run([model.loss_fn, model.accuracy_fn],
                                                                        feed_dict={
                                                                            model.data: test_data,
                                                                            model.label: test_label
                                                                        })
                validation_loss += validation_loss_val
                validation_accuracy += validation_accuracy_val
                i_batch = i_batch + 1
        except tf.errors.OutOfRangeError:
            pass
        validation_loss = validation_loss / float(i_batch)
        validation_accuracy = validation_accuracy / float(i_batch)
        print("Validation Loss: {0} - Accuracy: {1}".format(validation_loss, validation_accuracy))

        # add step on validation and training set
    pass

    train_writer.close()
    eval_writer.close()