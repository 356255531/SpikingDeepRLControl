from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.backend import set_image_dim_ordering

set_image_dim_ordering("tf")



def batch_generator(batch_size, flag):

    (image_train, label_train), (image_test, label_test) = mnist.load_data()
    image_train = image_train.reshape(-1, 28 * 28)
    image_test = image_test.reshape(-1, 28 * 28)
    label_train = label_train.reshape(-1, 1)
    label_test = label_test.reshape(-1, 1)

    mini_batch = np.ndarray(shape=(batch_size, 28*28))
    labels = np.zeros(shape=(batch_size, 1))

    index = 0

    if flag == "training":
        while 1:

            mini_batch = image_train[index:index+batch_size,:]
            labels = label_train[index:index+batch_size,:]
            index = index + batch_size

            yield mini_batch, labels

    if flag == "validation":
        while 1:
            mini_batch = image_test[index:index + batch_size, :]
            labels = label_test[index:index + batch_size, :]
            index = index + batch_size

            yield mini_batch, labels


# training_generator = batch_generator(32,"training")
# image, label = training_generator.next()
#
# print image.shape
# print label.shape
#
#
# training_generator = batch_generator(32,"validation")
# image, label = training_generator.next()
#
# print image.shape
# print label.shape