import numpy as np
from keras.optimizers import Adam
from keras.backend import set_image_dim_ordering

from keras.layers import Dense
from keras.models import Sequential

from keras.callbacks import (ModelCheckpoint, TensorBoard)


class Q_learning_network(object):

    set_image_dim_ordering("tf")

    def __init__(self, batch_size, input_shape, nb_classes, weights_file):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape =  nb_classes
        self.weights_file = weights_file

    def build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.input_shape))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.output_shape, activation='softmax'))

        return model

    def run_training(self, training_data, label):
        model = self.build_model()
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'],
                          )

        model.train_on_batch(x=training_data, y=label)
        model.save_weights(self.weights_file)

    def prediction(self, input):
        model = self.build_model()
        model.loss_weights( self.weights_file)
        output = model.predict(input)
        return output





        # checkpoint = ModelCheckpoint(weights_file,
        #                              monitor='val_acc',
        #                              verbose=1,
        #                              save_best_only=True,
        #                              mode='max')

        # tensorboard = TensorBoard(log_dir=weights_file + 'Graph',
        #                           histogram_freq=0,
        #                           write_graph=True,
        #                           write_images=True
        #                           )

        # model.fit_generator(training_generator,
        #                         samples_per_epoch=5000,
        #                         nb_epoch=20,
        #                         validation_data=validation_generator,
        #                         nb_val_samples=300
        #                         )




from keras.datasets import mnist
from keras.utils import np_utils

model = Q_learning_network(batch_size=32, input_shape=28*28, nb_classes=10)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)


def batch_generator(batch_size, data_set, label):
    index = 0

    while 1:

        data_batch = data_set[index:index+batch_size, :]
        label_batch = label[index:index+batch_size, :]
        index = index + batch_size

        yield data_batch, label_batch


training_batch_generator = batch_generator(32, X_train, y_train)
validation_batch_generator = batch_generator(32, X_test, y_test)


model.run_training(weights_file="/home/huangbo/Desktop/",
                   training_generator=training_batch_generator,
                   validation_generator=validation_batch_generator
                   )




