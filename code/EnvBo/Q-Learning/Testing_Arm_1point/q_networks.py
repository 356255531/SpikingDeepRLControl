#!/usr/bin/python
import numpy as np
import os
import sys

from keras.layers import Activation, Dense, Input 
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import RMSprop



NUM_OF_HIDDEN_NEURONS = 100
QNETWORK_NAME = 'online_network'
TARGETNET_NAME = 'target_network'
TAU = 0.0001                         # soft update / low pass filter


class QNetworks:
    def __init__(self, num_of_actions, num_of_states, num_of_hidden_neurons=NUM_OF_HIDDEN_NEURONS, tau=TAU): 
        self.NUM_OF_ACTIONS = num_of_actions
        self.NUM_OF_HIDDEN_NEURONS = num_of_hidden_neurons
        self.NUM_OF_STATES = num_of_states
        self.TAU = tau

        self.online_net = self.init_model(QNETWORK_NAME)
        self.target_net = self.init_model(QNETWORK_NAME)

    def do_soft_update(self):
        weights = self.online_net.get_weights()
        target_weights = self.target_net.get_weights()
        for i in xrange(len(weights)):
            target_weights[i] = self.TAU*weights[i] + (1.0-self.TAU)*target_weights[i]
        self.target_net.set_weights(target_weights)
        return

    def do_hard_update(self):
        weights = self.online_net.get_weights()
        target_weights = self.target_net.get_weights()
        for i in xrange(len(weights)):
            target_weights[i] = weights[i]
        self.target_net.set_weights(target_weights)
        return

    def get_weights(self):
        # get weights of the online Q network
        return self.online_net.get_weights()

    def init_model(self, net_name):
        model = Sequential()

        model.add(Dense(self.NUM_OF_HIDDEN_NEURONS, input_shape=(self.NUM_OF_STATES,)))
        model.add(Activation('relu'))

        model.add(Dense(self.NUM_OF_HIDDEN_NEURONS))
        model.add(Activation('relu'))

        model.add(Dense(self.NUM_OF_HIDDEN_NEURONS))
        model.add(Activation('relu'))

        model.add(Dense(self.NUM_OF_ACTIONS))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='rmsprop')

        filename = net_name+'/'+net_name
        if os.path.isfile(filename+str(0)+'.txt'):
            weights = model.get_weights()
            for i in xrange(len(weights)):
                loaded_weights = np.loadtxt(filename+str(i)+'.txt')
                weights[i] = loaded_weights
            model.set_weights(weights)
        else:
            print 'No model', filename, 'found. Creating a new model.'

        return model

    def save_models(self):
        weights = self.online_net.get_weights()
        for i in xrange(len(weights)):
            np.savetxt(QNETWORK_NAME+'/'+QNETWORK_NAME+str(i)+'.txt', weights[i])

        weights = self.target_net.get_weights()
        for i in xrange(len(weights)):
            np.savetxt(TARGETNET_NAME+'/'+TARGETNET_NAME+str(i)+'.txt', weights[i])

        print("Saved models to disk.")