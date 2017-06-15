import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

with nengo.Network(seed=0) as model:

    model.config[nengo.Ensemble].neuron_type = nengo_dl.neurons.SoftLIFRate()
    model.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    model.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
    model.config[nengo.Connection].synapse = None

    # initialize encoder and decoder
    rng = np.random.RandomState(1)
    encoders = rng.normal(size=(1000, 784))
    decoders = rng.normal(size=(1000, 10))

    # network
    input_node = nengo.Node(size_in=784)
    layer = nengo.Ensemble(n_neurons=1000,
                           dimensions=784,
                           encoders=encoders,
                           )
    nengo.Connection(input_node, layer)
    output = nengo.Node(size_in=10)
    output_p = nengo.Probe(output)
    conn = nengo.Connection(layer.neurons,
                            output,
                            transform=decoders.T
                            )

with nengo_dl.Simulator(model, minibatch_size=60, step_blocks=1, device="/gpu:0", seed=2) as sim:

    from keras.datasets import mnist
    from keras.utils import np_utils

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    X_train_ = np.expand_dims(X_train, axis=1)
    X_test_ = np.expand_dims(X_test, axis=1)
    y_train_ = np.expand_dims(y_train, axis=1)
    y_test_ = np.expand_dims(y_test, axis=1)

    sim.train({input_node: X_train_},
              {output_p: y_train_},
              tf.train.MomentumOptimizer(5e-2, 0.9),
              n_epochs=10
              )


    sim.step(input_feeds={input_node: X_test_[0:60,:,:]})
    output = sim.data[output_p]
    prediction = np.squeeze(output, axis=1)

    # evaluate the model
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(np.argmax(y_test[0:60,:], axis=1), np.argmax(prediction, axis=1))
    print "the test acc is:", acc

    #print sim.loss({input_node: X_test_[0:6000, :, :]}, {output_p: y_test_[0:6000, :, :]}, "mse")
    #sim.save_params("/home/huangbo/SpikingDeepRLControl/huangbo_ws/networks/saved_weights/snn_weights")

# with nengo_dl.Simulator(model, minibatch_size=10000, step_blocks=1, device="/gpu:0", seed=1) as sim:
#
#     #sim.load_params("/home/huangbo/SpikingDeepRLControl/huangbo_ws/networks/saved_weights/snn_weights")
#
#     # sim.step(input_feeds={input_node: np.expand_dims(X_test_[123, :, :], axis=0)})
#     # output = sim.data[output_p]
#     # output = np.squeeze(output, axis=1)
#     # print output
#     # print np.argmax(output)
#     #
#     # import matplotlib.pyplot as plt
#     # plt.figure()
#     # plt.imshow(X_test[123,:].reshape(28,28))
#     # plt.show()
#
#
#     sim.step(input_feeds={input_node: X_test_})
#     output = sim.data[output_p]
#     prediction = np.squeeze(output, axis=1)
#
#     # evaluate the model
#     from sklearn.metrics import accuracy_score
#     acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
#     print "the test acc is:", acc
#

