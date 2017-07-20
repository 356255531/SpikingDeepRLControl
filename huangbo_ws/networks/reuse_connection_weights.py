import numpy as np
import nengo

num_items = 5

d_key = 2
d_value = 4
SEED = 7

rng = np.random.RandomState(seed=SEED)
keys = nengo.dists.UniformHypersphere(surface=True).sample(num_items, d_key, rng=rng)
values = nengo.dists.UniformHypersphere(surface=False).sample(num_items, d_value, rng=rng)

intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()


def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period / dt))
    if i_every != period / dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        i = int(round((t - dt) / dt))  # t starts at dt
        idx = (i // i_every) % len(x)
        return x[idx]

    return f


# Model constants
n_neurons = 200
dt = 0.001
period = 0.3
T = period * num_items
sample_every = 0.01

with nengo.Network() as train_model:
    # Create the inputs/outputs
    stim_keys = nengo.Node(cycle_array(keys, period, dt))
    stim_values = nengo.Node(cycle_array(values, period, dt))
    # Turn learning permanently on
    learning = nengo.Node([0])
    recall = nengo.Node(size_in=d_value)

    # Create the memory with a seed, so we can create the same ensemble again
    # in the new network
    memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons,
                            seed=SEED)

    # Learn the encoders/keys
    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)
    conn_in = nengo.Connection(stim_keys, memory, synapse=None,
                               learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)

    # Learn the decoders/values, initialized to a null function
    conn_out = nengo.Connection(memory, recall,
                                learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.zeros(d_value))

    # Create the error population
    error = nengo.Ensemble(n_neurons, d_value)
    nengo.Connection(learning, error.neurons, transform=[[10.0]] * n_neurons,
                     synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(stim_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)


with train_model:
    # Setup probes to save the weights
    p_dec = nengo.Probe(conn_out, 'weights', sample_every=sample_every)
    # In future versions of Nengo, you should be able to probe the ensemble
    # to get it's encoders, but for now we have to probe an attribute of
    # the learning rule
    p_enc = nengo.Probe(conn_in.learning_rule, 'scaled_encoders',
                        sample_every=sample_every)

# run the model and retrieve the encoders and decoders
with nengo.Simulator(train_model, dt=dt) as train_sim:
    train_sim.run(T)

enc = train_sim.data[p_enc][-1]
dec = train_sim.data[p_dec][-1]

with nengo.Network() as test_model:
    # Create the inputs/outputs
    stim_keys = nengo.Node(cycle_array(keys, period, dt))
    stim_values = nengo.Node(cycle_array(values, period, dt))
    # Turn learning off to show that our network still works
    learning = nengo.Node([-1])
    recall = nengo.Node(size_in=d_value)

    # Create the memory with zero eval points, since we're going to
    # define the decoders later anyways
    memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons,
                            encoders=enc, n_eval_points=0, seed=SEED)

    # Learn the encoders/keys
    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)
    conn_in = nengo.Connection(stim_keys, memory, synapse=None,
                               learning_rule_type=voja)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)

    # Learn the decoders/values, initialized to a null function
    conn_out = nengo.Connection(memory.neurons, recall,
                                learning_rule_type=nengo.PES(1e-3),
                                transform=dec)

    # Create the error population
    error = nengo.Ensemble(n_neurons, d_value)
    nengo.Connection(learning, error.neurons, transform=[[10.0]] * n_neurons,
                     synapse=None)

    # Calculate the error and use it to drive the PES rule
    nengo.Connection(stim_values, error, transform=-1, synapse=None)
    nengo.Connection(recall, error, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # Setup probes
    p_val = nengo.Probe(stim_values, synapse=0.005)
    p_recall = nengo.Probe(recall, synapse=0.005)

# run the network and plot the results for verification
with nengo.Simulator(test_model, dt=dt) as test_sim:
    test_sim.run(T)

import matplotlib.pyplot as plt

plt.plot(test_sim.data[p_val])
plt.figure()
plt.plot(test_sim.data[p_recall])
plt.show()