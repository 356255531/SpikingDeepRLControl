import nengo
import numpy as np
from nengo.processes import WhiteSignal

model = nengo.Network(label="Delayed connection")
with model:
    # We'll use white noise as input
    inp = nengo.Node(WhiteSignal(2, high=5), size_out=1, label="input")
    print inp

