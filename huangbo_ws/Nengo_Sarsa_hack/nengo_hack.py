import numpy as np
import nengo

model = nengo.Network()
with model:
    
    stim = nengo.Node(0)
    
    
    ens = nengo.Ensemble(100, 1)
    
    nengo.Connection(stim, ens)
    
    
    class LearningNode(nengo.Node):
        def __init__(self, n_neurons, dimensions):
            self.w = np.zeros((dimensions, n_neurons))
            super(LearningNode, self).__init__(self.update, size_in=n_neurons)
        def update(self, t, x):
            
            return np.dot(self.w, x)
    
    my_rule = LearningNode(100, 2)
    
    nengo.Connection(ens.neurons, my_rule)
    
    
