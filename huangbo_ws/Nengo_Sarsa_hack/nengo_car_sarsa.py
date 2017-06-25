import nengo
import numpy as np
import mountaincar

def mountain_car_func(t,x):
    mountain_car.apply_force(x[0])
    mountain_car.simulate_timesteps(1, 0.001)
    rew=mountain_car._get_reward()
    #print(rew)
    if rew == 1.:
        mountain_car.reset()
        print("We reached the top")
    return [mountain_car.x*0.01,mountain_car.x_d*0.1,rew]


def print_q(x):
    print(x)
    return 0

speed=0.1

mountain_car = mountaincar.MountainCar()

model = nengo.Network()

with model:

    car_node = nengo.Node(mountain_car_func,size_in=1)
    
    
    
    state = nengo.Ensemble(n_neurons=200, dimensions=2, radius=1.4)
    nengo.Connection(car_node[0:2],state[:2])
    #pos = nengo.Ensemble(n_neurons=500, dimensions=2)
    #target = nengo.Node(lambda t: [np.sin(t), np.cos(t)])
    
    #nengo.Connection(pos, pos, synapse=0.1)
    #nengo.Connection(pos, state[:2])
    #nengo.Connection(target, state[2:])
    
    
    q_a = nengo.Ensemble(n_neurons=200,
                         dimensions=3,
                         neuron_type=nengo.Direct()
                         )
    
    def initial_function(state):
        return [0,0,0]
        #pos = state[:2]
        #targ = state[2:]
        
        #delta = targ-pos
        return [delta[0], -delta[0], delta[1], -delta[1]]
        
    
    c = nengo.Connection(state, q_a, function=initial_function,
                         learning_rule_type=nengo.PES(learning_rate=1e-4))
    
    def select(x):
        v = [0,0,0]
        v[np.argmax(x)] = 1
        return v
    a = nengo.Node(lambda t, x: x, size_in=3)
    nengo.Connection(q_a, a, function=select)
    
    def act(a):
        return a[0]-a[2],0#return a[0]-a[1], a[2]-a[3]
            
    v = nengo.Ensemble(100, 2,neuron_type=nengo.Direct())
    nengo.Connection(a, v, function=act,synapse=None)
    #nengo.Connection(v, pos, transform=speed)
    nengo.Connection(v[0],car_node,synapse=None)
    
    r = nengo.Node(lambda t, x: x[:3]*x[3], size_in=8)
    #nengo.Connection(target, r[4:6])
    #nengo.Connection(pos, r[4:6], transform=-1)
    nengo.Connection(car_node[2], r[3],transform=30.)
    nengo.Connection(a, r[:3])
    
    noise=nengo.Node([0,0,0])
    nengo.Connection(noise[2],q_a[2],transform=1.)
    nengo.Connection(noise[0],q_a[0],transform=1.)
    nengo.Connection(noise[1],q_a[1],transform=1.)
    
    
    gamma=0.9
    slow=0.1
    
    error = nengo.Node(None, size_in=3)
    nengo.Connection(r, error, synapse=slow)
    nengo.Connection(q_a, error, transform=gamma)
    nengo.Connection(q_a, error, transform=-1, synapse=slow)
    
    nengo.Connection(error, c.learning_rule, transform=-1)
    #asd=nengo.Ensemble(200,1)
    #nengo.Connection(q_a,asd,function=print_q)