import matplotlib.pyplot as plt
import numpy as np
import nengo

m = 200  # number of training points
n = 100  # number of neurons

# --- create x and y points for target function (square)
rng = np.random.RandomState(0)
x = rng.uniform(-1, 1, size=(m, 1))         # random points along x-axis
y = x**2 + rng.normal(0, 0.1, size=(m, 1))  # square of x points plus noise

with nengo.Network() as model:
    a = nengo.Ensemble(n, 1)
    b = nengo.Ensemble(n, 1)
    c = nengo.Connection(a, b, eval_points=x, function=y)

with nengo.Simulator(model) as sim:
    pass

x2 = np.linspace(-1, 1, 100).reshape(-1, 1)
x2, _, y2 = nengo.utils.connection.eval_point_decoding(c, sim, eval_points=x2)

plt.plot(x, y, 'k.')
plt.plot(x2, y2, 'b-')
plt.show()