import neural_network as nn
import definitions as d
import numpy as np

rng = np.random.default_rng(0)

l1 = nn.InputLayer(3, "def")
l2 = nn.FullConnectLayer(l1, 5, d.relu, rng)
l3 = nn.FullConnectLayer(l2, 7, d.relu, rng)

l4 = nn.InputLayer(3, "bac")
l5 = nn.FullConnectLayer(l4, copy=l2)
l6 = nn.FullConnectLayer(l5, copy=l3)

l7 = nn.MergeLayer([l3, l6])
l8 = nn.FullConnectLayer(l7, 1, d.linear, rng)

net = nn.NeuralNetwork(l8, d.l2_loss)


set = {
    "def": rng.random([3, 5]),
    "bac": rng.random([3, 5]),
    "output": rng.random([1, 5])
}

net.train(set, set, 2, "output", rng)

