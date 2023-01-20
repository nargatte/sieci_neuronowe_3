import numpy as np

# Activation functions

def relu_n(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0) * 1

relu = {
    "n": relu_n,
    "d": relu_d
}


def linear_n(x):
    return x

def linear_d(x):
    return np.ones(x.shape)

linear = {
    "n": linear_n,
    "d": linear_d
}


def sigmoid_n(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid_n(x) * (1 - sigmoid_n(x))

sigmoid = {
    "n": sigmoid_n,
    "d": sigmoid_d
}


def gaussian_n(x):
    return np.exp(-np.power(x, 2))

def gaussian_d(x):
    return -2 * x * np.exp(-np.power(x, 2))

gaussian = {
    "n": gaussian_n,
    "d": gaussian_d
}


def step_n(x):
    return (x > 0) * 1

def step_d(x):
    return np.zeros(x.shape)

step = {
    "n": step_n,
    "d": step_d
}


def softmax_n(x):
    exps = np.exp(x)
    sum = np.sum(exps, axis=0)
    return exps / sum

def softmax_d(x):
    ret = None
    for i in range(x.shape[1]):
        s = softmax_n(x[:, i]).reshape((x.shape[0], -1))
        ret = np.diagflat(s) - np.dot(s, s.T) if ret is None else np.vstack((ret, np.diagflat(s) - np.dot(s, s.T)))
    return ret

softmax = {
    "n": softmax_n,
    "d": softmax_d
}


# Loss functions

def l1_loss_n(predicted, expected):
    return np.abs(predicted - expected)

def l1_loss_d(predicted, expected):
    return np.sign(predicted - expected)

l1_loss = {
    "n": l1_loss_n,
    "d": l1_loss_d
}


def l2_loss_n(predicted, expected):
    return np.power(predicted - expected, 2)

def l2_loss_d(predicted, expected):
    return 2 * (predicted - expected)

l2_loss = {
    "n": l2_loss_n,
    "d": l2_loss_d
}


def cross_entropy_loss_n(predicted, expected):
    return -np.sum(expected * np.log(predicted), axis=0).reshape((1, -1))

def cross_entropy_loss_d(predicted, expected):
    return -expected / predicted

cross_entropy_loss = {
    "n": cross_entropy_loss_n,
    "d": cross_entropy_loss_d
}


def hinge_loss_n(predicted, expected):
    # hinge expects y = +-1, so need to transform from [0,1] to [-1,1]
    predicted = 2 * predicted - 1
    expected = 2 * expected - 1
    ret = np.maximum(0, 1 - expected * predicted)
    return ret

def hinge_loss_d(predicted, expected):
    # hinge expects y = +-1, so need to transform from [0,1] to [-1,1]
    product = (2 * predicted - 1) * (2 * expected - 1)

    # d/dy L(y) = -y_expected, but we applied transformation above, so need to adjust it
    ret = (product < 1) * (2 - 4 * expected)
    return ret

hinge_loss = {
    "n": hinge_loss_n,
    "d": hinge_loss_d
}