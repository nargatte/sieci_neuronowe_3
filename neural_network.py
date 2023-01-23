import numpy as np
import math


class FullConnectWeights:
    def __init__(self, input_size, output_size, rng):
        xavier_factor = math.sqrt(6 / (input_size + output_size))
        self.W = rng.random([output_size, input_size]) * 2 * xavier_factor - xavier_factor
        self.b = rng.random([output_size, 1]) * 2 * xavier_factor - xavier_factor
        self.reset()

    def reset(self):
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.update_count = 0

    def update_derivative(self, dW, db):
        self.dW += dW
        self.db += db
        self.update_count += 1


class FullConnectLayer:
    def __init__(self, input_layer, output_size=None, activation=None, rng=None, dropout_rate=1.0, copy=None):
        self.input_layer = input_layer
        self.rng = rng
        self.dropout_rate = dropout_rate
        if copy is None:
            self.output_size = output_size
            self.activation = activation
            self.weights = FullConnectWeights(input_layer.output_size, output_size, rng)
        else:
            self.output_size = copy.output_size
            self.activation = copy.activation
            self.weights = copy.weights

    def propagate_forward(self, x, is_training=False):
        self.x = x
        self.z = np.dot(self.weights.W, x) + self.weights.b
        r = self.rng.binomial(1, self.dropout_rate, self.z.shape)
        if is_training:
            return r / self.dropout_rate * self.activation["n"](self.z)
        else:
            return self.activation["n"](self.z)

    def propagate_backward(self, da):
        da_dz = self.activation["d"](self.z)
        if da_dz.shape == self.z.shape:
            dz = da * da_dz
        # Jacobian
        else:
            input_size = da.shape[0]
            input_count = da.shape[1]
            dz = None
            for j in range(input_count):
                da_dz_slice = da_dz[j * input_size : (j + 1) * input_size, :]
                da_slice = da[:, j].reshape((-1, 1))
                
                dz = np.dot(da_dz_slice, da_slice) if dz is None else np.hstack((dz, np.dot(da_dz_slice, da_slice)))

        dW = np.dot(dz, self.x.T) / da.shape[1]
        db = np.reshape(np.sum(dz, 1), (-1, 1)) / da.shape[1]
        self.weights.update_derivative(dW, db)
        return np.dot(self.weights.W.T, dz)


class InputLayer:
    def __init__(self, output_size, name):
        self.output_size = output_size
        self.name = name

    def propagate_forward(self, inputs):
        return inputs[self.name]
    

class MergeLayerInput:
    def __init__(self, input_layer, merge_layer):
        self.input_layer = input_layer
        self.merge_layer = merge_layer

    def propagate_forward(self, x):
        self.x = x

    def propagate_backward(self):
        return self.da


class MergeLayer:
    def __init__(self, input_layers):
        self.output_size = sum([il.output_size for il in input_layers])
        self.merge_inputs = []
        for il in input_layers:
            mli = MergeLayerInput(il, self)
            self.merge_inputs.append(mli)
    
    def propagate_forward(self):
        xs = [mi.x for mi in self.merge_inputs]
        return np.concatenate(tuple(xs))

    def propagate_backward(self, da):
        u = 0
        for mi in self.merge_inputs:
            w = u + mi.input_layer.output_size
            mi.da = da[u:w, :]
            u = w


class NeuralNetwork:
    def __init__(self, output_layer, loss, rng):
        self.output_layer = output_layer
        self.loss = loss
        self.weights = []
        self.find_weights_req(output_layer)
        self.optimizers = [AdamOptimizer(w) for w in self.weights]
        self.rng = rng

    def find_weights_req(self, layer):
        if type(layer) == InputLayer:
            pass
        elif type(layer) == MergeLayer:
            for mi in layer.merge_inputs:
                self.find_weights_req(mi.input_layer)
        else:
            self.weights.append(layer.weights)
            self.find_weights_req(layer.input_layer)

    def propagate_forward(self, inputs, is_training=False):
        self.inputs = inputs
        self.a = self.propagate_forward_req(self.output_layer, is_training)
        return self.a
        
    def propagate_forward_req(self, layer, is_training=False):
        if type(layer) == InputLayer:
            return layer.propagate_forward(self.inputs)
        elif type(layer) == MergeLayer:
            for mi in layer.merge_inputs:
                x = self.propagate_forward_req(mi.input_layer)
                mi.propagate_forward(x)
            return layer.propagate_forward()
        else:
            x = self.propagate_forward_req(layer.input_layer)
            return layer.propagate_forward(x, is_training)

    def propagate_backward(self, expected):
        da = self.loss["d"](self.a, expected)
        self.propagate_backward_req(self.output_layer, da)

    def propagate_backward_req(self, layer, da):
        if type(layer) == InputLayer:
            pass
        elif type(layer) == MergeLayer:
            layer.propagate_backward(da)
            for mi in layer.merge_inputs:
                nda = mi.propagate_backward()
                self.propagate_backward_req(mi.input_layer, nda)
        else:
            nda = layer.propagate_backward(da)
            self.propagate_backward_req(layer.input_layer, nda)

    def calculate_loss(self, predicted, expected):
        return np.nanmean(self.loss["n"](predicted, expected))

    def update_weights(self):
        for ao in self.optimizers:
            ao.update_weights()

    def get_batch(self, set, batch_size, batch_number):
        size = next(iter(set.values())).shape[1]
        u = batch_size*batch_number
        w = min(batch_size * (batch_number + 1), size)
        return {n: v[:, u:w] for n, v in set.items()}

    def get_batches(self, set, batch_size):
        size = next(iter(set.values())).shape[1]
        nof_batches = math.ceil(size / batch_size)
        return [self.get_batch(set, batch_size, idx) for idx in range(nof_batches)]

    def shuffle_set(self, set, rng):
        size = next(iter(set.values())).shape[1]
        permutation = rng.permutation(np.array(range(size)))
        new_set = {}
        for name, arr in set.items():
            new_set[name] = arr[:, permutation]

        return new_set

    def train(self, train_set, test_set, batch_size, output_name, epoch_count=100):
        stagnant_count = 0
        epoch = 0
        test_losses = []
        while stagnant_count < 3:
            epoch += 1
            print(f"Epoch {epoch}: ", end='')
            train_set = self.shuffle_set(train_set, self.rng)
            train_batched = self.get_batches(train_set, batch_size)
            losses = []
            for batch in train_batched:
                predicted = self.propagate_forward(batch, True)
                expected = batch[output_name]
                losses.append(self.calculate_loss(predicted, expected))
                self.propagate_backward(expected)
                self.update_weights()
            train_loss = sum(losses) / len(losses)
            predicted = self.propagate_forward(test_set)
            test_loss = self.calculate_loss(predicted, test_set[output_name])
            print(f"train: {train_loss}, test: {test_loss}")
            test_losses.append(test_loss)
            if (len(test_losses) > 1 and test_losses[-1] >= test_losses[-2]):
                stagnant_count += 1
            else:
                stagnant_count = 0

    def predict(self, data):
        return self.propagate_forward(data)


class AdamOptimizer:
    def __init__(self, weights, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        self.vdW = np.zeros(weights.W.shape)
        self.vdb = np.zeros(weights.b.shape)
        self.sdW = np.zeros(weights.W.shape)
        self.sdb = np.zeros(weights.b.shape)
        self.weights = weights
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update_weights(self):
        if self.weights.update_count == 0:
            return

        self.t += 1

        self.weights.dW /= self.weights.update_count
        self.weights.db /= self.weights.update_count

        self.vdW = self.beta1 * self.vdW + (1 - self.beta1) * self.weights.dW
        self.vdb = self.beta1 * self.vdb + (1 - self.beta1) * self.weights.db

        v_corrected_dW = self.vdW / (1 - self.beta1 ** self.t)
        v_corrected_db = self.vdb / (1 - self.beta1 ** self.t)

        self.sdW = self.beta2 * self.sdW + (1 - self.beta2) * (np.square(self.weights.dW))
        self.sdb = self.beta2 * self.sdb + (1 - self.beta2) * (np.square(self.weights.db))

        s_corrected_dW = self.sdW / (1 - self.beta2 ** self.t)
        s_corrected_db = self.sdb / (1 - self.beta2 ** self.t)

        self.weights.W -= self.learning_rate * v_corrected_dW / (np.sqrt(s_corrected_dW) + self.epsilon)
        self.weights.b -= self.learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + self.epsilon)

        self.weights.reset()
