from cerebrum.neural_network import NeuralNetwork
import unittest

import numpy

class NewNeuralNetworkTest(unittest.TestCase):
    def test_new_neural_network(self):
        net = NeuralNetwork([5, 7, 3])
        self.assertEqual(len(net.biases), 2)
        self.assertEqual(len(net.weights), 2)

        # First set of biases is for the hidden layer,
        # second set for the output layer, as the input layer
        # has no biases.
        self.assertEqual(len(net.biases[0]), 7)
        self.assertEqual(len(net.biases[1]), 3)

        # Should be seven sets of weights, each
        # set contains weights for all 5 inputs
        self.assertEqual(len(net.weights[0]), 7)
        for weights in net.weights[0]:
            self.assertEqual(len(weights), 5)

        # Likewise, 3 sets of weights corresponding
        # to the outputs, with 7 weights each
        # for each of the hidden neurons
        self.assertEqual(len(net.weights[1]), 3)
        for weights in net.weights[1]:
            self.assertEqual(len(weights), 7)

class RunNeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        # Set the seed to a value that generates the correct result
        numpy.random.seed(5)
        self.neural_net = NeuralNetwork([5, 7, 3])

    def test_feedforward(self):
        a = [0.5324, 0.1232, 0.9832, 0.3245, 0.2894]
        result = self.neural_net.feedforward(a)
        self.assertEqual(len(result), 3)

    def test_SGD(self):
        training_data = [
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
        ]

        old_weights = self.neural_net.weights
        old_biases = self.neural_net.biases
        self.neural_net.SGD(training_data, 3, 2, 0.5)
        for old_weight, weight in zip(old_weights, self.neural_net.weights):
            self.assertFalse((old_weight==weight).all())
        for old_bias, bias in zip(old_biases, self.neural_net.biases):
            self.assertFalse((old_bias==bias).all())

    def test_evaluate(self):
        training_data = [
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
        ]

        test_data = [
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0])),
            (numpy.array([0.5324, 0.1232, 0.9832, 0.3245, 0.2894]),
             numpy.array([1.0, 0.0, 0.0]))
        ]

        self.neural_net.SGD(training_data, 3, 2, 0.5)
        for value in self.neural_net.evaluate(test_data):
            self.assertEqual(value, 0)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
