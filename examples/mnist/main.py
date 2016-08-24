from .load import transform
from cerebrum.neural_network import NeuralNetwork

if __name__ == "__main__":
    print("Loading MNIST training, validation, and test data...")
    training_data, validation_data, test_data = transform()
    print("Generating neural network...")
    net = NeuralNetwork([784, 192, 10])
    print("Training neural network...")
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
