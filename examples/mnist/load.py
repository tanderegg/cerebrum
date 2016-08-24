import pickle
import gzip
import numpy

def extract():
    """
    training_data: a tuple with two entires, the images and the numeral 0 to 9.
    Each image is 784 greyscale pixels, 0.0..1.0, and there are 50,000 images
    in total.
    validation_data: 10,000 images in the same format.
    test_data: 10,000 images in the same format.
    """
    f = gzip.open('examples/mnist/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='latin1'
    )
    f.close()
    return (training_data, validation_data, test_data)


def transform():
    """
    We need to shape all the data to match our neural network, the inputs
    in this case are a matrix of n 784x1 numpy vectors, and n 10x1 output
    vectors, corresponding to the input layer and output layer of the
    neural network.
    """
    training_data, validation_data, test_data = extract()

    # Training inputs come from the first item in the tuple, and we reshape
    # each 784 element array into a 784x1 numpy ndarray.
    training_inputs = [numpy.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [numpy.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])

    test_inputs = [numpy.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    return (list(training_data), list(validation_data), list(test_data))

def vectorized_result(j):
    """
    Converts a 0..9 integer to a 10 dimensional vector with only one element of
    1.0, the rest 0.0, with the 1.0 corresponding to an index equal to the
    original integer.  Thus, 7 becomes [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0].
    """
    e = numpy.zeros((10, 1))
    e[j] = 1.0
    return e
