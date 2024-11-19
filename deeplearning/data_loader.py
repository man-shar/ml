import pickle
import gzip

import numpy as np


def load_pickle_file(file_path):
    f = gzip.open(file_path, "rb")
    data = pickle.load(f, encoding="latin1")
    f.close()
    return data


def convert_digit_to_one_hot_vector(digit, len=10):
    if digit > 9.0:
        raise ValueError("Digit can't be greater than 9")

    if digit < 0.0:
        raise ValueError("Digit can't be less than 0")

    v = np.zeros((len, 1))
    v[digit] = 1
    return v


def load_data(file_path):
    data = load_pickle_file(file_path)
    training_data, validation_data, test_data = data

    # all three have tuples
    # a particular tuple is input, label
    # the input is array of 784
    # label is one digit which is an integer of np.int64
    # for our training data, it is better for us to have a vector of 10 digits, with the desired integer being 1
    # so 1 becomes (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    # but it's okay for the format to be as is for validation and test data
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_outputs = [convert_digit_to_one_hot_vector(d) for d in training_data[1]]
    training_data = list(zip(training_inputs, training_outputs))
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return (training_data, test_data, validation_data)
