import numpy as np
import math

def sigmoid_using_math(data):
    """
    The function using Math library to compute the sigmoid function
    :param data: scalar, data to compute sigmoid for
    :return: sig -- value after sigmoid implementation
    """

    sig = 1 / (1 + math.exp(-data))

    return sig

def sigmoid_for_array(x):
    """
    The function to compute sigmoid of an array using numpy
    :param x: scalar/numpy array, data to compute sigmoid for
    :return: s -- value after sigmoid implementation
    """

    s = 1 / (1 + np.exp(-x))

    return s
if __name__ == "__main__":
    print("Test Math sigmoid")
    print(sigmoid_using_math(4))

    print("Test the Numpy sigmoid")
    inputs = np.array([1, 2, 0.3, -3])
    print(sigmoid_for_array(inputs))
    print("Test numpy sigmoid using a scalar")
    print(sigmoid_for_array(4))
