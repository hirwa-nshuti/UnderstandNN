import numpy as np
import math


def sigmoid_math(data):
    """
    The function using Math library to compute the sigmoid function
    :param : scalar, data to compute sigmoid for
    :return: sig -- value after sigmoid implementation
    """

    sig = 1.0 / (1 + math.exp(-data))

    return sig


def sigmoid(data):
    """
    The function to compute sigmoid of an array using numpy
    :param : scalar/numpy array, data to compute sigmoid for
    :return: s -- value after sigmoid implementation
    """

    return 1 / (1 + np.exp(-data))


def tanh(data):
    """
    The function to calculate the hyperbolic tangent of input data
    :param data: a scalar/numpy array, input data to compute tanh value for
    :return: th -- The computed value after applying the tanh function
    """

    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    """
    Method to compute the rectified Linear Unit
    :param data: a scalar/numpy array, input data to compute relu value for
    :return: a scalar/numpy array with maximum between 0 and data elements
    """
    return np.maximum(0, data)


def leaky_relu(alpha, data):
    """
    Method for computing the leaky relu value.
    :param alpha: a leaky relu parameter for giving the negative slope
    :param data: a scalar/numpy array, input data to compute relu value for.
    :return: a scalar/numpy array with alpha * data element if element < 0 or data elements if > 0
    """
    return np.maximum(alpha*data, data)


def softmax(data):
    """
    Method fot computing the softmax value
    :param data: a numpy array to compute the values for.
    :return: a numpy array with probability densities of output values
    """
    e = np.exp(data)
    sf = e / np.sum(e)

    return sf

