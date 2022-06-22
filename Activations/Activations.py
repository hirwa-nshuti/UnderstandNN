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


def sigmoid(x):
    """
    The function to compute sigmoid of an array using numpy
    :param : scalar/numpy array, data to compute sigmoid for
    :return: s -- value after sigmoid implementation
    """

    s = 1 / (1 + np.exp(-x))
    return s


def tanh(data):
    """
    The function to calculate the hyperbolic tangent of input data
    :param data: a scalar/numpy array, input data to compute tanh value for
    :return: th -- The computed value after applying the tanh function
    """

    th = (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    return th
