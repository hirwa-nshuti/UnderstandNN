from Activations import sigmoid, sigmoid_math
import numpy as np

if __name__ == "__main__":
    print("Test Math sigmoid")
    print(sigmoid_math(4))
    print("Test the Numpy sigmoid")
    inputs = np.array([1, 2, 0.3, -3])
    print(sigmoid(inputs))
    print("Test numpy sigmoid using a scalar")
    print(sigmoid(4))
    print("Testing negative numbers")
    print(sigmoid(-5))
