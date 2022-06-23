from Activations import relu, leaky_relu
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    data = np.arange(-10, 10, 0.1)
    print("Performing Relu")
    rel = relu(data)
    print("performing Leaky Relu")
    lk_rel = leaky_relu(0.03, data)
    plt.plot(data, rel)
    plt.plot(data, lk_rel)
    plt.grid()
    plt.show()
