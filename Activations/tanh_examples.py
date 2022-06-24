from Activations import tanh
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    data = np.array([3, -6, 9, 199, 0.1, -0.2])
    print("Testing the tanh activation")
    print(tanh(data))
    data_2 = np.arange(-10., 10., 0.1)
    th = tanh(data_2)
    plt.plot(data_2, th)
    plt.grid()
    plt.show()
