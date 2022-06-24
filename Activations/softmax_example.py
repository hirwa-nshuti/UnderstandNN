import numpy as np
from Activations import softmax


if __name__ == "__main__":
    print("Testing the softmax function")
    arr = np.array([12, 9, -2, 7, -8])
    sft = softmax(arr)
    print(f"The softmax values is {sft}")
    print("Sum of the softmax distribution")
    print(sft.sum())
