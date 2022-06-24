# Activation function implementation with Numpy

* ### Sigmoid activation function

The sigmoid function is a mathematics function having a characteristic "S"-shaped curve or sigmoid curve. as writen here
[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)

Sometimes referred to as a logistic function it is also a non-linear function used in Deep Learning and Logistic regression

$S(x) = \frac{1}{1+e^{-x}}$

![Sigmoid activation Image from Wikipedia](../Images/Sigmoid%20Image.png)

Some constraints for sigmoid if the value of x is large positive number the function gives number close to 1
and if x is a negative large number the function results into a number close to 0.

* ### Softmax activation

It is a generalization of logistic function with many dimensions.

The softmax function takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval(0,1), and the components will add up to 1, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities. Definition from Wikipedia [Softmax](https://en.wikipedia.org/wiki/Softmax_function)