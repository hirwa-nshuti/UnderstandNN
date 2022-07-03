# Understanding Neural Networks from basics

A repository for deep diving into Deep Learning in Python most implementation will be using both Tensorflow and Pytorch framework

<a name='0'></a>

## Topics covered

| Sr. No | Topic Name                            | Markdowns                           | Code and Examples               |
|--------|:--------------------------------------|-------------------------------------|---------------------------------|
| 1.     | Numpy basics           | [Numpy basics](#1) | [Code Numpy basics](Basic_Numpy.ipynb) |
| 2.     | Activation functions                  | [Activation](#2) | [Code activations](Activations) |
| 3.     | Logistic regression as Neural network | [LR as NN](#3)               | [code LR as NN](Log_regression_from_Deeplearning.ipynb)|

<a name='1'></a>

## Numpy Basics

Numpy is a powerful mathematical tool used that allows fast and efficient computation

*****
Numpy contents:

* [Introduction](#1_1)
* [Numpy arrays](#1_2)
* [Operations](#1_3)
* [Further Reading](#1_4)

*****

<a name='2'></a>

## Activation function implementation with Numpy

******
Activation contents:

* [Sigmoid activation](#2_1)
* [Tanh activation](#2_2)
* [Rectified Linear Unit(reLu)](#2_3)
* [Softmax activation](#2_4)

******

<a name='2_1'></a>

* ### Sigmoid activation function

The sigmoid function is a mathematics function having a characteristic "S"-shaped curve or sigmoid curve. as writen here
[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)

Sometimes referred to as a logistic function it is also a non-linear function used in Deep Learning and Logistic regression
$S(x) = \frac{1}{1+e^{-x}}$

![Sigmoid activation Image designed using Matplotlib](Images/Sigmoid%20Image.png)

Some constraints for sigmoid if the value of x is large positive number the function gives number close to 1
and if x is a negative large number the function results into a number close to 0.

<a name='2_2'></a>

* ### Tanh activation function

Tou can read about RelU functions here [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
…$ s(z) = \frac{e^{-z_i}}{\sum_{i=1}^{k} e^{-z_i}}$he tanh function is a hyperbolic tangent function that is defined by a hyperbola instead of circle
for regular trigonometric functions its results is between -1 and 1 means that whenever we have a large negative number 
the function gives -1 and if it is a large positive input number the function gives 1.
You can read more about hyperbolic functions here [Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)

$tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^-{x}}$

![Tanh activation function Image designed using matplotlib](Images/Tanh%20Image.png)

<a name='2_3'></a>

* ### Rectified Linear Unit(ReLu)

The ReLu function is a linear function used to remove the negative part of the given array input,
Whenever the input is negative the function will give 0 and for positive inputs it will return that values.
You can read about RelU functions here [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

$f(x) = max(0, x)$
![Relu activation designed using Matplotlib](Images/Relu.png)
It may be simplified as 

```python
def relu(x):
    if x > 0:
        return x
    else:
        return 0
```

#### Here also we can have a leaky relu function which has a small slope for negative values

$f(x) = alpha * x$ if x < 0

$f(x) = x$ if x > 0

![Leaky Relu Designed using Matplotlib](Images/Leaky%20Relu.png)

<a name='2_4'></a>

* ### Softmax activation

It is a generalization of logistic function with many dimensions.

The softmax function takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval(0,1), and the components will add up to 1, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities. Definition from Wikipedia [Softmax](https://en.wikipedia.org/wiki/Softmax_function)

Always the sum of softmax distribution gives 1.

$ s(z) = \frac{e^{z_i}}{\sum_{i=1}^{k} e^{z_i}}$
or
$ s(z) = \frac{e^{-z_i}}{\sum_{i=1}^{k} e^{-z_i}}$


<a name='3'></a>

## Logistic Regression as a Neural network
