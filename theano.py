
# Python Machine Learning - Code Examples

# Parallelizing Neural Network Training with Theano library
#

import os
import theano
from theano import tensor as T
import numpy as np
import struct
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


#############################################################################
print(50 * '=')
print('First steps with Theano')
print(50 * '-')

# initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execute
net_input(2.0, 1.0, 0.5)


#############################################################################
print(50 * '=')
print('Configuring Theano')
print(50 * '-')

print('theano.config.floatX', theano.config.floatX)
theano.config.floatX = 'float32'

print('print(theano.config.device)', print(theano.config.device))


#############################################################################
print(50 * '=')
print('Working with array structures')
print(50 * '-')


# initialize
# if you are running Theano on 64 bit mode,
# you need to use dmatrix instead of fmatrix
x = T.fmatrix(name='x')
x_sum = T.sum(x, axis=0)

# compile
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# execute (Python list)
ary = [[1, 2, 3], [1, 2, 3]]
print('Column sum:', calc_sum(ary))

# execute (NumPy array)
ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Column sum:', calc_sum(ary))


# initialize
x = T.fmatrix(name='x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                             dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[x],
                            updates=update,
                            outputs=z)

# execute
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
for i in range(5):
    print('z%d:' % i, net_input(data))


"""
We can use the `givens` variable to insert values into the graph
before compiling it. Using this approach we can reduce the number
of transfers from RAM (via CPUs) to GPUs to speed up learning with
shared variables. If we use `inputs`, a datasets is transferred from
the CPU to the GPU multiple times, for example, if we iterate over a
dataset multiple times (epochs) during gradient descent. Via `givens`,
we can keep the dataset on the GPU if it fits (e.g., a mini-batch).
(theano documentation)
"""

# initialize
data = np.array([[1, 2, 3]],
                dtype=theano.config.floatX)
x = T.fmatrix(name='x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                             dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[],
                            updates=update,
                            givens={x: data},
                            outputs=z)

# execute
for i in range(5):
    print('z:', net_input())


#############################################################################
print(50 * '=')
print('Wrapping things up: A linear regression example')
print(50 * '-')

X_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0],
                      [5.0], [6.0], [7.0], [8.0], [9.0]],
                     dtype=theano.config.floatX)

y_train = np.asarray([1.0, 1.3, 3.1, 2.0, 5.0,
                      6.3, 6.6, 7.4, 8.0, 9.0],
                     dtype=theano.config.floatX)


def train_linreg(X_train, y_train, eta, epochs):

    costs = []
    # Initialize arrays
    eta0 = T.fscalar('eta0')
    y = T.fvector(name='y')
    X = T.fmatrix(name='X')
    w = theano.shared(np.zeros(
                      shape=(X_train.shape[1] + 1),
                      dtype=theano.config.floatX),
                      name='w')

    # calculate cost
    net_input = T.dot(X, w[1:]) + w[0]
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2))

    # perform gradient update
    gradient = T.grad(cost, wrt=w)
    update = [(w, w - eta0 * gradient)]

    # compile model
    train = theano.function(inputs=[eta0],
                            outputs=cost,
                            updates=update,
                            givens={X: X_train,
                                    y: y_train})

    for _ in range(epochs):
        costs.append(train(eta))

    return costs, w


costs, w = train_linreg(X_train, y_train, eta=0.001, epochs=10)

plt.plot(range(1, len(costs) + 1), costs)

plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Cost')
# plt.tight_layout()
# plt.savefig('./figures/cost_convergence.png', dpi=300)
plt.show()


def predict_linreg(X, w):
    Xt = T.matrix(name='X')
    net_input = T.dot(Xt, w[1:]) + w[0]
    predict = theano.function(inputs=[Xt], givens={w: w}, outputs=net_input)
    return predict(X)


plt.scatter(X_train, y_train, marker='s', s=50)
plt.plot(range(X_train.shape[0]),
         predict_linreg(X_train, w),
         color='gray',
         marker='o',
         markersize=4,
         linewidth=3)

plt.xlabel('x')
plt.ylabel('y')

# plt.tight_layout()
# plt.savefig('./figures/linreg.png', dpi=300)
plt.show()


#############################################################################
print(50 * '=')
print('Wrapping things up: A linear regression example')
print(50 * '-')


# note that first element (X[0] = 1) to denote bias unit

X = np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])


def net_input(X, w):
    z = X.dot(w)
    return z


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)


print('P(y=1|x) = %.3f' % logistic_activation(X, w)[0])


# W : array, shape = [n_output_units, n_hidden_units+1]
#          Weight matrix for hidden layer -> output layer.
# note that first column (A[:][0] = 1) are the bias units
W = np.array([[1.1, 1.2, 1.3, 0.5],
              [0.1, 0.2, 0.4, 0.1],
              [0.2, 0.5, 2.1, 1.9]])

# A : array, shape = [n_hidden+1, n_samples]
#          Activation of hidden layer.
# note that first element (A[0][0] = 1) is for the bias units

A = np.array([[1.0],
              [0.1],
              [0.3],
              [0.7]])

# Z : array, shape = [n_output_units, n_samples]
#          Net input of output layer.

Z = W.dot(A)
y_probas = logistic(Z)
print('Probabilities:\n', y_probas)

y_class = np.argmax(Z, axis=0)
print('predicted class label: %d' % y_class[0])


#############################################################################
print(50 * '=')
print('Estimating probabilities in multi-class'
      ' classification via the softmax function')
print(50 * '-')


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_activation(X, w):
    z = net_input(X, w)
    return softmax(z)


y_probas = softmax(Z)
print('Probabilities:\n', y_probas)

print('Sum of probabilities', y_probas.sum())

y_class = np.argmax(Z, axis=0)
print('Predicted class', y_class)


#############################################################################
print(50 * '=')
print('Broadening the output spectrum using a hyperbolic tangent')
print(50 * '-')


def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

# alternatives:
# from scipy.special import expit
# log_act = expit(z)
# tanh_act = np.tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle='--')
plt.axhline(0.5, color='black', linestyle='--')
plt.axhline(0, color='black', linestyle='--')
plt.axhline(-1, color='black', linestyle='--')

plt.plot(z, tanh_act,
         linewidth=2,
         color='black',
         label='tanh')
plt.plot(z, log_act,
         linewidth=2,
         color='lightgreen',
         label='logistic')

plt.legend(loc='lower right')
# plt.tight_layout()
# plt.savefig('./figures/activation.png', dpi=300)
plt.show()


