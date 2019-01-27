import numpy
from autograd import numpy as np
from autograd import grad
from autograd import hessian
import matplotlib.pyplot as plt

# Data
inputs = np.array(
    [[4, 1], [4.2, 1], [6, 1], [5.2, 1], [5.5, 1], [5.1, 1], [4.8, 1], [2.7, 1], [1.8, 1], [3.2, 1], [2.5, 1],
     [2.6, 1]])
targets = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


# Sigmoid function
def sigmoid(weights, inputs):
    return 1 / (1 + np.exp(-np.dot(inputs, weights)))


# Lossfunction
def loss(weights):
    pred = sigmoid(weights, inputs)
    return -np.sum(np.log(pred * targets + (1 - pred) * (1 - targets)))

# Newtons methode
def newton_optimizer(learning_rate, epochs, weights, loss_hist, weights_hist):
    breaker = 0
    for epoch in range(epochs):
        weights += learning_rate * np.linalg.solve(loss_hessian(weights), -loss_gradient(weights))
        loss_hist.append(loss(weights))
        weights_hist.append(weights)

        if (epoch > 1):
            if (loss_hist[-1] >= loss_hist[len(loss_hist) - 2]):
                learning_rate /= 10
                breaker += 1
        if (breaker > 10):
            return weights
    return weights

def plot_hist(hist):
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set(xlabel='epochs', ylabel='loss')
    ax.grid()


def plot_sigmoid(weights):
    x = numpy.array([numpy.linspace(1, 7, 1000).tolist(), numpy.ones(1000).tolist()]).T
    y = sigmoid(weights, x)
    fig, ax = plt.subplots()
    ax.plot(x[:, 0], y)
    ax.scatter(inputs[:, 0], targets)
    ax.set(xlabel='x', ylabel='y')
    ax.grid()


if __name__ == '__main__':
    loss_gradient = grad(loss)
    loss_hessian = hessian(loss)
    loss_hist = []
    weights_hist = []
    weights = np.array([3.0, -10.0])
    weights = newton_optimizer(0.001, 1000, weights, loss_hist, weights_hist)
    plot_hist(loss_hist)
    plot_sigmoid(weights)
    plt.show()
    print(weights)



