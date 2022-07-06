
import numpy as np


def add(x1, x2):
    y = x1 + x2

    def grad_fn(dy):
        return dy, dy

    return y, grad_fn


def sub(x1, x2):
    y = x1 - x2

    def grad_fn(dy):
        return dy, -dy

    return y, grad_fn


def matmul(x1, x2):
    y = np.dot(x1, x2)

    def grad_fn(dy):
        dx1 = np.dot(dy, x2.T)
        dx2 = np.dot(x1.T, dy)
        return dx1, dx2

    return y, grad_fn


def mul(x1, x2):
    y = x1 * x2

    def grad_fn(dy):
        dx1 = dy * x2
        dx2 = dy * x1
        return dx1, dx2

    return y, grad_fn


def div(x1, x2):
    y = x1 / x2

    def grad_fn(dy):
        dx1 = dy / x2
        dx2 = -dy * x1 / x2**2
        return dx1, dx2

    return y, grad_fn


def inv(x):
    y = 1 / x

    def grad_fn(dy):
        dx = -dy / x**2
        return dx

    return y, grad_fn


def exp(x):
    y = np.exp(x)

    def grad_fn(dy):
        return dy * y

    return y, grad_fn


def reshape(x, shape):
    origin = x.shape
    y = x.reshape(shape)

    def grad_fn(dy):
        return dy.reshape(origin)

    return y, grad_fn

