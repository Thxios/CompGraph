
from math import exp, sin, cos


def node_exp(node):
    def fn(*x):
        x, grad = node(*x)
        print('exp', x)

        def grad_fn(dy):
            dx = dy * exp(x)
            return grad(dx)

        y = exp(x)
        return y, grad_fn

    return fn

def node_inv(node):
    def fn(*x):
        x, grad = node(*x)
        print('inv', x)

        def grad_fn(dy):
            dx = -dy / x**2
            return grad(dx)

        y = 1/x
        return y, grad_fn

    return fn

def node_input():
    def fn(x):
        def grad_fn(dy):
            return dy

        return x, grad_fn

    return fn

def node_add(node1, node2):
    def fn(*x):
        x1, x2 = x
        x1, grad1 = node1(x1)
        x2, grad2 = node2(x2)
        print('add', x1, x2)

        def grad_fn(dy):
            dx1, dx2 = dy, dy
            return grad1(dx1), grad2(dx2)

        y = x1 + x2
        return y, grad_fn

    return fn

def node_sin(node):
    def fn(*x):
        x, grad = node(*x)
        print('sin', x)

        def grad_fn(dy):
            dx = dy * cos(x)
            return grad(dx)

        y = sin(x)
        return y, grad_fn

    return fn


