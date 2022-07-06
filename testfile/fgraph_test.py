
from ops.func_graph import *


def case1():
    x = node_input()
    x = node_inv(x)
    f = node_exp(x)

    def abs_f(x):
        return exp(1/x)

    out, _ = f(3)
    print(out)
    print(abs_f(3))


def case2():
    x = node_input()
    x = node_exp(x)

    y = node_input()
    y = node_sin(y)
    y = node_exp(y)

    x = node_add(x, y)
    f = node_inv(x)

    def abs_f(x, y):
        return 1 / (exp(x) + exp(sin(y)))

    print(f(1, 2))
    print(abs_f(1, 2))


def case3():
    x = node_input()
    x = node_inv(x)
    f = node_exp(x)

    def abs_f(x):
        return exp(1/x)

    def abs_grad(x):
        return -1 / x**2 * exp(1/x)

    feed = 3
    out, grad = f(feed)
    print(out, abs_f(feed))
    print(grad(1), abs_grad(feed))


def case4():
    x = node_input()
    x = node_exp(x)

    y = node_input()
    y = node_sin(y)
    y = node_exp(y)

    x = node_add(x, y)
    f = node_inv(x)

    def abs_f(x, y):
        return 1 / (exp(x) + exp(sin(y)))

    def abs_grad(x, y):
        tmp1 = exp(sin(y))
        tmp2 = exp(x) + tmp1
        return -exp(x) / tmp2**2, -cos(y) * tmp1 / tmp2**2

    feed = 1, 2
    out, grad = f(*feed)

    print(out, abs_f(*feed))
    print(grad(1), abs_grad(*feed))


if __name__ == '__main__':
    case3()
    # case2()
