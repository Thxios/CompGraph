
import numpy as np
from ops.dynamic_graph import Tensor
from ops.dg_ops import apply_fn, add, inv, exp, const_add, const_mul
from ops.dg_ops_overload import NewTensor



def vec(*args):
    return np.array(args, dtype=float)

def fn_add(x1, x2):
    return apply_fn(add, x1, x2)

def fn_inv(x):
    return apply_fn(inv, x)

def fn_exp(x):
    return apply_fn(exp, x)

def fn_const_add(x1, x2):
    return apply_fn(const_add, x1, x2)

def fn_const_mul(x1, x2):
    return apply_fn(const_mul, x1, x2)



def case1():
    def f(x, y):
        return fn_exp(fn_add(fn_inv(x), y))

    x1 = Tensor([
        [1, 2],
        [3, 4]
    ], requires_grad=True)
    x2 = Tensor([
        [-1, 0.5],
        [1, -2]
    ], requires_grad=True)
    y = f(x1, x2)
    print('y', y)
    y.backward()
    print(x1.grad)
    print(x2.grad)
    print(y.grad)


def case2():
    def f(x):
        return fn_exp(fn_const_add(fn_const_mul(x, -0.5), -3))

    x = Tensor(-2, requires_grad=True)
    print(x)
    y = f(x)
    print(y)

    y.backward()
    print(x.grad)


def case3():

    x1 = NewTensor(-4, requires_grad=True)
    x2 = NewTensor(2, requires_grad=True)

    y = (x1 + 3 + x1*x2) * (4*x2 + 5)

    print(y)
    y.backward()
    print(x1.grad)
    print(x2.grad)


if __name__ == '__main__':
    case3()

