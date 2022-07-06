
import numpy as np

from ops.dynamic_graph import Tensor
# from ops.dg_ops_overload import NewTensor as Tensor


def apply_fn(fn, *inputs, **kwargs) -> Tensor:
    x = map(lambda inp: inp.value if isinstance(inp, Tensor) else inp, inputs)
    y, grad_fn = fn(*x, **kwargs)
    input_tensors = tuple(filter(lambda inp: isinstance(inp, Tensor), inputs))

    if any(map(lambda inp: inp.requires_grad, input_tensors)):
        y_tensor = Tensor(
            y,
            requires_grad=True,
            inputs=input_tensors,
            grad_fn=grad_fn
        )
    else:
        y_tensor = Tensor(y)

    return y_tensor


def add(x1, x2):
    y = x1 + x2

    def grad_fn(dy):
        return dy, dy

    return y, grad_fn


def const_add(x, c):
    y = x + c

    def grad_fn(dy):
        return dy

    return y, grad_fn


def neg(x):
    y = -x

    def grad_fn(dy):
        return -dy

    return y, grad_fn


def mul(x1, x2):
    y = x1 * x2

    def grad_fn(dy):
        dx1 = dy * x2
        dx2 = dy * x1
        return dx1, dx2

    return y, grad_fn


def const_mul(x, c):
    y = c * x

    def grad_fn(dy):
        return c * dy

    return y, grad_fn


def sub(x1, x2):
    y = x1 - x2

    def grad_fn(dy):
        return dy, -dy

    return y, grad_fn


def const_sub(x, c):
    y = x - c

    def grad_fn(dy):
        return dy

    return y, grad_fn


def const_rsub(c, x):
    y = c - x

    def grad_fn(dy):
        return -dy

    return y, grad_fn


def div(x1, x2):
    y = x1 / x2

    def grad_fn(dy):
        dx1 = dy / x2
        dx2 = -dy * x1 / x2 ** 2
        return dx1, dx2

    return y, grad_fn


def const_div(x, c):
    y = x / c

    def grad_fn(dy):
        return dy / c

    return y, grad_fn


def const_rdiv(c, x):
    y = c / x

    def grad_fn(dy):
        return -dy * c / x ** 2

    return y, grad_fn


def inv(x):
    y = 1 / x

    def grad_fn(dy):
        dx = -dy / x ** 2
        return dx

    return y, grad_fn


def exp(x):
    y = np.exp(x)

    def grad_fn(dy):
        return dy * y

    return y, grad_fn


def matmul(x1, x2):
    y = np.dot(x1, x2)

    def grad_fn(dy):
        dx1 = np.dot(dy, x2.T)
        dx2 = np.dot(x1.T, dy)
        return dx1, dx2

    return y, grad_fn


def reshape(x, shape):
    origin = x.shape
    y = x.reshape(shape)

    def grad_fn(dy):
        return dy.reshape(origin)

    return y, grad_fn


