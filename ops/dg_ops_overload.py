
import numpy as np
from ops.dynamic_graph import Tensor
from ops.dg_ops import apply_fn as _apply_fn, add, const_add, sub, const_sub, const_rsub, mul, const_mul, \
    div, const_div, const_rdiv, neg, matmul, reshape

from ops.conv_util import conv2d


def apply_fn(fn, *inputs, **kwargs):
    # print(fn, inputs)
    x = map(lambda inp: inp.value if isinstance(inp, NewTensor) else inp, inputs)
    y, grad_fn = fn(*x, **kwargs)
    input_tensors = tuple(filter(lambda inp: isinstance(inp, NewTensor), inputs))

    if any(map(lambda inp: inp.requires_grad, input_tensors)):
        y_tensor = NewTensor(
            y,
            requires_grad=True,
            inputs=input_tensors,
            grad_fn=grad_fn
        )
    else:
        y_tensor = NewTensor(y)

    return y_tensor



class NewTensor(Tensor):

    def __add__(self, other):
        if isinstance(other, Tensor):
            return apply_fn(add, self, other)
        else:
            return apply_fn(const_add, self, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return apply_fn(sub, self, other)
        else:
            return apply_fn(const_sub, self, other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return apply_fn(sub, other, self)
        else:
            return apply_fn(const_rsub, other, self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return apply_fn(mul, self, other)
        else:
            return apply_fn(const_mul, self, other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return apply_fn(div, self, other)
        else:
            return apply_fn(const_div, self, other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return apply_fn(div, other, self)
        else:
            return apply_fn(const_rdiv, other, self)

    def __neg__(self):
        return apply_fn(neg, self)

    __rmul__ = __mul__
    __radd__ = __add__


def t_matmul(x1, x2):
    return apply_fn(matmul, x1, x2)

def t_reshape(x, shape):
    return apply_fn(reshape, x, shape)

def relu(x):
    # pos = (x >= 0).astype(np.float32)
    pos = (x >= 0).astype(np.float32)
    y = x * pos

    def grad_fn(dy):
        return dy * pos

    return y, grad_fn

def t_relu(x):
    return apply_fn(relu, x)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))

    def grad_fn(dy):
        return dy * y * (1-y)

    return y, grad_fn

def t_sigmoid(x):
    return apply_fn(sigmoid, x)

def tanh(x):
    y = np.tanh(x)

    def grad_fn(dy):
        return dy * (1-y**2)

    return y, grad_fn

def t_tanh(x):
    return apply_fn(tanh, x)

def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # e_x = np.exp(x)
    y = e_x / np.sum(e_x, axis=axis, keepdims=True)
    # print(np.sum(y, axis=axis))
    return y

def _one_hot(x):
    return np.eye(10)[x]

def softmax_crossentropy(logit, label):
    prob = _softmax(logit)
    # print('prob', prob)
    # ce = -label * np.log(prob)
    # ce = -np.sum(label * np.log(prob))
    # print(prob[label])
    _pred = prob[np.arange(prob.shape[0]), label]
    # print('pred', _pred)
    # print()
    ce = -np.sum(np.log(_pred + 1e-8))

    def grad_fn(dy):
        return dy * (prob - _one_hot(label))

    return ce, grad_fn

def t_softmax_crossentropy(logit, label):
    return apply_fn(softmax_crossentropy, logit, label)

def t_conv2d(x, kernel, stride=1, pad=0):
    return apply_fn(conv2d, x, kernel, stride=stride, pad=pad)


if __name__ == '__main__':
    a = np.array([
        [1, 3, -2],
        [3, 2, 0]
    ])
    b = np.array([
        [0, 1, 0],
        [0, 1, 0]
    ])
    print(np.sum(a, axis=-1))

    print(_softmax(a))
    print(_softmax(a, -1))

    ce, gf = softmax_crossentropy(a, b)
    print(ce)
    gd = gf(1)
    print(gd)

