
import numpy as np


def _as_iterable(obj):
    if not isinstance(obj, (list, tuple)):
        return obj,
    return obj


def _as_array(value):
    if not isinstance(value, np.ndarray):
        value = np.array(value, dtype=float)
    return value


def _reduce_grad(value, shape):
    if value.shape != shape:
        value = np.sum(value, axis=0)
    return value



class Tensor:
    def __init__(self, array, requires_grad=False, inputs=None, grad_fn=None):
        self._value = _as_array(array)

        self._requires_grad = requires_grad
        self._retain_grad = requires_grad and inputs is None

        self.inputs = inputs
        self.grad_fn = grad_fn

        if self.retain_grad:
            self._grad = np.zeros_like(self._value)
        else:
            self._grad = None

    def __repr__(self):
        return f'T[{self.value}]'

    @property
    def value(self):
        return self._value

    @property
    def grad(self):
        return self._grad

    @property
    def retain_grad(self):
        return self._retain_grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, flag):
        if flag == self.requires_grad:
            return

        if flag is False:
            if self.inputs is not None and any(map(lambda x: x.requires_grad, self.inputs)):
                raise RuntimeError('cannot change a non-leaf node of backpropagation')
            else:
                self._requires_grad = False
                self._grad = None
                self.clear_grad_fn()

        elif flag is True:
            self._requires_grad = True
            self._grad = np.zeros_like(self.value)

    def backward(self, gradient=None):
        assert self.requires_grad

        # print('grad', self, gradient, self.grad_fn)

        if gradient is None:
            gradient = np.ones_like(self.value, dtype=float)
        else:
            gradient = _reduce_grad(gradient, self._value.shape)

        if self.retain_grad:
            self._grad += gradient

        if self.grad_fn is not None:
            dxs = _as_iterable(self.grad_fn(gradient))

            for inp, dx in zip(self.inputs, dxs):
                if inp.requires_grad:
                    inp.backward(dx)

        self.clear_grad_fn()

    def clear_grad_fn(self):
        self.inputs = None
        self.grad_fn = None

    def zero_grad(self):
        assert self.retain_grad
        self._grad *= 0


if __name__ == '__main__':
    pass
