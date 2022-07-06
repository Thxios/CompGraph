
import numpy as np
from ops.dg_ops_overload import NewTensor as Tensor


class Parameter(Tensor):
    def __init__(self, array, name=None, requires_grad=True):
        super(Parameter, self).__init__(
            array,
            requires_grad=requires_grad,
            inputs=None,
            grad_fn=None
        )
        self.name = name

    def __repr__(self):
        return f'T {self.name if self.name is not None else ""}[{self.value}]'

    def update(self, delta):
        self._value += delta

    @staticmethod
    def random_normal(shape, mean=0., stddev=0.02, **kwargs):
        arr = np.random.normal(mean, stddev, shape)
        return Parameter(arr, **kwargs)


class Layer:
    def __init__(self):
        self.params = []

    def add_parameter(self, param):
        self.params.append(param)

