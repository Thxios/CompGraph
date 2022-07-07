
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
    def zeros(shape, **kwargs):
        arr = np.zeros(shape, dtype=np.float32)
        return Parameter(arr, **kwargs)


    @staticmethod
    def random_normal(shape, stddev=0.02, **kwargs):
        arr = np.random.normal(0, stddev, shape).astype(np.float32)
        return Parameter(arr, **kwargs)

    @staticmethod
    def he_initialization(shape, fan_in, **kwargs):
        stddev = np.sqrt(2 / fan_in)
        arr = np.random.normal(0, stddev, shape).astype(np.float32)
        return Parameter(arr, **kwargs)


class Layer:
    def __init__(self):
        self.params = []

    def add_parameter(self, param):
        self.params.append(param)

