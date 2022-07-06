


import numpy as np
from ops.dynamic_graph import Tensor


class TestTensor(Tensor):
    def __add__(self, other):
        print('__add__', self, other)

    def __radd__(self, other):
        print('__radd__', self, other)

    def __sub__(self, other):
        print('__sub__', self, other)

    def __rsub__(self, other):
        print('__rsub__', self, other)

    def __mul__(self, other):
        print('__mul__', self, other)

    def __rmul__(self, other):
        print('__rmul__', self, other)

    def __truediv__(self, other):
        print('__truediv__', self, other)

    def __rtruediv__(self, other):
        print('__rtruediv__', self, other)

    def __neg__(self):
        print('__neg__', self)

    def __eq__(self, other):
        print('__eq__', self, other)

    def __pow__(self, power, modulo=None):
        print('__pow__', self, power)

    def __rpow__(self, other):
        print('__rpow__', self, other)

    def __gt__(self, other):
        print('__gt__', self, other)

    def __lt__(self, other):
        print('__lt__', self, other)

    def __ge__(self, other):
        print('__ge__', self, other)

    def __le__(self, other):
        print('__le__', self, other)

    def __len__(self):
        print('__len__', self)
        return 3

    def __getitem__(self, item):
        print('__getitem__', self, item)
        return 3





def case1():
    x = TestTensor(2)
    print(x)

    x+3
    3+x
    x-3
    3-x
    x*3
    3*x
    x/3
    3/x
    -x
    x == 3
    3 == x
    x ** 3
    3 ** x
    x > 3
    x < 3
    x >= 3
    x <= 3
    len(x)
    x[3]


if __name__ == '__main__':
    case1()

