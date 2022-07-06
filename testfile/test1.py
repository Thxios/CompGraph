
import numpy as np
import torch

from ops.basic import *


def show(arr, symbol):
    print(f'{symbol}: shape {arr.shape}\n{arr}')


if __name__ == '__main__':
    a = np.array([
        [1, 3],
        [5, 7],
        [-1, -2],
        [-3, -5]
    ])
    b = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    # b = np.arange(2*4).reshape((4, 2))

    show(a, 'a')
    show(b, 'b')

    c, grad = matmul(a, b)
    show(c, 'c')

    da, db = grad(np.ones_like(c))
    show(da, 'da')
    show(db, 'db')

    print()
    ta = torch.tensor(a.astype(float), requires_grad=True)
    taa = ta * 3
    tb = torch.tensor(b.astype(float), requires_grad=True)
    show(taa, 'ta')
    show(tb, 'tb')
    tc = torch.matmul(taa, tb)
    print(tc)
    tc.backward(gradient=torch.ones_like(tc))
    print(ta.grad)
    print(tb.grad)
    print(tc.grad)
