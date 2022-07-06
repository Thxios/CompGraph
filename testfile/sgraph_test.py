
import numpy as np
from ops.static_graph import *



def case1():
    x = Input('x', requires_grad=False)
    y = Tensor(vec(2))
    f = Exp(Add(Inv(x), y))

    print(x.requires_grad, y.requires_grad, f.requires_grad)

    feed = {'x': vec(3)}
    print(feed)
    out = f(feed)
    print(out, type(out))
    f.backward()
    print(x.grad)
    print(y.grad)



if __name__ == '__main__':
    case1()
