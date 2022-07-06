

import numpy as np
from typing import List, Optional, Dict, Tuple, Callable


def vec(*args):
    return np.array(args)

def _as_iterable(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj

def _feed_node_inputs(nodes, inp):
    def feed(node):
        return node(inp)

    return [*map(feed, nodes)]


class GraphNode:
    def __init__(self, requires_grad=False, **kwargs):
        self.grad = None
        self.grad_fn = None

        self.requires_grad = requires_grad

    def __call__(self, x) -> np.ndarray:
        raise NotImplementedError

    def backward(self, gradient=None):
        # if not self.requires_grad:
        #     raise RuntimeError('has no grad')

        if self.requires_grad:
            self.grad = gradient



class Input(GraphNode):
    names = set()

    def __init__(self, name, **kwargs):
        super(Input, self).__init__(**kwargs)

        assert name not in Input.names, f'Input names must be unique, but name "{name}" is already exists.'
        Input.names.add(name)

        self.name = name

    def __call__(self, feed_dict: Dict[str, np.ndarray]) -> np.ndarray:
        assert self.name in feed_dict

        return feed_dict[self.name]


class Tensor(GraphNode):
    def __init__(self, init_value: np.ndarray, requires_grad=True, **kwargs):
        super(Tensor, self).__init__(requires_grad=requires_grad, **kwargs)

        self.value = init_value

    def __call__(self, x):
        return self.value


class Computation(GraphNode):
    def __init__(self, *node: GraphNode, **kwargs):
        super(Computation, self).__init__(**kwargs)
        self.prev = node
        self.pass_shape = None

        self.requires_grad = any(map(lambda x: x.requires_grad, node))

    def __call__(self, x) -> np.ndarray:
        x = _feed_node_inputs(self.prev, x)

        y, grad_fn = self.call(*x)

        if self.requires_grad:
            self.grad_fn = grad_fn
            self.pass_shape = y.shape

        return y

    def backward(self, gradient=None):
        super(Computation, self).backward(gradient)

        if self.requires_grad and self.grad_fn is not None:

            if gradient is None:
                gradient = np.ones(self.pass_shape)

            dxs = [*self.grad_fn(gradient)]

            for prev, dx in zip(self.prev, dxs):
                prev.backward(gradient=dx)

            self.grad_fn = None
            self.pass_shape = None

    def call(self, *args, **kwargs) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        raise NotImplementedError


class Exp(Computation):
    def call(self, x):
        y = np.exp(x)

        def grad_fn(dy):
            return dy * y

        return y, grad_fn


class Inv(Computation):
    def call(self, x):
        y = 1 / x

        def grad_fn(dy):
            return -dy / x**2

        return y, grad_fn


class Add(Computation):
    def call(self, x1, x2):
        y = x1 + x2

        def grad_fn(dy):
            return dy, dy

        return y, grad_fn




