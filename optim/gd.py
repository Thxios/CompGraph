import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class GradientDescent(Optimizer):

    def step(self):
        for param in self.params:
            gradient = param.grad
            delta = -self.lr * gradient
            param.update(delta)

class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super(Adam, self).__init__(params, lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {
            id(param): np.zeros_like(param.value, dtype=np.float32) for param in params
        }
        self.v = {
            id(param): np.zeros_like(param.value, dtype=np.float32) for param in params
        }
        self.t = {
            id(param): 0 for param in params
        }

    def step(self):
        for param in self.params:
            pid = id(param)
            grad = param.grad

            m, v, t = self.m[pid], self.v[pid], self.t[pid]
            assert id(m) == id(self.m[pid])
            t += 1

            m = self.beta1*m + (1-self.beta1)*grad
            v = self.beta2*v + (1-self.beta2)*grad**2

            m_hat = m / (1-self.beta1**t)
            v_hat = v / (1-self.beta2**t)

            delta = -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.m[pid], self.v[pid], self.t[pid] = m, v, t
            param.update(delta)


