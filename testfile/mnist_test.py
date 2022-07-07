
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

from ops.dg_ops_overload import NewTensor as Tensor, t_relu, t_matmul, \
    t_reshape, t_tanh, t_sigmoid, t_softmax_crossentropy, t_conv2d
from ops.param import Parameter
from optim.gd import GradientDescent, Adam


def dense():
    w1 = Parameter.he_initialization((28*28, 256), 28*28, name='w1')
    b1 = Parameter.zeros((256,), name='b1')

    w2 = Parameter.he_initialization((256, 10), 256, name='w2')
    b2 = Parameter.zeros((10,), name='b2')

    params = (w1, b2, w2, b2)

    def get_logit(x):
        x = t_matmul(x, w1) + b1
        x = t_relu(x)
        x = t_matmul(x, w2) + b2
        return x

    def get_loss(logit, y):
        loss = t_softmax_crossentropy(logit, y)
        return loss

    return params, get_logit, get_loss



def conv():
    k1 = Parameter.he_initialization((16, 1, 5, 5), 5*5*1)
    b1 = Parameter.zeros((16,))

    k2 = Parameter.he_initialization((32, 16, 3, 3), 3*3*16)
    b2 = Parameter.zeros((32,))

    k3 = Parameter.he_initialization((64, 32, 3, 3), 3*3*32)
    b3 = Parameter.zeros((64,))

    fcw1 = Parameter.he_initialization((3*3*64, 10), 3*3*64)
    fcb1 = Parameter.zeros((10,))

    params = (k1, b1, k2, b2, k3, b3, fcw1, fcb1)

    def get_logit(x):
        b = x.shape[0]

        x = t_conv2d(x, k1, stride=2, pad=0) + b1
        x = t_relu(x)
        x = t_conv2d(x, k2, stride=2, pad=1) + b2
        x = t_relu(x)
        x = t_conv2d(x, k3, stride=2, pad=1) + b3

        x = t_reshape(x, (b, -1))
        x = t_matmul(x, fcw1) + fcb1
        return x

    def get_loss(logit, y):
        # for ii in range(y.shape[0]):
        #     print(ii)
        #     print(logit.shape, y.shape)
        #     print(logit.value[ii], y[ii])
        loss = t_softmax_crossentropy(logit, y)
        return loss

    return params, get_logit, get_loss



if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_data()

    print(f'x_train {x_train.shape}')
    print(f'y_train {y_train.shape}')
    print(f'x_test {x_test.shape}')
    print(f'y_test {y_test.shape}')
    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    x_train = x_train.astype(np.float32)
    x_train = x_train / 255
    x_test = x_test.astype(np.float32)
    x_test = x_test / 255

    x_train = x_train.reshape((n_train_samples, 28, 28, 1))
    x_test = x_test.reshape((n_test_samples, 28, 28, 1))

    epochs = 3
    batch_size = 128

    # params_, get_logit_, get_loss_ = dense()
    params_, get_logit_, get_loss_ = conv()

    optimizer = GradientDescent(params_, lr=0.0008)
    # optimizer = Adam(params_, lr=0.001)

    indices = np.arange(n_train_samples)

    for epoch in range(epochs):
        np.random.shuffle(indices)

        i = 0
        for batch_idx in range(0, n_train_samples, batch_size):
            # break
            optimizer.zero_grad()

            st, ed = batch_idx, batch_idx+batch_size
            ind = indices[st: ed]


            x_, y_ = x_train[ind], y_train[ind]
            # x_, y_ = x_train[0:batch_size], y_train[0:batch_size]
            n_batch_samples = x_.shape[0]
            # print('y', y_)
            # y_ = one_hot(y_)

            x_ = Tensor(x_)

            lo = get_logit_(x_)
            loss_ = get_loss_(lo, y_)
            # print('logit', logit.value, logit.value.shape)
            loss_.backward()
            # print(i, 'loss', loss_.value / n_batch_samples)
            # print(b2)
            # print(b2.grad)

            optimizer.step()

            # print(w1.grad)
            # print(b1.grad)
            i += 1
            if i % 50 == 0:
                print(i, 'loss', loss_.value / n_batch_samples)

        correct_sum = 0
        loss_sum = 0.
        for batch_idx in range(0, n_test_samples, batch_size):
            st, ed = batch_idx, batch_idx+batch_size

            x_, y_ = x_test[st:ed], y_test[st:ed]

            lo = get_logit_(x_)
            loss_ = get_loss_(lo, y_).value
            pred = np.argmax(lo.value, axis=1)
            # print(y_, pred)
            correct_sum += np.sum((y_ == pred).astype(int))
            loss_sum += np.sum(loss_)

        print(f'epoch {epoch}')
        print(f'accuracy: {correct_sum / n_test_samples:.4f}')
        print(f'loss: {loss_sum / n_test_samples:.4f}')
        print()


