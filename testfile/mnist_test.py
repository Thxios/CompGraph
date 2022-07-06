
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

from ops.dg_ops_overload import NewTensor as Tensor, t_relu, t_matmul, t_reshape, t_softmax_crossentropy
from ops.param import Parameter
from optim.gd import GradientDescent, Adam





if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_data()

    print(f'x_train {x_train.shape}')
    print(f'y_train {y_train.shape}')
    print(f'x_test {x_test.shape}')
    print(f'y_test {y_test.shape}')
    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    x_train = x_train.reshape((n_train_samples, -1))
    x_train = x_train / 255
    x_test = x_test.reshape((n_test_samples, -1))
    x_test = x_test / 255

    w1 = Parameter.random_normal((28*28, 256), name='w1')
    b1 = Parameter.random_normal((256,), name='b1')

    w2 = Parameter.random_normal((256, 10), name='w2')
    b2 = Parameter.random_normal((10,), name='b2')
    # w2 = Parameter.random_normal((28*28, 10), name='w2')
    # b2 = Parameter.random_normal((10,), name='b2')
    # w2 = Parameter.random_normal((256, 128), name='w2')
    # b2 = Parameter.random_normal((128,), name='b2')

    # w3 = Parameter.random_normal((128, 10), name='w3')
    # b3 = Parameter.random_normal((10,), name='b3')


    params = (w1, b2, w2, b2)

    # optimizer = GradientDescent(params, lr=0.005)
    optimizer = Adam(params, lr=0.004)

    def get_logit(x):
        x = t_matmul(x, w1) + b1
        x = t_relu(x)
        x = t_matmul(x, w2) + b2
        # x = t_relu(x)
        # x = t_matmul(x, w3) + b3
        return x

    def get_loss(logit, y):

        loss = t_softmax_crossentropy(logit, y)

        return loss

    def one_hot(x) -> np.ndarray:
        return np.eye(10)[x]

    epochs = 3
    batch_size = 128
    indices = np.arange(n_train_samples)

    for epoch in range(epochs):
        np.random.shuffle(indices)

        i = 0
        for batch_idx in range(0, n_train_samples, batch_size):
            # break
            optimizer.zero_grad()

            st, ed = batch_idx, batch_idx+batch_size
            ind = indices[st: ed]

            n_batch_samples = ind.shape[0]

            x_, y_ = x_train[ind], y_train[ind]
            # x_, y_ = x_train[0:2], y_train[0:2]
            # print('y', y_)
            # y_ = one_hot(y_)

            x_ = Tensor(x_)

            lo = get_logit(x_)
            loss_ = get_loss(lo, y_)
            # print('logit', logit.value, logit.value.shape)
            loss_.backward()
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

            lo = get_logit(x_)
            loss_ = get_loss(lo, y_).value
            pred = np.argmax(lo.value, axis=1)
            # print(y_, pred)
            correct_sum += np.sum((y_ == pred).astype(int))
            loss_sum += np.sum(loss_)

        print(f'epoch {epoch}')
        print(f'accuracy: {correct_sum / n_test_samples:.4f}')
        print(f'loss: {loss_sum / n_test_samples:.4f}')
        print()



