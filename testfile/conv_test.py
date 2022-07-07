
import numpy as np
import torch
import torch.nn.functional as F

from ops.conv_util import im2col, col2im, conv2d
from ops.dg_ops import matmul



if __name__ == '__main__':
    n_batch = 2
    in_channels = 3
    out_channels = 4

    # x = np.ones([n_batch, 16, 16, in_channels], dtype=np.float32)
    x = np.random.normal(size=[n_batch, 16, 16, in_channels]).astype(np.float32)

    # kernel = np.ones([out_channels, in_channels, 5, 5], dtype=np.float32)
    kernel = np.random.normal(scale=0.1, size=[out_channels, in_channels, 5, 5]).astype(np.float32)
    '''
    x_col = im2col(x, kernel.shape, pad=2)
    # print(x_col.shape)
    # print(x_col)
    x_re = col2im(x_col, x.shape, kernel.shape, pad=2)
    # print('x_re')
    # print(x_col.shape)
    k_flat = kernel.reshape([out_channels, -1]).T
    # print(k_flat.shape)

    y_flat, grad_fn = matmul(x_col, k_flat)
    # print('y_flat', y_flat.shape)
    y = y_flat.reshape([n_batch, 16, 16, out_channels])
    # y = y.transpose([0, 3, 1, 2])
    # print(y)
    x_col_grad, k_flat_grad = grad_fn(np.ones_like(y_flat, dtype=np.float32))
    # print(x_col_grad.shape)
    x_grad = col2im(x_col_grad, x.shape, kernel.shape, pad=2)
    # print('x_grad', x_grad.shape)
    # print(x_grad)
    print('k_flat_grad', k_flat_grad.shape)
    k_grad = k_flat_grad.T.reshape([out_channels, in_channels, 5, 5])
    print(k_grad)

    print('\n------\n')
    '''

    y, grad_fn = conv2d(x, kernel, pad=2)
    x_grad, k_grad = grad_fn(np.ones_like(y, dtype=np.float32))
    print(k_grad)


    xt = torch.tensor(x.transpose([0, 3, 1, 2]), requires_grad=True)
    kt = torch.tensor(kernel, requires_grad=True)

    yt = F.conv2d(xt, kt, padding=2)
    # print(yt)
    # print(yt.size())
    yt.backward(torch.ones_like(yt))
    xt_grad = xt.grad.permute((0, 2, 3, 1))
    # print('xt_grad', xt_grad.size())
    # print(xt_grad)
    print('\n------\n')
    print(kt.grad)

