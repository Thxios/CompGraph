
import numpy as np


def im2col(x, filter_shape, stride=1, pad=0):
    n, h, w, c = x.shape
    _, _, f_h, f_w = filter_shape

    out_h = (h + 2*pad - f_h) // stride + 1
    out_w = (w + 2*pad - f_w) // stride + 1

    img = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')
    # out = np.zeros((n, f_h, f_w, out_h, out_w, c))
    col = np.zeros((n, out_h, out_w, c, f_h, f_w), dtype=np.float32)

    for h in range(f_h):
        h_end = h + stride * out_h
        for w in range(f_w):
            w_end = w + stride * out_w
            col[:, :, :, :, h, w] = img[:, h:h_end:stride, w:w_end:stride, :]

    col = col.reshape(n * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_shape, stride=1, pad=0):
    n, h, w, c = input_shape
    _, _, f_h, f_w = filter_shape
    out_h = (h + 2*pad - f_h) // stride + 1
    out_w = (w + 2*pad - f_w) // stride + 1
    col = col.reshape(n, out_h, out_w, c, f_h, f_w)

    img = np.zeros((n, h + 2*pad + stride-1, w + 2*pad + stride-1, c), dtype=np.float32)
    for y in range(f_h):
        y_max = y + stride*out_h
        for x in range(f_w):
            x_max = x + stride*out_w
            img[:, y:y_max:stride, x:x_max:stride, :] += col[:, :, :, :, y, x]

    img = img[:, pad: h+pad, pad: w+pad, :]
    return img


def conv2d(x, kernel, stride=1, pad=0):
    b, h, w, c = x.shape

    outc, inc, k_h, k_w = kernel.shape
    # kernel_shape = kernel.shape

    out_h = (h + 2*pad - k_h) // stride + 1
    out_w = (w + 2*pad - k_w) // stride + 1

    x_col = im2col(x, (outc, inc, k_h, k_w), stride, pad)
    k_flat = kernel.reshape((outc, -1))

    y = np.dot(x_col, k_flat.T)

    y = y.reshape((b, out_h, out_w, outc))

    def grad_fn(dy):
        dy = dy.reshape((-1, outc))
        x_col_grad = np.dot(dy, k_flat)
        k_flat_grad = np.dot(x_col.T, dy)

        x_grad = col2im(x_col_grad, (b, h, w, c), (outc, inc, k_h, k_w), stride, pad)
        k_grad = k_flat_grad.T.reshape((outc, inc, k_h, k_w))

        return x_grad, k_grad

    return y, grad_fn



