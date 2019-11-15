import numpy as np


class MatMul:
    def __init__(self, W):
        self.params = [W]
        # np.zeros(W.shape)
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        """逆伝播

        :param dout: 誤差逆伝播法における前の層の勾配
        :return: 勾配
        """
        W, = self.params
        # 転置
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # grads の参照先は変えずに値のみ代入
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # Repeatノード
        # ベクトルの形状を保つために axis=0 を指定
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
