import numpy as np


class Sigmoid:
    def __init__(self):
        # 学習するパラメータはない
        self.params = []

    def forward(self, x):
        """シグモイド関数における順伝播

        :param x: 前の層で線形変換された値
        :return: シグモイド関数で非線形変換した値
        """
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        """Initialization of Affine Layer
        層ごとに学習させるパラメータを保存しておく

        :param W: 重みの初期入力
        :param b: バイアスの初期入力
        """
        self.params = [W, b]

    def forward(self, x):
        """Affine Layer における順伝播

        :param x: 入力
        :return: 線形変換した出力
        """
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # すべての重みをリストにまとめる
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        """推論

        :param x: 初期入力
        :return: 全ての層を経た後の出力
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)
