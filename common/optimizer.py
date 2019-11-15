class SGD:
    def __init__(self, lr=0.01):
        """

        :param lr: Learning rate
        """
        self.lr = lr

    def update(self, params, grads):
        """勾配と逆方向に重みを更新（最急降下法）

        :param params: 重み
        :param grads: 勾配
        :return: 更新後の重み
        """
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
