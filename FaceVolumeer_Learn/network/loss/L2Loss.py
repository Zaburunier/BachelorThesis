from tensorflow import nn, math as tfmath, cast, float32, function, TensorSpec
from keras.losses import Loss


class L2Loss(Loss):
    '''
    Общий класс для функций потерь L2
    '''
    def __init__(self, weight = 1.0):
        super().__init__()
        self.weight = weight

    @function
    def call(self, y_true, y_pred):
        #normalizing_ratio = cast(tfmath.reduce_prod(y_true.points[:-1]), float32)
        return self.weight * (0.5 / cast(tfmath.reduce_prod(y_true.shape[:-1]), float32)) * nn.l2_loss(y_true - y_pred)
