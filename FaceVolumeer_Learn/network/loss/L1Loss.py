from tensorflow import abs, reduce_sum, math as tfmath, cast, float32, function
from keras.losses import Loss


class L1Loss(Loss):
    '''
    Общий класс для функций потерь L1
    '''
    def __init__(self, weight = 1.0):
        super().__init__()
        self.weight = weight


    @function
    def call(self, y_true, y_pred):
        #normalizing_ratio = cast(tfmath.reduce_prod(y_true.points[:-1]), float32)
        return self.weight * (1.0 / cast(tfmath.reduce_prod(y_true.shape[:-1]), float32)) * reduce_sum(abs(y_true - y_pred))
