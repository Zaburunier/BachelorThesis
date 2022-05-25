from network.loss.L2Loss import L2Loss
from tensorflow import reshape, function, TensorSpec, float32, nn, cast, math as tfmath
import learning_const


class ProjectionLoss(L2Loss):
    '''
    Класс для базовой L2-функции потерь по определению ракурса
    Используется в тех случаях тренировки, когда ракурс заранее известен
    '''

    def __init__(self, weight = 1.0):
        super().__init__()
        self.weight = weight

    @function(input_signature = (TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, 8)), TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, 8))))
    def call(self, y_true, y_pred):
        # Здесь переводим в такой формат, поскольку усреднение идёт по всем размерностям, кроме последней. В результате мы не поделим на 8
        return self.weight * (0.5 / cast(tfmath.reduce_prod(y_true.shape), float32)) * nn.l2_loss(reshape(y_true, y_true.shape + [1]) - reshape(y_pred, y_pred.shape + [1]))
        #super().call(reshape(y_true, y_true.shape + [1]), reshape(y_pred, y_pred.shape + [1]))
