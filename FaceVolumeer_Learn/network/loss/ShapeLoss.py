from network.loss.L2Loss import L2Loss
from tensorflow import function, TensorSpec, float32, nn, cast, math as tfmath
import learning_const

class ShapeLoss(L2Loss):
    '''
    Класс для базовой L2-функции потерь по реконструкции объёма поверхности
    Используется в тех случаях тренировки, когда точная форма заранее известна
    '''

    def __init__(self, weight = 1.0):
        super().__init__(weight)

    @function(input_signature=(TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3))))
    def call(self, y_true, y_pred):
        return self.weight * (0.5 / cast(tfmath.reduce_prod(y_true.shape[:-1]), float32)) * nn.l2_loss(y_true - y_pred)
            #super().call(y_true, y_pred)
