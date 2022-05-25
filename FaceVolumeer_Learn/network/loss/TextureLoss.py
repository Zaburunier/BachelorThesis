from network.loss.L1Loss import L1Loss
from network.loss.L2Loss import L2Loss
from tensorflow import abs, reduce_sum, math as tfmath, cast, float32, function, TensorSpec, math as tfmath
import learning_const


class TextureLoss(L1Loss):
    '''
    Класс для базовой L2-функции потерь по реконструкции текстуры лица
    Используется в тех случаях тренировки, когда нужная текстура заранее известна
    '''

    def __init__(self, weight = 1.0):
        super().__init__(weight)


    @function(input_signature=(TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 1))))
    def call(self, y_true, y_pred, mask):
        count_non_zero = cast(tfmath.count_nonzero(mask), float32)
        return self.weight * reduce_sum(abs(y_true - y_pred)) / count_non_zero
        #return self.weight * (1.0 / cast(tfmath.reduce_prod(y_true.shape[:-1]), float32)) * reduce_sum(abs(y_true - y_pred))
        #super().call(y_true, y_pred)
