from tensorflow import reduce_sum, cast, square, function, TensorSpec, float32, math as tfmath, abs
from keras.losses import Loss
import learning_const


class ReconstructionLoss(Loss):
    """
    Функция потерь по реконструкции изображения
    Здесь используется техника L2,1-метрики:
    - Расстояние между каждой парой пикселей в исходном и синтетическом изображении
    подсчитано как корень из суммы квадратов
    (в отличие от стандартного L2, где идёт половина суммы квадратов)
    - Суммирование по пикселям проводится как L1 (простая сумма модулей)
    - В качестве результата берётся среднее значение
    """

    def __init__(self, weight = 1.0):
        super().__init__()
        self.weight = weight


    @function(input_signature=(TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 1))))
    def call(self, y_true, y_pred, mask):
        '''
        Вызов подсчёта ф-ции потерь
        :param y_true:
        :param y_pred:
        :return:
        '''
        count_nonzero = cast(tfmath.count_nonzero(mask), float32)
        return self.weight * reduce_sum(abs(y_pred - y_true)) / count_nonzero
