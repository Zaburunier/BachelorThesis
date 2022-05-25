from tensorflow import abs, reduce_sum, map_fn
from tensorflow import image, math as tfmath, cast, float32, function, TensorSpec
from network.loss.L1Loss import L1Loss
import learning_const


class AlbedoSymmetryLoss(L1Loss):
    '''
    Функция потерь по симметрии лица с точки зрения отражений
    '''

    def __init__(self, weight = 1.0):
        super().__init__(weight)


    @function(input_signature=(TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3))))
    def call(self, y_true, y_pred):
        '''

        :param y_true:
        :param y_pred:
        :return:
        '''
        return self.weight * (1.0 / cast(tfmath.reduce_prod(y_true.shape[:-1]), float32)) * reduce_sum(abs(y_true - y_pred))
        #super().call(y_true, y_pred)


    @classmethod
    @function
    def FlipAlbedos(cls, albedo_data):
        '''
        Отражение текстуры альбедо по горизонтали (для подсчёта функции потерь)
        :param albedo_data: Текстуры альбедо
        :return:
        '''
        return map_fn(lambda instance: image.flip_left_right(instance), albedo_data)
