from tensorflow import abs, reduce_sum, exp, pow, divide, norm, sqrt, square
from tensorflow import cast, float32, not_equal, reduce_mean, function, TensorSpec
from keras.losses import Loss
import learning_const


class AlbedoConstancyLoss(Loss):
    '''
    Функция потерь по
    '''

    def __init__(self, weight = 1.0):
        super().__init__()
        self.weight = weight


    @function
    def call(self, y_true, y_pred):
        '''

        :param y_true:
        :param y_pred:
        :return:
        '''
        return sqrt(reduce_sum(square(y_true - y_pred), axis=[-1]) + 1.0e-6)


    @function(input_signature=(TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3))))
    def CalculateLoss(self, textures, albedo):
        '''

        :param textures:
        :param texture_visibility_mask:
        :param albedo:
        :return:
        '''
        textures_as_mask = cast(not_equal(reduce_sum(textures, axis = -1, keepdims = True), 0), dtype=float32)
        weights_u, weights_v = AlbedoConstancyLoss.CalculateAlbedoWeights(textures,
                                                                          textures_as_mask)
        albedo_constancy_loss_u = weights_u * pow(self.__call__(albedo[:, :-1, :, :],
                                                                albedo[:, 1:, :, :]),
                                                  0.8)

        albedo_constancy_loss_v = weights_v * pow(self.__call__(albedo[:, :, :-1, :],
                                                                albedo[:, :, 1:, :]),
                                                  0.8)

        albedo_constancy_loss_u = reduce_sum(albedo_constancy_loss_u)
        albedo_constancy_loss_v = reduce_sum(albedo_constancy_loss_v)

        return reduce_mean(albedo_constancy_loss_u) / reduce_sum(weights_u) + \
               reduce_mean(albedo_constancy_loss_v) / reduce_sum(weights_v)


    @classmethod
    @function
    def CalculateAlbedoWeights(cls, textures, visibility_mask, neighbours_amount=4.0):
        '''

        :param albedo_data:
        :param neighbours_amount:
        :return:
        '''

        # Если когда-то пригодится разное число соседей, то сделаем общую реализацию, а пока...
        if neighbours_amount != 4.0:
            print("Поддержка только для 4 соседей!")
            return

        textures_chromacity = textures
        textures_chromacity = divide(textures_chromacity,
                                     reduce_sum(textures_chromacity, axis=[-1], keepdims = True) + 1.0e-6)

        weights_u = exp(-15 * AlbedoConstancyLoss.Norm(textures_chromacity[:, :-1, :, :] -
                                                       textures_chromacity[:, 1:, :, :])) * \
                    visibility_mask[:, :-1, :, :]
        weights_v = exp(-15 * AlbedoConstancyLoss.Norm(textures_chromacity[:, :, :-1, :] -
                                                       textures_chromacity[:, :, 1:, :])) * \
                    visibility_mask[:, :, :-1, :]


        return weights_u + 1.0e-10, weights_v + 1.0e-10


    @classmethod
    @function
    def Norm(cls, tensor):
        return sqrt(reduce_sum(square(tensor)) + 1.0e-6) + 1.0e-6
