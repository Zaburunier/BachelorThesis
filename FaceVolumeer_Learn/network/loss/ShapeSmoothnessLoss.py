from network.loss.L2Loss import L2Loss
from tensorflow import reduce_sum, square, sqrt, math as tfmath, cast, float32, function
from tensorflow import function, TensorSpec, float32, nn, cast, math as tfmath
import learning_const


class ShapeSmoothnessLoss(L2Loss):
    '''
    Функция потерь по гладкости переходов формы
    Для каждого элемента берёт на 4 соседей,
    вычисляет среднее и смотрит на разницу этого среднего и элемента
    '''

    def __init__(self, weight = 1.0):
        super(ShapeSmoothnessLoss, self).__init__(weight)

        self.weight = weight

    @function(input_signature=(TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0] - 2, learning_const.TEXTURE_SIZE[1] - 2, 3)),
                               TensorSpec(dtype=float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0] - 2, learning_const.TEXTURE_SIZE[1] - 2, 3))))
    def call(self, y_true, y_pred):
        return self.weight * (0.5 / cast(tfmath.reduce_prod(y_true.shape[:-1]), float32)) * nn.l2_loss(y_true - y_pred)
        #super().call(y_true, y_pred)


    @classmethod
    @function
    def CalculateNeighboursAverageShape(cls, shape_data, neighbours_amount=4.0):
        '''
        Вычисление средних значений соседей для каждого элемента
        :param shape_data: Данные об объёме
        :param neighbours_amount: Число соседей
        :return: Тензор, где в каждом экземпляре бэтча на месте значения элемента стоит среднее значение соседей
        ВНИМАНИЕ! Форма тензора будет другая, т. к. для крайних элементов зн-я соседей
        не подсчитываются и не заполняются
        '''

        # Если когда-то пригодится разное число соседей, то сделаем общую реализацию, а пока...
        if neighbours_amount != 4.0:
            print("Поддержка только для 4 соседей!")
            return

        n_dims = len(shape_data.shape)
        if (n_dims == 4):
            return (shape_data[:, :-2, 1:-1, :] +
                    shape_data[:, 2:, 1:-1, :] +
                    shape_data[:, 1:-1, :-2, :] +
                    shape_data[:, 1:-1, 2:, :]
                    ) / neighbours_amount
        elif n_dims == 3:
            return (shape_data[:-2, 1:-1, :] +
                    shape_data[2:, 1:-1, :] +
                    shape_data[1:-1, :-2, :] +
                    shape_data[1:-1, 2:, :]
                    ) / neighbours_amount


    @classmethod
    @function
    def TrimShape(cls, shape_data):
        '''
        Обрезаем крайние элементы, которые нельзя сравнить, поскольку для них нет соседей
        :param shape_data: Данные об объёме
        :return: Тензор, где в каждом экземпляре бэтча обрезаны крайние значения
        '''
        return shape_data[:, 1:-1, 1:-1, :]
