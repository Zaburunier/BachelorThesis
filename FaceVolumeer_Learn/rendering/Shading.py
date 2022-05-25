import math

from tensorflow import Tensor, reshape, split, concat, constant, int64, device, function
from tensorflow import math as tfmath, gather, matmul, squeeze, tile
from tensorflow import ones_like, square, expand_dims, clip_by_value, Variable
from tensorflow_graphics.math import math_helpers as tfgmath_helpers
from tensorflow_graphics.math import spherical_harmonics as tfgmath_spherical_harmonics

import learning_const
from learning_const import *

# coeffs_shape = normal_as_spherical_coords.points[:-1] + [1]
with device("/job:localhost/replica:0/task:0/device:CPU:0"):
    l = Variable(tile(input=[[[0, 1, 1, 1, 2, 2, 2, 2, 2]]], multiples=[learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0] * learning_const.TEXTURE_SIZE[1], 1]), trainable = False)
                 #caching_device="/job:localhost/replica:0/task:0/device:GPU:0")
    m = Variable(tile(input=[[[0, -1, 0, 1, -2, -1, 0, 1, 2]]], multiples=[learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0] * learning_const.TEXTURE_SIZE[1], 1]), trainable = False)
                 #caching_device="/job:localhost/replica:0/task:0/device:GPU:0")


class Shading:
    '''
    Класс генератора текстур затенения
    '''

    def __init__(self):
        pass

    @classmethod
    def GenerateShade(cls,  base_model_triangle_data, base_model_triangle_2d_data, base_model_triangle_barycentric_data, lightning_data: Tensor, vertex_normals: Tensor,
                      texture_size=(192, 224), debug_shapes = False, debug_values = False):
        '''
        Создание текстуры затенения
        :param data_container: Словарь с исходными сведениями о базовой модели
        (с ключами "triangle", "triangle_2d", "barycentric", "vertex_to_triangle", "vertex_to_uv")
        :param lightning_data: Данные об освещении, полученные из кодировщика (размер BATCH_SIZE x 27)
        :param vertex_normals: Нормали вершин (размер BATCH_SIZE x const.MESH_VERTICES x 3)
        :param texture_size: Размер выходной текстуры
        :param debug_shapes: Выдавать ли в консоль информацию по формам получаемых тензоров?
        :param debug_values: Выдавать ли в консоль информацию по значениям получаемых тензоров?
        :return: Текстуры затенения заданного размера (BATCH_SIZE x texture_size x 3)
        '''

        # Данные для текстурных координат сразу переводим в линейную форму,
        # поскольку все прочие данные у нас представлены именно так
        triangle_2d_data = reshape(base_model_triangle_2d_data, shape=[-1, 1])

        vertex_2d_indices_0 = reshape(gather(base_model_triangle_data[0], triangle_2d_data), shape=[-1])
        vertex_2d_indices_1 = reshape(gather(base_model_triangle_data[1], triangle_2d_data), shape=[-1])
        vertex_2d_indices_2 = reshape(gather(base_model_triangle_data[2], triangle_2d_data), shape=[-1])

        # То же самое с барицентрическими координатами
        triangle_vertex_barycentric_ratios_0 = tile(reshape(base_model_triangle_barycentric_data[:, :, 0], shape=[1, -1, 1]),
                                                    [learning_const.BATCH_SIZE, 1, 1])
        triangle_vertex_barycentric_ratios_1 = tile(reshape(base_model_triangle_barycentric_data[:, :, 1], shape=[1, -1, 1]),
                                                    [learning_const.BATCH_SIZE, 1, 1])
        triangle_vertex_barycentric_ratios_2 = tile(reshape(base_model_triangle_barycentric_data[:, :, 2], shape=[1, -1, 1]),
                                                    [learning_const.BATCH_SIZE, 1, 1])

        # Получаем информацию о нормалях в двумерном пространстве (на генерируемом изображении)
        vertex_normals_2d_0 = gather(vertex_normals, vertex_2d_indices_0, axis=1)
        vertex_normals_2d_1 = gather(vertex_normals, vertex_2d_indices_1, axis=1)
        vertex_normals_2d_2 = gather(vertex_normals, vertex_2d_indices_2, axis=1)

        # Составляем двумерную карту нормалей
        vertex_normals_2d = vertex_normals_2d_0 * triangle_vertex_barycentric_ratios_0 + \
                            vertex_normals_2d_1 * triangle_vertex_barycentric_ratios_1 + \
                            vertex_normals_2d_2 * triangle_vertex_barycentric_ratios_2

        if (debug_values):
            print("Отладочная информация по результату работы метода GenerateShade (для одного образца)")
            print(f"\nИсходные нормали: {vertex_normals[0]};"
                  f"\nНормали после барицентрического преобразования: {vertex_normals_2d[0]};")

        # Из нормалей и света получаем текстуру затенения и преобразовываем её в нужный формат
        shading = Shading.Shading(lightning_data, vertex_normals_2d, debug_shapes, debug_values)
        return reshape(shading, shape=[-1, texture_size[0], texture_size[1], 3])


    @classmethod
    def Shading(cls, lightning_data, vertex_normals, debug_shapes = False, debug_values = False):
        '''
        Формирование текстуры затенения
        :param lightning_data: Данные об освещении, полученные от расшифровщика (размер BATCH_SIZE x const.LightParamDimensionSize)
        :param vertex_normals: Нормали вершин
        :param debug_shapes: Выдавать ли в консоль информацию по формам получаемых тензоров?
        :param debug_values: Выдавать ли в консоль информацию по значениям получаемых тензоров?
        :return:
        '''
        normals_shape = vertex_normals.get_shape().as_list()

        normal_x, normal_y, normal_z = split(expand_dims(vertex_normals, -1), axis=2, num_or_size_splits=3)

        pi = math.pi
        sh = [0] * 9
        sh[0] = 1 / math.sqrt(4 * pi) * ones_like(normal_x)
        sh[1] = ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))) * normal_z
        sh[2] = ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))) * normal_y
        sh[3] = ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))) * normal_x
        sh[4] = (pi / 4) * (1 / 2) * (math.sqrt(5 / (4 * pi))) * (
                2 * square(normal_z) - square(normal_x) - square(normal_y))
        sh[5] = (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))) * (normal_y * normal_z)
        sh[6] = (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))) * (normal_x * normal_z)
        sh[7] = (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))) * (normal_x * normal_y)
        sh[8] = (pi / 4) * (3 / 2) * (math.sqrt(5 / (12 * pi))) * (square(normal_x) - square(normal_y))

        sh = concat(sh, axis=3)

        L1, L2, L3 = split(lightning_data, num_or_size_splits=3, axis=1)
        L1 = expand_dims(L1, 1)
        L1 = tile(L1, multiples=[1, normals_shape[1], 1])
        L1 = expand_dims(L1, -1)

        L2 = expand_dims(L2, 1)
        L2 = tile(L2, multiples=[1, normals_shape[1], 1])
        L2 = expand_dims(L2, -1)

        L3 = expand_dims(L3, 1)
        L3 = tile(L3, multiples=[1, normals_shape[1], 1])
        L3 = expand_dims(L3, -1)

        # print('L1.get_shape()')
        # print(L1.get_shape())

        B1 = matmul(sh, L1)
        B2 = matmul(sh, L2)
        B3 = matmul(sh, L3)

        B = squeeze(concat([B1, B2, B3], axis=2))

        if (debug_values):
            print("Отладочная информация по результату работы метода Shading (для одного образца)")

        return B

    @classmethod
    @function()
    def GetNormalShading(cls, vertex_normals, harmonics_c_ratios):
        '''
        Оценка множителей для получения карты затенения
        :param vertex_normals: Нормали вершин (размер [BATCH_SIZE, MESH_VERTICES, 3])
        :param harmonics_c_ratios: Оцененные нейросетью коэффициенты разложения сферических гармоник
        размера [BATCH_SIZE, 27] (9 на каждый из трёх каналов)
        :return: Карта затенения (размер [BATCH_SIZE, MESH_VERTICES, 3])
        '''
        normal_as_spherical_coords = tfgmath_helpers.cartesian_to_spherical_coordinates(vertex_normals)
        #theta = expand_dims(normal_as_spherical_coords[:, :, 1], axis = -1)
        #phi = expand_dims(normal_as_spherical_coords[:, :, 2], axis = -1) + math.pi
        #print(f"Theta angle max = {tfmath.reduce_max(theta)} and min = {tfmath.reduce_min(theta)}")
        #print(f"Phi angle max = {tfmath.reduce_max(phi)} and min = {tfmath.reduce_min(phi)}")

        harmonics_y_ratios = tfgmath_spherical_harmonics.evaluate_spherical_harmonics(degree_l = l,
                                                                               order_m = m,
                                                                               theta = expand_dims(normal_as_spherical_coords[:, :, 1], axis = -1),
                                                                               phi = expand_dims(normal_as_spherical_coords[:, :, 2], axis = -1) + math.pi)
        harmonics_c_ratios = expand_dims(harmonics_c_ratios, axis = -2)
        #result = tfmath.reduce_sum(harmonics_c_ratios * harmonics_y_ratios, axis=2, keepdims=True)
        return tfmath.reduce_sum(harmonics_c_ratios * harmonics_y_ratios, axis=2, keepdims=True)


