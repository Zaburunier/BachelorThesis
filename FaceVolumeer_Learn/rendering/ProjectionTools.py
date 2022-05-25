from tensorflow import reshape, Tensor, float32, math, zeros, transpose, matmul, function
import numpy as np
from tensorflow import tile, expand_dims, linalg, convert_to_tensor, eye, concat, map_fn, TensorSpec, ones
from transformations import compose_matrix
from tensorflow_graphics.geometry.transformation.rotation_matrix_3d import from_euler, rotate
import learning_const


def CalculateTfgRotationMatrices(projection_data : Tensor) -> Tensor:
    '''
    Получение матриц вращения из данных по ракурсу от нейронной сети
    :param projection_data:
    :return:
    '''

    rotation_matrices = from_euler(angles = projection_data[:, :3])
    return rotation_matrices


def CalculateTfgViewMatrices(projection_data : Tensor) -> Tensor:
    rotation_matrices = CalculateTfgRotationMatrices(projection_data)
    rotation_matrices = concat([rotation_matrices, reshape(projection_data[:, 4:7], (projection_data.shape[0], 3, 1))], axis = 2)
    rotation_matrices = concat([rotation_matrices, tile(reshape([0.0, 0.0, 0.0, 1.0], (1, 1, 4)), (projection_data.shape[0], 1, 1))], axis = 1)
    return rotation_matrices



def CalculateViewMatrices(projection_data : Tensor) -> Tensor:
    #translations = projection_data[:, 4:7]
    #rotation_angles = projection_data[:, :3]
    #scales = ones(points = [translations.points[0], 3])
    return map_fn(fn = view_matrix_mapped, elems = [projection_data[:, 4:7], projection_data[:, :3], ones(shape = [projection_data.shape[0], 3])], fn_output_signature = TensorSpec(shape = (4, 4)))


def view_matrix_mapped(input):
    return view_matrix(input[0], input[1], input[2])


def view_matrix(translate : Tensor, angles : Tensor, scale : Tensor = None):
    M = eye(4)

    if translate is not None:
        T = convert_to_tensor([[1.0, 0.0, 0.0, translate[0]], [0.0, 1.0, 0.0, translate[1]], [0.0, 0.0, 1.0, translate[2]], [0.0, 0.0, 0.0, 1.0]], dtype = float32)
        #T = eye(4)
        #T[3, :] = translate[:3]
        M = linalg.matmul(M, T)

    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2])
        #R = concat([R, [[0.0, 0.0, 0.0]]], axis = 0)
        #R = concat([R, [[0.0], [0.0], [0.0], [1.0]]], axis = 1)
        M = linalg.matmul(M, R)

    if scale is not None:
        S = convert_to_tensor([[scale[0], 0.0, 0.0, 0.0], [0.0, scale[1], 0.0, 0.0], [0.0, 0.0, scale[2], 0.0], [0.0, 0.0, 0.0, 1.0]], dtype = float32)
        #S = eye(4)
        #S[0, 0] = scale[0]
        #S[1, 1] = scale[1]
        #S[2, 2] = scale[2]
        M = linalg.matmul(M, S)

    M /= M[3, 3]
    return M


def euler_matrix(ai, aj, ak):
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = convert_to_tensor([[cj * ck, sj * sc - cs, sj * cc + ss, 0.0], [cj * sk, sj * ss + cc, sj * cs - sc, 0.0], [-sj, cj * si, cj * ci, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return M


def RotateVectors(vectors, projection_matrices):
    '''
    Преобразование полученных векторов для соответствия ракурсу
    :param vectors: Векторы (размер BATCH_SIZE x ... x 3)
    :param projection_matrices: Матрицы вращения (размер BATCH_SIZE x 3 x 3)
    :return: Повёрнутые векторы (размер BATCH_SIZE x ... x 3)
    '''
    # Перестановка [0, 2, 1] означает, что вдоль первой оси, где индексируются экземпляры бэтча, мы ничего не трогаем
    rotated_vectors = rotate(point = vectors, matrix = expand_dims(projection_matrices, axis = 1))
    return rotated_vectors


@function(input_signature = (TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, 8)), ))
def CreateRotationMatrices(projection_data):
    '''
    Метод получения матрицы вращения из ответа нейросети
    :param projection_data: ответ нейросети размерности [batch_size, 8]
    :return: матрицы вращения размером [batch_size, 4, 3]
    '''
    batch_size = projection_data.shape[0]
    projection = transpose(reshape(projection_data, (batch_size, 4, 2)), perm = (0, 2, 1))
    row1_norm = linalg.norm(projection[:, 0, 0:3], axis = 1, keepdims = True)
    row2_norm = linalg.norm(projection[:, 1, 0:3], axis = 1, keepdims = True)
    row1 = projection[:, 0, 0:3] / (row1_norm + 1.0e-10)
    row2 = projection[:, 1, 0:3] / (row2_norm + 1.0e-10)
    row3 = linalg.cross(row1, row2)
    row1 = concat([reshape(row1, (batch_size, 1, 3)), zeros((batch_size, 1, 1))], axis = 2)
    row2 = concat([reshape(row2, (batch_size, 1, 3)), zeros((batch_size, 1, 1))], axis = 2)
    row3 = concat([reshape(row3, (batch_size, 1, 3)), zeros((batch_size, 1, 1))], axis = 2)
    return concat([row1, row2, row3], axis = 1)


@function(input_signature = (TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3)), TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, 3, 4))))
def RotatePointsWithMatrices(points, projection_matrices, ):
    '''
    Метод вращения формы лица матрицами
    :param points: Формы лица размера [batch_size, n_vertices, 3]
    :param projection_matrices: Матрицы проецирования размера [batch_size, 3, 4]
    :return:
    '''
    vertices = concat([points, ones(shape = (points.shape[0], points.shape[1], 1))], axis = 2)
    return transpose(matmul(projection_matrices, transpose(vertices, perm = (0, 2, 1))), perm = (0, 2, 1))