import math as m

import numba
from PIL import Image
from tensorflow import cast, not_equal, reduce_sum, multiply, float32, float64, greater, less, reduce_max, reduce_min
from tensorflow import linalg, math, transpose, constant, int32, concat, map_fn, vectorized_map, TensorSpec, print as tfprint
from tensorflow import reduce_mean, split, floor, clip_by_value, zeros, add_n, gather, convert_to_tensor, device
from tensorflow import reshape, shape, tile, range as tfrange, stack, gather_nd, expand_dims, ones, identity
from tensorflow import experimental, numpy_function, py_function, function, TensorArray, TensorArraySpec
from tensorflow import compat, debugging, while_loop, cond, squeeze, boolean_mask, int64, meshgrid, bool as tfbool, cond
#from tensorflow.keras.utils import save_img
import numpy as np
import tensorflow._api.v2.experimental.numpy as tnp

from tools.TensorTools import RemapTensor, TransformTensor
from typing import Tuple

import learning_const

sqrt2 = m.sqrt(2.0) + 1.0e-12


def ComputeNormals(vertices, triangles, vertex_triangle_map, debug_shapes = False, debug_values = False):
    '''
    Получение векторов нормали к поверхности лица
    :param vertices: Массив вершин (размер batch_size x const.MESH_VERTICES x 3)
    :param triangles: Массив треугольников (размер 3 x const.MESH_TRIANGLES)
    :param vertex_triangle_map:
    Список принадлежности вершины к треугольникам
    (пока не понял необходимость; возможно, просто во избежание переборов
    (размер T x main.MESH_VERTICES (T=8: maximum number of triangle each vertices can belong to))
    :param debug_shapes: Выдавать ли в консоль информацию по формам получаемых тензоров?
    :param debug_values: Выдавать ли в консоль информацию по значениям получаемых тензоров?
    :return: (Нормали вершин (размер batch_size x const.MESH_VERTICES x 3), нормали треугольников  (размер batch_size x const.MESH_TRIANGLES x 3))
    '''
    # Подготавливаем данные:
    # 1. Разбиваем массив треугольников на три массива для первых, вторых и третьих вершин
    triangle_vertex_indices_0, \
    triangle_vertex_indices_1, \
    triangle_vertex_indices_2 = split(triangles, num_or_size_splits=3, axis=0)

    # 2. Получаем размерности
    # (по-хорошему мы их знаем из настроек в main,
    # но на всякий случай здесь берём гарантированно достоверные данные)
    batch_size = shape(vertices)[0]
    triangles_amount = shape(triangles)[1]
    vertices_amount = shape(vertices)[1]
    T = 8 #points(vertex_triangle_map)[0]  # У нас 8

    if (debug_shapes):
        print(f"Определены следующие исходные данные:"
              f"\nРазмер бэтча = {batch_size};"
              f"\nЧисло треугольников = {triangles_amount};"
              f"\nЧисло вершин = {vertices_amount};")

    # 3. Готовим индексные тензоры для получения карты нормалей треугольников
    # Для каждого экземпляра в бэтче...
    # (тензор одномерный)
    batch_indexing_tensor = range(0, batch_size)
    batch_indexing_tensor = reshape(batch_indexing_tensor, (batch_size, 1))
    # Реплицируем для всего бэтча и получаем список индексов
    # (в результате получается таблица, т. к. до этого мы превратили исходный тензор в вектор-столбец)
    batch_indexing_megatensor = tile(batch_indexing_tensor, (1, triangles_amount))

    # Здесь мы получаем индексы из данных о треугольниках и
    # также реплицируем для всего бэтча
    # Пояснение: базовая 3D-модель для всех экземпляров бэтча - общая,
    # в связи с чем связи между вершинами и треугольниками не изменяются
    # (изменяются только положения вершин)
    # Каждый из полученных тензоров представляет собой таблицу
    # (т. к. исходные тензоры есть вектора-строки, а мы реплицируем "вниз", а не "вправо")
    triangle_vertex_indices_0_batch = tile(triangle_vertex_indices_0, (batch_size, 1))
    triangle_vertex_indices_1_batch = tile(triangle_vertex_indices_1, (batch_size, 1))
    triangle_vertex_indices_2_batch = tile(triangle_vertex_indices_2, (batch_size, 1))

    if (debug_shapes):
        print(f"Размерность массива индексов вершин треугольника (первых, вторых, третьих): {triangle_vertex_indices_0_batch.shape}")

    # Теперь мы объединяем всю навороченную херню в мегатензоры,
    # которые будут индексировать весь бэтч целиком
    triangle_vertex_indices_0 = stack([batch_indexing_megatensor, triangle_vertex_indices_0_batch], 2)
    triangle_vertex_indices_1 = stack([batch_indexing_megatensor, triangle_vertex_indices_1_batch], 2)
    triangle_vertex_indices_2 = stack([batch_indexing_megatensor, triangle_vertex_indices_2_batch], 2)

    # 4. Получаем карту нормалей треугольников

    # Формируем три массива под каждую из вершин треугольника
    # Здесь идём поиск по таблице: triangle_vertex_indices_N задаёт индексы в vertices,
    # из которых берём координату
    # В результате у нас всё отсортировано по треугольникам и мы можем получить карту,
    # которая будет полностью соответствовать списку треугольников
    first_vertices = gather_nd(params = vertices, indices = triangle_vertex_indices_0)
    second_vertices = gather_nd(params = vertices, indices = triangle_vertex_indices_1)
    third_vertices = gather_nd(params = vertices, indices = triangle_vertex_indices_2)

    # Наконец, формируем саму карту нормалей
    triangle_normals = linalg.cross(second_vertices - first_vertices,
                                    third_vertices - first_vertices)
    triangle_normals = linalg.l2_normalize(triangle_normals, dim=2)

    if (debug_shapes):
        print(f"Размерность карт нормалей треугольников: {triangle_normals.shape}")

    if (vertex_triangle_map is None):
        return None, triangle_normals

    # Заодно формируем маску (маску кого?)
    mask = expand_dims(tile(
        expand_dims(
            not_equal(
                vertex_triangle_map,
                triangles.shape[1] - 1),
            2),
        [1, 1, 3]),
        0)
    mask = cast(mask, vertices.dtype)

    # 5. Получаем карту нормалей вершин
    # Преобразуем в вектор-строку
    vertex_triangle_map = reshape(vertex_triangle_map, shape=[1, -1])
    # Наш мегатензор теперь реплицирован не по числу треугольников,
    # а по числу этих самых ассоциаций с вершиной
    batch_indexing_megatensor = tile(batch_indexing_tensor, (1, T * vertices_amount))
    # Получаем из нашей вектор-строки таблицу индексирования
    vertex_indices_batch = tile(vertex_triangle_map, (batch_size, 1))
    indices = stack([batch_indexing_megatensor, vertex_indices_batch], 2)

    # Наконец, получаем саму карту нормалей вершин
    # Получаем нужные данные по индексам
    vertex_normals = gather_nd(triangle_normals, indices)
    # Возвращаем формат данных в исходный
    vertex_normals = reshape(vertex_normals, shape=[-1, T, vertices_amount, 3])
    # Нормаль вершины будет являться средним всех нормалей треугольников, в которые вершина входит (до восьми)
    vertex_normals = reduce_sum(multiply(vertex_normals, mask), axis=1)
    # Средним, а не суммой, потому что здесь нормализация!
    vertex_normals = linalg.l2_normalize(vertex_normals, axis=2)

    if (debug_shapes):
        print(f"Размерность карт нормалей вершин: {vertex_normals.shape}")

    # 6. На всякий случай делаем проверку на то, что нормали смотрят от модели, а не внутрь
    v = vertices - reduce_mean(vertices, 1, keepdims=True)
    s = reduce_sum(multiply(v, vertex_normals), 1, keepdims=True)

    count_s_greater_0 = math.count_nonzero(greater(s, 0), axis=0, keepdims=True)
    count_s_less_0 = math.count_nonzero(less(s, 0), axis=0, keepdims=True)

    sign = 2 * cast(greater(count_s_greater_0, count_s_less_0), float32) - 1
    vertex_normals = multiply(vertex_normals, sign)
    triangle_normals = multiply(triangle_normals, sign)

    if (debug_values):
        print("Отладочная информация по результату работы метода ComputeNormals (для одного образца)."
              "\n\nВХОДНЫЕ ДАННЫЕ:"
              f"\n\nВершины: \n{vertices[0]}"
              f"\n\nТреугольники: {triangles};"
              "\n\nВЫХОДНЫЕ ДАННЫЕ:"
              f"\n\nНормали вершин: {vertex_normals[0]};"
              f"\n\nНормали треугольников : {triangle_normals[0]}.")

    return vertex_normals, triangle_normals


def CalculateBarycentricRatios(first_point_u, second_point_u, third_point_u, first_point_v, second_point_v, third_point_v, sought_point_u, sought_point_v, debug_values = False):
    '''
    Преобразование в барицентрические координаты
    (все параметры должны иметь одинаковый размер, на сам размер ограничений нет)
    :param first_point_u: U-координата первых точек треугольника
    :param second_point_u: U-координата вторых точек треугольника
    :param third_point_u: U-координата третьих точек треугольника
    :param first_point_v: V-координата первых точек треугольника
    :param second_point_v: V-координата вторых точек треугольника
    :param third_point_v: V-координата третьих точек треугольника
    :param sought_point_u: U-координата точек, для к-рых ищем координаты
    :param sought_point_v: V-координата точек, для к-рых ищем координаты
    :return: Кортеж из трёх векторов коэффициентов разложения в барицентрические координаты
    '''
    v0_u = second_point_u - first_point_u
    v0_v = second_point_v - first_point_v

    v1_u = third_point_u - first_point_u
    v1_v = third_point_v - first_point_v

    v2_u = sought_point_u - first_point_u
    v2_v = sought_point_v - first_point_v

    invDenom = 1 / (v0_u * v1_v - v1_u * v0_v + 1e-6)
    c1 = cast((v2_u * v1_v - v1_u * v2_v) * invDenom, dtype = float32)
    c2 = cast((v0_u * v2_v - v2_u * v0_v) * invDenom, dtype = float32)
    c0 = 1 - c1 - c2

    if (debug_values):
        print("Отладочная информация по результату работы метода CalculateBarycentricRatios (для одного образца).")
        n_dims = len(first_point_u.shape)
        if n_dims == 1:
            print(f"UV-координаты первой точки: ({first_point_u[0]}, {first_point_v[0]});"
                  f"\nUV-координаты второй точки: ({second_point_u[0]}, {second_point_v[0]});"
                  f"\nUV-координаты третьей точки: ({third_point_u[0]}, {third_point_v[0]});"
                  f"\nUV-координаты исследуемой точки: ({sought_point_u[0]}, {sought_point_v[0]});"
                  f"\nРезультат разложения: c0 = {c0[0]}, c1 = {c1[0]}, c2 = {c2[0]}")
        elif n_dims == 2:
            print(f"UV-координаты первой точки: ({first_point_u[0, 0]}, {first_point_v[0, 0]});"
                  f"\nUV-координаты второй точки: ({second_point_u[0, 0]}, {second_point_v[0, 0]});"
                  f"\nUV-координаты третьей точки: ({third_point_u[0, 0]}, {third_point_v[0, 0]});"
                  f"\nUV-координаты исследуемой точки: ({sought_point_u[0, 0]}, {sought_point_v[0, 0]});"
                  f"\nРезультат разложения: c0 = {c0[0, 0]}, c1 = {c1[0, 0]}, c2 = {c2[0, 0]}")
        else:
            print("Неподдерживаемый формат данных")

    return c0, c1, c2


def bilinear_sampler(images_batch, x, y):
    '''
    Отображение с помощью билинейной интерполяции
    :param images_batch: Изображения
    :param x:
    :param y:
    :return:
    '''
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - images_batch: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    batch_size = shape(images_batch)[0]
    image_height = shape(images_batch)[1]
    image_width = shape(images_batch)[2]
    n_colors = shape(images_batch)[3]

    max_y = cast(image_height - 1, 'int32')
    max_x = cast(image_width - 1, 'int32')
    zero = zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = cast(x, 'float32')
    y = cast(y, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = cast(floor(x), 'int32')
    x1 = x0 + 1
    y0 = cast(floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate images_batch boundaries
    x0 = clip_by_value(x0, zero, max_x)
    x1 = clip_by_value(x1, zero, max_x)
    y0 = clip_by_value(y0, zero, max_y)
    y1 = clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(images_batch, x0, y0)
    Ib = get_pixel_value(images_batch, x0, y1)
    Ic = get_pixel_value(images_batch, x1, y0)
    Id = get_pixel_value(images_batch, x1, y1)

    # recast as float for delta calculation
    x0 = cast(x0, 'float32')
    x1 = cast(x1, 'float32')
    y0 = cast(y0, 'float32')
    y1 = cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = expand_dims(wa, axis=3)
    wb = expand_dims(wb, axis=3)
    wc = expand_dims(wc, axis=3)
    wd = expand_dims(wd, axis=3)

    # compute output
    out = add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


def get_pixel_value(images_batch, x, y):
    '''
    Получение значений пикселей по заданным координатам
    :param images_batch: Набор изображений (размер BATCH_SIZE x IMAGE_SIZE[0] x IMAGE_SIZE[1] x 3)
    :param x: Координаты на изображении по ширине
    :param y: Координаты на изображении по высоте
    :return:
    '''
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - images_batch: tensor of points (B, H, W, C)
    - x: flattened tensor of points (B*H*W, )
    - y: flattened tensor of points (B*H*W, )
    Returns
    -------
    - output: tensor of points (B, H, W, C)
    """
    shape = images_batch.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = range(0, batch_size)
    batch_idx = reshape(batch_idx, (batch_size, 1, 1))
    b = tile(batch_idx, (1, height, width))

    indices = stack([b, y, x], 3)

    return gather_nd(images_batch, indices)


def save_images(images, image_path, inverse=True):
    if (inverse):
        images = .5 * (images + 1.0)

    for image in images:
        #Image.fromarray(image).save(image_path)
        #save_img(image_path, image)
        pass

    # return imsave(images, size, image_path)


def XYZtoUV(vertices, image_size = learning_const.IMAGE_SIZE, u_scale_ratio = 1.0, v_scale_ratio = 1.0, u_translation_ratio = 0.0, v_translation_ratio = 0.0, normalized = False):
    '''
    Преобразование вершин в текстурное пространство
    :param vertices: Список вершин (размер N x 3, где второй столбец задаёт вертикальную ось)
    :param u_scale_ratio: Множитель масштабирования для U-координаты
    :param v_scale_ratio: Множитель масштабирования для V-координаты
    :param u_translation_ratio: Смещение для U-координаты
    :param v_translation_ratio: Смещение для V-координаты
    :return: Список координат (размер N x 2)
    '''
    length = vertices.shape[0]
    x = cast(vertices[:, 0], dtype = float32)
    y = cast(vertices[:, 1], dtype = float32)
    z = cast(vertices[:, 2], dtype = float32)

    u = RemapTensor(-y, new_lowest = 0.0, new_highest = 1.0 if normalized else image_size[0])
    v = RemapTensor(experimental.numpy.arctan(x / z), new_lowest = 0.0, new_highest = 1.0 if normalized else image_size[1])

    return transpose(TransformUV(u, v, u_scale_ratio, v_scale_ratio, u_translation_ratio, v_translation_ratio))


def TransformUV(u, v, u_scale_ratio = 1.0, v_scale_ratio = 1.0, u_translation_ratio = 0.0, v_translation_ratio = 0.0):
    length = u.shape[0]
    return stack([(tile([u_scale_ratio], multiples = (length, )) * u + tile([u_translation_ratio], multiples = (length, ))),
                            (tile([v_scale_ratio], multiples = (length, )) * v + tile([v_translation_ratio], multiples = (length, )))])


def UVtoXYZ(vertices, uv_coords, texture):
    '''
    Преобразование текстурного пространства в вершины
    :param vertices: Координаты вершин
    :param uv_coords: UV-координаты вершин
    :param texture: Текстура
    :return: Цвета вершин
    '''
    length = vertices.shape[0]
    x = cast(vertices[:, 0], dtype=float32)
    y = cast(vertices[:, 1], dtype=float32)
    z = cast(vertices[:, 2], dtype=float32)

    ro = math.sqrt(x ** 2 + z ** 2)
    phi = uv_coords[:, 1]
    _z = uv_coords[:, 0]

    texture_x_coords = cast(_z * texture.shape[0], dtype = int32)
    texture_y_coords = cast(phi * texture.shape[1], dtype = int32)
    #xmax = reduce_max(texture_x_coords)
    #xmin = reduce_min(texture_x_coords)
    #ymax = reduce_max(texture_y_coords)
    #ymin = reduce_min(texture_y_coords)
    texture_coords = stack([texture_x_coords, texture_y_coords], axis = 1)

    vertex_colors = gather_nd(params = texture, indices = (texture_coords))
    vertex_colors = concat([vertex_colors, reshape(convert_to_tensor([1.0] * vertex_colors.shape[0]), shape = (-1, 1))], axis = 1)#map_fn(fn = lambda element : cast([element[0], element[1], element[2], 255.0], dtype = float32), elems = vertex_colors, fn_output_signature = TensorSpec(points = [4]))

    return vertex_colors / 255.0


def GetVertexUVColors(uv, normalize = True):
    length = len(uv)
    uv_tensor = cast(uv, dtype = float32)

    if (normalize):
        u = RemapTensor(uv_tensor[:, 0], new_lowest = 0.0, new_highest = 1.0)
        v = RemapTensor(uv_tensor[:, 1], new_lowest = 0.0, new_highest = 1.0)
    else:
        u = uv_tensor[:, 0]
        v = uv_tensor[:, 1]


    result = transpose(stack([v, u, zeros(shape=(length)), ones(shape=(length))]))

    return transpose(stack([v, u])), result.numpy()


def GetVertexCoordColors(vertices, normalize = True):
    length = len(vertices)
    vertices_tensor = cast(vertices, dtype=float32)

    if (normalize):
        x = RemapTensor(vertices_tensor[:, 0], new_lowest=0.0, new_highest=1.0)
        y = RemapTensor(vertices_tensor[:, 1], new_lowest=0.0, new_highest=1.0)
        z = RemapTensor(vertices_tensor[:, 2], new_lowest=0.0, new_highest=1.0)
    else:
        x = vertices_tensor[:, 0]
        y = vertices_tensor[:, 1]
        z = vertices_tensor[:, 2]

    result = transpose(stack([x, y, z, ones(shape=(length))]))

    return result.numpy()


def GetVertexDepthColors(vertices_z, normalize = True):
    z_tensor = cast(vertices_z, dtype = float32)
    length = len(vertices_z)
    depth_min = tile(math.reduce_min(z_tensor, axis = 0, keepdims = True), multiples = [length])
    depth_max = tile(math.reduce_max(z_tensor, axis = 0, keepdims = True), multiples = [length])
    depth = ((z_tensor - depth_min) * (ones(shape = [length]) - zeros(shape = [length])) / (depth_max - depth_min)) if normalize else z_tensor
    return transpose(([depth, depth, depth, ones(shape=[length])])).numpy()


@function
def UnwarpToImage(uv, colors, image_size = (192, 224)):
    '''
    Преобразование цветов модели в UV-развёртку (для одного экземпляра)
    :param uv: Текстурные координаты вершин
    :param colors: Цвета вершин
    :param image_size: Размер развёртки
    :return:
    '''
    # Масштабируем UV-координаты до размеров изображения
    #uv = concat([RemapTensor(uv[:, 0], 0.0, float(image_size[1])), RemapTensor(uv[:, 1], 0.0, float(image_size[0]))], axis = 1)


    return UnwarpSingleImageAsTensor(uv, colors, image_size)
    #return numpy_function(func = UnwarpSingleImageAsNumpy, inp = [uv, colors, image_size], Tout = float32)


@function()
def UnwarpToImageBatch(uv, colors, batch_size, image_size = (192, 224)):
    '''
    Преобразование цветов модели в UV-развёртку (для бэтча)
    :param uv: Текстурные координаты вершин
    :param colors: Цвета вершин
    :param image_size: Размер развёртки
    :return:
    '''
    '''tensor_arr = TensorArray(size = batch_size, element_shape = (image_size[0], image_size[1], 3), dtype = float32)
    #size_tensor = convert_to_tensor(image_size, dtype = int32)
    size_array = np.array(image_size, dtype = np.int32)
    for i in range(batch_size):
        tensor_arr = tensor_arr.write(i, UnwarpSingleImageAsTensor(uv[i], colors[i]))#zeros(points = (const.IMAGE_SIZE[0], const.IMAGE_SIZE[1], 3), dtype = float32))

    return tensor_arr.stack()'''
    return map_fn(fn = lambda input: UnwarpSingleImageAsTensor(input[0], input[1]), elems = (uv, colors), fn_output_signature = TensorSpec(shape = (image_size[0], image_size[1], 3)))
    #return vectorized_map(fn = lambda input: UnwarpSingleImageAsTensor(input[0], input[1]), elems = (uv, colors))



@function(jit_compile = True)
def UnwarpSingleImageAsTfNumpy(uv, colors, image_size):
    uv = tnp.asarray(uv, tnp.float32)
    colors = tnp.asarray(colors, tnp.float32)
    u0 = tnp.asarray(uv[:, 0], dtype = tnp.int32)
    u1 = u0 + 1
    v0 = tnp.asarray(uv[:, 1], dtype = tnp.int32)
    v1 = v0 + 1

    result = np.zeros(shape=[image_size[0], image_size[1], 3])

    for i in tnp.arange(image_size[0]):
        u_matching_mask = tnp.logical_or(u0 == i, u1 == i)
        if (tnp.equal(tnp.any(u_matching_mask), False)):
            continue

        for j in tnp.arange(image_size[1]):
            uv_matching_mask = tnp.logical_and(u_matching_mask, tnp.logical_or(v0 == j, v1 == j))
            if (tnp.equal(tnp.any(uv_matching_mask), False)):
                continue

            # Оставляем только те вершины, которые влияют на пиксель
            matching_vertices_uv = uv[uv_matching_mask]#boolean_mask(tensor=uv, mask=uv_matching_mask)
            matching_vertices_colors = colors[uv_matching_mask]#boolean_mask(tensor=colors, mask=uv_matching_mask)

            # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
            weights = 1.0 - tnp.sqrt(tnp.add(tnp.power(matching_vertices_uv[:, 0] - float(i), 2), tnp.power(matching_vertices_uv[:, 1] - float(j), 2))) / sqrt2

            # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
            weights = weights / tnp.sum(weights, axis = 0)

            weighted_colors = tnp.expand_dims(tnp.asarray(weights, tnp.float32), axis = -1) * tnp.asarray(matching_vertices_colors, tnp.float32)

            # Определяем итоговый цвет как скалярное произведение нормализованных весов и исходных цветов
            result[i, j] = tnp.sum(weighted_colors, axis = 0)

    return convert_to_tensor(result, float32)



@function(input_signature = (TensorSpec(dtype = float32, shape = (learning_const.MESH_VERTICES, 2)), TensorSpec(dtype = float32, shape = (learning_const.MESH_VERTICES, 3))))
def UnwarpSingleImageAsTensor(uv, colors):
    image_size = (192, 224)
    #print(uv)
    uv = cast(uv, dtype = float32)
    colors = cast(colors, dtype = float32)
    u0 = cast(uv[:, 0], dtype=int32)
    u1 = u0 + 1
    v0 = cast(uv[:, 1], dtype=int32)
    v1 = v0 + 1

    #result = zeros(points=[image_size[0], image_size[1], 3])

    row_array = TensorArray(dtype = float32, size = image_size[0], element_shape = (image_size[1], 3))
    i_range = convert_to_tensor(range(image_size[0]))
    for i in i_range:
        u_matching_mask = math.logical_or(u0 == i, u1 == i)
        if (math.equal(math.reduce_any(u_matching_mask), False)):
            continue

        #print(u_matching_mask)
        column_array = TensorArray(dtype = float32, size = image_size[1], element_shape = (3,))
        j_range = convert_to_tensor(range(image_size[1]))
        for j in j_range:
            uv_matching_mask = math.logical_and(u_matching_mask, math.logical_or(v0 == j, v1 == j))
            if (math.equal(math.reduce_any(uv_matching_mask), False)):
                continue

            # Оставляем только те вершины, которые влияют на пиксель
            #print(uv_matching_mask)
            matching_vertices_uv = uv[uv_matching_mask]  # boolean_mask(tensor=uv, mask=uv_matching_mask)
            matching_vertices_colors = colors[uv_matching_mask]  # boolean_mask(tensor=colors, mask=uv_matching_mask)

            # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
            weights = 1.0 - math.sqrt(math.add(math.pow(matching_vertices_uv[:, 0] - cast(i, float32), 2),
                                               math.pow(matching_vertices_uv[:, 1] - cast(j, float32), 2))) / sqrt2
            # weights = 1.0 - np.multiply(matching_vertices_uv[:, 0] - float(i), matching_vertices_uv[:, 1] - float(j))

            # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
            weights = weights / math.reduce_sum(weights, axis=0)

            weighted_colors = expand_dims(cast(weights, float32), axis=-1) * cast(matching_vertices_colors, float32)
            sum = math.reduce_sum(weighted_colors, axis=0)
            # Определяем итоговый цвет как скалярное произведение нормализованных весов и исходных цветов
            column_array = column_array.write(index = j, value = sum)

        row_array = row_array.write(index = i, value = column_array.stack())

    print(f"handle: {row_array.handle}; flow: {row_array.flow}; el_shape: {row_array.element_shape}")
    return cast(row_array.stack(), float32)


@function(input_signature = (TensorSpec(dtype = float64, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 2)), TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3))))
def UnwarpImageBatchAsTensor(uv, colors):
    #image_size = (192, 224)
    n_instances = learning_const.BATCH_SIZE#uv.points[0]
    n_rows = learning_const.TEXTURE_SIZE[0]
    n_columns = learning_const.TEXTURE_SIZE[1]
    batch_array = TensorArray(dtype = float32, dynamic_size = False, size = n_instances, element_shape = (n_rows, n_columns, 3))
    for m in tfrange(n_instances, dtype = int64):
        instance_uv = cast(uv[m], dtype = float32)
        instance_colors = cast(colors[m], dtype = float32)
        u0 = cast(instance_uv[:, 0], dtype=int64)
        #u1 = u0 + 1
        v0 = cast(instance_uv[:, 1], dtype=int64)
        #v1 = v0 + 1

        #result = zeros(points=[image_size[0], image_size[1], 3])
        row_array = TensorArray(dtype = float32, dynamic_size = False, size = n_rows, element_shape = (n_columns, 3))

        for i in tfrange(n_rows, dtype = int64):
            u_matching_mask = math.logical_or(u0 == i, u0 + 1 == i)
            if (math.equal(math.reduce_any(u_matching_mask), False)):
                continue

            column_array = TensorArray(dtype = float32, dynamic_size = False, size = n_columns, element_shape = (3,))

            for j in tfrange(n_columns, dtype = int64):
                uv_matching_mask = math.logical_and(u_matching_mask, math.logical_or(v0 == j, v0 + 1 == j))
                if (math.equal(math.reduce_any(uv_matching_mask), False)):
                    continue

                # Оставляем только те вершины, которые влияют на пиксель
                matching_vertices_uv = instance_uv[uv_matching_mask]  # boolean_mask(tensor=uv, mask=uv_matching_mask)
                matching_vertices_colors = instance_colors[uv_matching_mask]  # boolean_mask(tensor=colors, mask=uv_matching_mask)

                # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
                weights = 1.0 - math.sqrt(math.add(math.pow(matching_vertices_uv[:, 0] - float(i), 2),
                                                   math.pow(matching_vertices_uv[:, 1] - float(j), 2))) / sqrt2
                # weights = 1.0 - np.multiply(matching_vertices_uv[:, 0] - float(i), matching_vertices_uv[:, 1] - float(j))

                # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
                #weights = weights / math.reduce_sum(weights, axis=0)

                weighted_colors = expand_dims(cast(weights / math.reduce_sum(weights, axis=0), float32), axis=-1) * cast(matching_vertices_colors, float32)
                #sum = math.reduce_sum(weighted_colors, axis=0)
                # Определяем итоговый цвет как скалярное произведение нормализованных весов и исходных цветов
                column_array = column_array.write(index = cast(j, int32), value = math.reduce_sum(weighted_colors, axis=0))

            row_array = row_array.write(index = cast(i, int32), value = column_array.stack())

        batch_array = batch_array.write(index = cast(m, int32), value = row_array.stack())

    return cast(batch_array.stack(), float32)


@function(input_signature = (TensorSpec(dtype = float64, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 2)), TensorSpec(dtype = float32, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3))))
def UnwarpImageBatchAsTensor2(uv, colors):
    #image_size = (192, 224)
    n_instances = learning_const.BATCH_SIZE#uv.points[0]

    @function
    def BatchProcessor(inputs):
        instance_uv, instance_colors = inputs
        instance_uv = cast(instance_uv, dtype = float32)
        instance_colors = cast(instance_colors, dtype = float32)
        u0 = cast(instance_uv[:, 0], dtype=int64)
        #u1 = u0 + 1
        v0 = cast(instance_uv[:, 1], dtype=int64)
        #v1 = v0 + 1

        #result = zeros(points=[image_size[0], image_size[1], 3])
        n_rows = learning_const.TEXTURE_SIZE[0]
        n_columns = learning_const.TEXTURE_SIZE[1]
        row_array = TensorArray(dtype = float32, dynamic_size = False, size = n_rows, element_shape = (n_columns, 3))

        for i in tfrange(n_rows, dtype = int64):
            u_matching_mask = math.logical_or(u0 == i, u0 + 1 == i)
            if (math.equal(math.reduce_any(u_matching_mask), False)):
                continue

            column_array = TensorArray(dtype = float32, dynamic_size = False, size = n_columns, element_shape = (3,))

            for j in tfrange(n_columns, dtype = int64):
                uv_matching_mask = math.logical_and(u_matching_mask, math.logical_or(v0 == j, v0 + 1 == j))
                if (math.equal(math.reduce_any(uv_matching_mask), False)):
                    continue

                # Оставляем только те вершины, которые влияют на пиксель
                matching_vertices_uv = instance_uv[uv_matching_mask]  # boolean_mask(tensor=uv, mask=uv_matching_mask)
                matching_vertices_colors = instance_colors[uv_matching_mask]  # boolean_mask(tensor=colors, mask=uv_matching_mask)

                # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
                weights = 1.0 - math.sqrt(math.add(math.pow(matching_vertices_uv[:, 0] - float(i), 2),
                                                   math.pow(matching_vertices_uv[:, 1] - float(j), 2))) / sqrt2

                # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
                #weights = weights / math.reduce_sum(weights, axis=0)

                weighted_colors = expand_dims(cast(weights / math.reduce_sum(weights, axis=0), float32), axis=-1) * cast(matching_vertices_colors, float32)
                #sum = math.reduce_sum(weighted_colors, axis=0)
                # Определяем итоговый цвет как скалярное произведение нормализованных весов и исходных цветов
                column_array = column_array.write(index = cast(j, int32), value = math.reduce_sum(weighted_colors, axis=0))

            #cond(math.equal(math.reduce_any(u_matching_mask), False), zeros(points = (n_columns, 3)))
            row_array = row_array.write(index = cast(i, int32), value = column_array.stack())
        return row_array.stack()

    result = map_fn(fn = BatchProcessor, elems = (uv, colors), fn_output_signature = TensorSpec(shape = (learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)))

    return cast(result, float32)



def UnwarpSingleImage(uv, colors, image_size):
    u0 = cast(uv[:, 0], dtype = int32)
    u1 = u0 + 1
    v0 = cast(uv[:, 1], dtype = int32)
    v1 = v0 + 1

    result = zeros(shape=[image_size[0], image_size[1], 3])

    for i in tfrange(image_size[0]):
        u_matching_mask = math.logical_or(u0 == i, u1 == i)
        if (math.equal(math.reduce_any(u_matching_mask), False)):
            continue

        for j in tfrange(image_size[1]):
            uv_matching_mask = math.logical_and(u_matching_mask, math.logical_or(v0 == j, v1 == j))
            if (math.equal(math.reduce_any(uv_matching_mask), False)):
                continue

            # Оставляем только те вершины, которые влияют на пиксель
            matching_vertices_uv = uv[uv_matching_mask]#boolean_mask(tensor=uv, mask=uv_matching_mask)
            matching_vertices_colors = colors[uv_matching_mask]#boolean_mask(tensor=colors, mask=uv_matching_mask)

            # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
            weights = 1.0 - math.sqrt(math.add(math.pow(matching_vertices_uv[:, 0] - float(i), 2), math.pow(matching_vertices_uv[:, 1] - float(j), 2))) / sqrt2
            #weights = 1.0 - np.multiply(matching_vertices_uv[:, 0] - float(i), matching_vertices_uv[:, 1] - float(j))

            # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
            weights = weights / math.reduce_sum(weights, axis = 0)

            weighted_colors = expand_dims(cast(weights, float32), axis = -1) * cast(matching_vertices_colors, float32)

            # Определяем итоговый цвет как скалярное произведение нормализованных весов и исходных цветов
            result[i, j] = math.reduce_sum(weighted_colors, axis = 0)

    return cast(result, float32)


@function
def UnwarpSingleImageAsTensorLoops(uv, colors, image_size):
    uv = cast(uv, dtype = float32)
    colors = cast(colors, dtype = float32)
    u0 = cast(uv[:, 0], dtype=int32)
    u1 = u0 + 1
    v0 = cast(uv[:, 1], dtype=int32)
    v1 = v0 + 1

    #result = zeros(points=[image_size[0], image_size[1], 3])

    i = constant(0)
    i_loop_cond = lambda i: math.less(i, image_size[0])
    row_array = TensorArray(dtype = float32, size = image_size[0], dynamic_size = True, element_shape = (image_size[1], 3))

    def RowLoopBody(i):
        u_matching_mask = math.logical_or(u0 == i, u1 == i)
        column_array = TensorArray(dtype = float32, size = image_size[1], dynamic_size = True, element_shape = (3,))
        if (math.equal(math.reduce_any(u_matching_mask), False)):
            row_array.write(i, zeros(shape = (image_size[1], 3), dtype = float32))#.mark_used()
            return (i + 1,)

        j = constant(0)
        j_loop_cond = lambda j: math.less(j, image_size[1])

        def ColumnLoopBody(j):
            uv_matching_mask = math.logical_and(u_matching_mask, math.logical_or(v0 == j, v1 == j))
            if (math.equal(math.reduce_any(uv_matching_mask), False)):
                column_array.write(j, zeros(shape = (3, )))#.mark_used()
                return (j + 1,)


            # Оставляем только те вершины, которые влияют на пиксель
            matching_vertices_uv = uv[uv_matching_mask]  # boolean_mask(tensor=uv, mask=uv_matching_mask)
            matching_vertices_colors = colors[uv_matching_mask]  # boolean_mask(tensor=colors, mask=uv_matching_mask)

            # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
            weights = 1.0 - math.sqrt(math.add(math.pow(matching_vertices_uv[:, 0] - float(i), 2),
                                               math.pow(matching_vertices_uv[:, 1] - float(j), 2))) / sqrt2
            # weights = 1.0 - np.multiply(matching_vertices_uv[:, 0] - float(i), matching_vertices_uv[:, 1] - float(j))

            # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
            weights = weights / math.reduce_sum(weights, axis=0)

            weighted_colors = expand_dims(cast(weights, float32), axis=-1) * cast(matching_vertices_colors, float32)
            sum = math.reduce_sum(weighted_colors, axis=0)
            # Определяем итоговый цвет как скалярное произведение нормализованных весов и исходных цветов
            column_array.write(j, sum)#.mark_used()
            return (j + 1, )

        j_loop_result = while_loop(cond = j_loop_cond, body = ColumnLoopBody, loop_vars = [j], parallel_iterations = 32, maximum_iterations = image_size[1])
        row_array.write(index = i, value = column_array.stack())#.mark_used()
        return (i + 1, )

    i_loop_result = while_loop(cond = i_loop_cond, body = RowLoopBody, loop_vars = [i], parallel_iterations = 32, maximum_iterations = image_size[0])

    return cast(row_array.stack(), float32)


@function()
def UnwarpSingleImageFlatten(uv, colors):
    image_size = learning_const.TEXTURE_SIZE

    uv = cast(uv, dtype=float32)
    colors = cast(colors, dtype=float32)
    u0 = cast(uv[:, 0], dtype=int32)
    v0 = cast(uv[:, 1], dtype=int32)

    image_width = image_size[0]
    image_height = image_size[1]

    range1 = tile(tfrange(image_width), multiples = (image_height, ))
    range2 = tile(expand_dims(tfrange(image_height), axis = 1), multiples = (1, image_width))
    range2 = reshape(range2, (image_height * image_width, ))


    def LoopBody(inputs):
        i, j = inputs
        uv_matching_mask = math.logical_and(math.logical_or(u0 == i, u0 + 1 == i), math.logical_or(v0 == j, v0 + 1 == j))
        '''if (math.equal(math.reduce_any(uv_matching_mask), False)):
            return zeros(points = (3, ))'''

        def CalculateColor(uv_mask):
            matching_vertices_uv = boolean_mask(tensor=uv, mask=uv_mask)
            matching_vertices_colors = boolean_mask(tensor=colors, mask=uv_mask)

            # Определяем вес каждой вершины (влияние на пиксель зависит от удалённости)
            weights = 1.0 - math.sqrt(math.add(math.pow(matching_vertices_uv[:, 0] - cast(i, float32), 2),
                                               math.pow(matching_vertices_uv[:, 1] - cast(j, float32), 2))) / sqrt2

            # Нормализуем веса, поскольку нам необходимо, чтобы в сумме они давали единицу
            weights = weights / math.reduce_sum(weights, axis=0)

            weighted_colors = expand_dims(cast(weights, float32), axis=-1) * cast(matching_vertices_colors, float32)
            print(f":::: {weighted_colors}")
            return math.reduce_sum(weighted_colors, axis=0)

        result = cond(math.equal(math.reduce_any(uv_matching_mask), False), lambda: zeros(shape = (3, )), lambda: CalculateColor(identity(uv_matching_mask)))
        print(f"({i}, {j}), result = {result}")
        return result

    result = map_fn(fn = LoopBody, elems = (range1, range2), fn_output_signature = TensorSpec(shape = (3, )), parallel_iterations = 5376, swap_memory = True)
    return reshape(result, [image_width, image_height, 3])




'''#@function
def UnwarpSingleImageAsTensorOps(uv, colors, image_size):
    uv = cast(uv, dtype = float32)
    colors = cast(colors, dtype = float32)
    u0 = cast(uv[:, 0], dtype=int32)
    #u1 = u0 + 1
    v0 = cast(uv[:, 1], dtype=int32)
    #v1 = v0 + 1

    # Для начала применим маску для отсева лишних вершин
    image_mask = math.logical_and(math.logical_and(u0 >= 0, u0 + 1 < image_size[0]), math.logical_and(v0 >= 0, v0 + 1 < image_size[1]))
    u0 = boolean_mask(tensor = u0, mask = image_mask)
    v0 = boolean_mask(tensor = v0, mask = image_mask)
    image_colors = boolean_mask(tensor = colors, mask = image_mask)

    # Осталось только видимое
    # Дальше нам надо перейти к списку вершин, относящихся к каждой координате U или V
    grid = meshgrid(tfrange(0, image_size[0]), tfrange(0, image_size[1]))
    tiled_uv = tile(expand_dims(expand_dims(stack([u0, v0], axis = 0), axis = 0), axis = 0), [image_size[0], image_size[1], 1, 1])
    tiled_grid = tile(expand_dims(expand_dims(grid, axis = -1), axis = -1), [1, 1, const.MESH_VERTICES, 2])
    grid_colors = concat([tiled_grid, ])


    return None'''

