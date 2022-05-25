#from numba import cuda
from numpy import zeros, arange, full
from tensorflow import cast, gather, transpose, reshape, map_fn, boolean_mask, tile, logical_and, expand_dims, unstack
from tensorflow import math as tfmath, int32, RaggedTensor, RaggedTensorSpec, convert_to_tensor, unique_with_counts
from tensorflow import meshgrid, range as tfrange, TypeSpec, tuple as tftuple, dtypes, TensorSpec, stack, vectorized_map
from learning_const import IMAGE_SIZE, BATCH_SIZE
import time

import math


def DepthBufferTf2(vertices, triangles, triangle_visibility_mask, image_size = IMAGE_SIZE):
    '''
    Реализация буфера глубины (Z-buffer)
    :param vertices: Массив вершин (размер BATCH_SIZE x const.MESH_VERTICES x 3)
    :param triangles: Массив треугольников (размер 3 x const.MESH_TRIANGLES)
    :param triangle_visibility_mask: Маска видимости треугольников (размер BATCH_SIZE x const.MESH_TRIANGLES)
    :param image_size: размеры текстур
    :return: Карты сопоставления треугольников к пикселю (размер BATCH_SIZE x image_size[0] x image_size[1]),
    Буферы глубины (размер BATCH_SIZE x image_size[0] x image_size[1])
    '''
    batch_size = vertices.shape[0]
    vertices_amount = vertices.shape[1]
    triangles_amount = triangles.shape[1]

    triangle_maps = zeros(shape = [batch_size, image_size[0], image_size[1]])
    buffers = full([batch_size, image_size[0], image_size[1]], -math.inf)

    triangles = transpose(triangles)
    batch_vertices = unstack(vertices)
    batch_visibility_masks = unstack(triangle_visibility_mask)

    # Для каждого образца в бэтче
    for i in range(batch_size):
        # Отсекаем невидимые треугольники, чтобы не грузить ими видеокарту (лишние if-else нам ни к чему)
        instance_triangles = boolean_mask(tensor = triangles, mask = batch_visibility_masks[i])
        first_vertices = gather(params=batch_vertices[i],
                                indices=instance_triangles[:, 0])
        second_vertices = gather(params=batch_vertices[i],
                                 indices=instance_triangles[:, 1])
        third_vertices = gather(params=batch_vertices[i],
                                indices=instance_triangles[:, 2])

        # Теперь будем отсекать те треугольники, которые не подходят по текстурным координатам
        # (не попадут на изображение и потому обсчитывать их незачем)
        u_min = cast(tfmath.minimum(tfmath.minimum(first_vertices[:, 0], second_vertices[:, 0]),
                                    third_vertices[:, 0]),
                     dtype=int32)
        u_max = cast(tfmath.maximum(tfmath.maximum(first_vertices[:, 0], second_vertices[:, 0]),
                                    third_vertices[:, 0]),
                     dtype=int32) + 1

        v_min = cast(tfmath.minimum(tfmath.minimum(first_vertices[:, 1], second_vertices[:, 1]),
                                    third_vertices[:, 1]),
                     dtype=int32)
        v_max = cast(tfmath.maximum(tfmath.maximum(first_vertices[:, 1], second_vertices[:, 1]),
                                    third_vertices[:, 1]),
                     dtype=int32) + 1

        image_mask = logical_and(logical_and(u_min >= 0, u_max < 224), logical_and(v_min >= 0, v_max < 224))
        first_vertices = boolean_mask(tensor=first_vertices, mask=image_mask)
        second_vertices = boolean_mask(tensor=second_vertices, mask=image_mask)
        third_vertices = boolean_mask(tensor=third_vertices, mask=image_mask)
        u_min = boolean_mask(tensor=u_min, mask=image_mask)
        u_max = boolean_mask(tensor=u_max, mask=image_mask)
        v_min = boolean_mask(tensor=v_min, mask=image_mask)
        v_max = boolean_mask(tensor=v_max, mask=image_mask)

        # Теперь остались только подходящие треугольники
        # Для них нам нужно проверить все координаты в рамках той прямоугольной области,
        # которая задана в ((u_min, v_min), (u_max, v_max))
        def meshgridfunc(rect_tuple):
            u_range = tfrange(rect_tuple[0], rect_tuple[1])
            v_range = tfrange(rect_tuple[2], rect_tuple[3])
            grid_x, grid_y = meshgrid(u_range, v_range)
            return stack([grid_x, grid_y], axis = 0)

        elems = stack([u_min, u_max, v_min, v_max], axis = 1)
        grids = vectorized_map(fn = meshgridfunc, elems = elems)

        z_avg = (first_vertices[:, 2] + second_vertices[:, 2] + third_vertices[:, 2]) / 3

        # Вектора нам пригодятся для ... (для чего?)
        first_to_third_u = third_vertices[:, 0] - first_vertices[:, 0]
        first_to_third_v = third_vertices[:, 1] - first_vertices[:, 1]
        first_to_second_u = second_vertices[:, 0] - first_vertices[:, 0]
        first_to_second_v = second_vertices[:, 1] - first_vertices[:, 1]

        # Скалярные произведения нам пригодятся для ... (для чего?)
        dot_13_13 = first_to_third_u * first_to_third_u + first_to_third_v * first_to_third_v
        dot_13_12 = first_to_third_u * first_to_second_u + first_to_third_v * first_to_second_v
        dot_12_12 = first_to_second_u * first_to_second_u + first_to_second_v * first_to_second_v

        inverseDenominator = 1 / (dot_13_13 * dot_12_12 - dot_13_12 * dot_13_12)

        # Теперь вызываем CUDA-ядро для прошедших проверку
        buffer_device = cuda.device_array(shape = (image_size[0], image_size[1]))
        map_device = cuda.device_array(shape = (image_size[0], image_size[1]))
        DepthBufferCUDA2[32, 128](u_min.shape[0], u_min.numpy(), u_max.numpy(), v_min.numpy(), v_max.numpy(), z_avg.numpy(),
                                  first_vertices[:, 0].numpy(), first_vertices[:, 1].numpy(),
                                  first_to_third_u.numpy(), first_to_third_v.numpy(),
                                  first_to_second_u.numpy(), first_to_second_v.numpy(),
                                  buffer_device, map_device)

        triangle_maps[i] = buffer_device
        buffers[i] = map_device

    return convert_to_tensor(triangle_maps), convert_to_tensor(buffers)


#@cuda.jit
def DepthBufferCUDA2(length, u_mins, u_maxes, v_mins, v_maxes, z_avgs, first_vertices_u, first_vertices_v, first_to_third_u_array, first_to_third_v_array, first_to_second_u_array, first_to_second_v_array, out_buffer, out_mapping):
    blockSize = cuda.blockDim.x
    gridSize = cuda.gridDim.x

    blockId = cuda.blockIdx.x
    threadId = cuda.threadIdx.x

    for i in range(blockId * blockSize + threadId, length, gridSize * blockSize):
        first_to_third_u = first_to_third_u_array[i]
        first_to_third_v = first_to_third_v_array[i]
        first_to_second_u = first_to_second_u_array[i]
        first_to_second_v = first_to_second_v_array[i]

        dot_13_13 = first_to_third_u * first_to_third_u + first_to_third_v * first_to_third_v
        dot_13_12 = first_to_third_u * first_to_second_u + first_to_third_v * first_to_second_v
        dot_12_12 = first_to_second_u * first_to_second_u + first_to_second_v * first_to_second_v

        denominator = dot_13_13 * dot_12_12 - dot_13_12 * dot_13_12

        for u in range(u_mins[i], u_maxes[i] + 1):
            for v in range(v_mins[i], v_maxes[i] + 1):
                first_to_pixel_u = u - first_vertices_u[i]
                first_to_pixel_v = v - first_vertices_v[i]

                dot_13_1p = first_to_third_u * first_to_pixel_u + \
                        first_to_third_v * first_to_pixel_v
                dot_12_1p = first_to_second_u * first_to_pixel_u + \
                        first_to_second_v * first_to_pixel_v

                uu = (dot_12_12 * dot_13_1p - dot_13_12 * dot_12_1p) / denominator
                vv = (dot_13_13 * dot_12_1p - dot_13_12 * dot_13_1p) / denominator

                isValidForBuffer = uu + vv <= 1 \
                    if uu >= 0 and uu <= 1 and vv >= 0 and vv <= 1 \
                    else False

                isCloser = out_buffer[u, v] < z_avgs[i]

                if (isValidForBuffer and isCloser):
                    out_buffer[u, v] = z_avgs[i]
                    out_mapping[u, v] = i


def DepthBufferTf(vertices, triangles, triangle_visibility_mask, image_size = IMAGE_SIZE):
    '''
    Реализация буфера глубины (Z-buffer)
    :param vertices: Массив вершин (размер BATCH_SIZE x const.MESH_VERTICES x 3)
    :param triangles: Массив треугольников (размер 3 x const.MESH_TRIANGLES)
    :param triangle_visibility_mask: Маска видимости треугольников (размер BATCH_SIZE x const.MESH_TRIANGLES)
    :param image_size: размеры текстур
    :return: Карты сопоставления треугольников к пикселю (размер BATCH_SIZE x image_size[0] x image_size[1]),
    Буферы глубины (размер BATCH_SIZE x image_size[0] x image_size[1])
    '''

    batch_size = vertices.shape[0]
    vertices_amount = vertices.shape[1]
    triangles_amount = triangles.shape[1]

    # Отсекаем невидимые треугольники
    triangles = tile(input = expand_dims(transpose(triangles), axis = 0), multiples = (batch_size, 1, 1))
    # Подготаливаемся к заданию RaggedTensor
    results = [unique_with_counts(x = batch_elem) for batch_elem in triangle_visibility_mask]
    true_results = reshape(convert_to_tensor([boolean_mask(tensor = result.count, mask = result.y) for result in results]), shape = [-1])
    triangles = boolean_mask(tensor = triangles, mask = triangle_visibility_mask)
    triangles = RaggedTensor.from_row_lengths(values = triangles, row_lengths = true_results)
    '''def temp_fn(elem):
        print(f"This elem has {elem.points[0]} triangles")
        return elem

    temp = map_fn(fn = temp_fn, elems = triangles)'''


    # Получаем списки вершин треугольников (первые, вторые, третьи)
    #first_vertices = map_fn(fn = lambda instance_vertices : , elems = vertices, fn_output_signature = RaggedTensorSpec(points = [batch_size, None]))
    #second_vertices = map_fn()
    #third_vertices = map_fn()
    first_vertices = gather(params = vertices,
                               indices = triangles[0],
                               axis = 1)
    second_vertices = gather(params = vertices,
                                indices = triangles[1],
                                axis = 1)
    third_vertices = gather(params = vertices,
                               indices = triangles[2],
                               axis = 1)

    # Для каждого треугольника определяем диапазон пикселей, где мы можем претендовать на перезапись в буфер
    # Работаем сразу с тензорами
    # 0 - это U-координата,
    # 1 - это V-координата,
    # 2 - это Z-координата
    u_min = cast(tfmath.minimum(tfmath.minimum(first_vertices[:, :, 0], second_vertices[:, :, 0]),
                         third_vertices[:, :, 0]),
                 dtype = int32)
    u_max = cast(tfmath.maximum(tfmath.maximum(first_vertices[:, :, 0], second_vertices[:, :, 0]),
                         third_vertices[:, :, 0]),
                 dtype = int32) + 1

    v_min = cast(tfmath.minimum(tfmath.minimum(first_vertices[:, :, 1], second_vertices[:, :, 1]),
                         third_vertices[:, :, 1]),
                 dtype = int32)
    v_max = cast(tfmath.maximum(tfmath.maximum(first_vertices[:, :, 1], second_vertices[:, :, 1]),
                         third_vertices[:, :, 1]),
                 dtype = int32) + 1

    z_avg = (first_vertices[:, :, 2] + second_vertices[:, :, 2] + third_vertices[:, :, 2]) / 3

    image_mask = logical_and(logical_and(u_min >= 0, u_max < 224), logical_and(v_min >= 0, v_max < 224))
    first_vertices = boolean_mask(tensor = first_vertices, mask = image_mask)
    second_vertices = boolean_mask(tensor = second_vertices, mask = image_mask)
    third_vertices = boolean_mask(tensor = third_vertices, mask = image_mask)
    u_min = boolean_mask(tensor = u_min, mask = image_mask)
    u_max = boolean_mask(tensor = u_max, mask = image_mask)
    v_min = boolean_mask(tensor = v_min, mask = image_mask)
    v_max = boolean_mask(tensor = v_max, mask = image_mask)

    return

    # Вектора нам пригодятся для ... (для чего?)
    first_to_third_u =  third_vertices[:, :, 0] - first_vertices[:, :, 0]
    first_to_third_v = third_vertices[:, :, 1] - first_vertices[:, :, 1]

    first_to_second_u = second_vertices[:, :, 0] - first_vertices[:, :, 0]
    first_to_second_v = second_vertices[:, :, 1] - first_vertices[:, :, 1]

    # Скалярные произведения нам пригодятся для ... (для чего?)
    dot_13_13 = first_to_third_u * first_to_third_u + first_to_third_v * first_to_third_v
    dot_13_12 = first_to_third_u * first_to_second_u + first_to_third_v * first_to_second_v
    dot_12_12 = first_to_second_u * first_to_second_u + first_to_second_v * first_to_second_v

    inverseDenominator = 1 / (dot_13_13 * dot_12_12 - dot_13_12 * dot_13_12)

    # Дальше не знаю, как можно писать программу на языке тензоров,
    # так что пока будет простой цикл
    triangle_maps = zeros(shape = [vertices.shape[0], image_size[0], image_size[1]])
    buffers = full([vertices.shape[0], image_size[0], image_size[1]], -math.inf)

    # Для каждого образца в бэтче
    for i in range(batch_size):
        pass

    return triangle_maps, buffers


def DepthBuffer(vertices, triangles, triangle_visibility_mask, image_size = IMAGE_SIZE, debug_values = False):
    '''
    Применение буфера глубины
    :param vertices: Массив вершин (размер BATCH_SIZE x const.MESH_VERTICES x 3)
    :param triangles: Массив треугольников (размер 3 x const.MESH_TRIANGLES)
    :param triangle_visibility_mask: Маска видимости треугольников (размер BATCH_SIZE x const.MESH_TRIANGLES)
    :param image_size: размеры текстур
    :param debug_values: Выдавать ли в консоль информацию по значениям получаемых тензоров?
    :return: Карты сопоставления треугольников к пикселю (размер BATCH_SIZE x image_size[0] x image_size[1]),
    Буферы глубины (размер BATCH_SIZE x image_size[0] x image_size[1])
    '''
    batch_size = vertices.shape[0]
    vertices_amount = vertices.shape[1]
    triangles_amount = triangles.shape[1]

    # Для каждого образца в бэтче
    maps = []
    buffers = []


    for i in range (batch_size):
        triangles_flatten = cuda.to_device(reshape(transpose(triangles), [-1]).numpy())
        batch_vertices_flatten = cuda.to_device(reshape(vertices[i], [-1]).numpy())
        batch_visibility_mask_flatten = cuda.to_device(reshape(triangle_visibility_mask[i], [-1]).numpy())
        buffer_device = cuda.device_array(shape = (image_size[0], image_size[1]))
        map_device = cuda.device_array(shape = (image_size[0], image_size[1]))

        DepthBufferCuda[32, 128](batch_vertices_flatten,
                        triangles_flatten,
                        batch_visibility_mask_flatten,
                        vertices_amount,
                        triangles_amount,
                        buffer_device,
                        map_device)

        maps.append(map_device)
        buffers.append(buffer_device)

    if (debug_values):
        print("Отладочная информация по результату работы метода DepthBuffer (для одного образца).")
        print("ВХОДНЫЕ ДАННЫЕ"
              f"\nВершины: {vertices[0]};"
              f"\nТреугольники: {triangles};"
              "\n\nВЫХОДНЫЕ ДАННЫЕ"
              f"\nБуфер глубины: {buffers[0]};"
              f"\nКарта треугольников: {maps[0]}")

    return convert_to_tensor(maps), convert_to_tensor(buffers)


def DepthBufferCudaPreparing(vertices, triangles, triangle_visibility_mask, image_size = IMAGE_SIZE, debug_values = False):
    '''
    Применение буфера глубины
    :param vertices: Массив вершин (размер BATCH_SIZE x const.MESH_VERTICES x 3)
    :param triangles: Массив треугольников (размер 3 x const.MESH_TRIANGLES)
    :param triangle_visibility_mask: Маска видимости треугольников (размер BATCH_SIZE x const.MESH_TRIANGLES)
    :param image_size: размеры текстур
    :param debug_values: Выдавать ли в консоль информацию по значениям получаемых тензоров?
    :return: Карты сопоставления треугольников к пикселю (размер BATCH_SIZE x image_size[0] x image_size[1]),
    Буферы глубины (размер BATCH_SIZE x image_size[0] x image_size[1])
    '''
    pass


#@cuda.jit
def DepthBufferCuda(vertices, triangles, triangle_visibility_mask, vertices_amount, triangles_amount, out_buffer, out_mapping):
    '''
    Реализация Z-буфера через CUDA (применяется только для одного образца)
    :param vertices: Вершины одного бэтча (д. б. приведён в плоскую форму)
    :param triangles: Треугольники одного бэтча (д. б. приведён в плоскую форму)
    :param triangle_visibility_mask: Маска видимости треугольников одного образца
    :param vertices_amount: Число вершин
    :param triangles_amount: Число треугольников
    :param out_buffer: Выходной массив для значений буфера
    :param out_mapping: Выходной ассоциативный массив (какой треугольник к какому пикселю идёт)
    '''

    blockSize = cuda.blockDim.x
    gridSize = cuda.gridDim.x

    blockId = cuda.blockIdx.x
    threadId = cuda.threadIdx.x

    for i in range(blockId * blockSize + threadId, triangles_amount, gridSize * blockSize):
        if (triangle_visibility_mask[i] == False):
            continue

        first_vertex_index = triangles[i]
        second_vertex_index = triangles[i + vertices_amount]
        third_vertex_index = triangles[i + 2 * vertices_amount]

        # Эти координаты вершины будут также являться координатами
        # соответствующей точки изображения
        first_vertex_u = vertices[first_vertex_index]
        first_vertex_v = vertices[first_vertex_index + 1]
        second_vertex_u = vertices[second_vertex_index]
        second_vertex_v = vertices[second_vertex_index + 1]
        third_vertex_u = vertices[third_vertex_index]
        third_vertex_v = vertices[third_vertex_index + 1]

        u_min = int((min(first_vertex_u, second_vertex_u, third_vertex_u)))
        u_max = int((max(first_vertex_u, second_vertex_u, third_vertex_u))) + 1

        v_min = int((min(first_vertex_v, second_vertex_v, third_vertex_v)))
        v_max = int((max(first_vertex_v, second_vertex_v, third_vertex_v))) + 1

        # Если координаты не находятся в пределах сетки изображения,
        # то проведения операций бессмысленно
        if (u_min < 0 or v_min < 0 or
            u_max >= 224 or v_max >= 224):
            continue

        z_avg = (vertices[first_vertex_index + 2] +
                 vertices[second_vertex_index + 2] +
                 vertices[third_vertex_index + 2]) / 3

        first_to_third_u = third_vertex_u - first_vertex_u
        first_to_third_v = third_vertex_v - first_vertex_v
        first_to_second_u = second_vertex_u - first_vertex_u
        first_to_second_v = second_vertex_v - first_vertex_v

        dot_13_13 = first_to_third_u * first_to_third_u + \
                    first_to_third_v * first_to_third_v
        dot_13_12 = first_to_third_u * first_to_second_u + \
                    first_to_third_v * first_to_second_v
        dot_12_12 = first_to_second_u * first_to_second_u + \
                    first_to_second_v * first_to_second_v

        denominator = dot_13_13 * dot_12_12 - dot_13_12 * dot_13_12

        for u in range(u_min, u_max + 1):
            for v in range(v_min, v_max + 1):
                first_to_pixel_u = u - first_vertex_u
                first_to_pixel_v = v - first_vertex_v

                dot_13_1p = first_to_third_u * first_to_pixel_u + \
                        first_to_third_v * first_to_pixel_v

                dot_12_1p = first_to_second_u * first_to_pixel_u + \
                        first_to_second_v * first_to_pixel_v

                uu = (dot_12_12 * dot_13_1p - dot_13_12 * dot_12_1p) / denominator
                vv = (dot_13_13 * dot_12_1p - dot_13_12 * dot_13_1p) / denominator

                isValidForBuffer = uu + vv <= 1 \
                    if uu >= 0 and uu <= 1 and vv >= 0 and vv <= 1 \
                    else False

                isCloser = out_buffer[u, v] < z_avg

                if (isValidForBuffer and isCloser):
                    out_buffer[u, v] = z_avg
                    out_mapping[u, v] = i