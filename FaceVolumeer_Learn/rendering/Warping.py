from numpy import meshgrid, linspace, float32 as npfloat32
from tensorflow import reshape, int32, not_equal, tile, expand_dims, gather_nd
from tensorflow import transpose, split, greater, squeeze, cast

from rendering.RenderingTools import CalculateBarycentricRatios, bilinear_sampler
from rendering.DepthBuffer import DepthBufferTf2
import time


class Warping:
    def __init__(self):
        pass

    @classmethod
    def Warp(cls, base_model_triangle_data, base_model_vertex_to_uv_map_data, unwarped_textures, vertex_data, triangle_normals, output_size):
        '''

        :param base_model_triangle_data: Треугольники базовой модели(размер 3 x MESH_TRIANGLES)
        :param base_model_vertex_to_uv_map_data: Карта соответствия вершинных и текстурных координат (размер 2 x MESH_VERTICES)
        :param unwarped_textures: Текстуры, полученные после работы расшифровщика и получения карты теней
        (размер BATCH_SIZE x TEXTURE_SIZE[0] x TEXTURE_SIZE[1] x 3)
        :param vertex_data: Вершины (размер BATCH_SIZE x const.MESH_VERTICES x 3)
        :param triangle_normals: Нормали треугольников (размер BATCH_SIZE x const.MESH_TRIANGLES x 3)
        :param output_size: Размерность итоговой текстуры (текстура будет квадратной)
        :return: Текстуры, преобразованные в изображения (размер BATCH_SIZE x IMAGE_SIZE x IMAGE_SIZE x 3)
        '''
        batch_size = vertex_data.shape[0]

        vertex_u_mapping = tile(expand_dims(base_model_vertex_to_uv_map_data.data_u, axis = 0), [batch_size, 1])
        vertex_v_mapping = tile(expand_dims(base_model_vertex_to_uv_map_data.data_v, axis = 0), [batch_size, 1])

        vertices_u, vertices_v, vertices_z = split(axis=2, num_or_size_splits=3, value = vertex_data)

        vertices_u = squeeze(vertices_u - 1, axis = 2)
        vertices_v = squeeze(output_size - vertices_v, axis = 2)
        #vertices = transpose(concat(axis=1,
        #                           values=[vertices_v, vertices_u, vertices_z]))

        # Для удобства храним данные о первых, вторых и третьих вершинах треугольника раздельно
        triangle_vertex_indices_0 = tile(expand_dims(transpose(base_model_triangle_data[0]), axis = 0), [batch_size, 1])
        triangle_vertex_indices_1 = tile(expand_dims(transpose(base_model_triangle_data[1]), axis = 0), [batch_size, 1])
        triangle_vertex_indices_2 = tile(expand_dims(transpose(base_model_triangle_data[2]), axis = 0), [batch_size, 1])

        # Формируем маску видимости
        # Если нормаль треугольника по Z направлена от нас, то мы его не видим
        triangle_visibility_masks = greater(triangle_normals[:, :, 2], 0)

        # Применяем буфер глубины
        t0 = time.time()
        triangle_mapping, buffers = DepthBufferTf2(vertex_data,
                                                base_model_triangle_data,
                                                triangle_visibility_masks)

        print(f"Буфер работал {time.time() - t0} секунд")

        buffers_as_mask = cast(not_equal(buffers, 0), dtype = npfloat32)
        triangle_mapping_as_list = expand_dims(cast(reshape(triangle_mapping, [triangle_mapping.shape[0], -1]), int32), axis = 2)

        # Вычисляем барицентрические коэффициенты
        vertex_indices_0 = expand_dims(gather_nd(triangle_vertex_indices_0, triangle_mapping_as_list, batch_dims = 1), axis = 2)
        vertex_indices_1 = expand_dims(gather_nd(triangle_vertex_indices_1, triangle_mapping_as_list, batch_dims = 1), axis = 2)
        vertex_indices_2 = expand_dims(gather_nd(triangle_vertex_indices_2, triangle_mapping_as_list, batch_dims = 1), axis = 2)
        #
        pixel0_uu = gather_nd(vertices_u, vertex_indices_0, batch_dims = 1)
        pixel1_uu = gather_nd(vertices_u, vertex_indices_1, batch_dims = 1)
        pixel2_uu = gather_nd(vertices_u, vertex_indices_2, batch_dims = 1)
        pixel0_vv = gather_nd(vertices_v, vertex_indices_0, batch_dims = 1)
        pixel1_vv = gather_nd(vertices_v, vertex_indices_1, batch_dims = 1)
        pixel2_vv = gather_nd(vertices_v, vertex_indices_2, batch_dims = 1)

        u, v = meshgrid(linspace(0, output_size - 1, output_size, dtype = npfloat32),
                        linspace(0, output_size - 1, output_size, dtype = npfloat32))

        u = tile(reshape(u, [1, -1]), [batch_size, 1])
        v = tile(reshape(v, [1, -1]), [batch_size, 1])
        c0, c1, c2 = CalculateBarycentricRatios(pixel0_uu, pixel1_uu, pixel2_uu,
                                                pixel0_vv, pixel1_vv, pixel2_vv,
                                                u, v)

        #
        pixel_values_u_0 = gather_nd(vertex_u_mapping, vertex_indices_0, batch_dims = 1)
        pixel_values_u_1 = gather_nd(vertex_u_mapping, vertex_indices_1, batch_dims = 1)
        pixel_values_u_2 = gather_nd(vertex_u_mapping, vertex_indices_2, batch_dims = 1)
        pixel_values_v_0 = gather_nd(vertex_v_mapping, vertex_indices_0, batch_dims = 1)
        pixel_values_v_1 = gather_nd(vertex_v_mapping, vertex_indices_1, batch_dims = 1)
        pixel_values_v_2 = gather_nd(vertex_v_mapping, vertex_indices_2, batch_dims = 1)

        #
        pixel_values_u = reshape(c0 * pixel_values_u_0 +
                                 c1 * pixel_values_u_1 +
                                 c2 * pixel_values_u_2,
                                 [batch_size, output_size, output_size])
        pixel_values_v = reshape(c0 * pixel_values_v_0 +
                                 c1 * pixel_values_v_1 +
                                 c2 * pixel_values_v_2,
                                 [batch_size, output_size, output_size])

        images = bilinear_sampler(unwarped_textures, pixel_values_u, pixel_values_v)

        return images, buffers_as_mask


