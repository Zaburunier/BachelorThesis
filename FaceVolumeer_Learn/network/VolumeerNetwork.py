from numpy import load
from tensorflow import GradientTape, Variable, constant, float32, tile, reshape, not_equal, cast
from tensorflow import multiply, add, subtract, divide, Tensor, expand_dims, greater, math as tfmath
from keras.optimizers import Optimizer
from keras.layers import InputSpec
from keras.models import Model

import rendering.ProjectionTools
from learning_const import TEXTURE_SIZE, BASE_MODEL_DATA_DIRECTORY, BATCH_SIZE, IMAGE_SIZE, MESH_VERTICES
from data import NetworkDataGenerator
from data.base_model import ShapeData, TriangleData, Triangle2DData, \
    TriangleBarycentricData, VertexTriangleMapData, VertexUVMapData
from network.Decoder import ShapeDecoder, AlbedoDecoder
from network.Encoder import Encoder
from network.loss import ShapeLoss, ProjectionLoss, TextureLoss, ReconstructionLoss, \
    AlbedoSymmetryLoss, AlbedoConstancyLoss, ShapeSmoothnessLoss
from rendering import Shading, Warping, RenderingTools, ProjectionTools
from rendering.Warping import bilinear_sampler
from tools.LoadTools import LoadImagesByFilenames


class VolumeerNetwork(Model):
    def __init__(self, *args, **kwargs):
        super(VolumeerNetwork, self).__init__(*args, **kwargs)

        print("СОЗДАНИЕ НОВОЙ АВТОЭНКОДЕРНОЙ СЕТИ КЛАССА VolumeerNetwork")
        # Подготавливаем нейронные сети
        self.encoder = Encoder()
        self.shape_decoder = ShapeDecoder()
        self.albedo_decoder = AlbedoDecoder()
        print("--------------------------------------\nКОНФИГУРАЦИЯ УЗЛОВ АВТОЭНКОДЕРНОЙ СЕТИ ЗАВЕРШЕНА\n")

        print("ПОЛУЧЕНИЕ ИНФОРМАЦИИ О БАЗОВОЙ МОДЕЛИ\n--------------------------------------")
        # Загружаем информацию о базовой модели лица
        base_model_shape = ShapeData.ShapeData(BASE_MODEL_DATA_DIRECTORY + "3DMM_shape_basis.dat")
        base_model_expression = ShapeData.ShapeData(BASE_MODEL_DATA_DIRECTORY + "3DMM_exp_basis.dat")
        #points = np.divide(shape_data.data_mu + exp_data.data_mu, shape_std_data)
        self.batch_shape_mean = constant(
            tile(reshape(tfmath.divide(base_model_shape.data_mu + base_model_expression.data_mu,
                                       tile([1e4, 1e4, 1e4], [MESH_VERTICES])),
                         [1, -1]),
                 [BATCH_SIZE, 1]),
            float32)

        self.base_model_triangle_data = TriangleData.TriangleData(BASE_MODEL_DATA_DIRECTORY + "triangles.dat")

        self.base_model_triangle_2d_data = Triangle2DData.Triangle2DData(BASE_MODEL_DATA_DIRECTORY + "triangles_2d.dat")

        self.base_model_triangle_barycentric_data = TriangleBarycentricData. \
            TriangleBarycentricData(BASE_MODEL_DATA_DIRECTORY + "triangles_2d_barycoord.dat")

        self.base_model_triangle_vertex_map_data = VertexTriangleMapData.VertexTriangleMapData(
            BASE_MODEL_DATA_DIRECTORY + "vertex_triangle_mapping.dat")

        self.base_model_vertex_uv_map_data = VertexUVMapData. \
            VertexUVMapData(BASE_MODEL_DATA_DIRECTORY + "vertices_2d_u.dat",
                            BASE_MODEL_DATA_DIRECTORY + "vertices_2d_v.dat")
        self.batch_vertex_u_map_data = tile(reshape(self.base_model_vertex_uv_map_data.data_u[:-1], [1, 1, -1]),
                                            [BATCH_SIZE, 1, 1])
        self.batch_vertex_v_map_data = tile(reshape(self.base_model_vertex_uv_map_data.data_v[:-1], [1, 1, -1]),
                                            [BATCH_SIZE, 1, 1])

        self.batch_shape_std = constant(
            tile(reshape(load(BASE_MODEL_DATA_DIRECTORY + "std_shape.npy"), [1, -1]), [BATCH_SIZE, 1]), float32)

        projection_mean = constant(load(BASE_MODEL_DATA_DIRECTORY + "mean_projection.npy"), float32)
        self.batch_projection_mean = tile(reshape(projection_mean, [1, -1]), [BATCH_SIZE, 1])
        projection_std = constant(load(BASE_MODEL_DATA_DIRECTORY + "std_projection.npy"), float32)
        self.batch_projection_std = tile(reshape(projection_std, [1, -1]), [BATCH_SIZE, 1])

        print("--------------------------------------\nИНФОРМАЦИЯ О БАЗОВОЙ МОДЕЛИ ПОЛУЧЕНА\n")

        # Готовим место под вспомогательные д-е для обучения

        self._init_set_name("Volumeer")
        self.inputs = self.encoder.inputs
        self.input_spec = InputSpec(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.outputs = [self.shape_decoder.outputs, self.albedo_decoder.outputs,
                        self.encoder.projection_encoder.outputs, self.encoder.light_encoder.outputs]


    def call(self, inputs, training=None, mask=None):
        # 1. Проводим данные через кодировщик
        encoded_shape, encoded_albedo, encoded_projection, encoded_lightning = self.encoder(inputs)

        # 2. Данные для формы и альбедо пропускаем через расшифровщик
        decoded_shape = self.shape_decoder(encoded_shape)
        decoded_shape_as_list = reshape(
            bilinear_sampler(decoded_shape, self.batch_vertex_v_map_data, self.batch_vertex_u_map_data),
            shape=[BATCH_SIZE, -1])

        decoded_albedo = self.albedo_decoder(encoded_albedo)

        # 3. Проводим статистические преобразования над полученными величинами
        encoded_projection_normalized = VolumeerNetwork.ConvertToBasis(encoded_projection,
                                                                       self.batch_projection_mean,
                                                                       self.batch_projection_std)
        decoded_shape_normalized = VolumeerNetwork.ConvertToBasis(decoded_shape_as_list,
                                                                  self.batch_shape_mean,
                                                                  self.batch_shape_std)

        return decoded_shape_normalized, decoded_shape, decoded_albedo, encoded_projection_normalized, encoded_lightning


    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        super().compile(optimizer, loss, metrics, loss_weights,
                        weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

        self.encoder.compile(optimizer, loss, metrics, loss_weights,
                             weighted_metrics, run_eagerly,
                             steps_per_execution, **kwargs)

        self.shape_decoder.compile(optimizer, loss, metrics, loss_weights,
                                   weighted_metrics, run_eagerly,
                                   steps_per_execution, **kwargs)

        self.albedo_decoder.compile(optimizer, loss, metrics, loss_weights,
                                    weighted_metrics, run_eagerly,
                                    steps_per_execution, **kwargs)


    def train_step(self, data : dict):
        '''
        Шаг обучения сети
        :param data: Данные для обучения в списка файлов с изображениями
        :return: Сведения о метриках
        '''
        images = data.get("images")
        print(
            f"На вход приняты следующие данные: размер бэтча - {images.shape[0]}, размер изображения - {images.shape[1:3]}, число каналов - {images.shape[3]})")

        with GradientTape(persistent=True) as gradient_tape:
            # 1. Получаем ответ от нейронных сетей
            print("Пропускаем изображения через автоэнкодер...")
            shape_data, shape_data_2d, albedo_data, projection_data, lightning_data = self(images)
            print(f"Получены выходные параметры автоэнкодерной сети.Размерности:"
                  f"\nданные о формах - {shape_data.shape};"
                  f"\nоб альбедо - {albedo_data.shape};"
                  f"\nо ракурсах - {projection_data.shape};"
                  f"\nоб освещении - {lightning_data.shape}.")
            print("......................................")

            vertices = reshape(shape_data, [BATCH_SIZE, -1, 3])

            # 2. Формируем необходимые для синтеза текстур данные
            print("Анализ полученных значений...")
            print("Формируем матрицы вращения...")
            projection_matrices = rendering.ProjectionTools.CalculateProjectionMatrices(projection_data)
            print(f"Матрицы вращения успешно сформированы. Размерность - {projection_matrices.shape}")
            print("Формируем карты нормалей...")
            vertex_normals, triangle_normals = RenderingTools. \
                ComputeNormals(vertices=vertices,
                               triangles=self.base_model_triangle_data.data,
                               vertex_triangle_map=self.base_model_triangle_vertex_map_data.data)
            vertex_normals = ProjectionTools.RotateVectors(vectors=vertex_normals,
                                                           projection_matrices=projection_matrices)
            triangle_normals = ProjectionTools.RotateVectors(vectors=triangle_normals,
                                                             projection_matrices=projection_matrices)
            triangle_visibility_masks = greater(triangle_normals[:, :, 2], 0)
            print(f"Карты нормалей успешно сформированы."
                  f"\nРазмерность карт нормалей вершин - {vertex_normals.shape};"
                  f"\nРазмерность карт нормалей треугольников - {triangle_normals.shape}")
            print("......................................")

            # 3. Получаем карты теней, которая пригодятся для лицевой текстуры
            print("Получаем карту теней...")
            shading_texture = Shading.Shading.GenerateShade(self.base_model_triangle_data.data,
                                                            self.base_model_triangle_2d_data.data,
                                                            self.base_model_triangle_barycentric_data.data,
                                                            lightning_data,
                                                            vertex_normals,
                                                            TEXTURE_SIZE)
            print(f"Карта теней получена. Размерность - {shading_texture.shape}")

            # 4. Получаем лицевые текстуры
            # Имея текстуру альбедо (UV-развёртку) и карту теней, можно получить итоговую текстуру
            print("Формируем лицевую текстуру...")
            texture_data = 2.0 * multiply(.5 * (albedo_data + 1.0), shading_texture) - 1
            print(f"Лицевая текстура получена. Размерность - {texture_data.shape}")

            # 5. Преобразовываем полученные текстуры в синтетические изображения
            print("Синтезируем изображения...")
            synthesized_images, synthesized_image_masks = Warping.Warping.Warp(self.base_model_triangle_data.data,
                                                                               self.base_model_vertex_uv_map_data,
                                                                               texture_data,
                                                                               vertices,
                                                                               triangle_normals,
                                                                               output_size=IMAGE_SIZE[0])

            # Зачем это нужно - см. подпункт статьи Occlusion-aware Rendering
            synthesized_image_masks = expand_dims(synthesized_image_masks, -1)
            synthesized_images = multiply(synthesized_images, synthesized_image_masks) + multiply(images, 1 - synthesized_image_masks)
            print(f"Синтетические изображения сформированы. Размерность данных - {synthesized_images.shape}")
            print("......................................")

            # 6. Имея все данные, считаем значения функции потерь
            print("Оцениваем функцию потерь...\n")
            input_data_dict = {
                "image": images,
                "points": None,#data.get("shapes"),
                "albedo": None,
                "projection": data.get("projections"),
                "light": None,
                "texture": data.get("textures")
            }

            reconstruction_data_dict = {
                "image": synthesized_images,
                "points": shape_data,
                "shape_2d" : shape_data_2d,
                "albedo": albedo_data,
                "projection": projection_data,
                "light": lightning_data,
                "texture": texture_data
            }

            loss = self.EstimateLossFunctionValue(input_data_dict, reconstruction_data_dict)
            print(f"Итоговое значение функции потерь = {loss}.\n......................................")

        # Фиксируем прибыль за последний шаг и обновляем модель
        print("Изменяем параметры сети...\n")
        #print("Tape variables:")
        #for variable in gradient_tape.watched_variables():
        #    print(f"Variable named {variable.name} with points {variable.points} and operating type {variable.dtype}")

        gradients = gradient_tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Возвращаем словарь с данными по метрикам
        return {"total_loss" : loss}


    def EstimateLossFunctionValue(self, input_data: dict, reconstruction_data: dict) -> float:
        '''
        Оцениваем значение функции потерь для полученной реконструкции
        :param input_data: Словарь входных данных
        (должен иметь ключ "image", дополнительно может иметь те же ключи,
        что и у словаря выходных д-х, см. ниже)
        :param reconstruction_data: Словарь
        (должен иметь ключи "points", "albedo", "projection", "light", "texture_data", "image")
        :return: Одно число - функция потерь для переданных данных
        '''
        loss_value = 0.0
        # Базовая функция потерь по реконструкции (попиксельное сравнение)
        image_reconstruction_loss = ReconstructionLoss. \
            ReconstructionLoss().__call__(input_data.get("image"),
                                          reconstruction_data.get("image"))
        print(f"Потеря по реконструкции изображения = {image_reconstruction_loss}")
        loss_value += image_reconstruction_loss

        # Функция потерь по реконструкции изображения с помощью loss network


        # Базовая функция потерь по симметрии альбедо (попиксельное сравнение)
        albedo_symmetry_loss = AlbedoSymmetryLoss. \
            AlbedoSymmetryLoss().__call__(AlbedoSymmetryLoss.AlbedoSymmetryLoss.
                                          FlipAlbedos(reconstruction_data.get("albedo")),
                                          reconstruction_data.get("albedo"))
        print(f"Потеря по симметрии альбедо = {albedo_symmetry_loss}")
        loss_value += albedo_symmetry_loss

        # Базовая функция потерь по постоянству альбедо (сравнение с соседями)
        albedo_constancy_loss = AlbedoConstancyLoss.AlbedoConstancyLoss().\
            CalculateLoss(reconstruction_data.get("texture"),
                          reconstruction_data.get("albedo"))
        print(f"Потеря по постоянству альбедо = {albedo_constancy_loss}")
        loss_value += albedo_constancy_loss

        # Базовая функция потерь по гладкости поверхности (сравнение каждого пикселя с соседями)
        shape_smoothness_loss = ShapeSmoothnessLoss. \
            ShapeSmoothnessLoss().__call__(ShapeSmoothnessLoss.ShapeSmoothnessLoss.
                                           CalculateNeighboursAverageShape(reconstruction_data.get("shape_2d")),
                                           ShapeSmoothnessLoss.ShapeSmoothnessLoss.
                                           TrimShape(reconstruction_data.get("shape_2d")))
        print(f"Потеря по гладкости формы = {shape_smoothness_loss}")
        loss_value += shape_smoothness_loss

        if (input_data.get("points") != None):
            # Функция потерь по форме (в случае, если идеальное значение предоставлено)
            shape_reconstruction_loss = ShapeLoss.ShapeLoss(weight = 10.0).__call__(input_data.get("points"),
                                                                       reconstruction_data.get("points"))
            print(f"Потеря по точности формы = {shape_reconstruction_loss}")
            loss_value += shape_reconstruction_loss
        else:
            print("Невозможно оценить точность полученной формы, так как идеальные значения не были переданы.")

        if (input_data.get("projection") != None):
            # Функция потерь по ракурсу (в случае, если идеальное значение предоставлено)
            projection_reconstruction_loss = ProjectionLoss. \
                ProjectionLoss(weight = 5.0).__call__(input_data.get("projection"),
                                          reconstruction_data.get("projection"))
            print(f"Потеря по точности проекции = {projection_reconstruction_loss}")
            loss_value += projection_reconstruction_loss
        else:
            print("Невозможно оценить точность полученной проекции, так как идеальные значения не были переданы.")

        if (input_data.get("texture") != None):
            # Функция потерь по текстуре (в случае, если идеальное значение предоставлено)
            texture_reconstruction_loss = TextureLoss. \
                TextureLoss().__call__(input_data.get("texture"),
                                       reconstruction_data.get("texture"))
            print(f"Потеря по точности текстуры = {texture_reconstruction_loss}")
            loss_value += texture_reconstruction_loss
        else:
            print("Невозможно оценить точность полученной текстуры, так как идеальные значения не были переданы.")

        if (input_data.get("landmark") != None):
            # Функция потерь по ключевым точкам (в случае, если идеальное значение предоставлено)
            landmark_loss = 0.0
            print(f"Потеря по точности ключевых точек = {landmark_loss}")
            loss_value += landmark_loss
        else:
            print(
                "Невозможно оценить точность расположения ключевых точек, так как идеальные значения не были переданы.")

        return loss_value

    @classmethod
    def ConvertToBasis(cls, data: Tensor, mean_data: Tensor, std_data: Tensor) -> Tensor:
        '''
        Преобразование тензора из стандартного распределения в базис
        :param data: Исходный тензор
        :param mean_data: Тензор средних значений
        :param std_data: Тензор СКО
        :return: Преобразованный через mean_data + data * std_data
        '''
        return add(mean_data, multiply(data, std_data))

    @classmethod
    def ConvertFromBasis(cls, data: Tensor, mean_data: Tensor, std_data: Tensor) -> Tensor:
        '''
        Преобразованные тензора из базиса в стандартное распределение
        :param data: Исходный тензор
        :param mean_data: Тензор средних значений
        :param std_data: Тензор СКО
        :return: Преобразованный через (mean_data - data) / std_data
        '''
        return divide(subtract(data, mean_data), std_data)
