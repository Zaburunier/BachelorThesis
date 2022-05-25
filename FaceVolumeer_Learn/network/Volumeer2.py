import sys
import typing
import logging

import keras.metrics
import tensorflow_graphics.rendering.utils
import trimesh.visual
from trimesh.visual import TextureVisuals
from trimesh import Trimesh
import tensorflow as tf
from tensorflow_graphics.image.transformer import sample
from tensorflow_graphics.geometry.representation.mesh.normals import vertex_normals as compute_vertex_normals
from keras.models import Model
from keras.preprocessing.image import array_to_img

import visualizer.ModelVisualizer
from data.base_model.Triangle2DData import Triangle2DData
from tools.StatConversionTools import ConvertFromBase, ConvertToBase


import h5py, json, numpy as np

import learning_const
import rendering.ImageRendering
from learning_const import TEXTURE_SIZE, BATCH_SIZE, IMAGE_SIZE, MESH_VERTICES, BASE_MODEL_DATA_DIRECTORY
from rendering import Shading, Warping, RenderingTools, ProjectionTools
from data.base_model import ShapeData, TriangleData, VertexUVMapData
from network.Decoder import ShapeDecoder, AlbedoDecoder
from network.Encoder import Encoder
from tools.GradientAccumulator import GradientAccumulator
from tools.TensorTools import RemapTensor, TransformTensor

from network.loss import ShapeLoss, ProjectionLoss, TextureLoss, ReconstructionLoss, \
    AlbedoSymmetryLoss, AlbedoConstancyLoss, ShapeSmoothnessLoss, LandmarkLoss
from keras.metrics import Mean


MODEL_DIR = "D:\\Study\\Thesis\\FaceVolumeer_Learn\\basel"


class Volumeer2(Model):
    def __init__(self, *args, **kwargs):
        super(Volumeer2, self).__init__(*args, **kwargs)

        print("ПОЛУЧЕНИЕ ИНФОРМАЦИИ О БАЗОВОЙ МОДЕЛИ\n--------------------------------------")
        #self.base_model_trimesh = Volumeer2.LoadBaselModel(MODEL_DIR)
        self.base_model_trimesh = Volumeer2.Load3DMM(BASE_MODEL_DATA_DIRECTORY)
        self.logger = logging.getLogger("volumeer_log")

        base_model_triangle_2d_data =  Triangle2DData(learning_const.BASE_MODEL_DATA_DIRECTORY + "triangles_2d.dat").data
        self.faces_on_texture = tf.tile(tf.expand_dims(tf.gather(self.base_model_trimesh.faces, base_model_triangle_2d_data, axis=0), axis = 0), multiples = [learning_const.BATCH_SIZE, 1, 1, 1])

        self.mean_projection = tf.reshape(tf.convert_to_tensor(np.load(learning_const.BASE_MODEL_DATA_DIRECTORY + "mean_projection.npy"), dtype = tf.float32), shape= (1, 8))
        self.std_projection = tf.reshape(tf.convert_to_tensor(np.load(learning_const.BASE_MODEL_DATA_DIRECTORY + "std_projection.npy"), dtype = tf.float32), shape= (1, 8))

        self.std_shape = tf.convert_to_tensor(np.tile(np.array([[1e4, 1e4, 1e4]], dtype = np.float32), (1, learning_const.MESH_VERTICES, 1)), dtype = tf.float32)
        shape_data = ShapeData.ShapeData(learning_const.BASE_MODEL_DATA_DIRECTORY + "3DMM_shape_basis.dat")
        exp_data = ShapeData.ShapeData(learning_const.BASE_MODEL_DATA_DIRECTORY + "3DMM_exp_basis.dat")
        self.mean_shape = tf.convert_to_tensor(np.reshape(shape_data.data_mu + exp_data.data_mu, (1, learning_const.MESH_VERTICES, 3)) / self.std_shape, dtype = tf.float32)
        print("--------------------------------------\nИНФОРМАЦИЯ О БАЗОВОЙ МОДЕЛИ ПОЛУЧЕНА\n")

        print("СОЗДАНИЕ АВТОЭНКОДЕРНОЙ СЕТИ")
        # Подготавливаем нейронные сети
        self.encoder = Encoder()
        self.shape_decoder = ShapeDecoder()
        self.albedo_decoder = AlbedoDecoder()

        self.mean_metric = keras.metrics.Mean()
        print("--------------------------------------\nКОНФИГУРАЦИЯ УЗЛОВ АВТОЭНКОДЕРНОЙ СЕТИ ЗАВЕРШЕНА\n")

        # Схитрим и будем обновлять веса не каждый поданный бэтч, а после каждых нескольких
        self.accumulated_batches = tf.constant(learning_const.GRADIENT_BATCH_SIZE // learning_const.BATCH_SIZE)
        #self.batch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        #self.batch_gradients = GradientAccumulator()

        self.image_reconstruction_loss = ReconstructionLoss.ReconstructionLoss(weight=10.0)
        self.projection_loss = ProjectionLoss.ProjectionLoss(weight = 5.0)
        self.albedo_symmetry_loss = AlbedoSymmetryLoss.AlbedoSymmetryLoss(weight=1.0)
        self.albedo_constancy_loss = AlbedoConstancyLoss.AlbedoConstancyLoss(weight=10.0)
        self.shape_smoothness_loss = ShapeSmoothnessLoss.ShapeSmoothnessLoss(weight=1000.0)
        self.shape_reconstruction_loss = ShapeLoss.ShapeLoss(weight=10.0)
        self.texture_reconstruction_loss = TextureLoss.TextureLoss(weight=1.0)
        self.landmark_loss = LandmarkLoss.LandmarkLoss(weight = 0.1)

        self.total_loss_metric = Mean(name = "total_loss_metric")
        self.image_reconstruction_metric = Mean(name = "image_reconstruction_metric")
        self.projection_metric = Mean(name ="projection_metric")
        self.albedo_symmetry_metric = Mean(name = "albedo_symmetry_metric")
        self.albedo_constancy_metric = Mean(name = "albedo_constancy_metric")
        self.shape_smoothness_metric = Mean(name = "shape_smoothness_metric")
        self.shape_reconstruction_metric = Mean(name = "shape_reconstruction_metric")
        self.texture_reconstruction_metric = Mean(name = "texture_reconstruction_metric")
        self.landmark_metric = Mean(name = "landmark_metric")
        self.weight_regularizarion_metric = Mean("weight_regularization_metric")

        self.using_landmark_loss_flag = False
        self.using_reconstruction_loss_flag = False
        self.using_albedo_constancy_loss_flag = False


    @property
    def metrics(self):
        return [self.total_loss_metric, self.image_reconstruction_metric, self.projection_metric,
                self.albedo_symmetry_metric, self.albedo_constancy_metric,
                self.shape_smoothness_metric, self.shape_reconstruction_metric,
                self.texture_reconstruction_metric, self.landmark_metric, 
                self.weight_regularizarion_metric]


    '''@property
    def losses(self):
        return [self.image_reconstruction_loss,
                self.albedo_symmetry_loss, self.albedo_constancy_loss,
                self.shape_smoothness_loss, self.shape_reconstruction_loss,
                self.texture_reconstruction_loss, self.landmark_loss]'''


    @tf.function#(input_signature = (tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)), tf.dtypes.as_dtype(bool), None))
    def call(self, inputs : tf.Tensor, training : tf.bool = None, mask=None):
        #print("Пропускаем изображения через автоэнкодер...")
        # 1. Проводим данные через кодировщик
        #encoded_shape, encoded_albedo, encoded_projection, encoded_lightning = self.encoder(inputs, training = training, mask = mask)
        encoded_shape, encoded_albedo, encoded_projection, encoded_lightning = self.encoder(inputs, training = training, mask = mask)

        # 2. Данные для формы и альбедо пропускаем через расшифровщик
        decoded_shape = self.shape_decoder(encoded_shape, training, mask)
        decoded_albedo = self.albedo_decoder(encoded_albedo, training, mask)

        # 3. Проводим статистические преобразования данных
        converted_projection = self.mean_projection + encoded_projection * self.std_projection
        # Данные для усреднённой формы представлены в вершиных координатах, а не в текстурных, так что мы сначала переведём туда ответ нейросети
        sampled_shapes = sample(image=decoded_shape,
                                warp=tf.tile(tf.expand_dims(self.base_model_trimesh.visual.uv, axis=0), (learning_const.BATCH_SIZE, 1, 1)))
        sampled_shapes = self.mean_shape + sampled_shapes * self.std_shape

        '''print(f"Получены выходные параметры автоэнкодерной сети.Размерности:"
              f"\nданные о формах - {decoded_shape.points};"
              f"\nоб альбедо - {decoded_albedo.points};"
              f"\nо ракурсах - {encoded_projection.points};"
              f"\nоб освещении - {encoded_lightning.points}.\n")'''
        '''print(f"Checking nan values in output shapes: {tf.math.reduce_any(tf.math.is_nan(decoded_shape))}")
        print(f"Checking nan values in output albedoes: {tf.math.reduce_any(tf.math.is_nan(decoded_albedo))}")
        print(f"Checking nan values in output projections: {tf.math.reduce_any(tf.math.is_nan(encoded_projection))}")
        print(f"Checking nan values in output lights: {tf.math.reduce_any(tf.math.is_nan(encoded_lightning))}")'''
        #print("......................................")


        return decoded_shape, sampled_shapes, decoded_albedo, encoded_projection, converted_projection, encoded_lightning


    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):


        super(Volumeer2, self).compile(optimizer, loss, metrics, loss_weights,
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


    '''input_signature = (tf.TensorSpec(dtype = tf.float32, points = (const.BATCH_SIZE, const.IMAGE_SIZE[0], const.IMAGE_SIZE[1], 3)),
                                        (tf.TensorSpec(dtype = tf.float32, points = (const.BATCH_SIZE, const.TEXTURE_SIZE[0], const.TEXTURE_SIZE[1], 3)),
                                         tf.TensorSpec(dtype = tf.float32, points = (const.BATCH_SIZE, 8)),
                                         tf.TensorSpec(dtype = tf.float32, points = (const.BATCH_SIZE, const.MESH_VERTICES, 3))
                                         )
                     )'''


    @tf.function
    def train_step(self, data):
        '''
        Шаг обучения сети
        :param data: Данные для обучения в списка файлов с изображениями
        :return: Сведения о метриках
        '''
        input_images, input_image_masks, input_texture_masks = data[0]

        #print(f"На вход приняты следующие данные: размер бэтча - {input_images.points[0]}, размер изображения - {input_images.points[1:3]}, число каналов - {input_images.points[3]})\n")
        with tf.GradientTape() as gradient_tape:
            network_data = self(input_images, training = True)

            synthesized_data = self.PerformAnalyticRender_Learning(data[0], network_data, gradient_tape)

            total_loss, image_reconstruction_loss, projection_loss, albedo_symmetry_loss, albedo_constancy_loss, \
            shape_smoothness_loss, shape_reconstruction_loss, texture_reconstruction_loss, \
            landmark_loss, regularization_loss = self.CalculateLossFunctions(data[0], data[1], network_data, synthesized_data)

        gradients = gradient_tape.gradient(total_loss, self.trainable_weights)
        '''for i in range(len(gradients)):
            if (tf.math.reduce_any(tf.math.is_nan(gradients[i]))):
                print("GOT NAN GRADIENTS")
                print(f"List index:{i}; variable for this index: {self.trainable_variables[i]}")
                self.save_weights("weights on crash.hdf5")
                print("SAVED BEFORE CRASH")
                return None

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))'''

        def OnNanValuesAppeared():
            #self.save_weights("weights on crash.hdf5")
            #tf.print("CAUGHT NAN GRADIENTS! Cancelling model update.")
            pass

        def OnNanValuesNotAppeared():
            #tf.print("Gradients are ok, updating model...")
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))


        gradients_nan_flag = tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(grads)) for grads in gradients])
        tf.cond(gradients_nan_flag, OnNanValuesAppeared, OnNanValuesNotAppeared)

        self.total_loss_metric.update_state(total_loss)
        self.image_reconstruction_metric.update_state(image_reconstruction_loss)
        self.projection_metric.update_state(projection_loss)
        self.albedo_symmetry_metric.update_state(albedo_symmetry_loss)
        self.albedo_constancy_metric.update_state(albedo_constancy_loss)
        self.shape_smoothness_metric.update_state(shape_smoothness_loss)
        self.shape_reconstruction_metric.update_state(shape_reconstruction_loss)
        self.texture_reconstruction_metric.update_state(texture_reconstruction_loss)
        self.landmark_metric.update_state(landmark_loss)
        self.weight_regularizarion_metric.update_state(regularization_loss)
        return {"total_loss": self.total_loss_metric.result(),
                "projection_loss": self.projection_metric.result(),
                "image_reconstruction_loss": self.image_reconstruction_metric.result(),
                "albedo_symmetry_loss": self.albedo_symmetry_metric.result(),
                "albedo_constancy_loss": self.albedo_constancy_metric.result(),
                "shape_smoothness_loss": self.shape_smoothness_metric.result(),
                "shape_reconstruction_loss": self.shape_reconstruction_metric.result(),
                "texture_reconstruction_loss": self.texture_reconstruction_metric.result(),
                "landmark_loss": self.landmark_metric.result(),
                "regularization_loss": self.weight_regularizarion_metric.result()}


    @tf.function#(input_signature = ((tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)),
                #                    tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 1)),
                #                    tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 1))), 
                #                    (tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)),
                #                    tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, 8)),
                #                    tf.TensorSpec(dtype=tf.float32, shape=(learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3)))))
    def test_step(self, data):
        input_images, input_image_masks, input_texture_masks = data[0]
        input_textures, input_projections, input_shapes = data[1]
        # print(f"На вход приняты следующие данные: размер бэтча - {input_images.points[0]}, размер изображения - {input_images.points[1:3]}, число каналов - {input_images.points[3]})\n")

        network_data = self(input_images, training = True)

        synthesized_data = self.PerformAnalyticRender_Learning(data[0], network_data)

        total_loss, image_reconstruction_loss, projection_loss, albedo_symmetry_loss, albedo_constancy_loss, \
        shape_smoothness_loss, shape_reconstruction_loss, texture_reconstruction_loss, \
        landmark_loss, regularization_loss = self.CalculateLossFunctions(data[0], data[1], network_data, synthesized_data)

        self.total_loss_metric.update_state(total_loss)
        self.image_reconstruction_metric.update_state(image_reconstruction_loss)
        self.projection_metric.update_state(projection_loss)
        self.albedo_symmetry_metric.update_state(albedo_symmetry_loss)
        self.albedo_constancy_metric.update_state(albedo_constancy_loss)
        self.shape_smoothness_metric.update_state(shape_smoothness_loss)
        self.shape_reconstruction_metric.update_state(shape_reconstruction_loss)
        self.texture_reconstruction_metric.update_state(texture_reconstruction_loss)
        self.landmark_metric.update_state(landmark_loss)
        self.weight_regularizarion_metric.update_state(regularization_loss)

        return {"total_loss" : self.total_loss_metric.result(),
                "projection_loss": self.projection_metric.result(),
                "image_reconstruction_loss": self.image_reconstruction_metric.result(),
                "albedo_symmetry_loss": self.albedo_symmetry_metric.result(),
                "albedo_constancy_loss": self.albedo_constancy_metric.result(),
                "shape_smoothness_loss": self.shape_smoothness_metric.result(),
                "shape_reconstruction_loss": self.shape_reconstruction_metric.result(),
                "texture_reconstruction_loss": self.texture_reconstruction_metric.result(),
                "landmark_loss": self.landmark_metric.result(),
                "regularization_loss": self.weight_regularizarion_metric.result()}


    @tf.function
    def predict_step(self, data):
        self.logger.info("Начало обработки изображения...")
        input_images, input_image_masks, input_texture_masks = data
        # print(f"На вход приняты следующие данные: размер бэтча - {input_images.points[0]}, размер изображения - {input_images.points[1:3]}, число каналов - {input_images.points[3]})\n")

        self.logger.info("Подача изображения в нейросеть...")
        network_data = self(input_images, training = True)#False)
        self.logger.info("Аналитические преобразования...")
        render_data = self.PerformAnalyticRender(network_data)

        '''if (tf.config.functions_run_eagerly() == True):
            synthesized_images, synthesized_image_masks, unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices = render_data
            shape_data, vertices, albedo_data, projection_data, converted_projection_data, lightning_data = network_data
            input_images_numpy = input_images.numpy()
            input_image_masks_numpy = input_image_masks.numpy()
            input_texture_masks_numpy = input_texture_masks.numpy()
            shape_numpy = shape_data.numpy()
            albedo_numpy = albedo_data.numpy()
            texture_map_numpy = unwarped_textures.numpy()
            synthesized_images_numpy = synthesized_images.numpy()
            synthesized_image_mask_numpy = synthesized_image_masks.numpy()'''

        self.logger.info("Обработка изображения завершена.")
        return network_data, render_data


    @tf.function
    def PerformAnalyticRender_Learning(self, input_data, network_data, gradient_tape : tf.GradientTape = None):
        input_images, input_image_masks, input_texture_masks = input_data
        synthesized_images, synthesized_image_masks, unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices = self.PerformAnalyticRender(network_data, gradient_tape)

        if (tf.config.functions_run_eagerly() == True):
            image_before_mask_numpy = synthesized_images.numpy()
            textures_before_mask_numpy = unwarped_textures.numpy()
            normal_map_numpy = unwarped_normals.numpy()
            shading_map_numpy = unwarped_shadings.numpy()

        # tf.print("Накладываем маски...")
        # Убираем лишнее
        unwarped_textures = unwarped_textures * input_texture_masks
        # Здесь мы не просто убираем лишнее, но и заполняем края исходным изображением, чтобы не включать их в потери
        synthesized_images = synthesized_images * input_image_masks + input_images * (1.0 - input_image_masks)

        if (tf.config.functions_run_eagerly() == True):
            image_after_mask_numpy = synthesized_images.numpy()
            textures_after_mask_numpy = unwarped_textures.numpy()

        return synthesized_images, synthesized_image_masks, unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices



    @tf.function
    def PerformAnalyticRender(self, network_data, gradient_tape : tf.GradientTape = None):
        '''
        Метод преобразования данных из нейросети в текстуры и изображения
        :param input_data: Входные данные
        :param network_data: Ответ нейросети
        :param gradient_tape: Экземпляр класса для отслеживания градиентов
        :return: Кортеж данных: (синтезированные изображения, их маски, текстурные развёртки, теневые развёртки, карты нормалей)
        '''
        shape_data, vertices, albedo_data, projection_data, converted_projection_data, lightning_data = network_data

        #print("Проводим обратные преобразования...")

        # Формируем матрицы ракурса образцов
        projection_matrices = ProjectionTools.CreateRotationMatrices(converted_projection_data)
        rotated_vertices = ProjectionTools.RotatePointsWithMatrices(points = vertices, projection_matrices = projection_matrices)

        # Теперь нам нужно перейти от текстуры формы к текстуре нормалей
        # tf.print("Формируем карты нормалей...")
        batch_faces = tf.cast(tf.tile(input=tf.expand_dims(self.base_model_trimesh.faces,
                                                           axis=0),
                                      multiples=[learning_const.BATCH_SIZE, 1, 1]), dtype=tf.int32)

        vertex_normals = compute_vertex_normals(vertices=rotated_vertices, indices=batch_faces)
        rotated_vertex_normals = ProjectionTools.RotatePointsWithMatrices(points = vertex_normals, projection_matrices = projection_matrices)
        unwarped_normals_0 = tf.gather(rotated_vertex_normals, self.faces_on_texture[:, :, :, 0], axis=1, batch_dims = 1)
        unwarped_normals_1 = tf.gather(rotated_vertex_normals, self.faces_on_texture[:, :, :, 1], axis=1, batch_dims = 1)
        unwarped_normals_2 = tf.gather(rotated_vertex_normals, self.faces_on_texture[:, :, :, 2], axis=1, batch_dims = 1)
        unwarped_normals = (unwarped_normals_0 + unwarped_normals_1 + unwarped_normals_2) / 3.0
        #unwarped_normals = unwarped_normals / tf.math.reduce_euclidean_norm(unwarped_normals, axis = -1, keepdims = True)

        # Текстура нормали позволяет нам сформировать текстуру затенения
        # tf.print("Получаем карту теней...")
        unwarped_normals_flatten = tf.reshape(unwarped_normals, shape=(learning_const.BATCH_SIZE, -1, 3))
        # Умножаем на 10, чтобы выходы нейросети не стремились в бесконечность (световые коэффициенты потенциально могут быть в единице)
        unwarped_shadings_flatten = Shading.Shading.GetNormalShading(unwarped_normals_flatten, lightning_data)
        unwarped_shadings = tf.reshape(unwarped_shadings_flatten, shape = (learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 1))
        #unwarped_shadings = unwarped_shadings / tf.math.reduce_euclidean_norm(unwarped_shadings + 1.0e-10, axis = -1, keepdims = True)

        # Имея текстуру альбедо изначально и получив текстуру затенения, получаем итоговую текстуру
        # tf.print("Формируем лицевую текстуру...")
        unwarped_textures = tf.multiply(albedo_data, unwarped_shadings)

        # Здесь же можем наложить эту текстуру и получить модель головы
        if (gradient_tape == None):
            vertex_colors = rendering.ImageRendering.ConvertTextureToVertexColors(self.base_model_trimesh.visual.uv, unwarped_textures)
            synthesized = rendering.ImageRendering.RenderImages(vertices=rotated_vertices,
                                                                vertex_normals=rotated_vertex_normals,
                                                                vertex_colors=vertex_colors)
        else:
            with gradient_tape.stop_recording():
                vertex_colors = rendering.ImageRendering.ConvertTextureToVertexColors(self.base_model_trimesh.visual.uv, unwarped_textures)
                synthesized = rendering.ImageRendering.RenderImages(vertices=rotated_vertices,
                                                                    vertex_normals=rotated_vertex_normals,
                                                                    vertex_colors=vertex_colors)
        synthesized_images = synthesized[0] / 255.0
        synthesized_image_masks = synthesized[1]

        return synthesized_images, synthesized_image_masks, unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices


    @tf.function
    def CalculateLossFunctions(self, input_data, target_data, network_data, synthesized_data):
        '''
        Определение значения функции ошибок
        :param input_data: Входные данные (X)
        :param target_data: Целевые значения (Y)
        :param network_data: Ответ нейросети
        :param synthesized_data: Синтезированные данные
        :return:
        '''
        input_images, input_image_masks, input_texture_masks = input_data
        input_textures, input_projections, input_shapes = target_data
        shape_data, vertices, albedo_data, projection_data, converted_projection_data, lightning_data = network_data
        synthesized_images, synthesized_image_masks, unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices = synthesized_data

        visualizer.ModelVisualizer.VisualizeFromNetworkData(input_shapes[0], (self.mean_projection + input_projections * self.std_projection)[0], input_textures[0])
        visualizer.ModelVisualizer.VisualizeFromNetworkData(vertices[0], converted_projection_data[0], unwarped_textures[0])

        visualizer.ModelVisualizer.VisualizeFromNetworkData(input_shapes[1], (self.mean_projection + input_projections * self.std_projection)[1], input_textures[1])
        visualizer.ModelVisualizer.VisualizeFromNetworkData(vertices[1], converted_projection_data[1], unwarped_textures[1])

        if (tf.config.functions_run_eagerly() == True):
            input_images_numpy = input_images.numpy()
            input_image_masks_numpy = input_image_masks.numpy()
            input_texture_masks_numpy = input_texture_masks.numpy()
            input_textures_numpy = input_textures.numpy()
            shape_numpy = shape_data.numpy()
            albedo_numpy = albedo_data.numpy()
            texture_map_numpy = unwarped_textures.numpy()
            synthesized_images_numpy = synthesized_images.numpy()
            synthesized_image_mask_numpy = synthesized_image_masks.numpy()


        # tf.print("Оцениваем функцию потерь...")
        image_reconstruction_loss = self.image_reconstruction_loss.call(input_images, synthesized_images, input_image_masks)
        # tf.print(f"Потеря по реконструкции изображения = {image_reconstruction_loss}")

        projection_loss = self.projection_loss.call(input_projections, projection_data)
        # tf.print(f"Потеря по определению проекции = {projection_loss}")

        albedo_symmetry_loss = self.albedo_symmetry_loss.__call__(AlbedoSymmetryLoss.AlbedoSymmetryLoss.
                                                                  FlipAlbedos(albedo_data),
                                                                  albedo_data)
        # tf.print(f"Потеря по симметрии альбедо = {albedo_symmetry_loss}")

        albedo_constancy_loss = self.albedo_constancy_loss. \
            CalculateLoss(input_textures,
                          albedo_data)
        # tf.print(f"Потеря по постоянству альбедо = {albedo_constancy_loss}")

        shape_smoothness_loss = self.shape_smoothness_loss.__call__(ShapeSmoothnessLoss.ShapeSmoothnessLoss.
                                                                    CalculateNeighboursAverageShape(shape_data),
                                                                    ShapeSmoothnessLoss.ShapeSmoothnessLoss.
                                                                    TrimShape(shape_data))
        # tf.print(f"Потеря по гладкости формы = {shape_smoothness_loss}")

        shape_reconstruction_loss = self.shape_reconstruction_loss.__call__(input_shapes,
                                                                            vertices)
        # tf.print(f"Потеря по точности формы = {shape_reconstruction_loss}")

        texture_reconstruction_loss = self.texture_reconstruction_loss.call(input_textures, unwarped_textures, input_texture_masks)
        # tf.print(f"Потеря по точности текстуры = {texture_reconstruction_loss}")

        # Сравнивать ключевые точки будем по повёрнутым формам лица (чтобы приобщить сюда потерю по проекции, а не только дублировать потери по форме)
        projection_matrices = ProjectionTools.CreateRotationMatrices(self.mean_projection + input_projections * self.std_projection)
        rotated_input_vertices = ProjectionTools.RotatePointsWithMatrices(points = input_shapes, projection_matrices = projection_matrices)
        landmark_loss = self.landmark_loss.__call__(self.landmark_loss.GatherLandmarks(rotated_input_vertices),
                                                    self.landmark_loss.GatherLandmarks(rotated_vertices))
        # tf.print(f"Потеря по точности ключевых точек = {landmark_loss}")

        regularization_loss = 1.0e-01 * tf.add_n(self.losses)
        # tf.print(f"Потеря по регуляризации весов: {regularization_loss}")

        loss_value = projection_loss + shape_reconstruction_loss + texture_reconstruction_loss + albedo_symmetry_loss + shape_smoothness_loss# + regularization_loss# + landmark_loss + regularization_loss
        if (self.using_landmark_loss_flag):
            loss_value += landmark_loss

        if (self.using_reconstruction_loss_flag):
            loss_value += image_reconstruction_loss

        if (self.using_albedo_constancy_loss_flag):
            loss_value += albedo_constancy_loss

        # tf.print(f"Итоговое значение функции потерь = {loss_value}.\n......................................")
        return loss_value, image_reconstruction_loss, projection_loss, albedo_symmetry_loss, albedo_constancy_loss, \
               shape_smoothness_loss, shape_reconstruction_loss, texture_reconstruction_loss, \
               landmark_loss, regularization_loss


    def ResetBatchNormLayers(self):
        self.encoder.ResetBatchNormLayers()
        self.shape_decoder.ResetBatchNormLayers()
        self.albedo_decoder.ResetBatchNormLayers()


    @classmethod
    def LoadBaselModel(cls, path : str) -> Trimesh:
        '''
        Загрузка базельской модели из h5 и json-файлов в заданной папке
        :param path: Папка, в которой лежат файлы
        :return: Мэш модели головы и текстурные координаты
        '''
        with h5py.File("model2019_fullHead.h5") as model_file:
            print("Файл model2019_fullHead.h5 успешно открыт, считываем данные о вершинах...")
            #print(f"Outer model keys: {model_file.keys()}")
            #print("Color keys: %s" % model_file["color"].keys())
            #print("Color model keys: %s" % model_file["color"]["model"].keys())
            #print("Expr keys: %s" % model_file["expression"].keys())
            #print("Shape keys: %s" % model_file["points"].keys())
            #print("Model keys: %s" % model_file["points"]["model"].keys())
            # Get the vertices
            vertices = np.add(list(model_file["points"]["model"]["mean"]),
                              list(model_file["expression"]["model"]["mean"]))
            print("Данные о вершинах успешно прочитаны.\n......................................")

        with open("model2019_textureMapping.json") as mapping_file:
            print("Файл model2019_textureMapping.json успешно открыт, считываем данные о треугольниках и текстурных координатах...")
            map_data = json.load(mapping_file)
            faces = np.array(map_data["triangles"])
            uv = np.array(map_data["textureMapping"]["pointData"])
            # print(map_data)
            print("Данные о треугольниках и текстурных координатах успешно прочитаны.\n......................................")

        return Trimesh(vertices=np.reshape(vertices, newshape=(-1, 3)), faces=faces, visual = TextureVisuals(uv = uv))


    @classmethod
    def Load3DMM(cls, directory: str) -> Trimesh:
        '''
        Загрузка модели лица из данных в базовой статье
        :param directory: Папка с файлами данных
        :return: Мэш модели с текстурными координатами
        '''
        shape_data = ShapeData.ShapeData(directory + "3DMM_shape_basis.dat")
        shape_std_data = np.tile(np.array([1e4, 1e4, 1e4]),
                                 MESH_VERTICES)  # np.load(const.BASE_MODEL_DATA_DIRECTORY + "std_shape.npy")
        exp_data = ShapeData.ShapeData(directory + "3DMM_exp_basis.dat")

        shape = np.divide(shape_data.data_mu + exp_data.data_mu, shape_std_data)

        triangle_data = TriangleData.TriangleData(directory + "triangles.dat")
        uv_data = VertexUVMapData.VertexUVMapData(directory + "vertices_2d_u.dat", directory + "vertices_2d_v.dat")
        u = RemapTensor(uv_data.data_u[:-1])
        v = RemapTensor(uv_data.data_v[:-1])
        u = TransformTensor(u, scale_ratio=.96, translation_ratio=0.04)
        v = TransformTensor(v, scale_ratio=1.15, translation_ratio=-0.085)
        normalized_uv = tf.stack([v, -u], axis=1)

        result = Trimesh(vertices = np.reshape(shape, newshape = [MESH_VERTICES, 3]),
                         faces = np.transpose(triangle_data.data[:, :-1]),
                         visual = TextureVisuals(uv = normalized_uv))
        result.fix_normals()
        return result


    @classmethod
    def ApplyImageMask(cls, input_images, synthesized_images, synthesized_image_masks):
        '''
        Заполняем незакрашенные области синтезированного изображения исходным изображением (чтобы область вокруг лица не считалась в функции ошибок)
        :param input_images: Исходные изображения
        :param synthesized_images: Синтезированные изображения
        :param synthesized_image_masks: Маска синтезированных изображений
        :return: Преобразованные синтезированные изображения
        '''
        return synthesized_image_masks * synthesized_images + (1.0 - synthesized_image_masks) * input_images



'''self.batch_gradients(gradients)
self.batch_counter.assign_add(1)
# Обновляем модель, если накопили достаточное число градиентов
def ApplyGrads():
    tf.print("! ПРИМЕНЯЕМ НАКОПЛЕННЫЕ ГРАДИЕНТЫ !")
    self.batch_counter.assign(0)
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
    self.optimizer.apply_gradients(zip(self.batch_gradients.gradients, self.trainable_variables))
    self.batch_gradients.reset()

def NoApplyGrads():
    # tf.print("Не изменяем параметры сети, поскольку процесс накопления ещё не завершён")
    pass
tf.cond(tf.math.equal(self.batch_counter % const.GRADIENT_BATCH_SIZE, 0), ApplyGrads, NoApplyGrads)'''