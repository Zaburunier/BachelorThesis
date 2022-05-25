from keras.losses import Loss
from keras.metrics import Metric
from tensorflow import constant, float32

from network.loss import ShapeLoss, ProjectionLoss, TextureLoss, ReconstructionLoss, \
    AlbedoSymmetryLoss, AlbedoConstancyLoss, ShapeSmoothnessLoss


# from network.loss.ReconstructionNetworkLoss import ReconstructionNetworkLoss


class NetworkLoss(Loss):
    def call(self, y_true, y_pred):
        return NetworkLoss.EstimateLossFunctionValue(y_true, y_pred)


    @classmethod
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
        #return constant(0.0, dtype = float32)

        # Базовая функция потерь по реконструкции (попиксельное сравнение)
        image_reconstruction_loss = ReconstructionLoss. \
            ReconstructionLoss(weight = 10.0).__call__(input_data.get("images"),
                                          reconstruction_data.get("images"))
        print(f"Потеря по реконструкции изображения = {image_reconstruction_loss}")
        loss_value += image_reconstruction_loss

        # Функция потерь по реконструкции изображения с помощью loss network


        # Базовая функция потерь по симметрии альбедо (попиксельное сравнение)
        albedo_symmetry_loss = AlbedoSymmetryLoss. \
            AlbedoSymmetryLoss(weight = 3.0).__call__(AlbedoSymmetryLoss.AlbedoSymmetryLoss.
                                          FlipAlbedos(reconstruction_data[6]),
                                          reconstruction_data[6])
        print(f"Потеря по симметрии альбедо = {albedo_symmetry_loss}")
        loss_value += albedo_symmetry_loss

        # Базовая функция потерь по постоянству альбедо (сравнение с соседями)
        albedo_constancy_loss = AlbedoConstancyLoss.AlbedoConstancyLoss(weight = 5.0).\
            CalculateLoss(reconstruction_data[1],
                          reconstruction_data[6])
        print(f"Потеря по постоянству альбедо = {albedo_constancy_loss}")
        loss_value += albedo_constancy_loss

        # Базовая функция потерь по гладкости поверхности (сравнение каждого пикселя с соседями)
        shape_smoothness_loss = ShapeSmoothnessLoss. \
            ShapeSmoothnessLoss(weight = 2.0).__call__(ShapeSmoothnessLoss.ShapeSmoothnessLoss.
                                           CalculateNeighboursAverageShape(reconstruction_data[4]),
                                           ShapeSmoothnessLoss.ShapeSmoothnessLoss.
                                           TrimShape(reconstruction_data[4]))
        print(f"Потеря по гладкости формы = {shape_smoothness_loss}")
        loss_value += shape_smoothness_loss

        # Функция потерь по форме (в случае, если идеальное значение предоставлено)
        shape_reconstruction_loss = ShapeLoss.ShapeLoss(weight = 10.0).__call__(input_data[3],
                                                                              reconstruction_data[3])
        print(f"Потеря по точности формы = {shape_reconstruction_loss}")
        loss_value += shape_reconstruction_loss

        # Функция потерь по ракурсу (в случае, если идеальное значение предоставлено)
        '''projection_reconstruction_loss = ProjectionLoss. \
            ProjectionLoss(weight = 5.0).__call__(input_data[2],
                                                reconstruction_data[2])
        print(f"Потеря по точности проекции = {projection_reconstruction_loss}")
        loss_value += projection_reconstruction_loss'''

        # Функция потерь по текстуре (в случае, если идеальное значение предоставлено)
        texture_reconstruction_loss = TextureLoss. \
            TextureLoss(weight = 5.0).__call__(input_data[1],
                                   reconstruction_data[1])
        print(f"Потеря по точности текстуры = {texture_reconstruction_loss}")
        loss_value += texture_reconstruction_loss

        '''if (input_data[3] != None):
            # Функция потерь по форме (в случае, если идеальное значение предоставлено)
            shape_reconstruction_loss = ShapeLoss.ShapeLoss(weight = 10.0).__call__(input_data[3],
                                                                       reconstruction_data[3])
            print(f"Потеря по точности формы = {shape_reconstruction_loss}")
            loss_value += shape_reconstruction_loss
        else:
            print("Невозможно оценить точность полученной формы, так как идеальные значения не были переданы.")

        if (input_data[2] != None):
            # Функция потерь по ракурсу (в случае, если идеальное значение предоставлено)
            projection_reconstruction_loss = ProjectionLoss. \
                ProjectionLoss(weight = 5.0).__call__(input_data[2],
                                          reconstruction_data[2])
            print(f"Потеря по точности проекции = {projection_reconstruction_loss}")
            loss_value += projection_reconstruction_loss
        else:
            print("Невозможно оценить точность полученной проекции, так как идеальные значения не были переданы.")

        if (input_data[1] != None):
            # Функция потерь по текстуре (в случае, если идеальное значение предоставлено)
            texture_reconstruction_loss = TextureLoss. \
                TextureLoss().__call__(input_data[1],
                                       reconstruction_data[1])
            print(f"Потеря по точности текстуры = {texture_reconstruction_loss}")
            loss_value += texture_reconstruction_loss
        else:
            print("Невозможно оценить точность полученной текстуры, так как идеальные значения не были переданы.")

        if (False): # input_data.get("landmarks") != None
            # Функция потерь по ключевым точкам (в случае, если идеальное значение предоставлено)
            landmark_loss = 0.0
            print(f"Потеря по точности ключевых точек = {landmark_loss}")
            loss_value += landmark_loss
        else:
            print(
                "Невозможно оценить точность расположения ключевых точек, так как идеальные значения не были переданы.")'''

        return loss_value
