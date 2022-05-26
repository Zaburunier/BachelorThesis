from PySide6.QtWidgets import QLabel, QComboBox, QPushButton, QCheckBox, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QObject, QThread, QSize
import const
import learning_const
import visualizer.ModelVisualizer
from viewing.ModelViewer import ModelViewer
from trimesh import visual
from network.Volumeer2 import Volumeer2
from tensorflow_graphics.geometry.representation.mesh.normals import vertex_normals as compute_vertex_normals
from tensorflow import linalg
from PIL import Image
from keras.preprocessing.image import array_to_img
from time import struct_time, localtime
import time

MODEL_SAVE_FORMATS = ["obj", "stl"]


class ModelViewerWidgetHandler(QObject):
    viewingThread = QThread()


    def __init__(self, mainWindow):
        '''
        Инициализация виджетов окна, относящихся к изучению результата
        '''
        super(ModelViewerWidgetHandler, self).__init__()

        self.mainWindow = mainWindow
        self.mainWindow.onImageSelectionReset.connect(self.ResetState)

        self.prediction_data = None

        # 1. Разместим здесь логотип для красоты :)
        self.logoLabel = QLabel()
        pixmap = QPixmap("resources/face_icon.png").scaled(180, 180, Qt.KeepAspectRatio)
        self.logoLabel.setPixmap(pixmap)
        #self.mainLayout.addWidget(label, 0, 2)

        # 2. Несколько кнопок с режимом просмотра
        self.includeProjectionCheckBox = QCheckBox(text = "Применить оцененный ракурс")
        self.includeProjectionCheckBox.setFont(const.CHECKBOX_TEXT_FONT)
        self.includeProjectionCheckBox.setMinimumSize(QSize(220, 20))
        self.includeProjectionCheckBox.setMaximumSize(QSize(280, 20))

        self.includeTextureCheckBox = QCheckBox(text = "Наложить оцененную текстуру")
        self.includeTextureCheckBox.setFont(const.CHECKBOX_TEXT_FONT)
        self.includeTextureCheckBox.setMinimumSize(QSize(220, 20))
        self.includeTextureCheckBox.setMaximumSize(QSize(280, 20))

        self.viewResultButton = QPushButton(text ="Просмотреть результат")
        self.viewResultButton.setFont(const.BUTTON_TEXT_FONT)
        self.viewResultButton.clicked.connect(self.OnShowModelButtonClicked)
        self.viewResultButton.setMinimumSize(QSize(220, 50))
        self.viewResultButton.setMaximumSize(QSize(280, 50))

        # 3. Выбор опций сохранения
        # Здесь выпадающий список из нескольких форматов, активен всегда
        self.modelFormatOptionsBox = QComboBox()
        self.modelFormatOptionsBox.addItems(MODEL_SAVE_FORMATS)

        # 4. Кнопка сохранения 3D-модели
        # Неактивна вплоть до завершения работы первой обработки
        self.saveModelButton = QPushButton(text ="Сохранить\nтрёхмерную модель")
        self.saveModelButton.setFont(const.BUTTON_TEXT_FONT)
        self.saveModelButton.setMinimumSize(QSize(220, 50))
        self.saveModelButton.setMaximumSize(QSize(280, 60))
        self.saveModelButton.clicked.connect(self.OnSaveModelButtonClicked)

        self.saveTextureCheckBox = QCheckBox(text="С текстурой")
        self.saveTextureCheckBox.setFont(const.CHECKBOX_TEXT_FONT)

        # 5. Кнопка сохранения результатов работы нейросети
        # Неактивна вплоть до завершения работы первой обработки
        self.saveDataButton = QPushButton(text ="Сохранить результат\nобработки в папку")
        self.saveDataButton.setFont(const.BUTTON_TEXT_FONT)
        self.saveDataButton.setMinimumSize(QSize(220, 50))
        self.saveDataButton.setMaximumSize(QSize(280, 60))
        self.saveDataButton.clicked.connect(self.OnSaveDataButtonClicked)

        # 6. Объект, работающий с просмотром полученных данных
        self.modelViewer = ModelViewer(self.includeProjectionCheckBox, self.includeTextureCheckBox)
        #self.modelViewer.moveToThread(self.viewingThread)
        #self.viewingThread.started.connect(self.modelViewer.run)
        #self.modelViewer.onFinished.connect(self.viewingThread.quit)

        # 7. Диалоговое окно для сохранения данных (выбор папки)
        self.chooseDataSaveDirectoryDialog = QFileDialog(self.saveDataButton)
        self.chooseDataSaveDirectoryDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.chooseDataSaveDirectoryDialog.setDirectory("C:\\")
        self.chooseDataSaveDirectoryDialog.setWindowTitle("Выбор места сохранения результатов обработки")

        # 8. Диалоговое окно для сохранения модели (выбор папки и имени файла)
        self.chooseModelSaveDirectoryDialog = QFileDialog(self.saveDataButton)
        self.chooseModelSaveDirectoryDialog.setFileMode(QFileDialog.FileMode.Directory)
        self.chooseModelSaveDirectoryDialog.setDirectory("C:\\")
        self.chooseModelSaveDirectoryDialog.setWindowTitle("Выбор места сохранения результатов обработки")

        self.ResetState()


    def OnImageProcessingBegin(self):
        self.ResetState()


    def OnImageProcessingCompleted(self, prediction_data):
        self.viewResultButton.setDisabled(False)
        self.saveModelButton.setDisabled(False)
        self.saveDataButton.setDisabled(False)
        self.prediction_data = prediction_data

        #self.viewingThread.start()
        shape_data, vertices, albedo_data, projection_data, converted_projection_data, \
        lightning_data, synthesized_image, synthesized_image_mask, \
        unwarped_texture, unwarped_shading, unwarped_normal, rotated_vertices = self.prediction_data
        self.modelViewer.SetData(vertices = vertices, texture = unwarped_texture, projection = converted_projection_data)


    def ResetState(self):
        self.prediction_data = None
        self.modelViewer.SetData(None, None, None)
        self.viewResultButton.setDisabled(True)
        self.saveModelButton.setDisabled(True)
        self.saveDataButton.setDisabled(True)

        #self.viewingThread.quit()


    def OnShowModelButtonClicked(self, cheched):
        if (self.prediction_data is None):
            print("Нет предсказанных данных, демонстрация невозможна.")
            return None

        #self.viewingThread.start()
        self.modelViewer.run()


    def OnSaveModelButtonClicked(self, checked):
        if (self.prediction_data is None):
            print("Невозможно сохранить модель, поскольку нет данных о ней")
            return
        model_format = MODEL_SAVE_FORMATS[self.modelFormatOptionsBox.currentIndex()]

        shape_data, vertices, albedo_data, projection_data, converted_projection_data, \
        lightning_data, synthesized_image, synthesized_image_mask, \
        unwarped_texture, unwarped_shading, unwarped_normal, rotated_vertices = self.prediction_data

        trimesh_model = Volumeer2.Load3DMM(learning_const.BASE_MODEL_DATA_DIRECTORY)
        if (self.saveTextureCheckBox.checkState() == Qt.CheckState.Checked):
            trimesh_model.visual = visual.TextureVisuals(uv = trimesh_model.visual.uv, image = unwarped_texture)

        trimesh_model.vertices = vertices

        trimesh_model.vertex_normals = compute_vertex_normals(vertices=vertices, indices=trimesh_model.faces)
        trimesh_model.fix_normals()

        #trimesh_model.show()

        saving_filename = QFileDialog.getSaveFileName(parent = self.mainWindow, dir = "C:\\", filter = f"3D Model Files ( *.{model_format})")[0]
        if (saving_filename is None or saving_filename == ""):
            print("Не выбран файл для сохранения, выходим")
            return

        trimesh_model.export(file_obj=saving_filename)


    def OnSaveDataButtonClicked(self, checked):
        if (self.prediction_data is None):
            print("Невозможно сохранить модель, поскольку нет данных о ней")
            return

        shape_data, vertices, albedo_data, projection_data, converted_projection_data, \
        lightning_data, synthesized_image, synthesized_image_mask, \
        unwarped_texture, unwarped_shading, unwarped_normal, rotated_vertices = self.prediction_data

        saving_directory = QFileDialog.getExistingDirectory(parent = self.mainWindow, dir = "C:\\") + "/"
        print(saving_directory)

        filename_prefix = f"{localtime().tm_mday}-{localtime().tm_mon}-{localtime().tm_year}_{localtime().tm_hour}!{localtime().tm_min}.{localtime().tm_sec} "

        array_to_img(shape_data).save(fp = saving_directory + filename_prefix + "volume_2d_map.png", format = "png")
        array_to_img(shape_data / linalg.norm(shape_data, axis = 2, keepdims = True)).save(fp = saving_directory + filename_prefix + "volume_2d_map_normalized.png", format = "png")
        array_to_img(albedo_data).save(fp = saving_directory + filename_prefix + "albedo_2d_map.png", format = "png")
        array_to_img(albedo_data / linalg.norm(albedo_data, axis = 2, keepdims = True)).save(fp = saving_directory + filename_prefix + "albedo_2d_map_normalized.png", format = "png")
        array_to_img(unwarped_shading).save(fp = saving_directory + filename_prefix + "shading_2d_map.png", format = "png")
        array_to_img(unwarped_normal).save(fp = saving_directory + filename_prefix + "normals_2d_map.png", format = "png")
        array_to_img(synthesized_image).save(fp = saving_directory + filename_prefix + "reconstructed_image.png", format = "png")
        array_to_img(synthesized_image_mask).save(fp = saving_directory + filename_prefix + "reconstructed_image_mask.png", format = "png")
        array_to_img(unwarped_texture).save(fp = saving_directory + filename_prefix + "reconstructed_texture.png", format = "png")
