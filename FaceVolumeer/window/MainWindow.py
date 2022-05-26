from PySide6.QtWidgets import QMainWindow, QWidget
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal

from window.ImageLoaderWidgetHandler import ImageLoaderWidgetHandler
from window.ImageProcessorWidgetHandler import ImageProcessorWidgetHandler
from window.ModelViewerWidgetHandler import ModelViewerWidgetHandler


# Класс базового окна приложения
class MainWindow(QMainWindow):
    onImageSelectionReset = Signal()


    def __init__(self):
        super(MainWindow, self).__init__()

        self.setCentralWidget(QWidget(parent = self))
        # Инициализируем блоки приложения
        self.imageLoader = ImageLoaderWidgetHandler(self)
        self.imageProcessor = ImageProcessorWidgetHandler(self)
        self.modelViewer = ModelViewerWidgetHandler(self)

        # Добавляем все виджеты на наше окно
        self.SetupWindow()

        # Соединяем все части программы
        self.ConnectSlotsAndSignals()

    def resizeEvent(self, event):
        #print("resize")
        self.imageLoader.OnWindowResize()
        QMainWindow.resizeEvent(self, event)


    # Настройка базового вида окна
    def SetupWindow(self):
        # Настраиваем внешний вид: размер, заголовок, иконка
        self.resize(1280, 720)
        self.setMinimumWidth(960)
        self.setMinimumHeight(540)
        self.setMaximumWidth(1600)
        self.setMaximumHeight(900)
        self.setWindowTitle("FaceVolumeer")
        icon = QIcon("resources/eye_icon.png")
        self.setWindowIcon(icon)

        # Начинаем добавление виджетов
        # Конфигурируем сетку виджетов главного окна
        self.outerLayout = QHBoxLayout(parent = self.centralWidget())

        self.imageLoaderLayout = QVBoxLayout()
        self.imageLoaderLayout.addSpacing(30)
        self.imageLoaderLayout.addWidget(self.imageLoader.processImageButton, 1)
        self.imageLoaderLayout.addWidget(self.imageLoader.imageAreaLabel, 2)
        self.imageLoaderLayout.addWidget(self.imageLoader.imageFilenameLabel, 1)
        self.imageLoaderLayout.addWidget(self.imageLoader.loadImageButton, 1)
        self.imageLoaderLayout.addSpacing(30)

        self.imageProcessorLayout = QVBoxLayout()
        logProgressLayout = QHBoxLayout()
        logProgressLayout.addSpacing(30)
        logProgressLayout.addWidget(self.imageProcessor.imageProcessingProgressBar, 3)
        logProgressLayout.addSpacing(20)
        logProgressLayout.addWidget(self.imageProcessor.imageProcessingLog, 5)
        logProgressLayout.addSpacing(30)
        self.imageProcessorLayout.addSpacing(30)
        self.imageProcessorLayout.addLayout(logProgressLayout, 1)
        self.imageProcessorLayout.addSpacing(30)
        self.imageProcessorLayout.addWidget(self.imageProcessor.viewerOpenButton, 1)
        self.imageProcessorLayout.addSpacing(30)

        self.modelViewerLayout = QVBoxLayout()
        self.modelViewerLayout.addSpacing(30)
        self.modelViewerLayout.addWidget(self.modelViewer.logoLabel, 0)
        self.modelViewerLayout.addStretch(2)
        self.modelViewerLayout.addWidget(self.modelViewer.viewResultButton, 1)
        self.modelViewerLayout.addWidget(self.modelViewer.includeProjectionCheckBox, 1)
        self.modelViewerLayout.addWidget(self.modelViewer.includeTextureCheckBox, 1)
        self.modelViewerLayout.addStretch(2)
        self.modelViewerLayout.addWidget(self.modelViewer.modelFormatOptionsBox, 1)
        self.modelViewerLayout.addWidget(self.modelViewer.saveModelButton, 1)
        self.modelViewerLayout.addSpacing(5)
        self.modelViewerLayout.addWidget(self.modelViewer.saveTextureCheckBox, 1)
        self.modelViewerLayout.addSpacing(10)
        self.modelViewerLayout.addStretch(1)
        self.modelViewerLayout.addSpacing(10)
        self.modelViewerLayout.addWidget(self.modelViewer.saveDataButton, 1)
        self.modelViewerLayout.addSpacing(30)

        self.outerLayout.addSpacing(30)
        self.outerLayout.addLayout(self.imageLoaderLayout, 3)
        self.outerLayout.addStretch(1)
        self.outerLayout.addLayout(self.imageProcessorLayout, 5)
        self.outerLayout.addStretch(1)
        self.outerLayout.addLayout(self.modelViewerLayout, 3)
        self.outerLayout.addSpacing(30)


        self.centralWidget().setLayout(self.outerLayout)


    def ConnectSlotsAndSignals(self):
        # Идём по порядку:
        # 1. Связываем события выбора изображения
        #self.imageLoader.loadImageButton.clicked.connect()
        self.imageLoader.processImageButton.clicked.connect(self.OnProcessImageButtonClicked)
        self.imageLoader.onImageSelectionChanged.connect(self.OnImageSelectionChanged)
        self.imageProcessor.onImageProcessingCompleted.connect(self.modelViewer.OnImageProcessingCompleted)
        self.imageProcessor.onImageProcessingBegin.connect(self.modelViewer.OnImageProcessingBegin)


    def OnImageSelectionChanged(self, fileName):
        if (fileName is None or fileName == ""):
            self.onImageSelectionReset.emit()


    def OnProcessImageButtonClicked(self):
        self.imageProcessor.ProcessImage(self.imageLoader.selectedFileName)





