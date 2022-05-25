import typing

from PySide6.QtWidgets import QPushButton, QProgressBar, QTextEdit, QPlainTextEdit
from PySide6.QtCore import Qt, QSize, QThread, Signal, QObject
from PySide6.QtGui import QImage
from processing.ImageProcessor import ImageProcessor
from PIL import Image


class ImageProcessorWidgetHandler(QObject):
    processingThread = QThread()

    onImageProcessingBegin = Signal()
    onImageProcessingCompleted = Signal(tuple)


    def __init__(self, mainWindow):
        '''
        Инициализация виджетов окна, относящихся к работе с нейросетью
        '''
        # 1. Шкала прогресса обработки изображения
        # Пустует при входе в приложение, активируется при нажатии кнопки обработки изображения,
        # заполняется по мере получения сообщений, сбрасывается после нажатия перехода к просмотру результата
        super(ImageProcessorWidgetHandler, self).__init__()

        self.mainWindow = mainWindow
        self.mainWindow.onImageSelectionReset.connect(self.ResetState)

        self.imageProcessingProgressBar = QProgressBar()
        self.imageProcessingProgressBar.setValue(0.0)
        self.imageProcessingProgressBar.setOrientation(Qt.Orientation.Vertical)
        self.imageProcessingProgressBar.setMinimumSize(QSize(20, 200))
        self.imageProcessingProgressBar.setMaximumSize(QSize(20, 1200))

        # 2. Текстовый лог с сообщениями о стадии обработки
        # По аналогии со шкалой: сначала пустой, потом заполняется сообщениями, потом сбрасывается
        self.imageProcessingLog = QPlainTextEdit()
        self.imageProcessingLog.setMinimumSize(QSize(120, 200))
        self.imageProcessingLog.setMaximumSize(QSize(800, 1200))

        # 3. Кнопка для перехода к просмотру 3D-модели
        # Становится активной только после завершения обработки изображения и получения результата
        self.viewerOpenButton = QPushButton()
        self.viewerOpenButton.setDisabled(True)

        # Здесь же держим объект обработчика, чтобы вызывать его в нужные моменты
        self.imageProcessor = ImageProcessor()
        self.imageProcessor.logHandler.onLogReceived.connect(self.OnProcessLogReceived)
        self.imageProcessor.onProcessingCompleted.connect(self.OnImageProcessingCompleted)
        self.imageProcessor.moveToThread(self.processingThread)
        self.processingThread.started.connect(self.imageProcessor.run)
        self.imageProcessor.onFinished.connect(self.processingThread.quit)
        #self.imageProcessor.finished.connect(self.imageProcessor.deleteLater)
        #self.processingThread.finished.connect(self.processingThread.deleteLater)


    def ProcessImage(self, fileName : str):
        '''
        Запуск процесса обработки изображения
        :param fileName: Путь до файла
        :return:
        '''
        try:
            img = Image.open(fileName)
        except FileNotFoundError:
            return


        self.imageProcessor.img = img
        self.processingThread.start()
        self.onImageProcessingBegin.emit()


    def OnProcessLogReceived(self, message : str, progress : float):
        self.imageProcessingLog.appendPlainText(message)
        self.imageProcessingProgressBar.setValue(int(progress * 100))


    def OnImageProcessingCompleted(self, predicted_data):
        self.onImageProcessingCompleted.emit(predicted_data)
        self.imageProcessingProgressBar.setValue(0.0)
        self.imageProcessingLog.clear()


    def ResetState(self):
        self.viewerOpenButton.setDisabled(True)


    def __del__(self):
        try:
            self.processingThread.quit()
            self.processingThread.wait()
        except:
            pass