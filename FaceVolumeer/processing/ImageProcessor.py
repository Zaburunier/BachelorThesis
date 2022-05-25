import logging
import time
import typing

from PySide6.QtCore import QThread, Signal, Slot, QObject
from PIL import Image
from network.NetworkFacade import NetworkFacade
from tensorflow import config



class LogHandler(QObject, logging.StreamHandler):
    TOTAL_LOGS_AMOUNT = 15
    onLogReceived = Signal(str, float)
    onProcessingCompleted = Signal()

    def __init__(self, stream = None):
        super(LogHandler, self).__init__(stream)
        self.logCounter = 0


    def emitSignal(self, record) -> None:
        #super(LogHandler, self).emit(record)
        self.logCounter += 1
        self.onLogReceived.emit(record, self.logCounter / float(self.TOTAL_LOGS_AMOUNT))
        if (self.logCounter == self.TOTAL_LOGS_AMOUNT):
            self.onProcessingCompleted.emit()
            self.logCounter = 0



class ImageProcessor(QObject):
    onProcessingCompleted = Signal(tuple)
    onFinished = Signal()

    def __init__(self):
        super(ImageProcessor,self).__init__()
        config.run_functions_eagerly(True)
        self.processor = NetworkFacade()
        self.logHandler = LogHandler()
        self.processor.GetInfoLogger().addHandler(self.logHandler)
        self.img = None


    def run(self):
        # to do
        if self.img is None:
            self.logHandler.emitSignal("Невозможно начать обработку из-за отсутствия данных")
            return
        else:
            self.logHandler.emitSignal("Изображение принято на обработку")
            print(":)")
            prediction_data = self.processor.PassSingleImage(self.img)
            self.onProcessingCompleted.emit(prediction_data)
            self.onFinished.emit()