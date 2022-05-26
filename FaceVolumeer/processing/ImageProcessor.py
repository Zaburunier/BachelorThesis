import logging
import sys
import time
import typing

from PySide6.QtCore import QThread, Signal, Slot, QObject
from PIL import Image
from network.NetworkFacade import NetworkFacade
from tensorflow import config


class LogHandlerWrap(QObject):
    TOTAL_LOGS_AMOUNT = 16
    onLogReceived = Signal(str, float)
    onProcessingCompleted = Signal()


    def __init__(self):
        super(LogHandlerWrap, self).__init__()
        self.logHandler = LogHandler(sys.stdout, self)
        self.logCounter = 0


    def onSignalEmitted(self, record) -> None:
        #super(LogHandlerWrap, self).emit(record)
        self.logCounter += 1
        self.onLogReceived.emit(record.msg, self.logCounter / float(self.TOTAL_LOGS_AMOUNT))
        if (self.logCounter == self.TOTAL_LOGS_AMOUNT):
            self.onProcessingCompleted.emit()
            self.logCounter = 0


class LogHandler(logging.StreamHandler):

    def __init__(self, stream, wrapper : LogHandlerWrap):
        super(LogHandler, self).__init__(stream)
        self.setLevel("INFO")
        self.logWrapper = wrapper


    def emit(self, record: logging.LogRecord) -> None:
        super(LogHandler, self).emit(record)

        self.logWrapper.onSignalEmitted(record)



class ImageProcessor(QObject):
    onProcessingCompleted = Signal(tuple)
    onFinished = Signal()

    def __init__(self):
        super(ImageProcessor,self).__init__()
        config.run_functions_eagerly(True)
        self.processor = NetworkFacade()
        self.logHandlerWrap = LogHandlerWrap()
        self.processor.GetInfoLogger().addHandler(self.logHandlerWrap.logHandler)
        self.img = None


    def run(self):
        # to do
        if self.img is None:
            self.logHandlerWrap.onSignalEmitted(logging.LogRecord("", 0, "", 0, "Невозможно начать обработку из-за отсутствия данных\n", None, None))
            return
        else:
            self.logHandlerWrap.onSignalEmitted(logging.LogRecord("", 0, "", 0, "Изображение принято на обработку\n", None, None))
            #print(":)")
            prediction_data = self.processor.PassSingleImage(self.img)
            self.logHandlerWrap.onSignalEmitted(
                logging.LogRecord("", 0, "", 0, "\nГотово!\n", None, None))
            self.onProcessingCompleted.emit(prediction_data)
            self.onFinished.emit()