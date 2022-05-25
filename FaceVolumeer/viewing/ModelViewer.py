from PySide6.QtCore import QThread, Signal, Slot, QObject, Qt
from PySide6.QtWidgets import QCheckBox
from visualizer import ModelVisualizer
import pyglet.app


class ModelViewer(QObject):
    onFinished = Signal()

    def __init__(self, projectionCheckBox : QCheckBox, textureCheckBox : QCheckBox):
        super(ModelViewer, self).__init__()

        self.projectionCheckBox = projectionCheckBox
        self.textureCheckBox = textureCheckBox

        self.vertices = None
        self.texture = None
        self.projection = None


    def SetData(self, vertices, texture = None, projection = None):
        self.vertices = vertices
        self.texture = texture
        self.projection = projection


    def run(self):
        if (self.vertices is None):
            print("Нет данных для визуализации, возврат...")
            return

        ModelVisualizer.VisualizeFromNetworkData(shape = self.vertices.numpy(),
                                                 projection = self.projection.numpy() if self.projectionCheckBox.isChecked() == True else None,
                                                 texture = self.texture.numpy() if self.textureCheckBox.isChecked() == True else None)

        self.onFinished.emit()