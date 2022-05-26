import os.path

import const
from PySide6.QtWidgets import QPushButton, QLabel, QFileDialog, QDialog
from PySide6.QtGui import QPixmap, QImage, QFont, QColor
from PySide6.QtCore import QSize, Qt, Signal, QObject

DEFAULT_LABEL_TEXT = "Выберите изображение для продолжения работы"


class ImageLoaderWidgetHandler(QObject):
    onImageSelectionChanged = Signal(str)

    def __init__(self, mainWindow):
        '''
        Иницализация виджетов окна, относящихся к выбору входных данных
        '''
        super(ImageLoaderWidgetHandler, self).__init__()
        self.NO_IMAGE_PIXMAP = QPixmap()
        self.NO_IMAGE_PIXMAP.fill(QColor.fromRgb(163, 216, 255))

        self.mainWindow = mainWindow
        self.selectedFileName = ""
        self.selectedImage = None
        # Во-первых, кнопка загрузки изображения:
        # 1. Привязаться по нажатию на неё к интерфейсу выбора файла
        # 2. При выборе файла сохранить его (и не терять :) );
        # 3. На главном экране отображать название файла;
        # 4. Там же отображать выбранное изображение в уменьшенном виде.
        self.loadImageButton = QPushButton()
        self.loadImageButton.setCheckable(False)
        self.loadImageButton.clicked.connect(self.OnLoadImageButtonClicked)
        self.loadImageButton.setMinimumSize(QSize(220, 50))
        self.loadImageButton.setMaximumSize(QSize(280, 50))
        self.loadImageButton.setText("Выбрать...")
        self.loadImageButton.setFont(const.BUTTON_TEXT_FONT)

        # Здесь же храним настроенное диалоговое окно для выбора изображения
        self.chooseImageDialog = QFileDialog(self.loadImageButton)
        self.chooseImageDialog.setFileMode(QFileDialog.ExistingFile)
        self.chooseImageDialog.setDirectory("C:\\")
        self.chooseImageDialog.setWindowTitle("Выбор входных данных")
        self.chooseImageDialog.setNameFilter("Image Files (*.png *.jpg)")

        # Во-вторых, текстовое поле с названием выбранного файла:
        # 1. В отсутствие изображения мы держим placeholder;
        # 2. При выборе изображения мы переключаемся на название файла.
        self.imageFilenameLabel = QLabel(parent = self.mainWindow)
        self.imageFilenameLabel.setText(DEFAULT_LABEL_TEXT)
        self.imageFilenameLabel.setFont(QFont("Arial", pointSize = 11, weight = 1, italic = False))
        self.imageFilenameLabel.setStyleSheet(const.IMAGE_NOT_LOADED_LABEL_STYLESHEET)
        self.imageFilenameLabel.setMinimumSize(QSize(200, 20))
        self.imageFilenameLabel.setMaximumSize(QSize(900, 60))
        self.imageFilenameLabel.setWordWrap(True)

        # В-третьих, окно для предпросмотра изображения.
        # По аналогии с текстовым полем, прицепляемся к событию выбора изображения
        self.imageAreaLabel = QLabel(parent = self.mainWindow)
        self.imageAreaLabel.setMinimumSize(QSize(80, 120))
        self.imageAreaLabel.setMaximumSize(QSize(400, 600))
        #self.imageAreaLabel.setPixmap(NO_IMAGE_PIXMAP)

        # В-четвёртых, кнопка для начала обработки изображения
        # Располагаем здесь, сигнал от нажатия будет через посредника (главное окно) передаваться на следующий этап
        self.processImageButton = QPushButton()
        self.processImageButton.setCheckable(False)
        self.processImageButton.setDisabled(True)
        self.processImageButton.clicked.connect(self.OnProcessImageButtonClicked)
        self.processImageButton.setMinimumSize(QSize(220, 50))
        self.processImageButton.setMaximumSize(QSize(280, 50))
        self.processImageButton.setText("Оценить поверхность ...")
        self.processImageButton.setFont(const.BUTTON_TEXT_FONT)


    def OnLoadImageButtonClicked(self, checkable):
        '''
        Реакция на нажатие пользователем кнопки выбора изображения
        :param checkable: Неиспользуемое свойство кнопки
        :return:
        '''
        # Пользователь отказался выбирать файл, ничего не делаем
        if (self.chooseImageDialog.exec() == QFileDialog.Rejected):
            return

        # Пользователь выбрал файл, проверяем корректность
        selectedFileName = self.chooseImageDialog.selectedFiles()
        # В списке выбранных нет ни одного файла, выходим
        if len(selectedFileName) == 0:
            self.ResetImageSelection()
            return

        # Избавляемся от множественного выбора
        selectedFileName = selectedFileName[0]

        # Пользователь выбрал некорректный файл, выходим
        if selectedFileName.find(".") == -1 or selectedFileName[selectedFileName.find(".") + 1:] not in ["png", "jpg"]:
            errorDialog = QDialog(parent = self.loadImageButton)
            errorDialog.setWindowTitle("ОШИБКА! Некорректный формат")
            errorDialog.open()
            self.ResetImageSelection()
            return

        userImage = QImage(selectedFileName)
        if userImage is None:
            errorDialog = QDialog(parent = self.loadImageButton)
            errorDialog.setWindowTitle("ОШИБКА! Изображение не загружено")
            errorDialog.open()
            self.ResetImageSelection()
            return

        self.imageAreaLabel.setPixmap(QPixmap.fromImage(userImage).scaled(self.imageAreaLabel.size(), Qt.KeepAspectRatio))
        self.imageFilenameLabel.setText(selectedFileName)
        self.imageFilenameLabel.setStyleSheet(const.IMAGE_LOADED_LABEL_STYLESHEET)
        self.processImageButton.setEnabled(True)
        self.onImageSelectionChanged.emit(selectedFileName)
        self.selectedImage = userImage
        self.selectedFileName = selectedFileName



    def ResetImageSelection(self):
        '''
        Метод смены состояния виджетов при сбросе выбора изображения
        :return:
        '''
        #self.imageAreaLabel.setPixmap(QPixmap.)
        self.imageFilenameLabel.setText(DEFAULT_LABEL_TEXT)
        self.imageFilenameLabel.setStyleSheet(const.IMAGE_NOT_LOADED_LABEL_STYLESHEET)
        self.processImageButton.setEnabled(False)
        self.onImageSelectionChanged.emit(None)
        self.selectedImage = None
        self.selectedFileName = ""


    def OnProcessImageButtonClicked(self, checked):
        '''
        Реакция на нажатие пользователем кнопки обработки изображения
        :return:
        '''
        pass


    def OnWindowResize(self):
        if self.selectedImage is None:
            return

        self.imageAreaLabel.setPixmap(QPixmap.fromImage(self.selectedImage).scaled(self.imageAreaLabel.size(), Qt.KeepAspectRatio))

