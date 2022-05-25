from window.MainWindow import MainWindow
from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    application = QApplication()
    window = MainWindow()

    window.show()

    application.exec()

