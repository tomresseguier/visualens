import sys
from PyQt5.QtWidgets import QApplication
from .visualens import Visualens


def main() :
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    vl = Visualens()

    sys.exit(app.exec())


if __name__ == "__main__" :
    main()
