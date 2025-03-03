import gc


from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from APP import APP_ROOT, loadQssStyleSheet, init_progress_effect
from APP.Make import Start, Train, ProgressBar

def handle_exception(exc_type, exc_value, exc_traceback):
    """异常处理函数"""
    message = f"An exception of type {exc_type.__name__} occurred.\n{exc_value}\n{exc_traceback.tb_frame}"
    QMessageBox.critical(None, "Error", message)


if __name__ == "__main__":
    import sys
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)

    train_ui = Train(app)
    loadQssStyleSheet(app, train_ui)
    train_ui.show()
    progress_bar = ProgressBar(train_ui)
    init_progress_effect(progress_bar.ProgressBar)
    sys.exit(app.exec_())
