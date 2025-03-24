from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ultralytics.utils import PROGRESS_BAR

from APP.Designer.DesignerPy import progressbarUI


class ProgressBar(QWidget, progressbarUI.Ui_Progress):
    def __init__(self, parent=None, f=Qt.Dialog):
        super().__init__(parent, f)
        self.setupUi(self)
        self.min = 0
        self.max = 100
        self.ProgressBar.setRange(self.min, self.max)
        self.eventConnect()
        self.is_show = False
        self.stop = False

    def eventConnect(self):
        PROGRESS_BAR.Start_Signal.connect(self.start)
        PROGRESS_BAR.Set_Value_Signal.connect(self.setValue)#, Qt.ConnectionType.DirectConnection)
        PROGRESS_BAR.Reset_Signal.connect(self.reset)
        PROGRESS_BAR.Show_Signal.connect(self.showProgress)
        PROGRESS_BAR.Close_Signal.connect(self.close)

    def reset(self):
        self.ProgressBar.reset()

    def setValue(self, mes):
        value = mes[0]
        text = mes[1]
        self.ProgressBar.setValue(value)
        if text != "":
            self.Show_mes_te.append(text)
        QApplication.processEvents()

    def showProgress(self, mes) -> None:
        title, head_txt = mes
        self.setWindowTitle(title)
        if self.is_show:
            return
        else:
            self.is_show = True
            self.Show_mes_te.clear()
            self.Show_mes_te.append(head_txt)
            self.show()

    def start(self, range):
        self.min = range[0]
        self.max = range[1]
        self.ProgressBar.setRange(self.min, self.max)
        self.ProgressBar.setValue(self.min)
        self.ProgressBar.setFormat(f"%v/{self.max}")

    def closeEvent(self, event:QCloseEvent) -> None:
        if self.ProgressBar.value() != self.max:
            if PROGRESS_BAR.permit_stop and not PROGRESS_BAR.isStop():
                req = QMessageBox.information(self, "提示", "是否中断加载", QMessageBox.Yes | QMessageBox.No)
                if req == QMessageBox.Yes:
                    PROGRESS_BAR.stop()
                    event.accept()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            self.is_show = False
            PROGRESS_BAR.stop()
            event.accept()
