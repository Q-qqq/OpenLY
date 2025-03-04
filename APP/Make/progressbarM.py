
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ultralytics.utils import PROGRESS_BAR
from APP.Design import progressbarQT_ui


class ProgressBar(QWidget, progressbarQT_ui.Ui_Progress):
    """
        进度条类
    Attributes:
        is_show(bool):  是否正在显示进度条
        stop(bool):  进度条是否停止
    """
    def __init__(self, parent=None, f=Qt.Dialog):
        super().__init__(parent, f)
        self.setupUi(self)
        self.eventConnect()
        self.is_show = False
        self.stop = False

    def eventConnect(self):
        """信号槽连接"""
        PROGRESS_BAR.Start_Signal.connect(self.start, Qt.ConnectionType.DirectConnection)
        PROGRESS_BAR.Set_Value_Signal.connect(self.setValue, Qt.ConnectionType.DirectConnection)
        PROGRESS_BAR.Reset_Signal.connect(self.reset,Qt.ConnectionType.DirectConnection)
        PROGRESS_BAR.Close_Signal.connect(self.close, Qt.ConnectionType.DirectConnection)


    def reset(self):
        """重置进度条"""
        self.ProgressBar.reset()

    def setValue(self, value, text):
        """
        设置进度条的值
        Args:
            value(int):  进度条的值
            text(str):  显示的文本
        """
        self.ProgressBar.setValue(value)
        if text != "":
            self.Show_mes_te.append(text)
        QApplication.processEvents()

    def start(self, title, head_txt, range):
        """开始进度条
        Args:
            range(list):  进度条的范围
            title(str):  进度条的标题
            head_txt(str):  进度条的头部文本
        """
        self.setWindowTitle(title)
        if not self.is_show:
            self.is_show = True
            self.Show_mes_te.clear()
            self.Show_mes_te.append(head_txt)
            self.show()
        self.ProgressBar.setRange(range[0], range[1])
        self.ProgressBar.setValue(range[0])
        self.ProgressBar.setFormat(f"%v/{range[1]}")

    def closeEvent(self, event:QCloseEvent) -> None:
        """关闭进度条"""
        if self.ProgressBar.value() != self.ProgressBar.maximum():
            if PROGRESS_BAR.permit_stop and not PROGRESS_BAR.isStop():  #允许停止且不在停止状态
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
