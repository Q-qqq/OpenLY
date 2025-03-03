from pathlib import Path
from multiprocessing.pool import ThreadPool
import threading

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ultralytics.utils import NUM_THREADS,ThreadingLocked

from APP  import PROJ_SETTINGS
from APP.Utils.label import ShowLabel
from APP.Data import format_im_files
from APP.Utils import get_widget


class ImageScroll(QScrollArea):
    """供选择图像的滚动界面
    Args:
        hor_layout(QHBoxLayout): 水平布局，用于存放图像GroupBox
        groupBox(QGroupBox)：图像容器，容纳图像label
    """

    def __init__(self, parent, images_label):
        super().__init__(parent)
        self.images_label = images_label
        self.images_label.setMinimumHeight(self.height() - 15)
        self.setHLayout()
        self.setSmoothAnimal()

    def setHLayout(self):
        self.setContentsMargins(0, 0, 0, 0)
        self.hor_layout = QHBoxLayout()
        self.hor_layout.setMargin(0)
        self.hor_layout.setSpacing(5)
        self.widget = QWidget()
        self.widget.setContentsMargins(0,0 ,0 ,0)
        self.widget.setLayout(self.hor_layout)
        self.setWidget(self.widget)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignLeft)
        self.verticalScrollBar().setEnabled(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalScrollBar().setEnabled(True)
        self.horizontalScrollBar().valueChanged.connect(self.horBarValueChangedSlot)
        self.hor_layout.addWidget(self.images_label)

    def setSmoothAnimal(self):
        self.animal = QPropertyAnimation(self.horizontalScrollBar(), b"value", self)
        self.animal.setEasingCurve(QEasingCurve.OutCubic)
        self.animal.setDuration(100)


    def nextImage(self):
        show_file = self.images_label.im_files[self.images_label.current_selected]
        show_ind = self.images_label.show_files.index(show_file)
        if show_ind < len(self.images_label.show_files)-1:
            next_file = self.images_label.show_files[show_ind+1]
            all_ind = self.images_label.im_files.index(next_file)
            self.images_label.clickImage(all_ind)
            v = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(min(v + self.images_label.widths[all_ind] + self.images_label.spacing, self.horizontalScrollBar().maximum()))

    def lastImage(self):
        show_file = self.images_label.im_files[self.images_label.current_selected]
        show_ind = self.images_label.show_files.index(show_file)
        if show_ind > 0:
            last_file = self.images_label.show_files[show_ind - 1]
            all_ind = self.images_label.im_files.index(last_file)
            self.images_label.clickImage(all_ind)
            v = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(max(0,v - self.images_label.widths[all_ind] - self.images_label.spacing))



    def resizeEvent(self, arg__1:QResizeEvent) -> None:
        self.images_label.setMinimumHeight(self.height() - 30)
        self.images_label.setMaximumHeight(self.height() - 30)
        self.images_label.setSize()
        self.images_label.update()
        self.horBarValueChangedSlot()


    def wheelEvent(self, arg__1:QWheelEvent) -> None:
        hor_bar = self.horizontalScrollBar()
        v = hor_bar.value()
        new_v = v - arg__1.angleDelta().y()*2
        new_v = max(min(new_v, hor_bar.maximum()), hor_bar.minimum())
        self.setValue(new_v)

    def setValue(self, value):
        if value == self.horizontalScrollBar().value():
            return

        #停止
        self.animal.stop()
        #开始
        self.animal.setStartValue(self.horizontalScrollBar().value())
        self.animal.setEndValue(value)
        self.animal.start()

    def horBarValueChangedSlot(self):
        value = self.horizontalScrollBar().value()
        self.images_label.loadPix(value, self.width())





