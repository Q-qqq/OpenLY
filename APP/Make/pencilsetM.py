from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from APP.Designer import pencil_setUI

class PencilSet(QWidget, pencil_setUI.Ui_Form):
    def __init__(self, parent, img_label, f=Qt.Tool):
        super().__init__(parent, f)
        self.setupUi(self)
        self.img_label = img_label
        self.eventConnect()

    def eventConnect(self):
        self.Pencil_add_rb.clicked.connect(self.pencilModeChanged)
        self.Pencil_sub_pb.clicked.connect(self.pencilModeChanged)
        self.Pencil_size_hs.valueChanged.connect(self.pencilSizeChanged)

    def show(self) -> None:
        if self.img_label.pencil_mode == "add":
            self.Pencil_add_rb.setChecked(True)
        else:
            self.Pencil_sub_pb.setChecked(True)
        h, w = self.img_label.img.shape[:2]
        self.Pencil_size_hs.setMaximum(max(h,w))
        self.Pencil_size_sb.setMaximum(max(h,w))
        super().show()

    def pencilModeChanged(self):
        if self.Pencil_add_rb.isChecked():
            self.img_label.pencil_mode = "add"
        else:
            self.img_label.pencil_mode = "sub"

    def pencilSizeChanged(self):
        self.img_label.line_width = self.Pencil_size_hs.value()