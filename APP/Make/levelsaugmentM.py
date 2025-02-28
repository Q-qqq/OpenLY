import copy

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import cv2
import numpy as np

from ultralytics.utils import cv2_readimg

from APP.Designer.DesignerPy import levels_augmentUI
from APP.Utils.plotting import HistFigure



class LevelsAugment(QWidget, levels_augmentUI.Ui_Form):
    def __init__(self, parent,img_label, f = Qt.Tool):
        super().__init__(parent, f)
        self.setupUi(self)
        self.initUi()
        self.img_label = img_label
        self.rgb_levels = [0, 1.0, 255, 0, 255] #输入黑场阈值，输入灰场值，输入白场阈值，输出黑场阈值，输出白场阈值，
        self.r_levels = [0, 1.0, 255, 0, 255]
        self.g_levels = [0, 1.0, 255, 0, 255]
        self.b_levels = [0, 1.0, 255, 0, 255]
        self.channel = "RGB"
        self.eventConnect()


    def initUi(self):
        self.hist_figure = HistFigure()
        self.hist_show_gl = QGridLayout(self.Hist_show_F)
        self.hist_show_gl.setObjectName("hist_show_gl")
        self.hist_show_gl.setMargin(0)
        self.hist_show_gl.addWidget(self.hist_figure,0,0)

    def eventConnect(self):
        self.In_shadow_hs.valueChanged.connect(lambda :self.hsChangedSlot(self.In_shadow_hs))
        self.In_gray_hs.valueChanged.connect(lambda: self.hsChangedSlot(self.In_gray_hs))
        self.In_light_hs.valueChanged.connect(lambda: self.hsChangedSlot(self.In_light_hs))
        self.Out_shadow_hs.valueChanged.connect(lambda: self.hsChangedSlot(self.Out_shadow_hs))
        self.Out_light_hs.valueChanged.connect(lambda: self.hsChangedSlot(self.Out_light_hs))

        self.In_shadow_sb.editingFinished.connect(lambda: self.sbChangedSlot(self.In_shadow_sb))
        self.In_gray_dsb.editingFinished.connect(lambda: self.sbChangedSlot(self.In_gray_dsb))
        self.In_light_sb.editingFinished.connect(lambda: self.sbChangedSlot(self.In_light_sb))
        self.Out_shadow_sb.editingFinished.connect(lambda: self.sbChangedSlot(self.Out_shadow_sb))
        self.Out_light_sb.editingFinished.connect(lambda: self.sbChangedSlot(self.Out_light_sb))

        self.Channels_cbb.currentTextChanged.connect(self.channelChangedSlot)
        self.Hide_augment_cb.stateChanged.connect(self.hideAugmentSlot)
        self.Init_pb.clicked.connect(self.initClicked)

    def show(self) -> None:
        self.setHSLevels(self.rgb_levels)
        self.hist_figure.plot(self.img_label.img, self.channel)
        super().show()

    def getLevels(self, channel):
        levels = [0, 1.0, 255, 0, 255]
        if channel == "RGB":
            levels = self.rgb_levels
        elif channel == "R":
            levels = self.r_levels
        elif channel == "G":
            levels = self.g_levels
        elif channel == "B":
            levels = self.b_levels
        return levels

    def setHSLevels(self, levels):

        self.In_shadow_hs.setValue(levels[0])
        self.In_gray_hs.setValue(levels[1]*100)
        self.In_light_hs.setValue(levels[2])
        self.Out_shadow_hs.setValue(levels[3])
        self.Out_light_hs.setValue(levels[4])

    def setLevels(self):
        levels = [
            self.In_shadow_hs.value(),
            self.In_gray_hs.value() / 100,
            self.In_light_hs.value(),
            self.Out_shadow_hs.value(),
            self.Out_light_hs.value()
        ]
        if self.channel == "RGB":
            self.rgb_levels = levels
        elif self.channel == "R":
            self.r_levels = levels
        elif self.channel == "G":
            self.g_levels = levels
        elif self.channel == "B":
            self.b_levels = levels

    def sbChangedSlot(self, sb):
        if sb == self.In_shadow_sb:
            self.In_shadow_hs.setValue(sb.value())
        elif sb == self.In_gray_dsb:
            self.In_gray_hs.setValue(sb.value()*100)
        elif sb == self.In_light_sb:
            self.In_light_hs.setValue(sb.value())
        elif sb == self.Out_shadow_sb:
            self.Out_shadow_sb.setValue(sb.value())
        elif sb == self.Out_light_sb:
            self.Out_light_sb.setValue(sb.value())
        self.setLevels()
        self.levelsCompute()


    def hsChangedSlot(self, hs):
        if hs == self.In_shadow_hs:
            self.In_shadow_sb.setValue(hs.value())
        elif hs == self.In_gray_hs:
            self.In_gray_dsb.setValue(hs.value()/100)
        elif hs == self.In_light_hs:
            self.In_light_sb.setValue(hs.value())
        elif hs == self.Out_shadow_hs:
            self.Out_shadow_sb.setValue(hs.value())
        elif hs == self.Out_light_hs:
            self.Out_light_sb.setValue(hs.value())
        self.setLevels()
        self.levelsCompute()


    def channelChangedSlot(self):
        self.setLevels()
        self.channel = self.Channels_cbb.currentText()
        self.hist_figure.plot(self.img_label.img, self.channel)
        levels = self.getLevels(self.channel)
        self.setHSLevels(levels)
        self.levelsCompute()

    def hideAugmentSlot(self):
        if self.Hide_augment_cb.isChecked():
            self.img_label.levels_img = None
            self.img_label.update()
        else:
            self.levelsCompute()

    def initClicked(self):
        self.setHSLevels([0,1.0,255,0,255])
        self.rgb_levels = [0, 1.0, 255, 0, 255]
        self.r_levels = [0, 1.0, 255, 0, 255]
        self.g_levels = [0, 1.0, 255, 0, 255]
        self.b_levels = [0, 1.0, 255, 0, 255]

    def levelsCompute(self):
        def table_Compute(levels):
            Sin = levels[0]
            Mt = levels[1]
            Hin = levels[2]
            Sout = levels[3]
            Hout = levels[4]
            Sin = min(max(Sin, 0), Hin - 2)  # Sin, 黑场阈值, 0<=Sin<Hin
            Hin = min(Hin, 255)  # Hin, 白场阈值, Sin<Hin<=255
            Mt = min(max(Mt, 0.01), 9.99)  # Mt, 灰场调节值, 0.01~9.99
            Sout = min(max(Sout, 0), Hout - 2)  # Sout, 输出黑场阈值, 0<=Sout<Hout
            Hout = min(Hout, 255)  # Hout, 输出白场阈值, Sout<Hout<=255
            difIn = Hin - Sin
            difOut = Hout - Sout
            table = np.zeros(256, np.uint16)
            for i in range(256):
                V1 = min(max(255 * (i - Sin) / difIn, 0), 255)  # 输入动态线性拉伸
                V2 = 255 * np.power(V1 / 255, 1 / Mt)  # 灰场伽马调节
                table[i] = min(max(Sout + difOut * V2 / 255, 0), 255)  # 输出线性拉伸
            return table

        imgTone = copy.deepcopy(self.img_label.img)
        if self.channel == "RGB":
            table = table_Compute(self.rgb_levels)
            imgTone = cv2.LUT(imgTone, table)
        else:
            table_b = table_Compute(self.b_levels)
            table_g = table_Compute(self.g_levels)
            table_r = table_Compute(self.r_levels)
            imgTone[:, :, 0] = cv2.LUT(imgTone[:, :, 0], table_b)
            imgTone[:, :, 1] = cv2.LUT(imgTone[:, :, 1], table_g)
            imgTone[:, :, 2] = cv2.LUT(imgTone[:, :, 2], table_r)

        if not self.Hide_augment_cb.isChecked():
            self.img_label.levels_img = imgTone.astype(np.uint8).copy()
            self.img_label.update()


    def closeEvent(self, event:QCloseEvent) -> None:
        self.img_label.levels_img = None
        self.img_label.update()
        super().closeEvent(event)
