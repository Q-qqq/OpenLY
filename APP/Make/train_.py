# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'train.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import os
from pathlib import Path
from natsort import ns,natsorted
from PIL import  Image

from ultralytics.others import public_method


import torchvision
from ultralytics.data import dataset
import torch
from ultralytics.others.util import *
import time
import numpy as np
import cv2
from APP.Make import Mylabel,My_other_widght
import onnxruntime


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1430, 881)
        self.actionnew_file = QAction(MainWindow)
        self.actionnew_file.setObjectName(u"actionnew_file")
        self.actionopen_project = QAction(MainWindow)
        self.actionopen_project.setObjectName(u"actionopen_project")
        self.actionsave = QAction(MainWindow)
        self.actionsave.setObjectName(u"actionsave")
        self.actionsave_as = QAction(MainWindow)
        self.actionsave_as.setObjectName(u"actionsave_as")
        self.actionexit = QAction(MainWindow)
        self.actionexit.setObjectName(u"actionexit")
        self.actionback_start = QAction(MainWindow)
        self.actionback_start.setObjectName(u"actionback_start")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_6 = QGridLayout(self.centralwidget)
        self.gridLayout_6.setSpacing(0)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.tabWidget_2 = QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.gridLayout = QGridLayout(self.tab_5)
        self.gridLayout.setObjectName(u"gridLayout")
        self.down_image_pb = QPushButton(self.tab_5)
        self.down_image_pb.setObjectName(u"down_image_pb")
        self.down_image_pb.setMinimumSize(QSize(0, 30))
        font = QFont()
        font.setFamily(u"\u5b8b\u4f53")
        font.setPointSize(12)
        self.down_image_pb.setFont(font)

        self.gridLayout.addWidget(self.down_image_pb, 2, 1, 1, 1)

        self.up_image_pb = QPushButton(self.tab_5)
        self.up_image_pb.setObjectName(u"up_image_pb")
        self.up_image_pb.setMinimumSize(QSize(0, 30))
        self.up_image_pb.setFont(font)

        self.gridLayout.addWidget(self.up_image_pb, 2, 0, 1, 1)

        self.licon_imagePB = QPushButton(self.tab_5)
        self.licon_imagePB.setObjectName(u"licon_imagePB")
        font1 = QFont()
        font1.setFamily(u"Agency FB")
        font1.setPointSize(14)
        font1.setBold(True)
        font1.setWeight(75)
        self.licon_imagePB.setFont(font1)

        self.gridLayout.addWidget(self.licon_imagePB, 0, 0, 1, 2)

        self.frame = QFrame(self.tab_5)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.imageLB = Mylabel.mylabel_only_show_rect(self.frame)
        self.imageLB.setObjectName(u"imageLB")
        self.imageLB.setGeometry(QRect(0, 0, 781, 491))
        self.imageLB.setMaximumSize(QSize(2000, 800))
        self.imageLB.setFrameShape(QFrame.Box)
        self.imageLB.setScaledContents(False)
        self.imageLB.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.frame, 1, 0, 1, 2)

        self.tabWidget_2.addTab(self.tab_5, "")
        self.tab_6 = QWidget()
        self.tab_6.setObjectName(u"tab_6")
        self.gridLayout_8 = QGridLayout(self.tab_6)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.widget = QWidget(self.tab_6)
        self.widget.setObjectName(u"widget")
        self.gridLayout_7 = QGridLayout(self.widget)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.horizontalSpacer = QSpacerItem(684, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer, 0, 0, 1, 1)

        self.updata_RMP_PB = QPushButton(self.widget)
        self.updata_RMP_PB.setObjectName(u"updata_RMP_PB")
        self.updata_RMP_PB.setMinimumSize(QSize(0, 0))
        self.updata_RMP_PB.setMaximumSize(QSize(16777215, 16777215))

        self.gridLayout_7.addWidget(self.updata_RMP_PB, 0, 1, 1, 1)

        self.RMP_GLY = QGridLayout()
        self.RMP_GLY.setObjectName(u"RMP_GLY")

        self.gridLayout_7.addLayout(self.RMP_GLY, 1, 0, 1, 2)


        self.gridLayout_8.addWidget(self.widget, 0, 0, 1, 1)

        self.tabWidget_2.addTab(self.tab_6, "")
        self.tab_7 = QWidget()
        self.tab_7.setObjectName(u"tab_7")
        self.gridLayout_12 = QGridLayout(self.tab_7)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.widget_2 = QWidget(self.tab_7)
        self.widget_2.setObjectName(u"widget_2")
        self.gridLayout_9 = QGridLayout(self.widget_2)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.horizontalSpacer_2 = QSpacerItem(684, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer_2, 0, 0, 1, 1)

        self.updata_per_class_RMP_PB = QPushButton(self.widget_2)
        self.updata_per_class_RMP_PB.setObjectName(u"updata_per_class_RMP_PB")

        self.gridLayout_9.addWidget(self.updata_per_class_RMP_PB, 0, 1, 1, 1)

        self.per_class_RMP_GLY = QGridLayout()
        self.per_class_RMP_GLY.setSpacing(6)
        self.per_class_RMP_GLY.setObjectName(u"per_class_RMP_GLY")

        self.gridLayout_9.addLayout(self.per_class_RMP_GLY, 1, 0, 1, 2)

        self.gridLayout_9.setRowStretch(0, 1)

        self.gridLayout_12.addWidget(self.widget_2, 0, 0, 1, 1)

        self.tabWidget_2.addTab(self.tab_7, "")

        self.gridLayout_5.addWidget(self.tabWidget_2, 0, 0, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)

        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 1, 1, 1)

        self.run_to_lowPB = QPushButton(self.centralwidget)
        self.run_to_lowPB.setObjectName(u"run_to_lowPB")

        self.gridLayout_6.addWidget(self.run_to_lowPB, 2, 3, 1, 1)

        self.makeVL = QVBoxLayout()
        self.makeVL.setSpacing(1)
        self.makeVL.setObjectName(u"makeVL")
        self.makeVL.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.makedatasetPB = QPushButton(self.centralwidget)
        self.makedatasetPB.setObjectName(u"makedatasetPB")
        self.makedatasetPB.setMinimumSize(QSize(0, 30))
        self.makedatasetPB.setFont(font)

        self.makeVL.addWidget(self.makedatasetPB)

        self.makenetPB = QPushButton(self.centralwidget)
        self.makenetPB.setObjectName(u"makenetPB")
        self.makenetPB.setMinimumSize(QSize(0, 30))
        self.makenetPB.setFont(font)

        self.makeVL.addWidget(self.makenetPB)

        self.trainPB = QPushButton(self.centralwidget)
        self.trainPB.setObjectName(u"trainPB")
        self.trainPB.setMinimumSize(QSize(0, 30))
        self.trainPB.setFont(font)

        self.makeVL.addWidget(self.trainPB)

        self.tryPB = QPushButton(self.centralwidget)
        self.tryPB.setObjectName(u"tryPB")
        self.tryPB.setMinimumSize(QSize(0, 30))
        self.tryPB.setFont(font)

        self.makeVL.addWidget(self.tryPB)

        self.save_as_onnxPB = QPushButton(self.centralwidget)
        self.save_as_onnxPB.setObjectName(u"save_as_onnxPB")
        self.save_as_onnxPB.setMinimumSize(QSize(0, 30))
        self.save_as_onnxPB.setFont(font)

        self.makeVL.addWidget(self.save_as_onnxPB)

        self.sort_TW = QTableWidget(self.centralwidget)
        if (self.sort_TW.columnCount() < 2):
            self.sort_TW.setColumnCount(2)
        brush = QBrush(QColor(0, 0, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        __qtablewidgetitem = QTableWidgetItem()
        __qtablewidgetitem.setForeground(brush);
        self.sort_TW.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.sort_TW.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.sort_TW.rowCount() < 20):
            self.sort_TW.setRowCount(20)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(0, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(1, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(2, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(3, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(4, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(5, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(6, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(7, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(8, __qtablewidgetitem10)
        __qtablewidgetitem11 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(9, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(10, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(11, __qtablewidgetitem13)
        __qtablewidgetitem14 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(12, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(13, __qtablewidgetitem15)
        __qtablewidgetitem16 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(14, __qtablewidgetitem16)
        __qtablewidgetitem17 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(15, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(16, __qtablewidgetitem18)
        __qtablewidgetitem19 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(17, __qtablewidgetitem19)
        __qtablewidgetitem20 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(18, __qtablewidgetitem20)
        __qtablewidgetitem21 = QTableWidgetItem()
        self.sort_TW.setVerticalHeaderItem(19, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        __qtablewidgetitem22.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(0, 0, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        __qtablewidgetitem23.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(0, 1, __qtablewidgetitem23)
        __qtablewidgetitem24 = QTableWidgetItem()
        __qtablewidgetitem24.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(1, 0, __qtablewidgetitem24)
        __qtablewidgetitem25 = QTableWidgetItem()
        __qtablewidgetitem25.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(1, 1, __qtablewidgetitem25)
        __qtablewidgetitem26 = QTableWidgetItem()
        __qtablewidgetitem26.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(2, 0, __qtablewidgetitem26)
        __qtablewidgetitem27 = QTableWidgetItem()
        __qtablewidgetitem27.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(2, 1, __qtablewidgetitem27)
        __qtablewidgetitem28 = QTableWidgetItem()
        __qtablewidgetitem28.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(3, 0, __qtablewidgetitem28)
        __qtablewidgetitem29 = QTableWidgetItem()
        __qtablewidgetitem29.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(3, 1, __qtablewidgetitem29)
        __qtablewidgetitem30 = QTableWidgetItem()
        __qtablewidgetitem30.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(4, 0, __qtablewidgetitem30)
        __qtablewidgetitem31 = QTableWidgetItem()
        __qtablewidgetitem31.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(4, 1, __qtablewidgetitem31)
        __qtablewidgetitem32 = QTableWidgetItem()
        __qtablewidgetitem32.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(5, 0, __qtablewidgetitem32)
        __qtablewidgetitem33 = QTableWidgetItem()
        __qtablewidgetitem33.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(5, 1, __qtablewidgetitem33)
        __qtablewidgetitem34 = QTableWidgetItem()
        __qtablewidgetitem34.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(6, 0, __qtablewidgetitem34)
        __qtablewidgetitem35 = QTableWidgetItem()
        __qtablewidgetitem35.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(6, 1, __qtablewidgetitem35)
        __qtablewidgetitem36 = QTableWidgetItem()
        __qtablewidgetitem36.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(7, 0, __qtablewidgetitem36)
        __qtablewidgetitem37 = QTableWidgetItem()
        __qtablewidgetitem37.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(7, 1, __qtablewidgetitem37)
        __qtablewidgetitem38 = QTableWidgetItem()
        __qtablewidgetitem38.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(8, 0, __qtablewidgetitem38)
        __qtablewidgetitem39 = QTableWidgetItem()
        __qtablewidgetitem39.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(8, 1, __qtablewidgetitem39)
        __qtablewidgetitem40 = QTableWidgetItem()
        __qtablewidgetitem40.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(9, 0, __qtablewidgetitem40)
        __qtablewidgetitem41 = QTableWidgetItem()
        __qtablewidgetitem41.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(9, 1, __qtablewidgetitem41)
        __qtablewidgetitem42 = QTableWidgetItem()
        __qtablewidgetitem42.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(10, 0, __qtablewidgetitem42)
        __qtablewidgetitem43 = QTableWidgetItem()
        __qtablewidgetitem43.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(10, 1, __qtablewidgetitem43)
        __qtablewidgetitem44 = QTableWidgetItem()
        __qtablewidgetitem44.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(11, 0, __qtablewidgetitem44)
        __qtablewidgetitem45 = QTableWidgetItem()
        __qtablewidgetitem45.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(11, 1, __qtablewidgetitem45)
        __qtablewidgetitem46 = QTableWidgetItem()
        __qtablewidgetitem46.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(12, 0, __qtablewidgetitem46)
        __qtablewidgetitem47 = QTableWidgetItem()
        __qtablewidgetitem47.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(12, 1, __qtablewidgetitem47)
        __qtablewidgetitem48 = QTableWidgetItem()
        __qtablewidgetitem48.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(13, 0, __qtablewidgetitem48)
        __qtablewidgetitem49 = QTableWidgetItem()
        __qtablewidgetitem49.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(13, 1, __qtablewidgetitem49)
        __qtablewidgetitem50 = QTableWidgetItem()
        __qtablewidgetitem50.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(14, 0, __qtablewidgetitem50)
        __qtablewidgetitem51 = QTableWidgetItem()
        __qtablewidgetitem51.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(14, 1, __qtablewidgetitem51)
        __qtablewidgetitem52 = QTableWidgetItem()
        __qtablewidgetitem52.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(15, 0, __qtablewidgetitem52)
        __qtablewidgetitem53 = QTableWidgetItem()
        __qtablewidgetitem53.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(15, 1, __qtablewidgetitem53)
        __qtablewidgetitem54 = QTableWidgetItem()
        __qtablewidgetitem54.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(16, 0, __qtablewidgetitem54)
        __qtablewidgetitem55 = QTableWidgetItem()
        __qtablewidgetitem55.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(16, 1, __qtablewidgetitem55)
        __qtablewidgetitem56 = QTableWidgetItem()
        __qtablewidgetitem56.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(17, 0, __qtablewidgetitem56)
        __qtablewidgetitem57 = QTableWidgetItem()
        __qtablewidgetitem57.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(17, 1, __qtablewidgetitem57)
        __qtablewidgetitem58 = QTableWidgetItem()
        __qtablewidgetitem58.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(18, 0, __qtablewidgetitem58)
        __qtablewidgetitem59 = QTableWidgetItem()
        __qtablewidgetitem59.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(18, 1, __qtablewidgetitem59)
        __qtablewidgetitem60 = QTableWidgetItem()
        __qtablewidgetitem60.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(19, 0, __qtablewidgetitem60)
        __qtablewidgetitem61 = QTableWidgetItem()
        __qtablewidgetitem61.setTextAlignment(Qt.AlignCenter);
        self.sort_TW.setItem(19, 1, __qtablewidgetitem61)
        self.sort_TW.setObjectName(u"sort_TW")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sort_TW.sizePolicy().hasHeightForWidth())
        self.sort_TW.setSizePolicy(sizePolicy)
        self.sort_TW.setAutoFillBackground(False)
        self.sort_TW.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.sort_TW.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.sort_TW.setProperty("showDropIndicator", False)
        self.sort_TW.setRowCount(20)

        self.makeVL.addWidget(self.sort_TW)


        self.gridLayout_6.addLayout(self.makeVL, 0, 2, 1, 2)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.progressB = QProgressBar(self.centralwidget)
        self.progressB.setObjectName(u"progressB")
        self.progressB.setMaximum(100)
        self.progressB.setValue(0)

        self.verticalLayout_2.addWidget(self.progressB)

        self.outTE = QTextEdit(self.centralwidget)
        self.outTE.setObjectName(u"outTE")

        self.verticalLayout_2.addWidget(self.outTE)


        self.gridLayout_6.addLayout(self.verticalLayout_2, 1, 0, 2, 4)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setMinimumSize(QSize(0, 0))
        self.tabWidget.setMaximumSize(QSize(360, 16777215))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_13 = QGridLayout(self.tab)
        self.gridLayout_13.setSpacing(0)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.gridLayout_13.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.tab)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 343, 796))
        self.gridLayout_14 = QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.label_5 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_5.setObjectName(u"label_5")
        font2 = QFont()
        font2.setFamily(u"\u5e7c\u5706")
        font2.setPointSize(12)
        self.label_5.setFont(font2)

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_5)

        self.label2 = QLabel(self.scrollAreaWidgetContents_3)
        self.label2.setObjectName(u"label2")
        self.label2.setFont(font2)

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label2)

        self.label5 = QLabel(self.scrollAreaWidgetContents_3)
        self.label5.setObjectName(u"label5")
        font3 = QFont()
        font3.setFamily(u"Arial Narrow")
        font3.setPointSize(12)
        self.label5.setFont(font3)

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.label5)

        self.learn_rate_finalDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.learn_rate_finalDSB.setObjectName(u"learn_rate_finalDSB")
        self.learn_rate_finalDSB.setDecimals(10)
        self.learn_rate_finalDSB.setMaximum(10000.000000000000000)
        self.learn_rate_finalDSB.setSingleStep(0.000000000000000)
        self.learn_rate_finalDSB.setStepType(QAbstractSpinBox.DefaultStepType)

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.learn_rate_finalDSB)

        self.label3 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3.setObjectName(u"label3")
        self.label3.setFont(font3)

        self.formLayout_3.setWidget(11, QFormLayout.LabelRole, self.label3)

        self.epoch_numSB = QSpinBox(self.scrollAreaWidgetContents_3)
        self.epoch_numSB.setObjectName(u"epoch_numSB")
        self.epoch_numSB.setMaximum(10000)
        self.epoch_numSB.setSingleStep(0)

        self.formLayout_3.setWidget(11, QFormLayout.FieldRole, self.epoch_numSB)

        self.label3_2 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_2.setObjectName(u"label3_2")
        self.label3_2.setFont(font3)

        self.formLayout_3.setWidget(12, QFormLayout.LabelRole, self.label3_2)

        self.iou_cbb = QComboBox(self.scrollAreaWidgetContents_3)
        self.iou_cbb.addItem("")
        self.iou_cbb.addItem("")
        self.iou_cbb.addItem("")
        self.iou_cbb.setObjectName(u"iou_cbb")

        self.formLayout_3.setWidget(12, QFormLayout.FieldRole, self.iou_cbb)

        self.label4_10 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_10.setObjectName(u"label4_10")
        self.label4_10.setFont(font3)

        self.formLayout_3.setWidget(13, QFormLayout.LabelRole, self.label4_10)

        self.device_CBB = QComboBox(self.scrollAreaWidgetContents_3)
        self.device_CBB.addItem("")
        self.device_CBB.addItem("")
        self.device_CBB.setObjectName(u"device_CBB")

        self.formLayout_3.setWidget(13, QFormLayout.FieldRole, self.device_CBB)

        self.label4_11 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_11.setObjectName(u"label4_11")
        self.label4_11.setFont(font3)

        self.formLayout_3.setWidget(14, QFormLayout.LabelRole, self.label4_11)

        self.learning_mode_CBB = QComboBox(self.scrollAreaWidgetContents_3)
        self.learning_mode_CBB.addItem("")
        self.learning_mode_CBB.addItem("")
        self.learning_mode_CBB.addItem("")
        self.learning_mode_CBB.setObjectName(u"learning_mode_CBB")

        self.formLayout_3.setWidget(14, QFormLayout.FieldRole, self.learning_mode_CBB)

        self.model_nameCBB = QComboBox(self.scrollAreaWidgetContents_3)
        self.model_nameCBB.setObjectName(u"model_nameCBB")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.model_nameCBB)

        self.net_nameCBB = My_other_widght.my_combobox_show_nets(self.scrollAreaWidgetContents_3)
        self.net_nameCBB.setObjectName(u"net_nameCBB")

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.net_nameCBB)

        self.label5_2 = QLabel(self.scrollAreaWidgetContents_3)
        self.label5_2.setObjectName(u"label5_2")
        self.label5_2.setFont(font3)

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label5_2)

        self.learn_rate_initDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.learn_rate_initDSB.setObjectName(u"learn_rate_initDSB")
        self.learn_rate_initDSB.setMouseTracking(False)
        self.learn_rate_initDSB.setFocusPolicy(Qt.WheelFocus)
        self.learn_rate_initDSB.setAccelerated(False)
        self.learn_rate_initDSB.setDecimals(10)
        self.learn_rate_initDSB.setMaximum(10000.000000000000000)
        self.learn_rate_initDSB.setSingleStep(0.000000000000000)

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.learn_rate_initDSB)

        self.label5_3 = QLabel(self.scrollAreaWidgetContents_3)
        self.label5_3.setObjectName(u"label5_3")
        self.label5_3.setFont(font3)

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.label5_3)

        self.learning_rate_mode_cbb = QComboBox(self.scrollAreaWidgetContents_3)
        self.learning_rate_mode_cbb.addItem("")
        self.learning_rate_mode_cbb.addItem("")
        self.learning_rate_mode_cbb.addItem("")
        self.learning_rate_mode_cbb.addItem("")
        self.learning_rate_mode_cbb.addItem("")
        self.learning_rate_mode_cbb.setObjectName(u"learning_rate_mode_cbb")

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.learning_rate_mode_cbb)

        self.label3_3 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_3.setObjectName(u"label3_3")
        self.label3_3.setFont(font3)

        self.formLayout_3.setWidget(7, QFormLayout.LabelRole, self.label3_3)

        self.label3_4 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_4.setObjectName(u"label3_4")
        self.label3_4.setFont(font3)

        self.formLayout_3.setWidget(8, QFormLayout.LabelRole, self.label3_4)

        self.label3_5 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_5.setObjectName(u"label3_5")
        self.label3_5.setFont(font3)

        self.formLayout_3.setWidget(9, QFormLayout.LabelRole, self.label3_5)

        self.label3_6 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_6.setObjectName(u"label3_6")
        self.label3_6.setFont(font3)

        self.formLayout_3.setWidget(10, QFormLayout.LabelRole, self.label3_6)

        self.label3_7 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_7.setObjectName(u"label3_7")
        self.label3_7.setFont(font3)

        self.formLayout_3.setWidget(5, QFormLayout.LabelRole, self.label3_7)

        self.label3_8 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_8.setObjectName(u"label3_8")
        self.label3_8.setFont(font3)

        self.formLayout_3.setWidget(6, QFormLayout.LabelRole, self.label3_8)

        self.momentumDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.momentumDSB.setObjectName(u"momentumDSB")
        self.momentumDSB.setDecimals(10)
        self.momentumDSB.setMaximum(10000.000000000000000)
        self.momentumDSB.setSingleStep(0.000000000000000)

        self.formLayout_3.setWidget(5, QFormLayout.FieldRole, self.momentumDSB)

        self.weight_decayDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.weight_decayDSB.setObjectName(u"weight_decayDSB")
        self.weight_decayDSB.setDecimals(10)
        self.weight_decayDSB.setMaximum(10000.000000000000000)
        self.weight_decayDSB.setSingleStep(0.000000000000000)

        self.formLayout_3.setWidget(6, QFormLayout.FieldRole, self.weight_decayDSB)

        self.warmup_epochsSB = QSpinBox(self.scrollAreaWidgetContents_3)
        self.warmup_epochsSB.setObjectName(u"warmup_epochsSB")
        self.warmup_epochsSB.setMaximum(10000)
        self.warmup_epochsSB.setSingleStep(0)

        self.formLayout_3.setWidget(7, QFormLayout.FieldRole, self.warmup_epochsSB)

        self.warmup_bias_lrDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.warmup_bias_lrDSB.setObjectName(u"warmup_bias_lrDSB")
        self.warmup_bias_lrDSB.setDecimals(10)
        self.warmup_bias_lrDSB.setMaximum(10000.000000000000000)
        self.warmup_bias_lrDSB.setSingleStep(0.000000000000000)

        self.formLayout_3.setWidget(8, QFormLayout.FieldRole, self.warmup_bias_lrDSB)

        self.warmup_momentumDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.warmup_momentumDSB.setObjectName(u"warmup_momentumDSB")
        self.warmup_momentumDSB.setDecimals(10)
        self.warmup_momentumDSB.setMaximum(10000.000000000000000)
        self.warmup_momentumDSB.setSingleStep(0.000000000000000)

        self.formLayout_3.setWidget(9, QFormLayout.FieldRole, self.warmup_momentumDSB)

        self.optimizers_cbb = QComboBox(self.scrollAreaWidgetContents_3)
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.setObjectName(u"optimizers_cbb")

        self.formLayout_3.setWidget(10, QFormLayout.FieldRole, self.optimizers_cbb)

        self.label4_17 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_17.setObjectName(u"label4_17")
        self.label4_17.setFont(font3)

        self.formLayout_3.setWidget(15, QFormLayout.LabelRole, self.label4_17)

        self.multi_scale_able_cb = QCheckBox(self.scrollAreaWidgetContents_3)
        self.multi_scale_able_cb.setObjectName(u"multi_scale_able_cb")
        font4 = QFont()
        font4.setFamily(u"\u5b8b\u4f53")
        font4.setPointSize(14)
        self.multi_scale_able_cb.setFont(font4)

        self.formLayout_3.setWidget(15, QFormLayout.FieldRole, self.multi_scale_able_cb)

        self.label4_18 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_18.setObjectName(u"label4_18")
        self.label4_18.setFont(font3)

        self.formLayout_3.setWidget(16, QFormLayout.LabelRole, self.label4_18)

        self.multi_scale_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.multi_scale_DSB.setObjectName(u"multi_scale_DSB")
        self.multi_scale_DSB.setDecimals(10)
        self.multi_scale_DSB.setMaximum(10000.000000000000000)
        self.multi_scale_DSB.setSingleStep(0.000000000000000)

        self.formLayout_3.setWidget(16, QFormLayout.FieldRole, self.multi_scale_DSB)


        self.gridLayout_14.addLayout(self.formLayout_3, 0, 0, 2, 1)

        self.add_modelPB = QPushButton(self.scrollAreaWidgetContents_3)
        self.add_modelPB.setObjectName(u"add_modelPB")
        self.add_modelPB.setMaximumSize(QSize(31, 16777215))

        self.gridLayout_14.addWidget(self.add_modelPB, 0, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 746, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_14.addItem(self.verticalSpacer, 1, 1, 2, 1)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label3_9 = QLabel(self.scrollAreaWidgetContents_3)
        self.label3_9.setObjectName(u"label3_9")
        self.label3_9.setFont(font3)

        self.gridLayout_4.addWidget(self.label3_9, 1, 0, 1, 1)

        self.label4_16 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_16.setObjectName(u"label4_16")
        self.label4_16.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_16, 7, 0, 1, 1)

        self.label4_9 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_9.setObjectName(u"label4_9")
        self.label4_9.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_9, 12, 0, 1, 1)

        self.label4_7 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_7.setObjectName(u"label4_7")
        self.label4_7.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_7, 10, 0, 1, 1)

        self.label4_6 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_6.setObjectName(u"label4_6")
        self.label4_6.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_6, 9, 0, 1, 1)

        self.anchor_tDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.anchor_tDSB.setObjectName(u"anchor_tDSB")
        self.anchor_tDSB.setDecimals(10)
        self.anchor_tDSB.setMaximum(10000.000000000000000)
        self.anchor_tDSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.anchor_tDSB, 8, 1, 1, 1)

        self.label4_3 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_3.setObjectName(u"label4_3")
        self.label4_3.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_3, 3, 0, 1, 1)

        self.fl_gamma_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.fl_gamma_DSB.setObjectName(u"fl_gamma_DSB")
        self.fl_gamma_DSB.setDecimals(10)
        self.fl_gamma_DSB.setMaximum(1.000000000000000)
        self.fl_gamma_DSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.fl_gamma_DSB, 7, 1, 1, 1)

        self.label4_4 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_4.setObjectName(u"label4_4")
        self.label4_4.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_4, 4, 0, 1, 1)

        self.test_iou_thres_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.test_iou_thres_DSB.setObjectName(u"test_iou_thres_DSB")
        self.test_iou_thres_DSB.setDecimals(10)
        self.test_iou_thres_DSB.setMaximum(10000.000000000000000)
        self.test_iou_thres_DSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.test_iou_thres_DSB, 12, 1, 1, 1)

        self.label4_8 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_8.setObjectName(u"label4_8")
        self.label4_8.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_8, 11, 0, 1, 1)

        self.label4_14 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_14.setObjectName(u"label4_14")
        self.label4_14.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_14, 8, 0, 1, 1)

        self.val_iou_thres_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.val_iou_thres_DSB.setObjectName(u"val_iou_thres_DSB")
        self.val_iou_thres_DSB.setDecimals(10)
        self.val_iou_thres_DSB.setMaximum(10000.000000000000000)
        self.val_iou_thres_DSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.val_iou_thres_DSB, 10, 1, 1, 1)

        self.label4_13 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_13.setObjectName(u"label4_13")
        self.label4_13.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_13, 6, 0, 1, 1)

        self.gr_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.gr_DSB.setObjectName(u"gr_DSB")
        self.gr_DSB.setDecimals(10)
        self.gr_DSB.setMaximum(1.000000000000000)
        self.gr_DSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.gr_DSB, 6, 1, 1, 1)

        self.cls_pwDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.cls_pwDSB.setObjectName(u"cls_pwDSB")
        self.cls_pwDSB.setDecimals(10)
        self.cls_pwDSB.setMaximum(10000.000000000000000)
        self.cls_pwDSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.cls_pwDSB, 0, 1, 1, 1)

        self.label4_15 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_15.setObjectName(u"label4_15")
        self.label4_15.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_15, 2, 0, 1, 1)

        self.label4_12 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_12.setObjectName(u"label4_12")
        self.label4_12.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_12, 0, 0, 1, 1)

        self.cls_lossDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.cls_lossDSB.setObjectName(u"cls_lossDSB")
        self.cls_lossDSB.setDecimals(10)
        self.cls_lossDSB.setMaximum(10000.000000000000000)
        self.cls_lossDSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.cls_lossDSB, 5, 1, 1, 1)

        self.obj_pwDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.obj_pwDSB.setObjectName(u"obj_pwDSB")
        self.obj_pwDSB.setDecimals(10)
        self.obj_pwDSB.setMaximum(10000.000000000000000)
        self.obj_pwDSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.obj_pwDSB, 2, 1, 1, 1)

        self.obj_lossDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.obj_lossDSB.setObjectName(u"obj_lossDSB")
        self.obj_lossDSB.setDecimals(10)
        self.obj_lossDSB.setMaximum(10000.000000000000000)
        self.obj_lossDSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.obj_lossDSB, 4, 1, 1, 1)

        self.test_conf_thres_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.test_conf_thres_DSB.setObjectName(u"test_conf_thres_DSB")
        self.test_conf_thres_DSB.setDecimals(10)
        self.test_conf_thres_DSB.setMaximum(10000.000000000000000)
        self.test_conf_thres_DSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.test_conf_thres_DSB, 11, 1, 1, 1)

        self.cls_smooth_SB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.cls_smooth_SB.setObjectName(u"cls_smooth_SB")
        self.cls_smooth_SB.setDecimals(10)
        self.cls_smooth_SB.setMaximum(2.000000000000000)
        self.cls_smooth_SB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.cls_smooth_SB, 1, 1, 1, 1)

        self.giou_lossDSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.giou_lossDSB.setObjectName(u"giou_lossDSB")
        self.giou_lossDSB.setDecimals(10)
        self.giou_lossDSB.setMaximum(10000.000000000000000)
        self.giou_lossDSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.giou_lossDSB, 3, 1, 1, 1)

        self.label4_5 = QLabel(self.scrollAreaWidgetContents_3)
        self.label4_5.setObjectName(u"label4_5")
        self.label4_5.setFont(font3)

        self.gridLayout_4.addWidget(self.label4_5, 5, 0, 1, 1)

        self.val_conf_thres_DSB = QDoubleSpinBox(self.scrollAreaWidgetContents_3)
        self.val_conf_thres_DSB.setObjectName(u"val_conf_thres_DSB")
        self.val_conf_thres_DSB.setDecimals(10)
        self.val_conf_thres_DSB.setMaximum(10000.000000000000000)
        self.val_conf_thres_DSB.setSingleStep(0.000000000000000)

        self.gridLayout_4.addWidget(self.val_conf_thres_DSB, 9, 1, 1, 1)


        self.gridLayout_14.addLayout(self.gridLayout_4, 2, 0, 1, 1)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.gridLayout_13.addWidget(self.scrollArea_3, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.layoutWidget = QWidget(self.tab_2)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(10, 10, 171, 111))
        self.formLayout_2 = QFormLayout(self.layoutWidget)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label4 = QLabel(self.layoutWidget)
        self.label4.setObjectName(u"label4")
        self.label4.setFont(font3)

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label4)

        self.class_numSB = QSpinBox(self.layoutWidget)
        self.class_numSB.setObjectName(u"class_numSB")
        self.class_numSB.setMaximum(10000)

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.class_numSB)

        self.label4_2 = QLabel(self.layoutWidget)
        self.label4_2.setObjectName(u"label4_2")
        self.label4_2.setFont(font3)

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label4_2)

        self.img_sizeSB = QSpinBox(self.layoutWidget)
        self.img_sizeSB.setObjectName(u"img_sizeSB")
        self.img_sizeSB.setMaximum(10000)

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.img_sizeSB)

        self.label1 = QLabel(self.layoutWidget)
        self.label1.setObjectName(u"label1")
        self.label1.setFont(font3)

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.label1)

        self.batch_sizeSB = QSpinBox(self.layoutWidget)
        self.batch_sizeSB.setObjectName(u"batch_sizeSB")
        self.batch_sizeSB.setMaximum(10000)

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.batch_sizeSB)

        self.label1_9 = QLabel(self.layoutWidget)
        self.label1_9.setObjectName(u"label1_9")
        self.label1_9.setFont(font3)

        self.formLayout_2.setWidget(3, QFormLayout.LabelRole, self.label1_9)

        self.image_type_CBB = QComboBox(self.layoutWidget)
        self.image_type_CBB.addItem("")
        self.image_type_CBB.addItem("")
        self.image_type_CBB.setObjectName(u"image_type_CBB")

        self.formLayout_2.setWidget(3, QFormLayout.FieldRole, self.image_type_CBB)

        self.layoutWidget1 = QWidget(self.tab_2)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(10, 130, 282, 176))
        self.gridLayout_2 = QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(6)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.cache_img_cb = QCheckBox(self.layoutWidget1)
        self.cache_img_cb.setObjectName(u"cache_img_cb")

        self.gridLayout_2.addWidget(self.cache_img_cb, 2, 0, 1, 1)

        self.single_cls_cb = QCheckBox(self.layoutWidget1)
        self.single_cls_cb.setObjectName(u"single_cls_cb")

        self.gridLayout_2.addWidget(self.single_cls_cb, 3, 1, 1, 1)

        self.augment_cb = QCheckBox(self.layoutWidget1)
        self.augment_cb.setObjectName(u"augment_cb")

        self.gridLayout_2.addWidget(self.augment_cb, 5, 0, 1, 1)

        self.augment_hsv_cb = QCheckBox(self.layoutWidget1)
        self.augment_hsv_cb.setObjectName(u"augment_hsv_cb")

        self.gridLayout_2.addWidget(self.augment_hsv_cb, 6, 1, 1, 1)

        self.lr_flip_cb = QCheckBox(self.layoutWidget1)
        self.lr_flip_cb.setObjectName(u"lr_flip_cb")

        self.gridLayout_2.addWidget(self.lr_flip_cb, 7, 0, 1, 1)

        self.extract_bounding_boxes_cb = QCheckBox(self.layoutWidget1)
        self.extract_bounding_boxes_cb.setObjectName(u"extract_bounding_boxes_cb")

        self.gridLayout_2.addWidget(self.extract_bounding_boxes_cb, 3, 0, 1, 1)

        self.rect_cb = QCheckBox(self.layoutWidget1)
        self.rect_cb.setObjectName(u"rect_cb")

        self.gridLayout_2.addWidget(self.rect_cb, 4, 0, 1, 1)

        self.border_cb = QCheckBox(self.layoutWidget1)
        self.border_cb.setObjectName(u"border_cb")

        self.gridLayout_2.addWidget(self.border_cb, 6, 0, 1, 1)

        self.rect_size_SB = QSpinBox(self.layoutWidget1)
        self.rect_size_SB.setObjectName(u"rect_size_SB")

        self.gridLayout_2.addWidget(self.rect_size_SB, 4, 1, 1, 1)

        self.auto_anchor_cb = QCheckBox(self.layoutWidget1)
        self.auto_anchor_cb.setObjectName(u"auto_anchor_cb")

        self.gridLayout_2.addWidget(self.auto_anchor_cb, 1, 0, 1, 1)

        self.auto_batch_size_cb = QCheckBox(self.layoutWidget1)
        self.auto_batch_size_cb.setObjectName(u"auto_batch_size_cb")

        self.gridLayout_2.addWidget(self.auto_batch_size_cb, 1, 1, 1, 1)

        self.ud_flip_cb = QCheckBox(self.layoutWidget1)
        self.ud_flip_cb.setObjectName(u"ud_flip_cb")

        self.gridLayout_2.addWidget(self.ud_flip_cb, 7, 1, 1, 1)

        self.val_able_cb = QCheckBox(self.layoutWidget1)
        self.val_able_cb.setObjectName(u"val_able_cb")

        self.gridLayout_2.addWidget(self.val_able_cb, 0, 0, 1, 1)

        self.updata_cache_label_cb = QCheckBox(self.layoutWidget1)
        self.updata_cache_label_cb.setObjectName(u"updata_cache_label_cb")

        self.gridLayout_2.addWidget(self.updata_cache_label_cb, 2, 1, 1, 1)

        self.layoutWidget2 = QWidget(self.tab_2)
        self.layoutWidget2.setObjectName(u"layoutWidget2")
        self.layoutWidget2.setGeometry(QRect(10, 320, 195, 191))
        self.gridLayout_3 = QGridLayout(self.layoutWidget2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label1_3 = QLabel(self.layoutWidget2)
        self.label1_3.setObjectName(u"label1_3")
        self.label1_3.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_3, 1, 0, 1, 1)

        self.label1_5 = QLabel(self.layoutWidget2)
        self.label1_5.setObjectName(u"label1_5")
        self.label1_5.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_5, 3, 0, 1, 1)

        self.label1_4 = QLabel(self.layoutWidget2)
        self.label1_4.setObjectName(u"label1_4")
        self.label1_4.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_4, 2, 0, 1, 1)

        self.label1_2 = QLabel(self.layoutWidget2)
        self.label1_2.setObjectName(u"label1_2")
        self.label1_2.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_2, 0, 0, 1, 1)

        self.label1_6 = QLabel(self.layoutWidget2)
        self.label1_6.setObjectName(u"label1_6")
        self.label1_6.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_6, 4, 0, 1, 1)

        self.scale_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.scale_DSB.setObjectName(u"scale_DSB")
        self.scale_DSB.setDecimals(10)
        self.scale_DSB.setMaximum(1.000000000000000)
        self.scale_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.scale_DSB, 2, 1, 1, 1)

        self.hsv_h_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.hsv_h_DSB.setObjectName(u"hsv_h_DSB")
        self.hsv_h_DSB.setDecimals(10)
        self.hsv_h_DSB.setMaximum(1.000000000000000)
        self.hsv_h_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.hsv_h_DSB, 4, 1, 1, 1)

        self.label1_7 = QLabel(self.layoutWidget2)
        self.label1_7.setObjectName(u"label1_7")
        self.label1_7.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_7, 5, 0, 1, 1)

        self.translate_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.translate_DSB.setObjectName(u"translate_DSB")
        self.translate_DSB.setDecimals(10)
        self.translate_DSB.setMaximum(1.000000000000000)
        self.translate_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.translate_DSB, 1, 1, 1, 1)

        self.shear_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.shear_DSB.setObjectName(u"shear_DSB")
        self.shear_DSB.setDecimals(10)
        self.shear_DSB.setMaximum(180.000000000000000)
        self.shear_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.shear_DSB, 3, 1, 1, 1)

        self.degrees_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.degrees_DSB.setObjectName(u"degrees_DSB")
        self.degrees_DSB.setDecimals(10)
        self.degrees_DSB.setMaximum(180.000000000000000)
        self.degrees_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.degrees_DSB, 0, 1, 1, 1)

        self.hsv_s_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.hsv_s_DSB.setObjectName(u"hsv_s_DSB")
        self.hsv_s_DSB.setDecimals(10)
        self.hsv_s_DSB.setMaximum(1.000000000000000)
        self.hsv_s_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.hsv_s_DSB, 5, 1, 1, 1)

        self.label1_8 = QLabel(self.layoutWidget2)
        self.label1_8.setObjectName(u"label1_8")
        self.label1_8.setFont(font3)

        self.gridLayout_3.addWidget(self.label1_8, 6, 0, 1, 1)

        self.hsv_v_DSB = QDoubleSpinBox(self.layoutWidget2)
        self.hsv_v_DSB.setObjectName(u"hsv_v_DSB")
        self.hsv_v_DSB.setDecimals(10)
        self.hsv_v_DSB.setMaximum(1.000000000000000)
        self.hsv_v_DSB.setSingleStep(0.000100000000000)

        self.gridLayout_3.addWidget(self.hsv_v_DSB, 6, 1, 1, 1)

        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.gridLayout_11 = QGridLayout(self.tab_3)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.textEdit = QTextEdit(self.tab_3)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setReadOnly(True)

        self.gridLayout_11.addWidget(self.textEdit, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.gridLayout_10 = QGridLayout(self.tab_4)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.textEdit_2 = QTextEdit(self.tab_4)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setReadOnly(True)

        self.gridLayout_10.addWidget(self.textEdit_2, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_4, "")

        self.gridLayout_6.addWidget(self.tabWidget, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.tabWidget.raise_()
        self.run_to_lowPB.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1430, 23))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionback_start)
        self.menu.addAction(self.actionnew_file)
        self.menu.addAction(self.actionopen_project)
        self.menu.addAction(self.actionsave)
        self.menu.addAction(self.actionsave_as)
        self.menu.addAction(self.actionexit)

        self.retranslateUi(MainWindow)

        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)

        self.model = None
        self.model_name = ''
        self.image_path = ['']
        self.num = 0
        self.curser = QTextCursor.End
        self.trainPB.clicked.connect(self.train)
        self.licon_imagePB.clicked.connect(self.licon_imagePB_clicked)
        self.tryPB.clicked.connect(self.testPB_clicked)
        self.run_to_lowPB.clicked.connect(self.cursor_to_lowPB_clicked)
        self.down_image_pb.clicked.connect(self.down_img_clicked)
        self.up_image_pb.clicked.connect(self.up_img_clicked)
        #self.save_as_onnxPB.clicked.connect(self.save_as_onnx_clicked)
        self.net_nameCBB.updata_nets_signal.connect(self.updata_net_clicked)
        self.updata_RMP_PB.clicked.connect(self.updata_RMP_PB_clicked)
        self.updata_per_class_RMP_PB.clicked.connect(self.updata_per_class_RMP_PB_clicked)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow-train", None))
        self.actionnew_file.setText(QCoreApplication.translate("MainWindow", u"new project", None))
        self.actionopen_project.setText(QCoreApplication.translate("MainWindow", u"open project", None))
        self.actionsave.setText(QCoreApplication.translate("MainWindow", u"save", None))
        self.actionsave_as.setText(QCoreApplication.translate("MainWindow", u"save as", None))
        self.actionexit.setText(QCoreApplication.translate("MainWindow", u"exit", None))
        self.actionback_start.setText(QCoreApplication.translate("MainWindow", u"back start", None))
        self.down_image_pb.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u5f20", None))
        self.up_image_pb.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e00\u5f20", None))
        self.licon_imagePB.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u56fe\u50cf", None))
        self.imageLB.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), QCoreApplication.translate("MainWindow", u"\u624b\u52a8\u6d4b\u8bd5", None))
        self.updata_RMP_PB.setText(QCoreApplication.translate("MainWindow", u"\u5237\u65b0", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_6), QCoreApplication.translate("MainWindow", u"RMP", None))
        self.updata_per_class_RMP_PB.setText(QCoreApplication.translate("MainWindow", u"\u5237\u65b0", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_7), QCoreApplication.translate("MainWindow", u"per class RMP", None))
        self.run_to_lowPB.setText(QCoreApplication.translate("MainWindow", u"\u5e95", None))
        self.makedatasetPB.setText(QCoreApplication.translate("MainWindow", u"\u5236\u4f5c\u6570\u636e\u96c6", None))
        self.makenetPB.setText(QCoreApplication.translate("MainWindow", u"\u6784\u5efa\u795e\u7ecf\u7f51\u7edc\u7ed3\u6784", None))
        self.trainPB.setText(QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3", None))
        self.tryPB.setText(QCoreApplication.translate("MainWindow", u"\u6d4b\u8bd5", None))
        self.save_as_onnxPB.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58onnx\u6a21\u578b", None))
        ___qtablewidgetitem = self.sort_TW.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"\u79cd\u7c7b\u7d22\u5f15", None));
        ___qtablewidgetitem1 = self.sort_TW.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"\u79cd\u7c7b\u540d", None));

        __sortingEnabled = self.sort_TW.isSortingEnabled()
        self.sort_TW.setSortingEnabled(False)
        self.sort_TW.setSortingEnabled(__sortingEnabled)

        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u540d\u79f0\uff1a", None))
        self.label2.setText(QCoreApplication.translate("MainWindow", u"\u7f51\u7edc\u7ed3\u6784\u540d\u79f0\uff1a", None))
        self.label5.setText(QCoreApplication.translate("MainWindow", u"\u6700\u7ec8\u5b66\u4e60\u7387\uff1a", None))
        self.label3.setText(QCoreApplication.translate("MainWindow", u"epoch num\uff1a", None))
        self.label3_2.setText(QCoreApplication.translate("MainWindow", u"iou\uff1a", None))
        self.iou_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"Giou", None))
        self.iou_cbb.setItemText(1, QCoreApplication.translate("MainWindow", u"Diou", None))
        self.iou_cbb.setItemText(2, QCoreApplication.translate("MainWindow", u"Ciou", None))

        self.label4_10.setText(QCoreApplication.translate("MainWindow", u"device\uff1a", None))
        self.device_CBB.setItemText(0, QCoreApplication.translate("MainWindow", u"GPU", None))
        self.device_CBB.setItemText(1, QCoreApplication.translate("MainWindow", u"CPU", None))

        self.label4_11.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u5f0f\u9009\u62e9\uff1a", None))
        self.learning_mode_CBB.setItemText(0, QCoreApplication.translate("MainWindow", u"Yolo", None))
        self.learning_mode_CBB.setItemText(1, QCoreApplication.translate("MainWindow", u"\u8bed\u4e49\u5206\u5272", None))
        self.learning_mode_CBB.setItemText(2, QCoreApplication.translate("MainWindow", u"\u591a\u5206\u7c7b", None))

        self.label5_2.setText(QCoreApplication.translate("MainWindow", u"\u521d\u59cb\u5b66\u4e60\u7387\uff1a", None))
        self.label5_3.setText(QCoreApplication.translate("MainWindow", u"\u5b66\u4e60\u7387\u6a21\u5f0f\uff1a", None))
        self.learning_rate_mode_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"cos_1_2_lr", None))
        self.learning_rate_mode_cbb.setItemText(1, QCoreApplication.translate("MainWindow", u"cos_lr_2_zero", None))
        self.learning_rate_mode_cbb.setItemText(2, QCoreApplication.translate("MainWindow", u"line_1_2_lr", None))
        self.learning_rate_mode_cbb.setItemText(3, QCoreApplication.translate("MainWindow", u"line_lr_2_zero", None))
        self.learning_rate_mode_cbb.setItemText(4, QCoreApplication.translate("MainWindow", u"constant", None))

        self.label3_3.setText(QCoreApplication.translate("MainWindow", u"\u9884\u70ed\u5b66\u4e60\u5468\u671f\u6570\uff1a", None))
        self.label3_4.setText(QCoreApplication.translate("MainWindow", u"\u9884\u70ed\u5b66\u4e60\u521d\u59cb\u5b66\u4e60\u7387\uff1a", None))
        self.label3_5.setText(QCoreApplication.translate("MainWindow", u"\u9884\u70ed\u5b66\u4e60\u52a8\u91cf\uff1a", None))
        self.label3_6.setText(QCoreApplication.translate("MainWindow", u"\u4f18\u5316\u5668\uff1a", None))
        self.label3_7.setText(QCoreApplication.translate("MainWindow", u"\u52a8\u91cf\uff1a", None))
        self.label3_8.setText(QCoreApplication.translate("MainWindow", u"\u8870\u51cf\u7cfb\u6570\uff1a", None))
        self.optimizers_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"Adam", None))
        self.optimizers_cbb.setItemText(1, QCoreApplication.translate("MainWindow", u"AdamW", None))
        self.optimizers_cbb.setItemText(2, QCoreApplication.translate("MainWindow", u"RMSProp", None))
        self.optimizers_cbb.setItemText(3, QCoreApplication.translate("MainWindow", u"SGD", None))

        self.label4_17.setText(QCoreApplication.translate("MainWindow", u"multi_scale_able\uff1a", None))
        self.multi_scale_able_cb.setText("")
        self.label4_18.setText(QCoreApplication.translate("MainWindow", u"multi_scale\uff1a", None))
        self.add_modelPB.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.label3_9.setText(QCoreApplication.translate("MainWindow", u"\u79cd\u7c7b\u6b63\u8d1f\u6837\u672c\u503c\uff1a", None))
        self.label4_16.setText(QCoreApplication.translate("MainWindow", u"fl_gamma\uff1a", None))
        self.label4_9.setText(QCoreApplication.translate("MainWindow", u"\u6d4b\u8bd5\u7528iou\u9608\u503c\uff1a", None))
        self.label4_7.setText(QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1\u7528iou\u9608\u503c\uff1a", None))
        self.label4_6.setText(QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1\u7528\u7f6e\u4fe1\u5ea6\u9608\u503c\uff1a", None))
        self.label4_3.setText(QCoreApplication.translate("MainWindow", u"box\u635f\u5931\u52a0\u6743\u503c\uff1a", None))
        self.label4_4.setText(QCoreApplication.translate("MainWindow", u"\u7f6e\u4fe1\u5ea6\u635f\u5931\u52a0\u6743\u503c\uff1a", None))
        self.label4_8.setText(QCoreApplication.translate("MainWindow", u"\u6d4b\u8bd5\u7528\u7f6e\u4fe1\u5ea6\u9608\u503c\uff1a", None))
        self.label4_14.setText(QCoreApplication.translate("MainWindow", u"\u9884\u9009\u6846\u5224\u65ad\u9608\u503c\uff1a", None))
        self.label4_13.setText(QCoreApplication.translate("MainWindow", u"\u6b63\u6837\u672c\u7f6e\u4fe1\u5ea6\u6743\u503c\uff1a", None))
        self.label4_15.setText(QCoreApplication.translate("MainWindow", u"\u7f6e\u4fe1\u5ea6\u635f\u5931\u6b63\u4f8b\u6743\u503c\uff1a", None))
        self.label4_12.setText(QCoreApplication.translate("MainWindow", u"\u5206\u7c7b\u635f\u5931\u6b63\u4f8b\u6743\u503c\uff1a", None))
        self.label4_5.setText(QCoreApplication.translate("MainWindow", u"\u5206\u7c7b\u635f\u5931\u52a0\u6743\u503c\uff1a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u53c2\u6570", None))
        self.label4.setText(QCoreApplication.translate("MainWindow", u"class num\uff1a", None))
        self.label4_2.setText(QCoreApplication.translate("MainWindow", u"image size\uff1a", None))
        self.label1.setText(QCoreApplication.translate("MainWindow", u"batch size\uff1a", None))
        self.label1_9.setText(QCoreApplication.translate("MainWindow", u"image type\uff1a", None))
        self.image_type_CBB.setItemText(0, QCoreApplication.translate("MainWindow", u"color", None))
        self.image_type_CBB.setItemText(1, QCoreApplication.translate("MainWindow", u"gray", None))

        self.cache_img_cb.setText(QCoreApplication.translate("MainWindow", u"\u7f13\u5b58\u56fe\u50cf", None))
        self.single_cls_cb.setText(QCoreApplication.translate("MainWindow", u"\u5355\u4e00\u79cd\u7c7b", None))
        self.augment_cb.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u589e\u5f3a", None))
        self.augment_hsv_cb.setText(QCoreApplication.translate("MainWindow", u"hsv\u8272\u5f69\u7a7a\u95f4\u968f\u673a\u589e\u76ca", None))
        self.lr_flip_cb.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u968f\u673a\u5de6\u53f3\u7ffb\u8f6c", None))
        self.extract_bounding_boxes_cb.setText(QCoreApplication.translate("MainWindow", u"\u5206\u7c7b\u63d0\u53d6\u7279\u5f81", None))
        self.rect_cb.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u9002\u5e94\u6539\u8fdb\u6cd5", None))
        self.border_cb.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u968f\u673a\u4eff\u5c04\u53d8\u6362", None))
        self.auto_anchor_cb.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u52a8\u9884\u9009\u6846", None))
        self.auto_batch_size_cb.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u52a8batch size", None))
        self.ud_flip_cb.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u968f\u673a\u4e0a\u4e0b\u7ffb\u8f6c", None))
        self.val_able_cb.setText(QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1\u96c6", None))
        self.updata_cache_label_cb.setText(QCoreApplication.translate("MainWindow", u"\u5fc5\u987b\u66f4\u65b0\u6807\u7b7e\u7f13\u5b58", None))
        self.label1_3.setText(QCoreApplication.translate("MainWindow", u"translate\uff1a", None))
        self.label1_5.setText(QCoreApplication.translate("MainWindow", u"shear\uff1a", None))
        self.label1_4.setText(QCoreApplication.translate("MainWindow", u"scale\uff1a", None))
        self.label1_2.setText(QCoreApplication.translate("MainWindow", u"degrees\uff1a", None))
        self.label1_6.setText(QCoreApplication.translate("MainWindow", u"HSV_H\uff1a", None))
        self.label1_7.setText(QCoreApplication.translate("MainWindow", u"HSV_S\uff1a", None))
        self.label1_8.setText(QCoreApplication.translate("MainWindow", u"HSV_V\uff1a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"\u6570\u636e\u96c6\u53c2\u6570", None))
        self.textEdit.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'SimSun'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u6a21\u578b\u540d\u79f0\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u4fdd\u5b58\u6a21\u578b\u7684\u540d\u79f0\uff0c\u540e\u7f00.pt\uff0cbest_modelname.pt\u4e3a\u5bf9\u6d4b\u8bd5\u96c6\u7684\u6700\u4f18\u6a21\u578b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u7f51\u7edc\u7ed3\u6784\u540d\u79f0"
                        "</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u8bad\u7ec3\u7f51\u7edc\u7684.yaml\u6587\u4ef6\u540d\u79f0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u521d\u59cb\u5b66\u4e60\u7387</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt;\">\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u4f18\u5316\u5668\u7684\u521d\u59cb\u5b66\u4e60\u7387\uff0c\u4e0d\u5b9c\u8fc7\u5927\uff0c\u9ed8\u8ba40.0001</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u6700\u7ec8\u5b66\u4e60\u7387</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt;\""
                        ">\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u7ed3\u675f\u8bad\u7ec3\u65f6\u7684\u5b66\u4e60\u7387\u4e3a\u521d\u59cb\u5b66\u4e60\u7387*\u6700\u7ec8\u5b66\u4e60\u7387</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u5b66\u4e60\u7387\u6a21\u5f0f\uff1a</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">cos_1_2_lr:</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u5b66\u4e60\u7387\u4ece1\u5230\u6700\u7ec8\u5b66\u4e60\u7387\u4f59\u5f26\u4e0b\u964d\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-"
                        "indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">cos_lr_2_zero:</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u5b66\u4e60\u7387\u4ece\u6700\u7ec8\u5b66\u4e60\u7387\u52300\u4f59\u5f26\u4e0b\u964d\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">line_1_2_lr:</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u5b66\u4e60\u7387\u4ece1\u5230\u6700\u7ec8\u5b66\u4e60\u7387\u7ebf\u6027\u4e0b\u964d\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:"
                        "'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">line_lr_2_zero:</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u5b66\u4e60\u7387\u4ece\u6700\u7ec8\u5b66\u4e60\u7387\u52300\u7ebf\u6027\u4e0b\u964d\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">constant:</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u5b66\u4e60\u7387\u662f\u5e38\u91cf-\u6700\u7ec8\u5b66\u4e60\u7387</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u52a8\u91cf\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u9632\u6b62\u5c40\u90e8\u6700\u4f18\u5316</span></p>\n"
"<p style=\" "
                        "margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u8870\u51cf\u7cfb\u6570\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u9632\u6b62\u8fc7\u62df\u5408\uff0c\u4f7f\u6743\u91cd\u8d8b\u8fd1\u4e8e0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9884\u70ed\u5b66\u4e60\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u521a\u5f00\u59cb\u8bad\u7ec3\u65f6\u7684\u70ed\u8eab\u8bad\u7ec3\uff0c\u8d77\u59cb\u4f7f\u7528\u5f88\u5c0f\u7684\u5b66\u4e60\u7387\uff0c\u5728\u6307\u5b9a\u8bad\u7ec3\u5468\u6b21\u5185\u7ebf\u6027\u4e0a\u5347\u81f3\u6a21\u578b\u5f00\u59cb\u8bad\u7ec3\u7684\u5b66\u4e60\u7387\uff0c\u5f00\u59cb\u6b63\u5f0f\u8bad\u7ec3</span></p>\n"
"<p style=\" margin-t"
                        "op:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9884\u70ed\u5b66\u4e60\u5468\u671f\u6570\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u9884\u70ed\u5b66\u4e60\u7684\u8bad\u7ec3\u6b21\u6570\uff0c\u5c0f\u4e8e\u8bad\u7ec3\u5468\u671f\u7684\u5341\u5206\u4e4b\u4e00</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9884\u70ed\u5b66\u4e60\u521d\u59cb\u5b66\u4e60\u7387\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u9700\u8981\u975e\u5e38\u5c0f\u7684\u503c</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9884"
                        "\u70ed\u5b66\u4e60\u52a8\u91cf\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u9632\u6b62\u5c40\u90e8\u6700\u4f18\u5316</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u4f18\u5316\u5668\uff1a</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">adam\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u81ea\u9002\u5e94\u52a8\u91cf\u4f18\u5316\u5668\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" f"
                        "ont-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">adamw\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u81ea\u9002\u5e94\u52a8\u91cf\u6743\u91cd\u8870\u51cf\u4f18\u5316\u5668\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">rmsprop\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u6307\u6570\u52a0\u6743\u5e73\u5747\u4f18\u5316\u5668\uff08\u81ea\u9002\u5e94\u68af\u5ea6\uff09\uff1b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">  </span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt; font-weight:600;\">SGD\uff1a</span><span style=\" font-family:'\u5b8b\u4f53"
                        "'; font-size:11pt;\">\u968f\u673a\u68af\u5ea6\u4e0b\u964d\u4f18\u5316\u5668</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">epoch num\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u8bad\u7ec3\u5468\u671f\u6570\uff0c\u4e00\u822c100+</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">iou\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u771f\u5b9e\u6846\u4e0e\u9884\u6d4b\u6846\u76f8\u4f3c\u5ea6\u7684\u8ba1\u7b97\u65b9\u5f0f</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">dev"
                        "ice\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u8bad\u7ec3\u4f7f\u7528\u663e\u5361\uff1aCPU\u6216GPU\uff08cuda:0\uff09</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u6a21\u5f0f\u9009\u62e9\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">yolo\u3001\u8bed\u4e49\u5206\u5272\u3001\u591a\u5206\u7c7b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">multi_scale_able\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u4f7f\u80fd\u5bf9\u7f51\u7edc\u8f93\u5165\u56fe\u50cf\u8fdb\u884c\u968f\u673a\u7f29\u653e</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-in"
                        "dent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">multi_scale\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u7f29\u653e\u56fe\u50cf\u8303\u56f4\uff08-m,m\uff09</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u5206\u7c7b\u635f\u5931\u6b63\u4f8b\u6743\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u8ba1\u7b97\u5206\u7c7b\u635f\u5931\u65f6\uff0c\u7ed9\u6b63\u4f8b\u635f\u5931\u7684\u6743\u503c</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u79cd\u7c7b\u6b63\u8d1f\u6837\u672c\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u79cd\u7c7b\u4ee5o"
                        "ne-hot\u5f62\u5f0f\u8ba1\u7b97\u635f\u5931\uff0c\u5176\u4e2d\u6b63\u6837\u672c\u503c\u4e3a1-0.5x\uff0c\u8d1f\u6837\u672c\u503c\u4e3a0.5x</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u7f6e\u4fe1\u5ea6\u635f\u5931\u6b63\u4f8b\u6743\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u8ba1\u7b97\u7f6e\u4fe1\u5ea6\u635f\u5931\u65f6\uff0c\u7ed9\u6b63\u4f8b\u635f\u5931\u7684\u6743\u503c</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">box\u635f\u5931\u52a0\u6743\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">box\u635f\u5931\u503c\u4e58\u4ee5\u8be5\u52a0\u6743\u503c\uff0c\u52a0\u5febbox\u635f\u5931\u6536\u655b</span></p>\n"
"<p s"
                        "tyle=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u7f6e\u4fe1\u5ea6\u635f\u5931\u52a0\u6743\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u540c\u4e0a</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u5206\u7c7b\u635f\u5931\u52a0\u6743\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u540c\u4e0a</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u6b63\u6837\u672c\u7f6e\u4fe1\u5ea6\u6743\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">(1-gr) + gr*iou "
                        "  \u9650\u5236 0 - 1</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">fl_gamma</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt;\">\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u5f53gamma\u503c\u5927\u4e8e0\uff0cfocal_loss\u88ab\u542f\u7528</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9884\u9009\u6846\u5224\u65ad\u9608\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u9608\u503c\u9700\u5927\u4e8e1\u3002\u9ed8\u8ba44\uff0c\u9650\u5236\u9884\u6d4b\u4e0e\u9884\u9009\u6846\u7684wh\u6bd4\u503c\u4e3a(1/t,t)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-"
                        "indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9a8c\u8bc1\u7528\u7f6e\u4fe1\u5ea6\u9608\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u6d4b\u8bd5\u9a8c\u8bc1\u96c6\u65f6\u7684\u7f6e\u4fe1\u5ea6\u9608\u503c\uff0c\u9ed8\u8ba40.1</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u9a8c\u8bc1\u7528iou\u9608\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u6d4b\u8bd5\u9a8c\u8bc1\u96c6\u65f6\u7684iou\u9608\u503c\uff0c\u8d8a\u5c0f\uff0c\u540c\u79cd\u7c7b\u91cd\u53e0\u8d8a\u5c11</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u6d4b\u8bd5\u7528\u7f6e\u4fe1\u5ea6\u9608\u503c"
                        "\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u624b\u52a8\u6d4b\u8bd5\u65f6\uff0c\u7f6e\u4fe1\u5ea6\u9608\u503c\uff0c\u9ed8\u8ba40.5</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u5b8b\u4f53'; font-size:12pt; font-weight:600;\">\u6d4b\u8bd5\u7528iou\u9608\u503c\uff1a</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:11pt;\">\u624b\u52a8\u6d4b\u8bd5\u65f6\uff0ciou\u9608\u503c\uff0c\u9ed8\u8ba40.3</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'\u5b8b\u4f53'; font-size:10pt;\"><br /></p></body></html>", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"\u8bf4\u660e1", None))
        self.textEdit_2.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'SimSun'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">class_num\uff1a</span><span style=\" font-size:11pt;\">\u6570\u636e\u96c6\u6807\u7b7e\u7c7b\u522b\u6570\u91cf</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">image_size\uff1a</span><span style=\" font-size:11pt;\">\u7f51\u7edc\u6a21\u578b\u8f93\u5165\u56fe\u50cf\u7684size\uff0c\u81f3\u5c11\u4e3a2\u7684\u500d\u6570</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margi"
                        "n-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">batch_size\uff1a</span><span style=\" font-size:11pt;\">\u6279\u5904\u7406\u7684\u6bcf\u6279\u56fe\u50cf\u6570\u91cf</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">image_type\uff1a</span><span style=\" font-size:11pt;\">\u56fe\u50cf\u7c7b\u578b\uff1acolor/gray  \u5f69\u8272\u6216\u8005\u7070\u5ea6\u5bf9\u5e94\u901a\u90533/1</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u9a8c\u8bc1\u96c6\uff1a</span><span style=\" font-size:11pt;\">\u662f\u5426\u5b58\u5728\u9a8c\u8bc1\u96c6</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style="
                        "\" font-size:12pt; font-weight:600;\">\u81ea\u52a8\u9884\u9009\u6846\uff1a</span><span style=\" font-size:11pt;\">\u5f53\u8bbe\u5b9a\u7684\u9884\u9009\u6846\u4e0d\u7b26\u5408\u8981\u6c42\u662f\uff0c\u7a0b\u5e8f\u81ea\u52a8\u805a\u7c7b\u9884\u9009\u6846</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u81ea\u52a8batch size\uff1a</span><span style=\" font-size:11pt;\">\u81ea\u52a8\u9009\u62e9\u6700\u9002\u5408\u7684batch size</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u7f13\u5b58\u56fe\u50cf\uff1a</span><span style=\" font-size:11pt;\">\u5c06\u6240\u6709\u56fe\u50cf\u4e00\u6b21\u6027\u52a0\u8f7d\uff0c\u5bf9\u5185\u5b58\u5360\u7528\u8f83\u5927</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0p"
                        "x; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u5206\u7c7b\u63d0\u53d6\u7279\u5f81\uff1a</span><span style=\" font-size:11pt;\">\u4ece\u56fe\u50cf\u4e2d\u622a\u53d6\u7279\u5f81\uff0c\u5206\u7c7b\u4fdd\u5b58</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u5355\u4e00\u79cd\u7c7b\uff1a</span><span style=\" font-size:12pt;\">\u5c06\u6570\u636e\u96c6\u6240\u6709\u79cd\u7c7b\u7edf\u4e00\u4e3a\u4e00\u4e2a\u79cd\u7c7b</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u56fe\u50cf\u9002\u5e94\u6539\u8fdb\u6cd5\uff1a</span><span style=\" font-size:11pt;\">\u662f\u5426\u5bf9\u56fe\u50cf\u957f\u5bbd\u8fdb\u884c\u6539\u8fdb\uff0c\u957f\u6216\u5bbd\u4e3aimage_size\uff0c\u53e6\u4e00\u8fb9\u4e3arect_size\u7684"
                        "\u500d\u6570\uff0c\u9ed8\u8ba432</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u6570\u636e\u589e\u5f3a\uff1a</span><span style=\" font-size:11pt;\">\u662f\u5426\u5bf9\u56fe\u50cf\u8fdb\u884c\u6570\u636e\u589e\u5f3a\uff1a\u4eff\u5c04\u53d8\u6362\uff0c\u8272\u5f69\u53d8\u6362\u7b49</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u56fe\u50cf\u968f\u673a\u4eff\u5c04\u53d8\u6362\uff1a</span><span style=\" font-size:11pt;\">\u662f\u5426\u5728\u6570\u636e\u589e\u5f3a\u65f6\u5bf9\u56fe\u50cf\u8fdb\u884c\u968f\u673a\u4eff\u5c04\u53d8\u6362\uff1a\u89d2\u5ea6\u3001\u79fb\u52a8\u3001\u7f29\u653e\u3001\u9519\u5207</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
                        "<span style=\" font-size:12pt; font-weight:600;\">hsv\u8272\u5f69\u7a7a\u95f4\u968f\u673a\u589e\u76ca\uff1a</span><span style=\" font-size:11pt;\">\u662f\u5426\u5728\u6570\u636e\u589e\u5f3a\u65f6\u5bf9\u56fe\u50cf\u7684hsv\u8272\u5f69\u7a7a\u95f4\u8fdb\u884c\u968f\u673a\u589e\u76ca\uff0c\u53ea\u6709\u5f69\u8272\u56fe\u50cf\u65f6\u53ef\u7528</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u56fe\u50cf\u968f\u673a\u5de6\u53f3\u7ffb\u8f6c\uff1a</span><span style=\" font-size:11pt;\">\u662f\u5426\u5728\u6570\u636e\u589e\u5f3a\u65f6\u5bf9\u56fe\u50cf\u8fdb\u884c\u968f\u673a\u5de6\u53f3\u7ffb\u8f6c\u6982\u738750%</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">\u56fe\u50cf\u968f\u673a\u4e0a\u4e0b\u7ffb\u8f6c\uff1a</span><span style=\" font-size:"
                        "11pt;\">\u540c\u4e0a</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">degrees\uff1a</span><span style=\" font-size:11pt;\">\u968f\u673a\u65cb\u8f6c\u89d2\u5ea6\uff1a\uff08-d\uff0cd\uff09 0 - 180</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">translate\uff1a</span><span style=\" font-size:11pt;\">\u968f\u673a\u79fb\u52a8\u50cf\u7d20\u6570\uff1a\uff08-t*img_size,t*img_size\uff09</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">scale\uff1a</span><span style=\" font-size:11pt;\">\u968f\u673a\u7f29\u653e\u6bd4\u503c\uff1a\uff081-s,1+s\uff09,\uff080\uff0c1\uff09</span></p>\n"
"<p style=\" margin-top:0px; margin"
                        "-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">shear\uff1a</span><span style=\" font-size:11pt;\">\u968f\u673a\u9519\u5207\u89d2\u5ea6\uff1a\uff08-s,s\uff09 0 - 180</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">HSV_H\uff1a</span><span style=\" font-size:11pt;\">HSV\u8272\u76f8\u589e\u76ca</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">HSV_S\uff1a</span><span style=\" font-size:11pt;\">HSV\u9971\u548c\u5ea6\u589e\u76ca</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">HSV_V\uff1a</span><span style=\" font-size:11pt;"
                        "\">HSV\u8272\u8c03\u589e\u76ca</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt;\"><br /></p></body></html>", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate("MainWindow", u"\u8bf4\u660e2", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u6587\u4ef6", None))
    # retranslateUi


    # region 选择图像
    def licon_imagePB_clicked(self):
        try:
            image_filename, _ = QFileDialog.getOpenFileName(None, "测试图像", "./", "*.jpg *.png *bmp")
            if image_filename == "":
                return
            image_dir = Path(image_filename).parent
            # 获取所有文件名
            full_path = os.walk(image_dir)
            for dirname, fileport, image_names in full_path:  # 文件夹路径，文件夹，文件
                if fileport != []:
                    break
            for img_name in image_names:
                hz = os.path.splitext(img_name)[-1]
                if hz not in [".jpg", ".jpeg", ".bmp", ".png"]:
                    image_names.remove(img_name)
            image_names = natsorted(image_names, alg=ns.PATH)
            self.image_path = [dirname + "\\" + img_name for img_name in image_names]
            self.imgs = [public_method.load_img(img_path, self.image_type_CBB.currentText()) for img_path in
                         self.image_path]
            self.num = image_names.index(os.path.split(image_filename)[-1])

            self.show_img(self.imgs[self.num])
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())

    # endregion
    # region 下一张图像
    def down_img_clicked(self):
        if self.num < len(self.image_path) - 1:
            self.num += 1
            self.show_img(self.imgs[self.num])

    # endregion
    # region上一张图像
    def up_img_clicked(self):
        if self.num > 0:
            self.num -= 1
            self.show_img(self.imgs[self.num])

    # endregion

    # region 显示图像
    def show_img(self, img):
        try:
            self.imageLB.setGeometry(0, 0, self.frame.width(), self.frame.height())
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if self.image_type_CBB.currentText() == "color" else img
            img, r0, radio, dwh = public_method.pad_img(img, (self.imageLB.height(), self.imageLB.width()),
                                                        self.image_type_CBB.currentText())

            pix = public_method.array2pixmap(img, self.image_type_CBB.currentText())
            self.imageLB.box = None
            self.imageLB.setPixmap(pix)
            self.imageLB.update()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())

    # endregion

    # region 适应大小
    def resizeEvent(self, e):
        try:
            self.show_img(self.imgs[self.num])
        except:
            None

    # endregion

    # region 聚焦到文本最底部
    def cursor_to_lowPB_clicked(self):
        self.outTE.moveCursor(QTextCursor.End)

    # endregion

    # region 加载所有网络结构名称
    def load_nets_name(self):
        try:
            self.net_nameCBB.clear()
            nets_path = self.project_path + "\\nets\\"

            nets = os.walk(nets_path)
            for dirname, fileport, nets_name in nets:
                if fileport != []:
                    break
            nets_name = [os.path.splitext(net_name)[0] for net_name in nets_name]
            self.net_nameCBB.addItems(nets_name)
            self.net_nameCBB.showPopup()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())

    # endregion

    # region 加载所有模型名称
    def load_models_name(self):
        try:
            self.model_nameCBB.clear()
            models_path = self.project_path + "\\runs\\models\\"
            modeles = os.walk(models_path)
            modelss_name = []
            for dirname, fileport, modeles_name in modeles:
                if fileport != []:
                    break
            for model_name in modeles_name:
                if os.path.splitext(model_name)[-1] == ".onnx":
                    continue
                modelss_name.append(os.path.splitext(model_name)[0])
            self.model_nameCBB.addItems(modelss_name)
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())

    # endregion

    # region 加载项目
    def load_project(self):
        try:
            # 加载训练信息
            with open(self.project_path + "\\config\\learning_config.config", "r") as f:
                lines = f.readlines()
            learning_config = {}
            for line in lines:
                key, value = line.split("=")
                learning_config[key.rstrip().lstrip()] = value.rstrip()
            # 获取所有网络结构和模型
            self.load_nets_name()
            self.load_models_name()

            self.net_nameCBB.setCurrentText(learning_config["net name"])
            self.batch_sizeSB.setValue(int(learning_config["batch size"]))
            self.learn_rate_initDSB.setValue(float(learning_config["learn rate init"]))
            self.learn_rate_finalDSB.setValue(float(learning_config["learn rate final"]))
            self.momentumDSB.setValue(float(learning_config["momentum"]))
            self.weight_decayDSB.setValue(float(learning_config["weight decay"]))
            self.learning_rate_mode_cbb.setCurrentIndex(int(learning_config["learning rate mode"]))
            self.epoch_numSB.setValue(int(learning_config["epoch num"]))
            self.class_numSB.setValue(int(learning_config["class num"]))
            self.img_sizeSB.setValue(int(learning_config["image size"]))
            self.giou_lossDSB.setValue(float(learning_config["giou loss weight"]))
            self.obj_lossDSB.setValue(float(learning_config["obj loss weight"]))
            self.cls_lossDSB.setValue(float(learning_config["cls loss weight"]))
            self.val_conf_thres_DSB.setValue(float(learning_config["val conf thres"]))
            self.val_iou_thres_DSB.setValue(float(learning_config["val iou thres"]))
            self.test_conf_thres_DSB.setValue(float(learning_config["test conf thres"]))
            self.test_iou_thres_DSB.setValue(float(learning_config["test iou thres"]))
            self.device_CBB.setCurrentIndex(int(learning_config["device"]))
            self.learning_mode_CBB.setCurrentIndex(int(learning_config["learning mode"]))
            # Table
            c1 = learning_config["sort index"].split(",")
            c2 = learning_config["sort name"].split(",")
            s = [c1, c2]
            for i in range(2):
                for j in range(len(c1)):
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setText(s[i][j])
                    self.sort_TW.setItem(j, i, item)
            self.sort_TW.setColumnWidth(0, 70)
            self.sort_TW.setColumnWidth(1, 180)

            self.gr_DSB.setValue(float(learning_config["gr"]))
            self.anchor_tDSB.setValue(float(learning_config["anchors thres"]))
            self.model_nameCBB.setCurrentText(learning_config["model name"])
            self.image_type_CBB.setCurrentIndex(int(learning_config["image type"]))
            self.cache_img_cb.setChecked(eval(learning_config["cache img"]))
            self.updata_cache_label_cb.setChecked(eval(learning_config["updata cache label"]))
            self.extract_bounding_boxes_cb.setChecked(eval(learning_config["extract bounding boxes"]))
            self.single_cls_cb.setChecked(eval(learning_config["single cls"]))
            self.rect_cb.setChecked(eval(learning_config["rect"]))
            self.rect_size_SB.setValue(int(learning_config["rect size"]))
            self.augment_cb.setChecked(eval(learning_config["augment"]))
            self.border_cb.setChecked(eval(learning_config["border"]))
            self.augment_hsv_cb.setChecked(eval(learning_config["augment hsv"]))
            self.lr_flip_cb.setChecked(eval(learning_config["lr flip"]))
            self.ud_flip_cb.setChecked(eval(learning_config["ud flip"]))
            self.degrees_DSB.setValue(float(learning_config["degrees"]))
            self.translate_DSB.setValue(float(learning_config["translate"]))
            self.scale_DSB.setValue(float(learning_config["scale"]))
            self.shear_DSB.setValue(float(learning_config["shear"]))
            self.hsv_h_DSB.setValue(float(learning_config["hsv h"]))
            self.hsv_s_DSB.setValue(float(learning_config["hsv s"]))
            self.hsv_v_DSB.setValue(float(learning_config["hsv v"]))
            self.iou_cbb.setCurrentIndex(int(learning_config["iou"]))
            self.auto_anchor_cb.setChecked(eval(learning_config["auto anchors"]))
            self.auto_batch_size_cb.setChecked(eval(learning_config["auto batch size"]))
            self.warmup_epochsSB.setValue(int(learning_config["warmup epochs"]))
            self.warmup_bias_lrDSB.setValue(float(learning_config["warmup bias lr"]))
            self.warmup_momentumDSB.setValue(float(learning_config["warmup momentum"]))
            self.optimizers_cbb.setCurrentIndex(int(learning_config["optimizers"]))
            self.val_able_cb.setChecked(eval(learning_config["val able"]))
            self.multi_scale_able_cb.setChecked(eval(learning_config["multi scale able"]))
            self.multi_scale_DSB.setValue(float(learning_config["multi scale"]))
            self.fl_gamma_DSB.setValue(float(learning_config["fl gamma"]))
            self.cls_smooth_SB.setValue(float(learning_config["cls smooth"]))
            self.cls_pwDSB.setValue(float(learning_config["cls pw"]))
            self.obj_pwDSB.setValue(float(learning_config["obj pw"]))


        except Exception as e:
            QMessageBox.warning(None, "Error", e.__str__())

    # endregion

    # region 点击更新net
    def updata_net_clicked(self):
        self.load_nets_name()

    # endregion

    '''# region 将模型保存为onnx
    def save_as_onnx_clicked(self):
        try:
            d = "cpu"
            device, _ = select_device(d)
            # 加载模型
            model_name = self.model_nameCBB.currentText()
            ckpt = torch.load(self.project_path + "//runs//models//" + self.model_nameCBB.currentText() + ".pt",
                              map_location=device)
            model = ckpt["model"].to(device)
            model.eval()
            # 生成输入图像
            channels = 1 if self.image_type_CBB.currentText() == "gray" else 3
            modelinput = torch.randn([1, channels, 1408, 704])
            modelinput = modelinput.float().to(device) / 255.0
            # 保存
            torch.onnx.export(model,  # model being run
                              modelinput,
                              # model input (or a tuple for multiple inputs)
                              self.project_path + "//runs//models//" + self.model_nameCBB.currentText() + ".onnx",
                              # where to save the model
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=11,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['modelInput'],  # the model's input names
                              output_names=['modelOutput'],  # the model's output names
                              dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                            'modelOutput': {0: 'batch_size'}})
            session = onnxruntime.InferenceSession(
                self.project_path + "//runs//models//" + self.model_nameCBB.currentText() + ".onnx", None);
            result = session.run(None, {session.get_inputs()[0].name: modelinput.numpy()});
            QMessageBox.information(self, "提示", "保存成功")
        except Exception as ex:
            QMessageBox.warning(self, "报错", ex.__str__())

    # endregion'''

    # region 测试
    def testPB_clicked(self):
        try:
            d = "" if self.device_CBB.currentText() == "GPU" else "cpu"
            device, _ = select_device(d)

            if self.model == None or self.model_name != self.model_nameCBB.currentText():
                t1 = time.clock()
                self.model_name = self.model_nameCBB.currentText()
                ckpt = torch.load(self.project_path + "//runs//models//" + self.model_name + ".pt",
                                  map_location=device)
                self.model = ckpt["model"]
                self.model.eval()
                t2 = time.clock()
                self.outTE.append("加载模型耗时：%1.5g ms" % ((t2 - t1) * 1000))

            self.model.model[-1].conf_thres = self.test_conf_thres_DSB.value()
            self.model.model[-1].iou_thres = self.test_iou_thres_DSB.value()
            img = self.imgs[self.num]
            shape = [0, 0]
            if self.rect_cb.isChecked():
                h, w = img.shape[:2]
                s = h / w
                if s > 1:
                    shape = [1, 1 / s]
                else:
                    shape = [s, 1]
                shape = np.ceil(np.array(shape) * self.img_sizeSB.value() / float(self.rect_size_SB.value())).astype(
                    np.int) * self.rect_size_SB.value()
            else:
                shape = [self.img_sizeSB.value(), self.img_sizeSB.value()]
            img_train, r0, radio, dwh = public_method.pad_img(img,
                                                              shape,
                                                              self.image_type_CBB.currentText())
            h1, w1 = img_train.shape[0:2]
            img_train = img_train.reshape(h1, w1, 1) if self.image_type_CBB.currentText() != "color" else img_train
            img_train = img_train[:, :, ::-1].transpose(2, 0,
                                                        1) if self.image_type_CBB.currentText() == "color" else img_train.transpose(
                2, 0, 1)  # BGR to RGB to 3*h*w
            img_train = torch.from_numpy(np.ascontiguousarray(img_train))
            t1 = time.clock()
            pre_out, _ = self.model(img_train.unsqueeze(0).float().to(device) / 255.0)
            output = pre_out
            t2 = time.clock()
            run_time = (t2 - t1) * 1000
            box = None
            conf = None
            cls = None
            if output != [None] and len(output[0]) > 0:
                h, w = img.shape[:2]
                clip_coords(output[0], shape)
                op = torch.zeros((len(output[0]), 6))
                for i, o in enumerate(output[0]):
                    op[i] = o
                box = xyxy2xywh(op[:, :4])
                box[:, 0] = (box[:, 0] - dwh[0]) / (radio[0] * r0)
                box[:, 1] = (box[:, 1] - dwh[1]) / (radio[1] * r0)
                box[:, 2] = box[:, 2] / (radio[0] * r0)
                box[:, 3] = box[:, 3] / (radio[1] * r0)
                conf = op[:, 4]
                cls = op[:, 5]
            self.outTE.append(f"box:{box}  conf:{conf}   cls:{cls} run time:%1.5g" % run_time + "\n")

            # 种类名
            cls_name = {}
            for j in range(self.class_numSB.value()):
                cls_name[int(self.sort_TW.item(j, 0).text())] = self.sort_TW.item(j, 1).text()
            if box != None:
                # 适应大小
                img, r, radio, (dw, dh) = public_method.pad_img(img, (self.imageLB.height(), self.imageLB.width()),
                                                                self.image_type_CBB.currentText())
                box[:, 0] = box[:, 0] * r * radio[0] + dw
                box[:, 1] = box[:, 1] * r * radio[1] + dh
                box[:, 2] = box[:, 2] * r * radio[0]
                box[:, 3] = box[:, 3] * r * radio[1]
                self.imageLB.box = box
                self.imageLB.conf = conf
                self.imageLB.cls = cls
                self.imageLB.cls_name = cls_name
                self.imageLB.run_time = run_time
                self.imageLB.img_pix = public_method.array2pixmap(img, self.image_type_CBB.currentText())
                self.imageLB.paint = True
                self.imageLB.update()

        except Exception as e:
            QMessageBox.warning(None, "Error", e.__str__())
    # endregion