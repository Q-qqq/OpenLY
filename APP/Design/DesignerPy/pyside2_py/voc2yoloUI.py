# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'voc2yoloQT.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(668, 270)
        Form.setStyleSheet(u"background-color: rgb(251, 255, 213);")
        self.gridLayout_7 = QGridLayout(Form)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.frame = QFrame(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setStyleSheet(u"background-color: rgb(223, 221, 255);")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.frame)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.VOC_path_le = QLineEdit(self.frame)
        self.VOC_path_le.setObjectName(u"VOC_path_le")
        font = QFont()
        font.setFamily(u"\u5b8b\u4f53")
        font.setPointSize(10)
        self.VOC_path_le.setFont(font)
        self.VOC_path_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.horizontalLayout.addWidget(self.VOC_path_le)

        self.Browse_voc_pb = QPushButton(self.frame)
        self.Browse_voc_pb.setObjectName(u"Browse_voc_pb")
        self.Browse_voc_pb.setMaximumSize(QSize(25, 16777215))
        self.Browse_voc_pb.setFont(font)
        self.Browse_voc_pb.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.horizontalLayout.addWidget(self.Browse_voc_pb)


        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 1, 1, 1)

        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        font1 = QFont()
        font1.setFamily(u"\u5b8b\u4f53")
        font1.setPointSize(11)
        self.label.setFont(font1)

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font1)

        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.YOLO_path_le = QLineEdit(self.frame)
        self.YOLO_path_le.setObjectName(u"YOLO_path_le")
        self.YOLO_path_le.setFont(font)
        self.YOLO_path_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.horizontalLayout_2.addWidget(self.YOLO_path_le)

        self.Browse_yolo_pb = QPushButton(self.frame)
        self.Browse_yolo_pb.setObjectName(u"Browse_yolo_pb")
        self.Browse_yolo_pb.setMaximumSize(QSize(25, 16777215))
        self.Browse_yolo_pb.setFont(font)
        self.Browse_yolo_pb.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.horizontalLayout_2.addWidget(self.Browse_yolo_pb)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 1, 1, 1)


        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.frame, 0, 0, 1, 3)

        self.horizontalSpacer = QSpacerItem(134, 67, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer, 0, 3, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(135, 55, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_2, 1, 0, 1, 1)

        self.frame_2 = QFrame(Form)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setStyleSheet(u"background-color: rgb(225, 255, 217);")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(5, 5, 5, 5)
        self.label_3 = QLabel(self.frame_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font1)

        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)

        self.Img_dn_le = QLineEdit(self.frame_2)
        self.Img_dn_le.setObjectName(u"Img_dn_le")
        self.Img_dn_le.setFont(font)
        self.Img_dn_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_4.addWidget(self.Img_dn_le, 0, 1, 1, 1)

        self.label_4 = QLabel(self.frame_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font1)

        self.gridLayout_4.addWidget(self.label_4, 1, 0, 1, 1)

        self.Annotation_dn_le = QLineEdit(self.frame_2)
        self.Annotation_dn_le.setObjectName(u"Annotation_dn_le")
        self.Annotation_dn_le.setFont(font)
        self.Annotation_dn_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_4.addWidget(self.Annotation_dn_le, 1, 1, 1, 1)


        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.frame_2, 1, 2, 1, 3)

        self.frame_3 = QFrame(Form)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setStyleSheet(u"background-color: rgb(215, 255, 242);")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.frame_3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(5, 5, 5, 5)
        self.label_5 = QLabel(self.frame_3)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font1)

        self.gridLayout_6.addWidget(self.label_5, 0, 0, 1, 1)

        self.Annotation_suffix_le = QLineEdit(self.frame_3)
        self.Annotation_suffix_le.setObjectName(u"Annotation_suffix_le")
        self.Annotation_suffix_le.setFont(font)
        self.Annotation_suffix_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_6.addWidget(self.Annotation_suffix_le, 0, 1, 1, 1)

        self.label_6 = QLabel(self.frame_3)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font1)

        self.gridLayout_6.addWidget(self.label_6, 1, 0, 1, 1)

        self.Sets_name_le = QLineEdit(self.frame_3)
        self.Sets_name_le.setObjectName(u"Sets_name_le")
        self.Sets_name_le.setFont(font)
        self.Sets_name_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_6.addWidget(self.Sets_name_le, 1, 1, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_6, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.frame_3, 2, 0, 1, 3)

        self.horizontalSpacer_3 = QSpacerItem(278, 55, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_3, 2, 3, 1, 2)

        self.horizontalSpacer_5 = QSpacerItem(135, 23, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_5, 3, 1, 1, 1)

        self.Convert_pb = QPushButton(Form)
        self.Convert_pb.setObjectName(u"Convert_pb")
        font2 = QFont()
        font2.setFamily(u"Agency FB")
        font2.setPointSize(11)
        font2.setBold(False)
        font2.setWeight(50)
        self.Convert_pb.setFont(font2)
        self.Convert_pb.setStyleSheet(u"background-color: rgb(255, 239, 14);")

        self.gridLayout_7.addWidget(self.Convert_pb, 3, 2, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(135, 23, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_4, 3, 4, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PNG TO YOLO", None))
        self.Browse_voc_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.label.setText(QCoreApplication.translate("Form", u"VOC\u6570\u636e\u96c6\uff08\u52a0\u8f7d\uff09\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"YOLO\u6570\u636e\u96c6\uff08\u5b58\u50a8\uff09\uff1a", None))
        self.Browse_yolo_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u56fe\u50cf\u6587\u4ef6\u5939\u540d\u79f0\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u6807\u7b7e\u6587\u4ef6\u5939\u540d\u79f0\uff1a", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"\u6807\u7b7e\u6587\u4ef6\u540e\u7f00\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"\u6570\u636e\u96c6\u5206\u62e3\u8def\u5f84\u540d\u79f0\uff1a", None))
        self.Convert_pb.setText(QCoreApplication.translate("Form", u"\u8f6c\u6362", None))
    # retranslateUi

