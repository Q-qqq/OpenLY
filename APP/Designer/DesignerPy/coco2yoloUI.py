# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'coco2yoloQT.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(754, 405)
        Form.setStyleSheet(u"")
        self.gridLayout_7 = QGridLayout(Form)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.frame = QFrame(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setStyleSheet(u"")
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
        self.COCO_path_le = QLineEdit(self.frame)
        self.COCO_path_le.setObjectName(u"COCO_path_le")
        font = QFont()
        font.setFamily(u"\u5b8b\u4f53")
        font.setPointSize(10)
        self.COCO_path_le.setFont(font)
        self.COCO_path_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.horizontalLayout.addWidget(self.COCO_path_le)

        self.Browse_coco_pb = QPushButton(self.frame)
        self.Browse_coco_pb.setObjectName(u"Browse_coco_pb")
        self.Browse_coco_pb.setMaximumSize(QSize(25, 16777215))
        self.Browse_coco_pb.setFont(font)
        self.Browse_coco_pb.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.horizontalLayout.addWidget(self.Browse_coco_pb)


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


        self.gridLayout_7.addWidget(self.frame, 0, 0, 1, 2)

        self.horizontalSpacer = QSpacerItem(321, 95, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(322, 96, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_2, 1, 0, 1, 1)

        self.frame_2 = QFrame(Form)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setStyleSheet(u"")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(5, 5, 5, 5)
        self.label_4 = QLabel(self.frame_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font1)

        self.gridLayout_4.addWidget(self.label_4, 1, 0, 1, 1)

        self.Train_img_dn_le = QLineEdit(self.frame_2)
        self.Train_img_dn_le.setObjectName(u"Train_img_dn_le")
        self.Train_img_dn_le.setFont(font)
        self.Train_img_dn_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_4.addWidget(self.Train_img_dn_le, 0, 1, 1, 1)

        self.Val_img_dn_le = QLineEdit(self.frame_2)
        self.Val_img_dn_le.setObjectName(u"Val_img_dn_le")
        self.Val_img_dn_le.setFont(font)
        self.Val_img_dn_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_4.addWidget(self.Val_img_dn_le, 1, 1, 1, 1)

        self.label_3 = QLabel(self.frame_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font1)

        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_5 = QLabel(self.frame_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font1)

        self.gridLayout_4.addWidget(self.label_5, 2, 0, 1, 1)

        self.Annotations_dn_le = QLineEdit(self.frame_2)
        self.Annotations_dn_le.setObjectName(u"Annotations_dn_le")
        self.Annotations_dn_le.setFont(font)
        self.Annotations_dn_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_4.addWidget(self.Annotations_dn_le, 2, 1, 1, 1)


        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.frame_2, 1, 1, 1, 2)

        self.frame_3 = QFrame(Form)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setStyleSheet(u"")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.frame_3)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(5, 5, 5, 5)
        self.label_6 = QLabel(self.frame_3)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font1)

        self.gridLayout_6.addWidget(self.label_6, 1, 0, 1, 1)

        self.label_7 = QLabel(self.frame_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font1)

        self.gridLayout_6.addWidget(self.label_7, 2, 0, 1, 1)

        self.Task_cbb = QComboBox(self.frame_3)
        self.Task_cbb.addItem("")
        self.Task_cbb.addItem("")
        self.Task_cbb.addItem("")
        self.Task_cbb.setObjectName(u"Task_cbb")
        self.Task_cbb.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_6.addWidget(self.Task_cbb, 2, 1, 1, 1)

        self.Annotation_type_cbb = QComboBox(self.frame_3)
        self.Annotation_type_cbb.addItem("")
        self.Annotation_type_cbb.addItem("")
        self.Annotation_type_cbb.setObjectName(u"Annotation_type_cbb")
        self.Annotation_type_cbb.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_6.addWidget(self.Annotation_type_cbb, 1, 1, 1, 1)

        self.label_8 = QLabel(self.frame_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font1)

        self.gridLayout_6.addWidget(self.label_8, 0, 0, 1, 1)

        self.Annotation_suffix_le = QLineEdit(self.frame_3)
        self.Annotation_suffix_le.setObjectName(u"Annotation_suffix_le")
        self.Annotation_suffix_le.setFont(font)
        self.Annotation_suffix_le.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_6.addWidget(self.Annotation_suffix_le, 0, 1, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_6, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.frame_3, 2, 0, 1, 2)

        self.horizontalSpacer_6 = QSpacerItem(321, 95, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_6, 2, 2, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(322, 23, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_5, 3, 0, 1, 1)

        self.Convert_pb = QPushButton(Form)
        self.Convert_pb.setObjectName(u"Convert_pb")
        font2 = QFont()
        font2.setFamily(u"Agency FB")
        font2.setPointSize(11)
        font2.setBold(False)
        font2.setWeight(50)
        self.Convert_pb.setFont(font2)
        self.Convert_pb.setStyleSheet(u"")

        self.gridLayout_7.addWidget(self.Convert_pb, 3, 1, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(321, 23, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_7.addItem(self.horizontalSpacer_4, 3, 2, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"COCO TO YOLO", None))
        self.Browse_coco_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.label.setText(QCoreApplication.translate("Form", u"COCO\u6570\u636e\u96c6\uff08\u52a0\u8f7d\uff09\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"YOLO\u6570\u636e\u96c6\uff08\u5b58\u50a8\uff09\uff1a", None))
        self.Browse_yolo_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u9a8c\u8bc1\u56fe\u50cf\u6587\u4ef6\u5939\u540d\u79f0\uff1a", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u8bad\u7ec3\u56fe\u50cf\u6587\u4ef6\u5939\u540d\u79f0\uff1a", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"\u6807\u7b7e\u6587\u4ef6\u5939\u540d\u79f0\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"\u6807\u7b7e\u7c7b\u578b\uff1a", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"\u4efb\u52a1\uff1a", None))
        self.Task_cbb.setItemText(0, QCoreApplication.translate("Form", u"detect", None))
        self.Task_cbb.setItemText(1, QCoreApplication.translate("Form", u"segment", None))
        self.Task_cbb.setItemText(2, QCoreApplication.translate("Form", u"keypoint", None))

        self.Annotation_type_cbb.setItemText(0, QCoreApplication.translate("Form", u"instances", None))
        self.Annotation_type_cbb.setItemText(1, QCoreApplication.translate("Form", u"person_keypoints", None))

        self.label_8.setText(QCoreApplication.translate("Form", u"\u6807\u7b7e\u540e\u7f00\uff1a", None))
        self.Convert_pb.setText(QCoreApplication.translate("Form", u"\u8f6c\u6362", None))
    # retranslateUi

