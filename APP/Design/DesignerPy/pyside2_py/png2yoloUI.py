# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'png2yoloQT.ui'
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
        Form.resize(319, 405)
        self.gridLayout_3 = QGridLayout(Form)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        font = QFont()
        font.setFamily(u"Arial")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setVerticalSpacing(15)
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        font1 = QFont()
        font1.setFamily(u"Arial")
        font1.setPointSize(10)
        self.label_5.setFont(font1)

        self.gridLayout.addWidget(self.label_5, 0, 0, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setFont(font1)

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font1)

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.Seg_suffix_cbb = QComboBox(self.groupBox)
        self.Seg_suffix_cbb.addItem("")
        self.Seg_suffix_cbb.addItem("")
        self.Seg_suffix_cbb.addItem("")
        self.Seg_suffix_cbb.addItem("")
        self.Seg_suffix_cbb.setObjectName(u"Seg_suffix_cbb")
        self.Seg_suffix_cbb.setFont(font1)

        self.gridLayout.addWidget(self.Seg_suffix_cbb, 0, 1, 1, 1)

        self.widget_2 = QWidget(self.groupBox)
        self.widget_2.setObjectName(u"widget_2")
        self.horizontalLayout_2 = QHBoxLayout(self.widget_2)
        self.horizontalLayout_2.setSpacing(2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.Seg_train_p_le = QLineEdit(self.widget_2)
        self.Seg_train_p_le.setObjectName(u"Seg_train_p_le")
        self.Seg_train_p_le.setFont(font1)

        self.horizontalLayout_2.addWidget(self.Seg_train_p_le)

        self.Seg_train_p_browse_pb = QPushButton(self.widget_2)
        self.Seg_train_p_browse_pb.setObjectName(u"Seg_train_p_browse_pb")
        self.Seg_train_p_browse_pb.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_2.addWidget(self.Seg_train_p_browse_pb)


        self.gridLayout.addWidget(self.widget_2, 1, 1, 1, 1)

        self.widget_3 = QWidget(self.groupBox)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_3 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_3.setSpacing(2)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.Seg_val_p_le = QLineEdit(self.widget_3)
        self.Seg_val_p_le.setObjectName(u"Seg_val_p_le")
        self.Seg_val_p_le.setFont(font1)

        self.horizontalLayout_3.addWidget(self.Seg_val_p_le)

        self.Seg_val_p_browse_pb = QPushButton(self.widget_3)
        self.Seg_val_p_browse_pb.setObjectName(u"Seg_val_p_browse_pb")
        self.Seg_val_p_browse_pb.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_3.addWidget(self.Seg_val_p_browse_pb)


        self.gridLayout.addWidget(self.widget_3, 2, 1, 1, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 5)

        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 3)

        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setVerticalSpacing(15)
        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font1)

        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 1)

        self.Ori_suffix_cbb = QComboBox(self.groupBox_2)
        self.Ori_suffix_cbb.addItem("")
        self.Ori_suffix_cbb.addItem("")
        self.Ori_suffix_cbb.addItem("")
        self.Ori_suffix_cbb.addItem("")
        self.Ori_suffix_cbb.setObjectName(u"Ori_suffix_cbb")
        self.Ori_suffix_cbb.setFont(font1)

        self.gridLayout_2.addWidget(self.Ori_suffix_cbb, 0, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font1)

        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font1)

        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)

        self.widget_4 = QWidget(self.groupBox_2)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_4 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_4.setSpacing(2)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.Ori_train_p_le = QLineEdit(self.widget_4)
        self.Ori_train_p_le.setObjectName(u"Ori_train_p_le")
        self.Ori_train_p_le.setFont(font1)

        self.horizontalLayout_4.addWidget(self.Ori_train_p_le)

        self.Ori_train_p_browse_pb = QPushButton(self.widget_4)
        self.Ori_train_p_browse_pb.setObjectName(u"Ori_train_p_browse_pb")
        self.Ori_train_p_browse_pb.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_4.addWidget(self.Ori_train_p_browse_pb)


        self.gridLayout_2.addWidget(self.widget_4, 1, 1, 1, 1)

        self.widget = QWidget(self.groupBox_2)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.Ori_val_p_le = QLineEdit(self.widget)
        self.Ori_val_p_le.setObjectName(u"Ori_val_p_le")
        self.Ori_val_p_le.setFont(font1)

        self.horizontalLayout.addWidget(self.Ori_val_p_le)

        self.Ori_val_p_browse_pb = QPushButton(self.widget)
        self.Ori_val_p_browse_pb.setObjectName(u"Ori_val_p_browse_pb")
        self.Ori_val_p_browse_pb.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout.addWidget(self.Ori_val_p_browse_pb)


        self.gridLayout_2.addWidget(self.widget, 2, 1, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 5)

        self.gridLayout_3.addWidget(self.groupBox_2, 1, 0, 1, 3)

        self.groupBox_3 = QGroupBox(Form)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setFont(font1)
        self.groupBox_3.setAlignment(Qt.AlignCenter)
        self.horizontalLayout_5 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font1)

        self.horizontalLayout_5.addWidget(self.label_7)

        self.Yolo_p_le = QLineEdit(self.groupBox_3)
        self.Yolo_p_le.setObjectName(u"Yolo_p_le")
        self.Yolo_p_le.setFont(font1)

        self.horizontalLayout_5.addWidget(self.Yolo_p_le)

        self.Yolo_p_browse_pb = QPushButton(self.groupBox_3)
        self.Yolo_p_browse_pb.setObjectName(u"Yolo_p_browse_pb")
        self.Yolo_p_browse_pb.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_5.addWidget(self.Yolo_p_browse_pb)


        self.gridLayout_3.addWidget(self.groupBox_3, 2, 0, 1, 3)

        self.horizontalSpacer_2 = QSpacerItem(100, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_2, 3, 0, 1, 1)

        self.Convert_pb = QPushButton(Form)
        self.Convert_pb.setObjectName(u"Convert_pb")
        font2 = QFont()
        font2.setFamily(u"Arial")
        font2.setPointSize(11)
        self.Convert_pb.setFont(font2)

        self.gridLayout_3.addWidget(self.Convert_pb, 3, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(100, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer, 3, 2, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"PNG TO YOLO", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"\u5206\u5272\u56fe\u50cf", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"\u56fe\u50cf\u540e\u7f00\uff1a", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u8bad\u7ec3\u8def\u5f84\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u9a8c\u8bc1\u8def\u5f84\uff1a", None))
        self.Seg_suffix_cbb.setItemText(0, QCoreApplication.translate("Form", u"png", None))
        self.Seg_suffix_cbb.setItemText(1, QCoreApplication.translate("Form", u"jpg", None))
        self.Seg_suffix_cbb.setItemText(2, QCoreApplication.translate("Form", u"bnp", None))
        self.Seg_suffix_cbb.setItemText(3, QCoreApplication.translate("Form", u"jpeg", None))

        self.Seg_train_p_browse_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.Seg_val_p_browse_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"\u539f\u56fe\u50cf", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"\u56fe\u50cf\u540e\u7f00\uff1a", None))
        self.Ori_suffix_cbb.setItemText(0, QCoreApplication.translate("Form", u"jpg", None))
        self.Ori_suffix_cbb.setItemText(1, QCoreApplication.translate("Form", u"png", None))
        self.Ori_suffix_cbb.setItemText(2, QCoreApplication.translate("Form", u"bnp", None))
        self.Ori_suffix_cbb.setItemText(3, QCoreApplication.translate("Form", u"jpeg", None))

        self.label_3.setText(QCoreApplication.translate("Form", u"\u8bad\u7ec3\u8def\u5f84\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u9a8c\u8bc1\u8def\u5f84\uff1a", None))
        self.Ori_train_p_browse_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.Ori_val_p_browse_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"YOLO", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"yolo\u8def\u5f84\uff1a", None))
        self.Yolo_p_browse_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.Convert_pb.setText(QCoreApplication.translate("Form", u"\u8f6c\u6362", None))
    # retranslateUi

