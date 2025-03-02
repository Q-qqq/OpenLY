# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'levels_augmentQT.ui'
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
        Form.resize(333, 507)
        Form.setStyleSheet(u"")
        self.gridLayout_3 = QGridLayout(Form)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setFamily(u"Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)

        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)

        self.Channels_cbb = QComboBox(Form)
        self.Channels_cbb.addItem("")
        self.Channels_cbb.addItem("")
        self.Channels_cbb.addItem("")
        self.Channels_cbb.addItem("")
        self.Channels_cbb.setObjectName(u"Channels_cbb")
        self.Channels_cbb.setStyleSheet(u"background-color: rgb(255, 255, 255);")

        self.gridLayout_3.addWidget(self.Channels_cbb, 0, 1, 1, 1)

        self.Init_pb = QPushButton(Form)
        self.Init_pb.setObjectName(u"Init_pb")
        self.Init_pb.setMaximumSize(QSize(50, 16777215))
        font1 = QFont()
        font1.setFamily(u"Arial")
        font1.setPointSize(10)
        font1.setBold(False)
        font1.setWeight(50)
        self.Init_pb.setFont(font1)

        self.gridLayout_3.addWidget(self.Init_pb, 0, 2, 1, 1)

        self.Hide_augment_cb = QCheckBox(Form)
        self.Hide_augment_cb.setObjectName(u"Hide_augment_cb")

        self.gridLayout_3.addWidget(self.Hide_augment_cb, 0, 3, 1, 1)

        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFont(font)
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(5)
        self.gridLayout.setVerticalSpacing(1)
        self.gridLayout.setContentsMargins(5, 2, 5, 2)
        self.In_shadow_sb = QSpinBox(self.groupBox)
        self.In_shadow_sb.setObjectName(u"In_shadow_sb")
        font2 = QFont()
        font2.setFamily(u"Arial")
        font2.setPointSize(10)
        self.In_shadow_sb.setFont(font2)
        self.In_shadow_sb.setMaximum(255)

        self.gridLayout.addWidget(self.In_shadow_sb, 3, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(75, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 3, 2, 1, 1)

        self.In_gray_dsb = QDoubleSpinBox(self.groupBox)
        self.In_gray_dsb.setObjectName(u"In_gray_dsb")
        self.In_gray_dsb.setFont(font1)
        self.In_gray_dsb.setMinimum(0.010000000000000)
        self.In_gray_dsb.setMaximum(9.990000000000000)
        self.In_gray_dsb.setValue(1.000000000000000)

        self.gridLayout.addWidget(self.In_gray_dsb, 3, 3, 1, 1)

        self.widget = QWidget(self.groupBox)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.In_shadow_hs = QSlider(self.widget)
        self.In_shadow_hs.setObjectName(u"In_shadow_hs")
        self.In_shadow_hs.setStyleSheet(u"")
        self.In_shadow_hs.setMaximum(255)
        self.In_shadow_hs.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.In_shadow_hs)

        self.In_gray_hs = QSlider(self.widget)
        self.In_gray_hs.setObjectName(u"In_gray_hs")
        self.In_gray_hs.setStyleSheet(u"")
        self.In_gray_hs.setMinimum(1)
        self.In_gray_hs.setMaximum(999)
        self.In_gray_hs.setValue(128)
        self.In_gray_hs.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.In_gray_hs)

        self.In_light_hs = QSlider(self.widget)
        self.In_light_hs.setObjectName(u"In_light_hs")
        self.In_light_hs.setStyleSheet(u"")
        self.In_light_hs.setMaximum(255)
        self.In_light_hs.setValue(255)
        self.In_light_hs.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.In_light_hs)


        self.gridLayout.addWidget(self.widget, 2, 0, 1, 7)

        self.In_light_sb = QSpinBox(self.groupBox)
        self.In_light_sb.setObjectName(u"In_light_sb")
        self.In_light_sb.setFont(font2)
        self.In_light_sb.setMaximum(255)
        self.In_light_sb.setValue(255)

        self.gridLayout.addWidget(self.In_light_sb, 3, 5, 1, 1)

        self.Hist_show_F = QFrame(self.groupBox)
        self.Hist_show_F.setObjectName(u"Hist_show_F")
        self.Hist_show_F.setMinimumSize(QSize(0, 150))
        self.Hist_show_F.setBaseSize(QSize(0, 0))
        self.Hist_show_F.setFrameShape(QFrame.StyledPanel)
        self.Hist_show_F.setFrameShadow(QFrame.Plain)

        self.gridLayout.addWidget(self.Hist_show_F, 0, 0, 1, 7)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(0, 25))
        self.label_5.setStyleSheet(u"background-color: qlineargradient(spread:pad,x1:0, y1:0, x2:1, y2:0,stop:0 #232323,stop:1 #FFFFFF);")

        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 7)

        self.horizontalSpacer_2 = QSpacerItem(76, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 3, 4, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox, 1, 0, 1, 4)

        self.groupBox_2 = QGroupBox(Form)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(5, 2, 5, 2)
        self.widget_2 = QWidget(self.groupBox_2)
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_2 = QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.Out_shadow_hs = QSlider(self.widget_2)
        self.Out_shadow_hs.setObjectName(u"Out_shadow_hs")
        self.Out_shadow_hs.setStyleSheet(u"")
        self.Out_shadow_hs.setMaximum(255)
        self.Out_shadow_hs.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.Out_shadow_hs)

        self.Out_light_hs = QSlider(self.widget_2)
        self.Out_light_hs.setObjectName(u"Out_light_hs")
        self.Out_light_hs.setStyleSheet(u"")
        self.Out_light_hs.setMaximum(255)
        self.Out_light_hs.setValue(255)
        self.Out_light_hs.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.Out_light_hs)


        self.gridLayout_2.addWidget(self.widget_2, 1, 0, 1, 5)

        self.Out_light_sb = QSpinBox(self.groupBox_2)
        self.Out_light_sb.setObjectName(u"Out_light_sb")
        self.Out_light_sb.setFont(font2)
        self.Out_light_sb.setMaximum(255)
        self.Out_light_sb.setValue(255)

        self.gridLayout_2.addWidget(self.Out_light_sb, 2, 3, 1, 1)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 25))
        self.label_2.setStyleSheet(u"background-color: qlineargradient(spread:pad,x1:0, y1:0, x2:1, y2:0,stop:0 #232323,stop:1 #FFFFFF);")

        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 5)

        self.horizontalSpacer_3 = QSpacerItem(225, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_3, 2, 2, 1, 1)

        self.Out_shadow_sb = QSpinBox(self.groupBox_2)
        self.Out_shadow_sb.setObjectName(u"Out_shadow_sb")
        self.Out_shadow_sb.setFont(font2)
        self.Out_shadow_sb.setMaximum(255)

        self.gridLayout_2.addWidget(self.Out_shadow_sb, 2, 1, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_2, 2, 0, 1, 4)

        self.verticalSpacer = QSpacerItem(20, 9, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer, 3, 0, 1, 1)

        self.gridLayout_3.setColumnStretch(1, 4)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label.setText(QCoreApplication.translate("Form", u"Channels:", None))
        self.Channels_cbb.setItemText(0, QCoreApplication.translate("Form", u"RGB", None))
        self.Channels_cbb.setItemText(1, QCoreApplication.translate("Form", u"R", None))
        self.Channels_cbb.setItemText(2, QCoreApplication.translate("Form", u"G", None))
        self.Channels_cbb.setItemText(3, QCoreApplication.translate("Form", u"B", None))

#if QT_CONFIG(tooltip)
        self.Channels_cbb.setToolTip(QCoreApplication.translate("Form", u"\u56fe\u50cf\u989c\u8272\u901a\u9053", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.Init_pb.setToolTip(QCoreApplication.translate("Form", u"\u521d\u59cb\u5316", None))
#endif // QT_CONFIG(tooltip)
        self.Init_pb.setText(QCoreApplication.translate("Form", u"\u521d\u59cb\u5316", None))
#if QT_CONFIG(tooltip)
        self.Hide_augment_cb.setToolTip(QCoreApplication.translate("Form", u"\u9690\u85cf\u589e\u5f3a\u5c0f\u59d1\u54e6", None))
#endif // QT_CONFIG(tooltip)
        self.Hide_augment_cb.setText(QCoreApplication.translate("Form", u"\u9690\u85cf", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Input Levels", None))
#if QT_CONFIG(tooltip)
        self.In_shadow_hs.setToolTip(QCoreApplication.translate("Form", u"\u8f93\u5165\u9ed1\u573a\u9608\u503c\uff1a\u9ed1\u7684\u66f4\u9ed1", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.In_gray_hs.setToolTip(QCoreApplication.translate("Form", u"\u8f93\u5165\u7070\u573a\u503c\uff1a\u7070\u7684\u53d8\u4eae\u6216\u53d8\u6697", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.In_light_hs.setToolTip(QCoreApplication.translate("Form", u"\u8f93\u5165\u767d\u573a\u9608\u503c\uff1a\u767d\u7684\u66f4\u767d", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.Hist_show_F.setToolTip(QCoreApplication.translate("Form", u"\u56fe\u50cf\u7070\u9636\u56fe", None))
#endif // QT_CONFIG(tooltip)
        self.label_5.setText("")
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"Output Levels", None))
#if QT_CONFIG(tooltip)
        self.Out_shadow_hs.setToolTip(QCoreApplication.translate("Form", u"\u8f93\u51fa\u9ed1\u573a\u9608\u503c\uff1a\u9ed1\u7684\u53d8\u767d", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.Out_light_hs.setToolTip(QCoreApplication.translate("Form", u"\u8f93\u51fa\u767d\u573a\u9608\u503c\uff1a\u767d\u7684\u53d8\u9ed1", None))
#endif // QT_CONFIG(tooltip)
        self.label_2.setText("")
    # retranslateUi

