# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'fast_selectQT.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QSpacerItem, QSpinBox,
    QStackedWidget, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(533, 197)
        Form.setStyleSheet(u"")
        self.gridLayout_5 = QGridLayout(Form)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.widget_5 = QWidget(Form)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_4 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_4.setSpacing(3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)

        self.gridLayout_5.addWidget(self.widget_5, 0, 1, 1, 1)

        self.widget_2 = QWidget(Form)
        self.widget_2.setObjectName(u"widget_2")
        self.horizontalLayout = QHBoxLayout(self.widget_2)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.gridLayout_5.addWidget(self.widget_2, 0, 0, 1, 1)

        self.frame = QFrame(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.vboxLayout = QVBoxLayout(self.frame)
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.vboxLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_2 = QFrame(self.frame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.Fast_sel_methods_cbb = QComboBox(self.frame_2)
        self.Fast_sel_methods_cbb.addItem("")
        self.Fast_sel_methods_cbb.addItem("")
        self.Fast_sel_methods_cbb.addItem("")
        self.Fast_sel_methods_cbb.setObjectName(u"Fast_sel_methods_cbb")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(11)
        self.Fast_sel_methods_cbb.setFont(font)

        self.horizontalLayout_5.addWidget(self.Fast_sel_methods_cbb)

        self.horizontalSpacer_3 = QSpacerItem(122, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.FF_color_ch_cbb = QComboBox(self.frame_2)
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.setObjectName(u"FF_color_ch_cbb")
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(8)
        self.FF_color_ch_cbb.setFont(font1)

        self.horizontalLayout_5.addWidget(self.FF_color_ch_cbb)

        self.horizontalSpacer_6 = QSpacerItem(121, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_6)

        self.Finish_pb = QPushButton(self.frame_2)
        self.Finish_pb.setObjectName(u"Finish_pb")
        self.Finish_pb.setMaximumSize(QSize(40, 20))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(9)
        self.Finish_pb.setFont(font2)

        self.horizontalLayout_5.addWidget(self.Finish_pb)

        self.Cancel_pb = QPushButton(self.frame_2)
        self.Cancel_pb.setObjectName(u"Cancel_pb")
        self.Cancel_pb.setMaximumSize(QSize(40, 20))
        self.Cancel_pb.setFont(font2)

        self.horizontalLayout_5.addWidget(self.Cancel_pb)


        self.vboxLayout.addWidget(self.frame_2)

        self.stackedWidget = QStackedWidget(self.frame)
        self.stackedWidget.setObjectName(u"stackedWidget")
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(10)
        self.stackedWidget.setFont(font3)
        self.Thred_args_page = QWidget()
        self.Thred_args_page.setObjectName(u"Thred_args_page")
        self.gridLayout_2 = QGridLayout(self.Thred_args_page)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.Thre_up_th_hs = QSlider(self.Thred_args_page)
        self.Thre_up_th_hs.setObjectName(u"Thre_up_th_hs")
        self.Thre_up_th_hs.setStyleSheet(u"")
        self.Thre_up_th_hs.setMaximum(255)
        self.Thre_up_th_hs.setValue(255)
        self.Thre_up_th_hs.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.Thre_up_th_hs, 1, 1, 1, 1)

        self.Thre_up_th_sb = QSpinBox(self.Thred_args_page)
        self.Thre_up_th_sb.setObjectName(u"Thre_up_th_sb")
        self.Thre_up_th_sb.setMaximum(255)
        self.Thre_up_th_sb.setValue(255)

        self.gridLayout_2.addWidget(self.Thre_up_th_sb, 1, 2, 1, 1)

        self.Thre_lo_th_sb = QSpinBox(self.Thred_args_page)
        self.Thre_lo_th_sb.setObjectName(u"Thre_lo_th_sb")
        self.Thre_lo_th_sb.setMaximum(255)

        self.gridLayout_2.addWidget(self.Thre_lo_th_sb, 0, 2, 1, 1)

        self.label_3 = QLabel(self.Thred_args_page)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_4 = QLabel(self.Thred_args_page)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)

        self.widget_4 = QWidget(self.Thred_args_page)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_3 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer = QSpacerItem(85, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.Thre_cre_sel_pb = QPushButton(self.widget_4)
        self.Thre_cre_sel_pb.setObjectName(u"Thre_cre_sel_pb")
        self.Thre_cre_sel_pb.setFont(font3)

        self.horizontalLayout_3.addWidget(self.Thre_cre_sel_pb)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)


        self.gridLayout_2.addWidget(self.widget_4, 2, 0, 1, 3)

        self.Thre_lo_th_hs = QSlider(self.Thred_args_page)
        self.Thre_lo_th_hs.setObjectName(u"Thre_lo_th_hs")
        self.Thre_lo_th_hs.setStyleSheet(u"")
        self.Thre_lo_th_hs.setMaximum(255)
        self.Thre_lo_th_hs.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.Thre_lo_th_hs, 0, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 3, 0, 1, 1)

        self.stackedWidget.addWidget(self.Thred_args_page)
        self.Grabcut_args_page = QWidget()
        self.Grabcut_args_page.setObjectName(u"Grabcut_args_page")
        self.gridLayout_4 = QGridLayout(self.Grabcut_args_page)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.GC_cre_mask_pb = QPushButton(self.Grabcut_args_page)
        self.GC_cre_mask_pb.setObjectName(u"GC_cre_mask_pb")

        self.gridLayout_4.addWidget(self.GC_cre_mask_pb, 1, 3, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(77, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_7, 1, 0, 1, 1)

        self.horizontalSpacer_9 = QSpacerItem(78, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_9, 1, 2, 1, 1)

        self.widget = QWidget(self.Grabcut_args_page)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.GC_fore_g_rb = QRadioButton(self.widget)
        self.GC_fore_g_rb.setObjectName(u"GC_fore_g_rb")
        self.GC_fore_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_fore_g_rb, 0, 0, 1, 1)

        self.GC_posi_fore_g_rb = QRadioButton(self.widget)
        self.GC_posi_fore_g_rb.setObjectName(u"GC_posi_fore_g_rb")
        self.GC_posi_fore_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_posi_fore_g_rb, 0, 1, 1, 1)

        self.GC_back_g_rb = QRadioButton(self.widget)
        self.GC_back_g_rb.setObjectName(u"GC_back_g_rb")
        self.GC_back_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_back_g_rb, 1, 0, 1, 1)

        self.GC_posi_back_g_rb = QRadioButton(self.widget)
        self.GC_posi_back_g_rb.setObjectName(u"GC_posi_back_g_rb")
        self.GC_posi_back_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_posi_back_g_rb, 1, 1, 1, 1)


        self.gridLayout_4.addWidget(self.widget, 0, 0, 1, 5)

        self.horizontalSpacer_8 = QSpacerItem(77, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_8, 1, 4, 1, 1)

        self.GC_cre_sel_pb = QPushButton(self.Grabcut_args_page)
        self.GC_cre_sel_pb.setObjectName(u"GC_cre_sel_pb")

        self.gridLayout_4.addWidget(self.GC_cre_sel_pb, 1, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_4.addItem(self.verticalSpacer_3, 2, 2, 1, 1)

        self.stackedWidget.addWidget(self.Grabcut_args_page)
        self.Floodfill_args_page = QWidget()
        self.Floodfill_args_page.setObjectName(u"Floodfill_args_page")
        self.gridLayout_3 = QGridLayout(self.Floodfill_args_page)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.FF_lo_diff_hs = QSlider(self.Floodfill_args_page)
        self.FF_lo_diff_hs.setObjectName(u"FF_lo_diff_hs")
        self.FF_lo_diff_hs.setStyleSheet(u"")
        self.FF_lo_diff_hs.setMaximum(255)
        self.FF_lo_diff_hs.setOrientation(Qt.Horizontal)

        self.gridLayout_3.addWidget(self.FF_lo_diff_hs, 0, 1, 1, 1)

        self.label_2 = QLabel(self.Floodfill_args_page)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)

        self.FF_up_diff_hs = QSlider(self.Floodfill_args_page)
        self.FF_up_diff_hs.setObjectName(u"FF_up_diff_hs")
        self.FF_up_diff_hs.setStyleSheet(u"")
        self.FF_up_diff_hs.setMaximum(255)
        self.FF_up_diff_hs.setValue(0)
        self.FF_up_diff_hs.setOrientation(Qt.Horizontal)

        self.gridLayout_3.addWidget(self.FF_up_diff_hs, 1, 1, 1, 1)

        self.FF_up_diff_sb = QSpinBox(self.Floodfill_args_page)
        self.FF_up_diff_sb.setObjectName(u"FF_up_diff_sb")
        self.FF_up_diff_sb.setMaximum(255)
        self.FF_up_diff_sb.setValue(0)

        self.gridLayout_3.addWidget(self.FF_up_diff_sb, 1, 2, 1, 1)

        self.label = QLabel(self.Floodfill_args_page)
        self.label.setObjectName(u"label")

        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)

        self.widget_3 = QWidget(self.Floodfill_args_page)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_2 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_5 = QSpacerItem(71, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)

        self.FF_cre_sel_pb = QPushButton(self.widget_3)
        self.FF_cre_sel_pb.setObjectName(u"FF_cre_sel_pb")
        self.FF_cre_sel_pb.setFont(font3)

        self.horizontalLayout_2.addWidget(self.FF_cre_sel_pb)

        self.FF_sel_seed_pb = QPushButton(self.widget_3)
        self.FF_sel_seed_pb.setObjectName(u"FF_sel_seed_pb")
        self.FF_sel_seed_pb.setFont(font3)

        self.horizontalLayout_2.addWidget(self.FF_sel_seed_pb)

        self.horizontalSpacer_4 = QSpacerItem(47, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)


        self.gridLayout_3.addWidget(self.widget_3, 2, 0, 2, 3)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_4, 4, 0, 1, 1)

        self.FF_lo_diff_sb = QSpinBox(self.Floodfill_args_page)
        self.FF_lo_diff_sb.setObjectName(u"FF_lo_diff_sb")
        self.FF_lo_diff_sb.setMaximum(255)

        self.gridLayout_3.addWidget(self.FF_lo_diff_sb, 0, 2, 1, 1)

        self.stackedWidget.addWidget(self.Floodfill_args_page)

        self.vboxLayout.addWidget(self.stackedWidget)


        self.gridLayout_5.addWidget(self.frame, 1, 0, 1, 2)


        self.retranslateUi(Form)
        self.Fast_sel_methods_cbb.activated.connect(self.stackedWidget.setCurrentIndex)
        self.FF_lo_diff_hs.valueChanged.connect(self.FF_lo_diff_sb.setValue)
        self.FF_up_diff_hs.valueChanged.connect(self.FF_up_diff_sb.setValue)
        self.FF_lo_diff_sb.valueChanged.connect(self.FF_lo_diff_hs.setValue)
        self.FF_up_diff_sb.valueChanged.connect(self.FF_up_diff_hs.setValue)
        self.Thre_lo_th_hs.valueChanged.connect(self.Thre_lo_th_sb.setValue)
        self.Thre_up_th_hs.valueChanged.connect(self.Thre_up_th_sb.setValue)
        self.Thre_up_th_sb.valueChanged.connect(self.Thre_up_th_hs.setValue)
        self.Thre_lo_th_sb.valueChanged.connect(self.Thre_lo_th_hs.setValue)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u5feb\u901f\u9009\u62e9", None))
        self.Fast_sel_methods_cbb.setItemText(0, QCoreApplication.translate("Form", u"threshold", None))
        self.Fast_sel_methods_cbb.setItemText(1, QCoreApplication.translate("Form", u"grabcut", None))
        self.Fast_sel_methods_cbb.setItemText(2, QCoreApplication.translate("Form", u"floodfill", None))

        self.FF_color_ch_cbb.setItemText(0, QCoreApplication.translate("Form", u"RGB", None))
        self.FF_color_ch_cbb.setItemText(1, QCoreApplication.translate("Form", u"R", None))
        self.FF_color_ch_cbb.setItemText(2, QCoreApplication.translate("Form", u"G", None))
        self.FF_color_ch_cbb.setItemText(3, QCoreApplication.translate("Form", u"B", None))

        self.Finish_pb.setText(QCoreApplication.translate("Form", u"\u5b8c\u6210", None))
        self.Cancel_pb.setText(QCoreApplication.translate("Form", u"\u53d6\u6d88", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u9608\u503c\u4e0b\u9650\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u9608\u503c\u4e0a\u9650\uff1a", None))
        self.Thre_cre_sel_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u9009\u533a", None))
        self.GC_cre_mask_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u63a9\u819c", None))
        self.GC_fore_g_rb.setText(QCoreApplication.translate("Form", u"\u524d\u666f", None))
        self.GC_posi_fore_g_rb.setText(QCoreApplication.translate("Form", u"\u53ef\u80fd\u7684\u524d\u666f", None))
        self.GC_back_g_rb.setText(QCoreApplication.translate("Form", u"\u80cc\u666f", None))
        self.GC_posi_back_g_rb.setText(QCoreApplication.translate("Form", u"\u53ef\u80fd\u7684\u80cc\u666f", None))
        self.GC_cre_sel_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u9009\u533a", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u989c\u8272\u5bb9\u5dee\u4e0a\u9650\uff1a", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u989c\u8272\u5bb9\u5dee\u4e0b\u9650\uff1a", None))
        self.FF_cre_sel_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u9009\u533a", None))
        self.FF_sel_seed_pb.setText(QCoreApplication.translate("Form", u"\u9009\u53d6\u79cd\u5b50\u70b9", None))
    # retranslateUi

