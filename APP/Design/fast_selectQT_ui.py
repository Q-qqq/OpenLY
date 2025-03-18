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
    QTabWidget, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(539, 231)
        Form.setWindowOpacity(1.000000000000000)
        Form.setStyleSheet(u"")
        self.gridLayout_6 = QGridLayout(Form)
        self.gridLayout_6.setSpacing(0)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_5 = QGridLayout(self.frame)
        self.gridLayout_5.setSpacing(2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_11 = QSpacerItem(450, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer_11, 3, 0, 1, 1)

        self.Cancel_pb = QPushButton(self.frame)
        self.Cancel_pb.setObjectName(u"Cancel_pb")
        self.Cancel_pb.setMaximumSize(QSize(40, 20))
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(9)
        self.Cancel_pb.setFont(font)

        self.gridLayout_5.addWidget(self.Cancel_pb, 3, 2, 1, 1)

        self.Fast_sel_methods_tw = QTabWidget(self.frame)
        self.Fast_sel_methods_tw.setObjectName(u"Fast_sel_methods_tw")
        self.threshold_tab = QWidget()
        self.threshold_tab.setObjectName(u"threshold_tab")
        self.gridLayout_2 = QGridLayout(self.threshold_tab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(0)
        self.gridLayout_2.setVerticalSpacing(15)
        self.gridLayout_2.setContentsMargins(-1, 20, -1, -1)
        self.label_3 = QLabel(self.threshold_tab)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)

        self.Thre_lo_th_hs = QSlider(self.threshold_tab)
        self.Thre_lo_th_hs.setObjectName(u"Thre_lo_th_hs")
        self.Thre_lo_th_hs.setStyleSheet(u"")
        self.Thre_lo_th_hs.setMaximum(255)
        self.Thre_lo_th_hs.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_2.addWidget(self.Thre_lo_th_hs, 0, 1, 1, 3)

        self.Thre_lo_th_sb = QSpinBox(self.threshold_tab)
        self.Thre_lo_th_sb.setObjectName(u"Thre_lo_th_sb")
        self.Thre_lo_th_sb.setMaximum(255)

        self.gridLayout_2.addWidget(self.Thre_lo_th_sb, 0, 4, 1, 1)

        self.label_4 = QLabel(self.threshold_tab)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)

        self.Thre_up_th_hs = QSlider(self.threshold_tab)
        self.Thre_up_th_hs.setObjectName(u"Thre_up_th_hs")
        self.Thre_up_th_hs.setStyleSheet(u"")
        self.Thre_up_th_hs.setMaximum(255)
        self.Thre_up_th_hs.setValue(255)
        self.Thre_up_th_hs.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_2.addWidget(self.Thre_up_th_hs, 1, 1, 1, 3)

        self.Thre_up_th_sb = QSpinBox(self.threshold_tab)
        self.Thre_up_th_sb.setObjectName(u"Thre_up_th_sb")
        self.Thre_up_th_sb.setMaximum(255)
        self.Thre_up_th_sb.setValue(255)

        self.gridLayout_2.addWidget(self.Thre_up_th_sb, 1, 4, 1, 1)

        self.horizontalSpacer = QSpacerItem(208, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 2, 0, 1, 2)

        self.Thre_cre_sel_pb = QPushButton(self.threshold_tab)
        self.Thre_cre_sel_pb.setObjectName(u"Thre_cre_sel_pb")
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(10)
        self.Thre_cre_sel_pb.setFont(font1)

        self.gridLayout_2.addWidget(self.Thre_cre_sel_pb, 2, 2, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(224, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_2, 2, 3, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 2, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 3, 2, 1, 1)

        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(3, 1)
        self.Fast_sel_methods_tw.addTab(self.threshold_tab, "")
        self.grabout_tab = QWidget()
        self.grabout_tab.setObjectName(u"grabout_tab")
        self.gridLayout_3 = QGridLayout(self.grabout_tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setVerticalSpacing(15)
        self.widget = QWidget(self.grabout_tab)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.GC_posi_fore_g_rb = QRadioButton(self.widget)
        self.GC_posi_fore_g_rb.setObjectName(u"GC_posi_fore_g_rb")
        self.GC_posi_fore_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_posi_fore_g_rb, 0, 3, 1, 1)

        self.GC_fore_g_rb = QRadioButton(self.widget)
        self.GC_fore_g_rb.setObjectName(u"GC_fore_g_rb")
        self.GC_fore_g_rb.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.GC_fore_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_fore_g_rb, 0, 1, 1, 1)

        self.GC_posi_back_g_rb = QRadioButton(self.widget)
        self.GC_posi_back_g_rb.setObjectName(u"GC_posi_back_g_rb")
        self.GC_posi_back_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_posi_back_g_rb, 1, 3, 1, 1)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_14, 0, 4, 1, 1)

        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_13, 0, 0, 1, 1)

        self.GC_back_g_rb = QRadioButton(self.widget)
        self.GC_back_g_rb.setObjectName(u"GC_back_g_rb")
        self.GC_back_g_rb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.GC_back_g_rb, 1, 1, 1, 1)

        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_15, 0, 2, 1, 1)


        self.gridLayout_3.addWidget(self.widget, 0, 0, 1, 5)

        self.horizontalSpacer_7 = QSpacerItem(110, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_7, 1, 0, 1, 1)

        self.GC_cre_sel_pb = QPushButton(self.grabout_tab)
        self.GC_cre_sel_pb.setObjectName(u"GC_cre_sel_pb")

        self.gridLayout_3.addWidget(self.GC_cre_sel_pb, 1, 1, 1, 1)

        self.horizontalSpacer_9 = QSpacerItem(110, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_9, 1, 2, 1, 1)

        self.GC_cre_mask_pb = QPushButton(self.grabout_tab)
        self.GC_cre_mask_pb.setObjectName(u"GC_cre_mask_pb")

        self.gridLayout_3.addWidget(self.GC_cre_mask_pb, 1, 3, 1, 1)

        self.horizontalSpacer_8 = QSpacerItem(110, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_8, 1, 4, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(403, 7, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_3, 2, 0, 1, 5)

        self.Fast_sel_methods_tw.addTab(self.grabout_tab, "")
        self.floodfill_tab = QWidget()
        self.floodfill_tab.setObjectName(u"floodfill_tab")
        self.gridLayout_4 = QGridLayout(self.floodfill_tab)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setVerticalSpacing(5)
        self.label = QLabel(self.floodfill_tab)
        self.label.setObjectName(u"label")

        self.gridLayout_4.addWidget(self.label, 1, 0, 1, 1)

        self.label_2 = QLabel(self.floodfill_tab)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_4.addWidget(self.label_2, 2, 0, 1, 1)

        self.FF_color_ch_cbb = QComboBox(self.floodfill_tab)
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.addItem("")
        self.FF_color_ch_cbb.setObjectName(u"FF_color_ch_cbb")
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(8)
        self.FF_color_ch_cbb.setFont(font2)

        self.gridLayout_4.addWidget(self.FF_color_ch_cbb, 0, 5, 1, 1)

        self.FF_lo_diff_hs = QSlider(self.floodfill_tab)
        self.FF_lo_diff_hs.setObjectName(u"FF_lo_diff_hs")
        self.FF_lo_diff_hs.setStyleSheet(u"")
        self.FF_lo_diff_hs.setMaximum(255)
        self.FF_lo_diff_hs.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_4.addWidget(self.FF_lo_diff_hs, 1, 1, 1, 4)

        self.FF_lo_diff_sb = QSpinBox(self.floodfill_tab)
        self.FF_lo_diff_sb.setObjectName(u"FF_lo_diff_sb")
        self.FF_lo_diff_sb.setMaximum(255)

        self.gridLayout_4.addWidget(self.FF_lo_diff_sb, 1, 5, 1, 1)

        self.FF_cre_sel_pb = QPushButton(self.floodfill_tab)
        self.FF_cre_sel_pb.setObjectName(u"FF_cre_sel_pb")
        self.FF_cre_sel_pb.setFont(font1)

        self.gridLayout_4.addWidget(self.FF_cre_sel_pb, 3, 1, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(80, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_5, 3, 4, 1, 2)

        self.horizontalSpacer_4 = QSpacerItem(105, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_4, 3, 2, 1, 1)

        self.FF_up_diff_hs = QSlider(self.floodfill_tab)
        self.FF_up_diff_hs.setObjectName(u"FF_up_diff_hs")
        self.FF_up_diff_hs.setStyleSheet(u"")
        self.FF_up_diff_hs.setMaximum(255)
        self.FF_up_diff_hs.setValue(0)
        self.FF_up_diff_hs.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_4.addWidget(self.FF_up_diff_hs, 2, 1, 1, 4)

        self.horizontalSpacer_6 = QSpacerItem(427, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_6, 0, 0, 1, 5)

        self.FF_up_diff_sb = QSpinBox(self.floodfill_tab)
        self.FF_up_diff_sb.setObjectName(u"FF_up_diff_sb")
        self.FF_up_diff_sb.setMaximum(255)
        self.FF_up_diff_sb.setValue(0)

        self.gridLayout_4.addWidget(self.FF_up_diff_sb, 2, 5, 1, 1)

        self.FF_sel_seed_pb = QPushButton(self.floodfill_tab)
        self.FF_sel_seed_pb.setObjectName(u"FF_sel_seed_pb")
        self.FF_sel_seed_pb.setFont(font1)

        self.gridLayout_4.addWidget(self.FF_sel_seed_pb, 3, 3, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(81, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_3, 3, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(28, 62, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_4.addItem(self.verticalSpacer_2, 4, 2, 1, 1)

        self.gridLayout_4.setColumnStretch(1, 1)
        self.gridLayout_4.setColumnStretch(2, 1)
        self.gridLayout_4.setColumnStretch(3, 1)
        self.Fast_sel_methods_tw.addTab(self.floodfill_tab, "")

        self.gridLayout_5.addWidget(self.Fast_sel_methods_tw, 1, 0, 1, 3)

        self.Finish_pb = QPushButton(self.frame)
        self.Finish_pb.setObjectName(u"Finish_pb")
        self.Finish_pb.setMaximumSize(QSize(40, 20))
        self.Finish_pb.setFont(font)

        self.gridLayout_5.addWidget(self.Finish_pb, 3, 1, 1, 1)

        self.Window_head_w = QWidget(self.frame)
        self.Window_head_w.setObjectName(u"Window_head_w")
        self.Window_head_w.setMinimumSize(QSize(0, 30))

        self.gridLayout_5.addWidget(self.Window_head_w, 0, 0, 1, 3)


        self.gridLayout_6.addWidget(self.frame, 0, 0, 1, 1)

        self.widget_2 = QWidget(Form)
        self.widget_2.setObjectName(u"widget_2")
        self.horizontalLayout = QHBoxLayout(self.widget_2)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.gridLayout_6.addWidget(self.widget_2, 0, 1, 1, 1)

        self.widget_5 = QWidget(Form)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_4 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_4.setSpacing(3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)

        self.gridLayout_6.addWidget(self.widget_5, 0, 2, 1, 1)


        self.retranslateUi(Form)
        self.FF_lo_diff_hs.valueChanged.connect(self.FF_lo_diff_sb.setValue)
        self.FF_up_diff_hs.valueChanged.connect(self.FF_up_diff_sb.setValue)
        self.FF_lo_diff_sb.valueChanged.connect(self.FF_lo_diff_hs.setValue)
        self.FF_up_diff_sb.valueChanged.connect(self.FF_up_diff_hs.setValue)
        self.Thre_lo_th_hs.valueChanged.connect(self.Thre_lo_th_sb.setValue)
        self.Thre_up_th_hs.valueChanged.connect(self.Thre_up_th_sb.setValue)
        self.Thre_up_th_sb.valueChanged.connect(self.Thre_up_th_hs.setValue)
        self.Thre_lo_th_sb.valueChanged.connect(self.Thre_lo_th_hs.setValue)

        self.Fast_sel_methods_tw.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u5feb\u901f\u9009\u62e9", None))
        self.Cancel_pb.setText(QCoreApplication.translate("Form", u"\u53d6\u6d88", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u9608\u503c\u4e0b\u9650\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"\u9608\u503c\u4e0a\u9650\uff1a", None))
        self.Thre_cre_sel_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u9009\u533a", None))
        self.Fast_sel_methods_tw.setTabText(self.Fast_sel_methods_tw.indexOf(self.threshold_tab), QCoreApplication.translate("Form", u"threshold", None))
        self.GC_posi_fore_g_rb.setText(QCoreApplication.translate("Form", u"\u53ef\u80fd\u7684\u524d\u666f", None))
        self.GC_fore_g_rb.setText(QCoreApplication.translate("Form", u"\u524d\u666f", None))
        self.GC_posi_back_g_rb.setText(QCoreApplication.translate("Form", u"\u53ef\u80fd\u7684\u80cc\u666f", None))
        self.GC_back_g_rb.setText(QCoreApplication.translate("Form", u"\u80cc\u666f", None))
        self.GC_cre_sel_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u9009\u533a", None))
        self.GC_cre_mask_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u63a9\u819c", None))
        self.Fast_sel_methods_tw.setTabText(self.Fast_sel_methods_tw.indexOf(self.grabout_tab), QCoreApplication.translate("Form", u"grabout", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u989c\u8272\u5bb9\u5dee\u4e0b\u9650\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"\u989c\u8272\u5bb9\u5dee\u4e0a\u9650\uff1a", None))
        self.FF_color_ch_cbb.setItemText(0, QCoreApplication.translate("Form", u"RGB", None))
        self.FF_color_ch_cbb.setItemText(1, QCoreApplication.translate("Form", u"R", None))
        self.FF_color_ch_cbb.setItemText(2, QCoreApplication.translate("Form", u"G", None))
        self.FF_color_ch_cbb.setItemText(3, QCoreApplication.translate("Form", u"B", None))

        self.FF_cre_sel_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa\u9009\u533a", None))
        self.FF_sel_seed_pb.setText(QCoreApplication.translate("Form", u"\u9009\u53d6\u79cd\u5b50\u70b9", None))
        self.Fast_sel_methods_tw.setTabText(self.Fast_sel_methods_tw.indexOf(self.floodfill_tab), QCoreApplication.translate("Form", u"floodfill", None))
        self.Finish_pb.setText(QCoreApplication.translate("Form", u"\u5b8c\u6210", None))
    # retranslateUi

