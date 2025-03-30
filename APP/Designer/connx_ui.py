# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'connx.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QLabel, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(583, 216)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(50, 16777215))

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.pt_model_iconLE = QLineEdit(self.centralwidget)
        self.pt_model_iconLE.setObjectName(u"pt_model_iconLE")

        self.gridLayout_2.addWidget(self.pt_model_iconLE, 0, 1, 1, 3)

        self.icon_pt_modelPB = QPushButton(self.centralwidget)
        self.icon_pt_modelPB.setObjectName(u"icon_pt_modelPB")
        self.icon_pt_modelPB.setMaximumSize(QSize(40, 16777215))

        self.gridLayout_2.addWidget(self.icon_pt_modelPB, 0, 4, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.img_widthSB = QSpinBox(self.centralwidget)
        self.img_widthSB.setObjectName(u"img_widthSB")
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(12)
        self.img_widthSB.setFont(font)
        self.img_widthSB.setMaximum(100000)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.img_widthSB)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_5)

        self.img_heightSB = QSpinBox(self.centralwidget)
        self.img_heightSB.setObjectName(u"img_heightSB")
        self.img_heightSB.setFont(font)
        self.img_heightSB.setMaximum(100000)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.img_heightSB)

        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font)

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_6)

        self.img_channelsSB = QSpinBox(self.centralwidget)
        self.img_channelsSB.setObjectName(u"img_channelsSB")
        self.img_channelsSB.setFont(font)
        self.img_channelsSB.setMaximum(10)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.img_channelsSB)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_4)


        self.gridLayout_2.addLayout(self.formLayout, 1, 0, 1, 2)

        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setFont(font)
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.conf_threDSB = QDoubleSpinBox(self.widget)
        self.conf_threDSB.setObjectName(u"conf_threDSB")
        self.conf_threDSB.setDecimals(6)
        self.conf_threDSB.setMaximum(1.000000000000000)
        self.conf_threDSB.setSingleStep(0.000001000000000)

        self.gridLayout.addWidget(self.conf_threDSB, 1, 1, 1, 1)

        self.label_3 = QLabel(self.widget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.iou_threDSB = QDoubleSpinBox(self.widget)
        self.iou_threDSB.setObjectName(u"iou_threDSB")
        self.iou_threDSB.setDecimals(6)
        self.iou_threDSB.setMaximum(1.000000000000000)
        self.iou_threDSB.setSingleStep(0.000001000000000)

        self.gridLayout.addWidget(self.iou_threDSB, 2, 1, 1, 1)

        self.output_with_nmsCB = QCheckBox(self.widget)
        self.output_with_nmsCB.setObjectName(u"output_with_nmsCB")

        self.gridLayout.addWidget(self.output_with_nmsCB, 0, 0, 1, 2)


        self.gridLayout_2.addWidget(self.widget, 1, 2, 1, 2)

        self.horizontalSpacer = QSpacerItem(392, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 2, 0, 1, 3)

        self.create_onnxPB = QPushButton(self.centralwidget)
        self.create_onnxPB.setObjectName(u"create_onnxPB")
        self.create_onnxPB.setFont(font)

        self.gridLayout_2.addWidget(self.create_onnxPB, 2, 3, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 583, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"pt\u6a21\u578b", None))
        self.icon_pt_modelPB.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u56fe\u50cfheight\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u56fe\u50cf\u901a\u9053\u6570\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u56fe\u50cfwidth\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"NMS-\u7f6e\u4fe1\u5ea6\u9608\u503c\uff1a", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"NMS-iou\u9608\u503c\uff1a", None))
        self.output_with_nmsCB.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u51fa\u662f\u5426\u7ecf\u8fc7NMS", None))
        self.create_onnxPB.setText(QCoreApplication.translate("MainWindow", u"\u751f\u6210onnx", None))
    # retranslateUi

