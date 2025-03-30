# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'k_means.ui'
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
from PySide6.QtWidgets import (QApplication, QFormLayout, QGridLayout, QLabel,
    QLineEdit, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QStatusBar,
    QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(855, 617)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.labels_pathLE = QLineEdit(self.centralwidget)
        self.labels_pathLE.setObjectName(u"labels_pathLE")

        self.gridLayout_2.addWidget(self.labels_pathLE, 0, 0, 1, 2)

        self.licon_labelsPB = QPushButton(self.centralwidget)
        self.licon_labelsPB.setObjectName(u"licon_labelsPB")
        self.licon_labelsPB.setMaximumSize(QSize(40, 16777215))

        self.gridLayout_2.addWidget(self.licon_labelsPB, 0, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(551, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 3, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(10)
        font.setBold(False)
        self.label.setFont(font)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.img_widthSB = QSpinBox(self.centralwidget)
        self.img_widthSB.setObjectName(u"img_widthSB")
        self.img_widthSB.setEnabled(True)
        self.img_widthSB.setMaximum(10000)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.img_widthSB)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_2)

        self.img_heightSB = QSpinBox(self.centralwidget)
        self.img_heightSB.setObjectName(u"img_heightSB")
        self.img_heightSB.setMaximum(10000)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.img_heightSB)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_3)

        self.epoch_numSB = QSpinBox(self.centralwidget)
        self.epoch_numSB.setObjectName(u"epoch_numSB")
        self.epoch_numSB.setMaximum(10000)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.epoch_numSB)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_4)

        self.boxes_numSB = QSpinBox(self.centralwidget)
        self.boxes_numSB.setObjectName(u"boxes_numSB")
        self.boxes_numSB.setMaximum(10000)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.boxes_numSB)

        self.trainPB = QPushButton(self.centralwidget)
        self.trainPB.setObjectName(u"trainPB")

        self.formLayout.setWidget(4, QFormLayout.SpanningRole, self.trainPB)


        self.gridLayout_2.addLayout(self.formLayout, 1, 0, 1, 1)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")

        self.gridLayout_2.addLayout(self.gridLayout, 1, 1, 1, 3)

        self.outTE = QTextEdit(self.centralwidget)
        self.outTE.setObjectName(u"outTE")

        self.gridLayout_2.addWidget(self.outTE, 2, 0, 1, 4)

        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setRowStretch(1, 5)
        self.gridLayout_2.setRowStretch(2, 1)
        self.gridLayout_2.setColumnStretch(0, 2)
        self.gridLayout_2.setColumnStretch(1, 2)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout_2.setColumnStretch(3, 12)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 855, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.licon_labelsPB.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"img_width:", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"img_height:", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"epoch_num:", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"boxes_num:", None))
        self.trainPB.setText(QCoreApplication.translate("MainWindow", u"train", None))
    # retranslateUi

