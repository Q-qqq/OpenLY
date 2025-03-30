# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'new_train.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGridLayout, QGroupBox, QLabel,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QStackedWidget,
    QStatusBar, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1083, 824)
        self.actionnew_project = QAction(MainWindow)
        self.actionnew_project.setObjectName(u"actionnew_project")
        self.actionopen_project = QAction(MainWindow)
        self.actionopen_project.setObjectName(u"actionopen_project")
        self.actionclose = QAction(MainWindow)
        self.actionclose.setObjectName(u"actionclose")
        self.actiongo_project_file = QAction(MainWindow)
        self.actiongo_project_file.setObjectName(u"actiongo_project_file")
        self.actionsave = QAction(MainWindow)
        self.actionsave.setObjectName(u"actionsave")
        self.actionsave_as = QAction(MainWindow)
        self.actionsave_as.setObjectName(u"actionsave_as")
        self.actiondatasets = QAction(MainWindow)
        self.actiondatasets.setObjectName(u"actiondatasets")
        self.actionnetwork = QAction(MainWindow)
        self.actionnetwork.setObjectName(u"actionnetwork")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(350, 0, 731, 621))
        self.frame.setStyleSheet(u"background-color: rgb(0, 0, 0);\n"
"background-color: rgb(181, 181, 181);")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.pushButton_3 = QPushButton(self.frame)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setMinimumSize(QSize(0, 0))
        self.pushButton_3.setMaximumSize(QSize(50, 16777215))
        self.pushButton_3.setStyleSheet(u"background-color: rgb(235, 235, 235);\n"
"font: 12pt \"\u5e7c\u5706\";")
        self.pushButton_3.setCheckable(True)
        self.pushButton_3.setAutoExclusive(True)

        self.gridLayout.addWidget(self.pushButton_3, 2, 0, 1, 1)

        self.pushButton_2 = QPushButton(self.frame)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setMinimumSize(QSize(0, 0))
        self.pushButton_2.setMaximumSize(QSize(50, 16777215))
        self.pushButton_2.setStyleSheet(u"background-color: rgb(235, 235, 235);\n"
"font: 12pt \"\u5e7c\u5706\";")
        self.pushButton_2.setCheckable(True)
        self.pushButton_2.setAutoExclusive(True)

        self.gridLayout.addWidget(self.pushButton_2, 1, 0, 1, 1)

        self.pushButton = QPushButton(self.frame)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMinimumSize(QSize(0, 0))
        self.pushButton.setMaximumSize(QSize(50, 16777215))
        self.pushButton.setLayoutDirection(Qt.LeftToRight)
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet(u"background-color: rgb(235, 235, 235);\n"
"font: 12pt \"\u5e7c\u5706\";")
        self.pushButton.setCheckable(True)
        self.pushButton.setAutoExclusive(True)

        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 290, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.stackedWidget = QStackedWidget(self.frame)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"background-color: rgb(226, 226, 226);")
        self.Test_PAGE = QWidget()
        self.Test_PAGE.setObjectName(u"Test_PAGE")
        self.stackedWidget.addWidget(self.Test_PAGE)
        self.Mean_metrics_PAGE = QWidget()
        self.Mean_metrics_PAGE.setObjectName(u"Mean_metrics_PAGE")
        self.stackedWidget.addWidget(self.Mean_metrics_PAGE)
        self.Per_class_metrics_PAGE = QWidget()
        self.Per_class_metrics_PAGE.setObjectName(u"Per_class_metrics_PAGE")
        self.stackedWidget.addWidget(self.Per_class_metrics_PAGE)

        self.gridLayout_2.addWidget(self.stackedWidget, 0, 1, 2, 1)

        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(10, 100, 311, 641))
        self.widget.setStyleSheet(u"")
        self.groupBox = QGroupBox(self.widget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(0, 10, 301, 253))
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.gridLayout_3 = QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        font1 = QFont()
        font1.setFamilies([u"\u5e7c\u5706"])
        font1.setPointSize(12)
        self.label_5.setFont(font1)

        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)

        self.model_nameCBB = QComboBox(self.groupBox)
        self.model_nameCBB.setObjectName(u"model_nameCBB")
        font2 = QFont()
        font2.setFamilies([u"\u5b8b\u4f53"])
        font2.setPointSize(12)
        self.model_nameCBB.setFont(font2)

        self.gridLayout_3.addWidget(self.model_nameCBB, 0, 1, 1, 1)

        self.add_modelPB = QPushButton(self.groupBox)
        self.add_modelPB.setObjectName(u"add_modelPB")
        self.add_modelPB.setMaximumSize(QSize(31, 16777215))
        font3 = QFont()
        font3.setFamilies([u"\u5b8b\u4f53"])
        font3.setPointSize(9)
        self.add_modelPB.setFont(font3)

        self.gridLayout_3.addWidget(self.add_modelPB, 0, 2, 1, 1)

        self.label2 = QLabel(self.groupBox)
        self.label2.setObjectName(u"label2")
        self.label2.setFont(font1)

        self.gridLayout_3.addWidget(self.label2, 1, 0, 1, 1)

        self.label4_10 = QLabel(self.groupBox)
        self.label4_10.setObjectName(u"label4_10")
        font4 = QFont()
        font4.setFamilies([u"Arial Narrow"])
        font4.setPointSize(12)
        self.label4_10.setFont(font4)

        self.gridLayout_3.addWidget(self.label4_10, 2, 0, 1, 1)

        self.label3_6 = QLabel(self.groupBox)
        self.label3_6.setObjectName(u"label3_6")
        self.label3_6.setFont(font4)

        self.gridLayout_3.addWidget(self.label3_6, 3, 0, 1, 1)

        self.label5_2 = QLabel(self.groupBox)
        self.label5_2.setObjectName(u"label5_2")
        self.label5_2.setFont(font4)

        self.gridLayout_3.addWidget(self.label5_2, 4, 0, 1, 1)

        self.label3 = QLabel(self.groupBox)
        self.label3.setObjectName(u"label3")
        self.label3.setFont(font4)

        self.gridLayout_3.addWidget(self.label3, 5, 0, 1, 1)

        self.label1 = QLabel(self.groupBox)
        self.label1.setObjectName(u"label1")
        self.label1.setFont(font4)

        self.gridLayout_3.addWidget(self.label1, 6, 0, 1, 1)

        self.net_nameCBB = QComboBox(self.groupBox)
        self.net_nameCBB.setObjectName(u"net_nameCBB")
        self.net_nameCBB.setFont(font2)

        self.gridLayout_3.addWidget(self.net_nameCBB, 1, 1, 1, 2)

        self.device_CBB = QComboBox(self.groupBox)
        self.device_CBB.addItem("")
        self.device_CBB.addItem("")
        self.device_CBB.setObjectName(u"device_CBB")
        self.device_CBB.setFont(font2)

        self.gridLayout_3.addWidget(self.device_CBB, 2, 1, 1, 2)

        self.optimizers_cbb = QComboBox(self.groupBox)
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.addItem("")
        self.optimizers_cbb.setObjectName(u"optimizers_cbb")
        self.optimizers_cbb.setFont(font2)

        self.gridLayout_3.addWidget(self.optimizers_cbb, 3, 1, 1, 2)

        self.learn_rate_initDSB = QDoubleSpinBox(self.groupBox)
        self.learn_rate_initDSB.setObjectName(u"learn_rate_initDSB")
        self.learn_rate_initDSB.setFont(font2)
        self.learn_rate_initDSB.setMouseTracking(False)
        self.learn_rate_initDSB.setFocusPolicy(Qt.WheelFocus)
        self.learn_rate_initDSB.setAccelerated(False)
        self.learn_rate_initDSB.setDecimals(10)
        self.learn_rate_initDSB.setMaximum(10000.000000000000000)
        self.learn_rate_initDSB.setSingleStep(0.000000000000000)

        self.gridLayout_3.addWidget(self.learn_rate_initDSB, 4, 1, 1, 2)

        self.epoch_numSB = QSpinBox(self.groupBox)
        self.epoch_numSB.setObjectName(u"epoch_numSB")
        self.epoch_numSB.setFont(font2)
        self.epoch_numSB.setMaximum(10000)
        self.epoch_numSB.setSingleStep(0)

        self.gridLayout_3.addWidget(self.epoch_numSB, 5, 1, 1, 2)

        self.batch_sizeSB = QSpinBox(self.groupBox)
        self.batch_sizeSB.setObjectName(u"batch_sizeSB")
        self.batch_sizeSB.setFont(font2)
        self.batch_sizeSB.setMaximum(10000)

        self.gridLayout_3.addWidget(self.batch_sizeSB, 6, 1, 1, 2)

        self.groupBox_2 = QGroupBox(self.widget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 260, 291, 341))
        self.groupBox_2.setFont(font)
        self.checkBox = QCheckBox(self.groupBox_2)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(170, 30, 71, 16))
        self.checkBox.setFont(font2)
        self.label1_11 = QLabel(self.groupBox_2)
        self.label1_11.setObjectName(u"label1_11")
        self.label1_11.setGeometry(QRect(20, 245, 79, 20))
        self.label1_11.setFont(font4)
        self.hsv_s_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.hsv_s_DSB_2.setObjectName(u"hsv_s_DSB_2")
        self.hsv_s_DSB_2.setGeometry(QRect(105, 218, 108, 20))
        self.hsv_s_DSB_2.setFont(font2)
        self.hsv_s_DSB_2.setDecimals(10)
        self.hsv_s_DSB_2.setMaximum(1.000000000000000)
        self.hsv_s_DSB_2.setSingleStep(0.000100000000000)
        self.label1_12 = QLabel(self.groupBox_2)
        self.label1_12.setObjectName(u"label1_12")
        self.label1_12.setGeometry(QRect(20, 218, 79, 20))
        self.label1_12.setFont(font4)
        self.label1_13 = QLabel(self.groupBox_2)
        self.label1_13.setObjectName(u"label1_13")
        self.label1_13.setGeometry(QRect(20, 191, 79, 20))
        self.label1_13.setFont(font4)
        self.label1_14 = QLabel(self.groupBox_2)
        self.label1_14.setObjectName(u"label1_14")
        self.label1_14.setGeometry(QRect(20, 137, 79, 20))
        self.label1_14.setFont(font4)
        self.hsv_h_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.hsv_h_DSB_2.setObjectName(u"hsv_h_DSB_2")
        self.hsv_h_DSB_2.setGeometry(QRect(105, 191, 108, 20))
        self.hsv_h_DSB_2.setFont(font2)
        self.hsv_h_DSB_2.setDecimals(10)
        self.hsv_h_DSB_2.setMaximum(1.000000000000000)
        self.hsv_h_DSB_2.setSingleStep(0.000100000000000)
        self.label1_15 = QLabel(self.groupBox_2)
        self.label1_15.setObjectName(u"label1_15")
        self.label1_15.setGeometry(QRect(20, 164, 79, 20))
        self.label1_15.setFont(font4)
        self.translate_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.translate_DSB_2.setObjectName(u"translate_DSB_2")
        self.translate_DSB_2.setGeometry(QRect(105, 110, 108, 20))
        self.translate_DSB_2.setFont(font2)
        self.translate_DSB_2.setDecimals(10)
        self.translate_DSB_2.setMaximum(1.000000000000000)
        self.translate_DSB_2.setSingleStep(0.000100000000000)
        self.label1_16 = QLabel(self.groupBox_2)
        self.label1_16.setObjectName(u"label1_16")
        self.label1_16.setGeometry(QRect(20, 83, 79, 20))
        self.label1_16.setFont(font4)
        self.scale_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.scale_DSB_2.setObjectName(u"scale_DSB_2")
        self.scale_DSB_2.setGeometry(QRect(105, 137, 108, 20))
        self.scale_DSB_2.setFont(font2)
        self.scale_DSB_2.setDecimals(10)
        self.scale_DSB_2.setMaximum(1.000000000000000)
        self.scale_DSB_2.setSingleStep(0.000100000000000)
        self.hsv_v_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.hsv_v_DSB_2.setObjectName(u"hsv_v_DSB_2")
        self.hsv_v_DSB_2.setGeometry(QRect(105, 245, 108, 20))
        self.hsv_v_DSB_2.setFont(font2)
        self.hsv_v_DSB_2.setDecimals(10)
        self.hsv_v_DSB_2.setMaximum(1.000000000000000)
        self.hsv_v_DSB_2.setSingleStep(0.000100000000000)
        self.degrees_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.degrees_DSB_2.setObjectName(u"degrees_DSB_2")
        self.degrees_DSB_2.setGeometry(QRect(105, 83, 108, 20))
        self.degrees_DSB_2.setFont(font2)
        self.degrees_DSB_2.setDecimals(10)
        self.degrees_DSB_2.setMaximum(180.000000000000000)
        self.degrees_DSB_2.setSingleStep(0.000100000000000)
        self.shear_DSB_2 = QDoubleSpinBox(self.groupBox_2)
        self.shear_DSB_2.setObjectName(u"shear_DSB_2")
        self.shear_DSB_2.setGeometry(QRect(105, 164, 108, 20))
        self.shear_DSB_2.setFont(font2)
        self.shear_DSB_2.setDecimals(10)
        self.shear_DSB_2.setMaximum(180.000000000000000)
        self.shear_DSB_2.setSingleStep(0.000100000000000)
        self.label1_17 = QLabel(self.groupBox_2)
        self.label1_17.setObjectName(u"label1_17")
        self.label1_17.setGeometry(QRect(20, 110, 79, 20))
        self.label1_17.setFont(font4)
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(350, 630, 691, 161))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1083, 23))
        self.menu_F = QMenu(self.menubar)
        self.menu_F.setObjectName(u"menu_F")
        self.menu_S = QMenu(self.menubar)
        self.menu_S.setObjectName(u"menu_S")
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu_F.menuAction())
        self.menubar.addAction(self.menu_S.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menu_F.addAction(self.actiongo_project_file)
        self.menu_F.addAction(self.actionnew_project)
        self.menu_F.addAction(self.actionopen_project)
        self.menu_F.addAction(self.actionsave)
        self.menu_F.addAction(self.actionsave_as)
        self.menu_F.addAction(self.actionclose)
        self.menu_S.addAction(self.actiondatasets)
        self.menu_S.addAction(self.actionnetwork)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionnew_project.setText(QCoreApplication.translate("MainWindow", u"new project", None))
        self.actionopen_project.setText(QCoreApplication.translate("MainWindow", u"open project", None))
        self.actionclose.setText(QCoreApplication.translate("MainWindow", u"exit", None))
        self.actiongo_project_file.setText(QCoreApplication.translate("MainWindow", u"project file", None))
        self.actionsave.setText(QCoreApplication.translate("MainWindow", u"save", None))
        self.actionsave_as.setText(QCoreApplication.translate("MainWindow", u"save as", None))
        self.actiondatasets.setText(QCoreApplication.translate("MainWindow", u"dataset", None))
        self.actionnetwork.setText(QCoreApplication.translate("MainWindow", u"network", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"\u5404\n"
"\u79cd\n"
"\u7c7b\n"
"\u8bc4\n"
"\u4ef7\n"
"\u6307\n"
"\u6807", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\u603b\n"
"\u8bc4\n"
"\u4ef7\n"
"\u6307\n"
"\u6807", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u624b\n"
"\u52a8\n"
"\u6d4b\n"
"\u8bd5", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u91cd\u8981\u53c2\u6570", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u540d\u79f0\uff1a", None))
        self.add_modelPB.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.label2.setText(QCoreApplication.translate("MainWindow", u"\u7f51\u7edc\u7ed3\u6784\u540d\u79f0\uff1a", None))
        self.label4_10.setText(QCoreApplication.translate("MainWindow", u"device\uff1a", None))
        self.label3_6.setText(QCoreApplication.translate("MainWindow", u"\u4f18\u5316\u5668\uff1a", None))
        self.label5_2.setText(QCoreApplication.translate("MainWindow", u"\u5b66\u4e60\u7387\uff1a", None))
        self.label3.setText(QCoreApplication.translate("MainWindow", u"\u5468\u671f\u6570\uff1a", None))
        self.label1.setText(QCoreApplication.translate("MainWindow", u"\u6279\u5927\u5c0f\uff1a", None))
        self.device_CBB.setItemText(0, QCoreApplication.translate("MainWindow", u"GPU", None))
        self.device_CBB.setItemText(1, QCoreApplication.translate("MainWindow", u"CPU", None))

        self.optimizers_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"Adam", None))
        self.optimizers_cbb.setItemText(1, QCoreApplication.translate("MainWindow", u"AdamW", None))
        self.optimizers_cbb.setItemText(2, QCoreApplication.translate("MainWindow", u"RMSProp", None))
        self.optimizers_cbb.setItemText(3, QCoreApplication.translate("MainWindow", u"SGD", None))

        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u589e\u5f3a", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"\u542f\u7528", None))
        self.label1_11.setText(QCoreApplication.translate("MainWindow", u"\u8272\u8c03\uff1a", None))
        self.label1_12.setText(QCoreApplication.translate("MainWindow", u"\u9971\u548c\u5ea6\uff1a", None))
        self.label1_13.setText(QCoreApplication.translate("MainWindow", u"\u8272\u76f8\uff1a", None))
        self.label1_14.setText(QCoreApplication.translate("MainWindow", u"\u7f29\u653e\uff1a", None))
        self.label1_15.setText(QCoreApplication.translate("MainWindow", u"\u9519\u5207\uff1a", None))
        self.label1_16.setText(QCoreApplication.translate("MainWindow", u"\u65cb\u8f6c\uff1a", None))
        self.label1_17.setText(QCoreApplication.translate("MainWindow", u"\u5e73\u79fb\uff1a", None))
        self.menu_F.setTitle(QCoreApplication.translate("MainWindow", u"\u6587\u4ef6\uff08F\uff09", None))
        self.menu_S.setTitle(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e\uff08S\uff09", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u5e2e\u52a9\uff08H\uff09", None))
    # retranslateUi

