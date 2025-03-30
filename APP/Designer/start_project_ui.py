# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'start_project.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QSpacerItem, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1165, 539)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setFamilies([u"\u65b0\u5b8b\u4f53"])
        font.setPointSize(22)
        self.label.setFont(font)

        self.verticalLayout.addWidget(self.label)

        self.delect_project_PB = QPushButton(self.centralwidget)
        self.delect_project_PB.setObjectName(u"delect_project_PB")
        self.delect_project_PB.setMaximumSize(QSize(100, 16777215))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(12)
        self.delect_project_PB.setFont(font1)

        self.verticalLayout.addWidget(self.delect_project_PB)

        self.project_path_LW = QListWidget(self.centralwidget)
        self.project_path_LW.setObjectName(u"project_path_LW")
        font2 = QFont()
        font2.setFamilies([u"\u5b8b\u4f53"])
        font2.setPointSize(14)
        self.project_path_LW.setFont(font2)

        self.verticalLayout.addWidget(self.project_path_LW)


        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        font3 = QFont()
        font3.setFamilies([u"\u5b8b\u4f53"])
        font3.setPointSize(20)
        self.groupBox.setFont(font3)
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(0)
        self.gridLayout.setVerticalSpacing(30)
        self.gridLayout.setContentsMargins(-1, 50, -1, -1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(100, 16777215))
        font4 = QFont()
        font4.setFamilies([u"\u5b8b\u4f53"])
        font4.setPointSize(12)
        self.label_4.setFont(font4)

        self.horizontalLayout_2.addWidget(self.label_4)

        self.new_project_dir_LE = QLineEdit(self.groupBox)
        self.new_project_dir_LE.setObjectName(u"new_project_dir_LE")
        self.new_project_dir_LE.setFont(font4)

        self.horizontalLayout_2.addWidget(self.new_project_dir_LE)

        self.licon_project_dir_pb = QPushButton(self.groupBox)
        self.licon_project_dir_pb.setObjectName(u"licon_project_dir_pb")
        self.licon_project_dir_pb.setMaximumSize(QSize(30, 16777215))
        font5 = QFont()
        font5.setFamilies([u"\u5b8b\u4f53"])
        font5.setPointSize(9)
        self.licon_project_dir_pb.setFont(font5)

        self.horizontalLayout_2.addWidget(self.licon_project_dir_pb)

        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 7)
        self.horizontalLayout_2.setStretch(2, 3)

        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(100, 16777215))
        self.label_3.setFont(font4)

        self.horizontalLayout_3.addWidget(self.label_3)

        self.new_project_name_LE = QLineEdit(self.groupBox)
        self.new_project_name_LE.setObjectName(u"new_project_name_LE")
        self.new_project_name_LE.setFont(font4)

        self.horizontalLayout_3.addWidget(self.new_project_name_LE)


        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 1, 1, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.create_new_project = QPushButton(self.groupBox)
        self.create_new_project.setObjectName(u"create_new_project")
        self.create_new_project.setMaximumSize(QSize(600, 16777215))
        self.create_new_project.setFont(font4)

        self.horizontalLayout_4.addWidget(self.create_new_project)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)


        self.gridLayout.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_4, 2, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 6, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 3, 0, 1, 1)

        self.line = QFrame(self.groupBox)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line, 4, 0, 1, 2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(100, 16777215))
        self.label_5.setFont(font4)

        self.horizontalLayout.addWidget(self.label_5)

        self.exist_project_dir_LE = QLineEdit(self.groupBox)
        self.exist_project_dir_LE.setObjectName(u"exist_project_dir_LE")
        self.exist_project_dir_LE.setFont(font4)

        self.horizontalLayout.addWidget(self.exist_project_dir_LE)

        self.licon_exit_project_dir_pb = QPushButton(self.groupBox)
        self.licon_exit_project_dir_pb.setObjectName(u"licon_exit_project_dir_pb")
        self.licon_exit_project_dir_pb.setMaximumSize(QSize(30, 16777215))
        self.licon_exit_project_dir_pb.setFont(font5)

        self.horizontalLayout.addWidget(self.licon_exit_project_dir_pb)


        self.gridLayout.addLayout(self.horizontalLayout, 5, 0, 1, 2)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_5)

        self.Add_exist_project_pb = QPushButton(self.groupBox)
        self.Add_exist_project_pb.setObjectName(u"Add_exist_project_pb")
        self.Add_exist_project_pb.setMaximumSize(QSize(600, 16777215))
        self.Add_exist_project_pb.setFont(font4)

        self.horizontalLayout_5.addWidget(self.Add_exist_project_pb)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_6)


        self.gridLayout.addLayout(self.horizontalLayout_5, 6, 0, 1, 1)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_7, 6, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 37, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 7, 0, 1, 1)

        self.gridLayout.setColumnStretch(0, 9)

        self.gridLayout_2.addWidget(self.groupBox, 0, 1, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1165, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Project", None))
        self.delect_project_PB.setText(QCoreApplication.translate("MainWindow", u"DELETE", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"New Project", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u65b0\u9879\u76ee\u8def\u5f84\uff1a", None))
        self.licon_project_dir_pb.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u65b0\u9879\u76ee\u540d\u79f0\uff1a", None))
        self.create_new_project.setText(QCoreApplication.translate("MainWindow", u"CREATE", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u73b0\u6709\u9879\u76ee\u8def\u5f84\uff1a", None))
        self.licon_exit_project_dir_pb.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.Add_exist_project_pb.setText(QCoreApplication.translate("MainWindow", u"Add", None))
    # retranslateUi

