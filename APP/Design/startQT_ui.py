# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'startQT.ui'
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
    QListWidgetItem, QPushButton, QSizePolicy, QSpacerItem,
    QSplitter, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.setWindowModality(Qt.WindowModality.WindowModal)
        Form.resize(861, 446)
        Form.setStyleSheet(u"")
        self.gridLayout_3 = QGridLayout(Form)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.Projs_lw = QListWidget(self.splitter)
        self.Projs_lw.setObjectName(u"Projs_lw")
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(14)
        self.Projs_lw.setFont(font)
        self.Projs_lw.setStyleSheet(u"")
        self.splitter.addWidget(self.Projs_lw)
        self.frame = QFrame(self.splitter)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(self.frame)
        self.groupBox.setObjectName(u"groupBox")
        font1 = QFont()
        font1.setFamilies([u"\u5b8b\u4f53"])
        font1.setPointSize(18)
        self.groupBox.setFont(font1)
        self.groupBox.setStyleSheet(u"")
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(16777215, 16777215))
        font2 = QFont()
        font2.setFamilies([u"\u5b8b\u4f53"])
        font2.setPointSize(12)
        self.label_3.setFont(font2)

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)

        self.New_pro_name_le = QLineEdit(self.groupBox)
        self.New_pro_name_le.setObjectName(u"New_pro_name_le")
        self.New_pro_name_le.setFont(font2)
        self.New_pro_name_le.setStyleSheet(u"")

        self.gridLayout.addWidget(self.New_pro_name_le, 1, 1, 1, 1)

        self.Create_new_pro_pb = QPushButton(self.groupBox)
        self.Create_new_pro_pb.setObjectName(u"Create_new_pro_pb")
        self.Create_new_pro_pb.setMaximumSize(QSize(600, 16777215))
        self.Create_new_pro_pb.setFont(font2)
        self.Create_new_pro_pb.setStyleSheet(u"")

        self.gridLayout.addWidget(self.Create_new_pro_pb, 1, 2, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setFont(font2)

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.New_pro_dir_le = QLineEdit(self.groupBox)
        self.New_pro_dir_le.setObjectName(u"New_pro_dir_le")
        self.New_pro_dir_le.setFont(font2)
        self.New_pro_dir_le.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.New_pro_dir_le)

        self.Browse_new_project_dir_pb = QPushButton(self.groupBox)
        self.Browse_new_project_dir_pb.setObjectName(u"Browse_new_project_dir_pb")
        self.Browse_new_project_dir_pb.setMaximumSize(QSize(30, 16777215))
        font3 = QFont()
        font3.setFamilies([u"\u5b8b\u4f53"])
        font3.setPointSize(9)
        self.Browse_new_project_dir_pb.setFont(font3)
        self.Browse_new_project_dir_pb.setStyleSheet(u"")

        self.horizontalLayout_2.addWidget(self.Browse_new_project_dir_pb)

        self.horizontalLayout_2.setStretch(0, 7)
        self.horizontalLayout_2.setStretch(1, 3)

        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 1, 1, 2)

        self.gridLayout.setColumnStretch(0, 2)
        self.gridLayout.setColumnStretch(1, 6)
        self.gridLayout.setColumnStretch(2, 1)

        self.verticalLayout.addWidget(self.groupBox)

        self.verticalSpacer = QSpacerItem(20, 225, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.groupBox_2 = QGroupBox(self.frame)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font1)
        self.groupBox_2.setStyleSheet(u"")
        self.groupBox_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.Exist_pro_path_le = QLineEdit(self.groupBox_2)
        self.Exist_pro_path_le.setObjectName(u"Exist_pro_path_le")
        self.Exist_pro_path_le.setFont(font2)
        self.Exist_pro_path_le.setStyleSheet(u"")

        self.horizontalLayout.addWidget(self.Exist_pro_path_le)

        self.Browse_exist_project_dir_pb = QPushButton(self.groupBox_2)
        self.Browse_exist_project_dir_pb.setObjectName(u"Browse_exist_project_dir_pb")
        self.Browse_exist_project_dir_pb.setMaximumSize(QSize(30, 16777215))
        self.Browse_exist_project_dir_pb.setFont(font3)
        self.Browse_exist_project_dir_pb.setStyleSheet(u"")

        self.horizontalLayout.addWidget(self.Browse_exist_project_dir_pb)


        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 1, 1, 1)

        self.Add_exist_project_pb = QPushButton(self.groupBox_2)
        self.Add_exist_project_pb.setObjectName(u"Add_exist_project_pb")
        self.Add_exist_project_pb.setMaximumSize(QSize(600, 16777215))
        self.Add_exist_project_pb.setFont(font2)
        self.Add_exist_project_pb.setStyleSheet(u"")

        self.gridLayout_2.addWidget(self.Add_exist_project_pb, 1, 2, 1, 1)

        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(100, 16777215))
        self.label_5.setFont(font2)

        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.verticalLayout.setStretch(0, 4)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 4)
        self.splitter.addWidget(self.frame)

        self.gridLayout_3.addWidget(self.splitter, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"DL\u542f\u52a8\u754c\u9762", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"New Project", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"\u65b0\u9879\u76ee\u8def\u5f84\uff1a", None))
        self.Create_new_pro_pb.setText(QCoreApplication.translate("Form", u"\u521b\u5efa", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u65b0\u9879\u76ee\u540d\u79f0\uff1a", None))
        self.Browse_new_project_dir_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"Open Project", None))
        self.Browse_exist_project_dir_pb.setText(QCoreApplication.translate("Form", u"...", None))
        self.Add_exist_project_pb.setText(QCoreApplication.translate("Form", u"\u6253\u5f00", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"\u73b0\u6709\u9879\u76ee\u8def\u5f84\uff1a", None))
    # retranslateUi

