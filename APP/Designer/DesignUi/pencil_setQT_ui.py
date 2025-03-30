# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pencil_setQT.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QRadioButton, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(316, 87)
        self.gridLayout_2 = QGridLayout(Form)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.widget = QWidget(Form)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.Pencil_add_rb = QRadioButton(self.widget)
        self.Pencil_add_rb.setObjectName(u"Pencil_add_rb")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(10)
        self.Pencil_add_rb.setFont(font)

        self.gridLayout.addWidget(self.Pencil_add_rb, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(167, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.Pencil_sub_pb = QRadioButton(self.widget)
        self.Pencil_sub_pb.setObjectName(u"Pencil_sub_pb")
        self.Pencil_sub_pb.setFont(font)

        self.gridLayout.addWidget(self.Pencil_sub_pb, 0, 2, 1, 1)


        self.gridLayout_2.addWidget(self.widget, 0, 0, 1, 2)

        self.Pencil_size_hs = QSlider(Form)
        self.Pencil_size_hs.setObjectName(u"Pencil_size_hs")
        self.Pencil_size_hs.setStyleSheet(u"/* \u57fa\u7840\u6ed1\u69fd */\n"
"QSlider::groove:horizontal {\n"
"    height: 4px;\n"
"    background: #e0e0e0;\n"
"    border-radius: 2px;\n"
"}\n"
"\n"
"/* \u6ed1\u52a8\u8fdb\u5ea6\u6307\u793a */\n"
"QSlider::sub-page:horizontal {\n"
"    background: #000000;\n"
"    border-radius: 2px;\n"
"}\n"
"\n"
"\n"
"/* \u6781\u7b80\u6ed1\u5757 */\n"
"QSlider::handle:horizontal {\n"
"    background: #646464;\n"
"    width: 16px;\n"
"	color: rgb(76, 76, 76);\n"
"    height: 16px;\n"
"    border-radius: 8px;\n"
"    margin: -6px 0; /* \u5782\u76f4\u5c45\u4e2d */\n"
"}\n"
"\n"
"\n"
"/* \u4ea4\u4e92\u53cd\u9988 */\n"
"QSlider::handle:hover {\n"
"    background: #232323;\n"
"}\n"
"\n"
"QSlider::handle:pressed {\n"
"    background: #000000;\n"
"}")
        self.Pencil_size_hs.setMinimum(1)
        self.Pencil_size_hs.setMaximum(10000)
        self.Pencil_size_hs.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.Pencil_size_hs, 1, 0, 1, 1)

        self.Pencil_size_sb = QSpinBox(Form)
        self.Pencil_size_sb.setObjectName(u"Pencil_size_sb")
        self.Pencil_size_sb.setFont(font)
        self.Pencil_size_sb.setMinimum(1)
        self.Pencil_size_sb.setMaximum(10000)

        self.gridLayout_2.addWidget(self.Pencil_size_sb, 1, 1, 1, 1)


        self.retranslateUi(Form)
        self.Pencil_size_hs.valueChanged.connect(self.Pencil_size_sb.setValue)
        self.Pencil_size_sb.valueChanged.connect(self.Pencil_size_hs.setValue)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u753b\u7b14\u5927\u5c0f", None))
        self.Pencil_add_rb.setText(QCoreApplication.translate("Form", u"\u7ed8\u5236", None))
        self.Pencil_sub_pb.setText(QCoreApplication.translate("Form", u"\u64e6\u9664", None))
    # retranslateUi

