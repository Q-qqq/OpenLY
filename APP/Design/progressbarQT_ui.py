# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'progressbarQT.ui'
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
from PySide6.QtWidgets import (QApplication, QProgressBar, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_Progress(object):
    def setupUi(self, Progress):
        if not Progress.objectName():
            Progress.setObjectName(u"Progress")
        Progress.resize(577, 180)
        self.verticalLayout = QVBoxLayout(Progress)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.ProgressBar = QProgressBar(Progress)
        self.ProgressBar.setObjectName(u"ProgressBar")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ProgressBar.sizePolicy().hasHeightForWidth())
        self.ProgressBar.setSizePolicy(sizePolicy)
        self.ProgressBar.setStyleSheet(u"")
        self.ProgressBar.setValue(24)

        self.verticalLayout.addWidget(self.ProgressBar)

        self.Show_mes_te = QTextEdit(Progress)
        self.Show_mes_te.setObjectName(u"Show_mes_te")
        self.Show_mes_te.setStyleSheet(u"")

        self.verticalLayout.addWidget(self.Show_mes_te)


        self.retranslateUi(Progress)

        QMetaObject.connectSlotsByName(Progress)
    # setupUi

    def retranslateUi(self, Progress):
        Progress.setWindowTitle(QCoreApplication.translate("Progress", u"\u52a0\u8f7d\u4e2d......", None))
    # retranslateUi

