# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'cc.ui'
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
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QFrame, QGridLayout,
    QHeaderView, QMainWindow, QMenu, QMenuBar,
    QScrollArea, QSizePolicy, QSpacerItem, QStatusBar,
    QTableWidget, QTableWidgetItem, QToolBox, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(440, 688)
        self.Save_a = QAction(MainWindow)
        self.Save_a.setObjectName(u"Save_a")
        self.Update_a = QAction(MainWindow)
        self.Update_a.setObjectName(u"Update_a")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.toolBox = QToolBox(self.centralwidget)
        self.toolBox.setObjectName(u"toolBox")
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(14)
        self.toolBox.setFont(font)
        self.toolBox.setStyleSheet(u"")
        self.toolBox.setFrameShape(QFrame.NoFrame)
        self.toolBox.setFrameShadow(QFrame.Plain)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.page.setGeometry(QRect(0, 0, 422, 79))
        self.gridLayout_3 = QGridLayout(self.page)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_7 = QScrollArea(self.page)
        self.scrollArea_7.setObjectName(u"scrollArea_7")
        self.scrollArea_7.setFrameShape(QFrame.NoFrame)
        self.scrollArea_7.setWidgetResizable(True)
        self.scrollAreaWidgetContents_7 = QWidget()
        self.scrollAreaWidgetContents_7.setObjectName(u"scrollAreaWidgetContents_7")
        self.scrollAreaWidgetContents_7.setGeometry(QRect(0, 0, 422, 79))
        self.gridLayout_27 = QGridLayout(self.scrollAreaWidgetContents_7)
        self.gridLayout_27.setObjectName(u"gridLayout_27")
        self.gridLayout_27.setContentsMargins(0, 0, 0, 0)
        self.tableWidget = QTableWidget(self.scrollAreaWidgetContents_7)
        if (self.tableWidget.columnCount() < 2):
            self.tableWidget.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.tableWidget.rowCount() < 2):
            self.tableWidget.setRowCount(2)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget.setItem(0, 0, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.tableWidget.setItem(0, 1, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.tableWidget.setItem(1, 0, __qtablewidgetitem6)
        self.tableWidget.setObjectName(u"tableWidget")
        font1 = QFont()
        font1.setFamilies([u"\u5b8b\u4f53"])
        font1.setPointSize(10)
        self.tableWidget.setFont(font1)
        self.tableWidget.verticalHeader().setVisible(False)

        self.gridLayout_27.addWidget(self.tableWidget, 0, 0, 1, 1)

        self.scrollArea_7.setWidget(self.scrollAreaWidgetContents_7)

        self.gridLayout_3.addWidget(self.scrollArea_7, 0, 0, 1, 1)

        self.toolBox.addItem(self.page, u"\u5168\u5c40\u53c2\u6570")
        self.page_6 = QWidget()
        self.page_6.setObjectName(u"page_6")
        self.page_6.setGeometry(QRect(0, 0, 422, 388))
        self.gridLayout_24 = QGridLayout(self.page_6)
        self.gridLayout_24.setObjectName(u"gridLayout_24")
        self.gridLayout_24.setContentsMargins(0, 0, 0, 0)
        self.scrollArea = QScrollArea(self.page_6)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setStyleSheet(u"gridline-color: rgb(0, 0, 0);")
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 422, 388))
        font2 = QFont()
        font2.setFamilies([u"\u5b8b\u4f53"])
        font2.setPointSize(12)
        self.scrollAreaWidgetContents.setFont(font2)
        self.gridLayout_4 = QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.tableWidget_2 = QTableWidget(self.scrollAreaWidgetContents)
        if (self.tableWidget_2.columnCount() < 2):
            self.tableWidget_2.setColumnCount(2)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(1, __qtablewidgetitem8)
        if (self.tableWidget_2.rowCount() < 33):
            self.tableWidget_2.setRowCount(33)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(0, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(1, __qtablewidgetitem10)
        __qtablewidgetitem11 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(2, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(3, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(4, __qtablewidgetitem13)
        __qtablewidgetitem14 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(5, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(6, __qtablewidgetitem15)
        __qtablewidgetitem16 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(7, __qtablewidgetitem16)
        __qtablewidgetitem17 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(8, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(9, __qtablewidgetitem18)
        __qtablewidgetitem19 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(10, __qtablewidgetitem19)
        __qtablewidgetitem20 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(11, __qtablewidgetitem20)
        __qtablewidgetitem21 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(12, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(13, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(14, __qtablewidgetitem23)
        __qtablewidgetitem24 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(15, __qtablewidgetitem24)
        __qtablewidgetitem25 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(16, __qtablewidgetitem25)
        __qtablewidgetitem26 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(17, __qtablewidgetitem26)
        __qtablewidgetitem27 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(18, __qtablewidgetitem27)
        __qtablewidgetitem28 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(19, __qtablewidgetitem28)
        __qtablewidgetitem29 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(20, __qtablewidgetitem29)
        __qtablewidgetitem30 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(21, __qtablewidgetitem30)
        __qtablewidgetitem31 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(22, __qtablewidgetitem31)
        __qtablewidgetitem32 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(23, __qtablewidgetitem32)
        __qtablewidgetitem33 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(24, __qtablewidgetitem33)
        __qtablewidgetitem34 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(25, __qtablewidgetitem34)
        __qtablewidgetitem35 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(26, __qtablewidgetitem35)
        __qtablewidgetitem36 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(27, __qtablewidgetitem36)
        __qtablewidgetitem37 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(28, __qtablewidgetitem37)
        __qtablewidgetitem38 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(29, __qtablewidgetitem38)
        __qtablewidgetitem39 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(30, __qtablewidgetitem39)
        __qtablewidgetitem40 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(31, __qtablewidgetitem40)
        __qtablewidgetitem41 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(32, __qtablewidgetitem41)
        __qtablewidgetitem42 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 0, __qtablewidgetitem42)
        __qtablewidgetitem43 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 1, __qtablewidgetitem43)
        __qtablewidgetitem44 = QTableWidgetItem()
        self.tableWidget_2.setItem(1, 0, __qtablewidgetitem44)
        __qtablewidgetitem45 = QTableWidgetItem()
        self.tableWidget_2.setItem(2, 0, __qtablewidgetitem45)
        __qtablewidgetitem46 = QTableWidgetItem()
        self.tableWidget_2.setItem(3, 0, __qtablewidgetitem46)
        __qtablewidgetitem47 = QTableWidgetItem()
        self.tableWidget_2.setItem(4, 0, __qtablewidgetitem47)
        __qtablewidgetitem48 = QTableWidgetItem()
        self.tableWidget_2.setItem(5, 0, __qtablewidgetitem48)
        __qtablewidgetitem49 = QTableWidgetItem()
        self.tableWidget_2.setItem(6, 0, __qtablewidgetitem49)
        __qtablewidgetitem50 = QTableWidgetItem()
        self.tableWidget_2.setItem(7, 0, __qtablewidgetitem50)
        __qtablewidgetitem51 = QTableWidgetItem()
        self.tableWidget_2.setItem(8, 0, __qtablewidgetitem51)
        __qtablewidgetitem52 = QTableWidgetItem()
        self.tableWidget_2.setItem(9, 0, __qtablewidgetitem52)
        __qtablewidgetitem53 = QTableWidgetItem()
        self.tableWidget_2.setItem(10, 0, __qtablewidgetitem53)
        __qtablewidgetitem54 = QTableWidgetItem()
        self.tableWidget_2.setItem(11, 0, __qtablewidgetitem54)
        __qtablewidgetitem55 = QTableWidgetItem()
        self.tableWidget_2.setItem(12, 0, __qtablewidgetitem55)
        __qtablewidgetitem56 = QTableWidgetItem()
        self.tableWidget_2.setItem(13, 0, __qtablewidgetitem56)
        __qtablewidgetitem57 = QTableWidgetItem()
        self.tableWidget_2.setItem(14, 0, __qtablewidgetitem57)
        __qtablewidgetitem58 = QTableWidgetItem()
        self.tableWidget_2.setItem(15, 0, __qtablewidgetitem58)
        __qtablewidgetitem59 = QTableWidgetItem()
        self.tableWidget_2.setItem(16, 0, __qtablewidgetitem59)
        __qtablewidgetitem60 = QTableWidgetItem()
        self.tableWidget_2.setItem(17, 0, __qtablewidgetitem60)
        __qtablewidgetitem61 = QTableWidgetItem()
        self.tableWidget_2.setItem(18, 0, __qtablewidgetitem61)
        __qtablewidgetitem62 = QTableWidgetItem()
        self.tableWidget_2.setItem(19, 0, __qtablewidgetitem62)
        __qtablewidgetitem63 = QTableWidgetItem()
        self.tableWidget_2.setItem(20, 0, __qtablewidgetitem63)
        __qtablewidgetitem64 = QTableWidgetItem()
        self.tableWidget_2.setItem(21, 0, __qtablewidgetitem64)
        __qtablewidgetitem65 = QTableWidgetItem()
        self.tableWidget_2.setItem(22, 0, __qtablewidgetitem65)
        __qtablewidgetitem66 = QTableWidgetItem()
        self.tableWidget_2.setItem(23, 0, __qtablewidgetitem66)
        __qtablewidgetitem67 = QTableWidgetItem()
        self.tableWidget_2.setItem(24, 0, __qtablewidgetitem67)
        __qtablewidgetitem68 = QTableWidgetItem()
        self.tableWidget_2.setItem(25, 0, __qtablewidgetitem68)
        __qtablewidgetitem69 = QTableWidgetItem()
        self.tableWidget_2.setItem(26, 0, __qtablewidgetitem69)
        __qtablewidgetitem70 = QTableWidgetItem()
        self.tableWidget_2.setItem(27, 0, __qtablewidgetitem70)
        __qtablewidgetitem71 = QTableWidgetItem()
        self.tableWidget_2.setItem(28, 0, __qtablewidgetitem71)
        __qtablewidgetitem72 = QTableWidgetItem()
        self.tableWidget_2.setItem(29, 0, __qtablewidgetitem72)
        __qtablewidgetitem73 = QTableWidgetItem()
        self.tableWidget_2.setItem(30, 0, __qtablewidgetitem73)
        __qtablewidgetitem74 = QTableWidgetItem()
        self.tableWidget_2.setItem(31, 0, __qtablewidgetitem74)
        __qtablewidgetitem75 = QTableWidgetItem()
        self.tableWidget_2.setItem(32, 0, __qtablewidgetitem75)
        self.tableWidget_2.setObjectName(u"tableWidget_2")
        self.tableWidget_2.setFont(font1)
        self.tableWidget_2.setFrameShape(QFrame.StyledPanel)
        self.tableWidget_2.setShowGrid(True)
        self.tableWidget_2.setGridStyle(Qt.SolidLine)
        self.tableWidget_2.setWordWrap(True)
        self.tableWidget_2.verticalHeader().setVisible(False)

        self.gridLayout_4.addWidget(self.tableWidget_2, 0, 0, 1, 1)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout_24.addWidget(self.scrollArea, 0, 0, 1, 1)

        self.toolBox.addItem(self.page_6, u"\u8bad\u7ec3\u53c2\u6570")
        self.page_7 = QWidget()
        self.page_7.setObjectName(u"page_7")
        self.page_7.setGeometry(QRect(0, 0, 422, 283))
        self.gridLayout_5 = QGridLayout(self.page_7)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setHorizontalSpacing(0)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_2 = QScrollArea(self.page_7)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setFrameShape(QFrame.NoFrame)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 422, 283))
        self.gridLayout_13 = QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.gridLayout_13.setContentsMargins(0, 0, 0, 0)
        self.tableWidget_3 = QTableWidget(self.scrollAreaWidgetContents_2)
        if (self.tableWidget_3.columnCount() < 2):
            self.tableWidget_3.setColumnCount(2)
        __qtablewidgetitem76 = QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(0, __qtablewidgetitem76)
        __qtablewidgetitem77 = QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(1, __qtablewidgetitem77)
        if (self.tableWidget_3.rowCount() < 10):
            self.tableWidget_3.setRowCount(10)
        __qtablewidgetitem78 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(0, __qtablewidgetitem78)
        __qtablewidgetitem79 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(1, __qtablewidgetitem79)
        __qtablewidgetitem80 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(2, __qtablewidgetitem80)
        __qtablewidgetitem81 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(3, __qtablewidgetitem81)
        __qtablewidgetitem82 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(4, __qtablewidgetitem82)
        __qtablewidgetitem83 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(5, __qtablewidgetitem83)
        __qtablewidgetitem84 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(6, __qtablewidgetitem84)
        __qtablewidgetitem85 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(7, __qtablewidgetitem85)
        __qtablewidgetitem86 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(8, __qtablewidgetitem86)
        __qtablewidgetitem87 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(9, __qtablewidgetitem87)
        __qtablewidgetitem88 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 0, __qtablewidgetitem88)
        __qtablewidgetitem89 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 1, __qtablewidgetitem89)
        __qtablewidgetitem90 = QTableWidgetItem()
        self.tableWidget_3.setItem(1, 0, __qtablewidgetitem90)
        __qtablewidgetitem91 = QTableWidgetItem()
        self.tableWidget_3.setItem(2, 0, __qtablewidgetitem91)
        __qtablewidgetitem92 = QTableWidgetItem()
        self.tableWidget_3.setItem(3, 0, __qtablewidgetitem92)
        __qtablewidgetitem93 = QTableWidgetItem()
        self.tableWidget_3.setItem(4, 0, __qtablewidgetitem93)
        __qtablewidgetitem94 = QTableWidgetItem()
        self.tableWidget_3.setItem(5, 0, __qtablewidgetitem94)
        __qtablewidgetitem95 = QTableWidgetItem()
        self.tableWidget_3.setItem(6, 0, __qtablewidgetitem95)
        __qtablewidgetitem96 = QTableWidgetItem()
        self.tableWidget_3.setItem(7, 0, __qtablewidgetitem96)
        __qtablewidgetitem97 = QTableWidgetItem()
        self.tableWidget_3.setItem(8, 0, __qtablewidgetitem97)
        __qtablewidgetitem98 = QTableWidgetItem()
        self.tableWidget_3.setItem(9, 0, __qtablewidgetitem98)
        self.tableWidget_3.setObjectName(u"tableWidget_3")
        self.tableWidget_3.setFont(font1)
        self.tableWidget_3.verticalHeader().setVisible(False)

        self.gridLayout_13.addWidget(self.tableWidget_3, 0, 0, 1, 1)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.gridLayout_5.addWidget(self.scrollArea_2, 0, 0, 1, 1)

        self.gridLayout_5.setRowStretch(0, 3)
        self.toolBox.addItem(self.page_7, u"\u9a8c\u8bc1/\u6d4b\u8bd5\u53c2\u6570")
        self.page_8 = QWidget()
        self.page_8.setObjectName(u"page_8")
        self.page_8.setGeometry(QRect(0, 0, 422, 250))
        self.gridLayout_26 = QGridLayout(self.page_8)
        self.gridLayout_26.setSpacing(0)
        self.gridLayout_26.setObjectName(u"gridLayout_26")
        self.gridLayout_26.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.page_8)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setFrameShape(QFrame.NoFrame)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 422, 250))
        self.gridLayout_15 = QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.gridLayout_15.setContentsMargins(0, 0, 0, 0)
        self.tableWidget_4 = QTableWidget(self.scrollAreaWidgetContents_3)
        if (self.tableWidget_4.columnCount() < 2):
            self.tableWidget_4.setColumnCount(2)
        __qtablewidgetitem99 = QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(0, __qtablewidgetitem99)
        __qtablewidgetitem100 = QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(1, __qtablewidgetitem100)
        if (self.tableWidget_4.rowCount() < 9):
            self.tableWidget_4.setRowCount(9)
        __qtablewidgetitem101 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(0, __qtablewidgetitem101)
        __qtablewidgetitem102 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(1, __qtablewidgetitem102)
        __qtablewidgetitem103 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(2, __qtablewidgetitem103)
        __qtablewidgetitem104 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(3, __qtablewidgetitem104)
        __qtablewidgetitem105 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(4, __qtablewidgetitem105)
        __qtablewidgetitem106 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(5, __qtablewidgetitem106)
        __qtablewidgetitem107 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(6, __qtablewidgetitem107)
        __qtablewidgetitem108 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(7, __qtablewidgetitem108)
        __qtablewidgetitem109 = QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(8, __qtablewidgetitem109)
        __qtablewidgetitem110 = QTableWidgetItem()
        self.tableWidget_4.setItem(0, 0, __qtablewidgetitem110)
        __qtablewidgetitem111 = QTableWidgetItem()
        self.tableWidget_4.setItem(0, 1, __qtablewidgetitem111)
        __qtablewidgetitem112 = QTableWidgetItem()
        self.tableWidget_4.setItem(1, 0, __qtablewidgetitem112)
        __qtablewidgetitem113 = QTableWidgetItem()
        self.tableWidget_4.setItem(2, 0, __qtablewidgetitem113)
        __qtablewidgetitem114 = QTableWidgetItem()
        self.tableWidget_4.setItem(3, 0, __qtablewidgetitem114)
        __qtablewidgetitem115 = QTableWidgetItem()
        self.tableWidget_4.setItem(4, 0, __qtablewidgetitem115)
        __qtablewidgetitem116 = QTableWidgetItem()
        self.tableWidget_4.setItem(5, 0, __qtablewidgetitem116)
        __qtablewidgetitem117 = QTableWidgetItem()
        self.tableWidget_4.setItem(6, 0, __qtablewidgetitem117)
        __qtablewidgetitem118 = QTableWidgetItem()
        self.tableWidget_4.setItem(7, 0, __qtablewidgetitem118)
        __qtablewidgetitem119 = QTableWidgetItem()
        self.tableWidget_4.setItem(8, 0, __qtablewidgetitem119)
        self.tableWidget_4.setObjectName(u"tableWidget_4")
        self.tableWidget_4.setFont(font1)
        self.tableWidget_4.verticalHeader().setVisible(False)

        self.gridLayout_15.addWidget(self.tableWidget_4, 0, 0, 1, 1)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.gridLayout_26.addWidget(self.scrollArea_3, 0, 0, 1, 1)

        self.gridLayout_26.setRowStretch(0, 2)
        self.toolBox.addItem(self.page_8, u"\u9884\u6d4b\u53c2\u6570")
        self.page_9 = QWidget()
        self.page_9.setObjectName(u"page_9")
        self.page_9.setGeometry(QRect(0, 0, 422, 246))
        self.gridLayout_17 = QGridLayout(self.page_9)
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.gridLayout_17.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_4 = QScrollArea(self.page_9)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setFrameShape(QFrame.NoFrame)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 422, 246))
        self.gridLayout_18 = QGridLayout(self.scrollAreaWidgetContents_4)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.gridLayout_18.setContentsMargins(0, 0, 0, 0)
        self.tableWidget_5 = QTableWidget(self.scrollAreaWidgetContents_4)
        if (self.tableWidget_5.columnCount() < 2):
            self.tableWidget_5.setColumnCount(2)
        __qtablewidgetitem120 = QTableWidgetItem()
        self.tableWidget_5.setHorizontalHeaderItem(0, __qtablewidgetitem120)
        __qtablewidgetitem121 = QTableWidgetItem()
        self.tableWidget_5.setHorizontalHeaderItem(1, __qtablewidgetitem121)
        if (self.tableWidget_5.rowCount() < 9):
            self.tableWidget_5.setRowCount(9)
        __qtablewidgetitem122 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(0, __qtablewidgetitem122)
        __qtablewidgetitem123 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(1, __qtablewidgetitem123)
        __qtablewidgetitem124 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(2, __qtablewidgetitem124)
        __qtablewidgetitem125 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(3, __qtablewidgetitem125)
        __qtablewidgetitem126 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(4, __qtablewidgetitem126)
        __qtablewidgetitem127 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(5, __qtablewidgetitem127)
        __qtablewidgetitem128 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(6, __qtablewidgetitem128)
        __qtablewidgetitem129 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(7, __qtablewidgetitem129)
        __qtablewidgetitem130 = QTableWidgetItem()
        self.tableWidget_5.setVerticalHeaderItem(8, __qtablewidgetitem130)
        __qtablewidgetitem131 = QTableWidgetItem()
        self.tableWidget_5.setItem(0, 0, __qtablewidgetitem131)
        __qtablewidgetitem132 = QTableWidgetItem()
        self.tableWidget_5.setItem(0, 1, __qtablewidgetitem132)
        __qtablewidgetitem133 = QTableWidgetItem()
        self.tableWidget_5.setItem(1, 0, __qtablewidgetitem133)
        __qtablewidgetitem134 = QTableWidgetItem()
        self.tableWidget_5.setItem(2, 0, __qtablewidgetitem134)
        __qtablewidgetitem135 = QTableWidgetItem()
        self.tableWidget_5.setItem(3, 0, __qtablewidgetitem135)
        __qtablewidgetitem136 = QTableWidgetItem()
        self.tableWidget_5.setItem(4, 0, __qtablewidgetitem136)
        __qtablewidgetitem137 = QTableWidgetItem()
        self.tableWidget_5.setItem(5, 0, __qtablewidgetitem137)
        __qtablewidgetitem138 = QTableWidgetItem()
        self.tableWidget_5.setItem(6, 0, __qtablewidgetitem138)
        __qtablewidgetitem139 = QTableWidgetItem()
        self.tableWidget_5.setItem(7, 0, __qtablewidgetitem139)
        __qtablewidgetitem140 = QTableWidgetItem()
        self.tableWidget_5.setItem(8, 0, __qtablewidgetitem140)
        self.tableWidget_5.setObjectName(u"tableWidget_5")
        self.tableWidget_5.setFont(font1)
        self.tableWidget_5.verticalHeader().setVisible(False)

        self.gridLayout_18.addWidget(self.tableWidget_5, 0, 0, 1, 1)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)

        self.gridLayout_17.addWidget(self.scrollArea_4, 0, 0, 1, 1)

        self.gridLayout_17.setRowStretch(0, 2)
        self.toolBox.addItem(self.page_9, u"\u53ef\u89c6\u5316\u53c2\u6570")
        self.page_10 = QWidget()
        self.page_10.setObjectName(u"page_10")
        self.page_10.setGeometry(QRect(0, 0, 422, 246))
        self.gridLayout_20 = QGridLayout(self.page_10)
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.gridLayout_20.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_5 = QScrollArea(self.page_10)
        self.scrollArea_5.setObjectName(u"scrollArea_5")
        self.scrollArea_5.setFrameShape(QFrame.NoFrame)
        self.scrollArea_5.setWidgetResizable(True)
        self.scrollAreaWidgetContents_5 = QWidget()
        self.scrollAreaWidgetContents_5.setObjectName(u"scrollAreaWidgetContents_5")
        self.scrollAreaWidgetContents_5.setGeometry(QRect(0, 0, 422, 246))
        self.gridLayout_21 = QGridLayout(self.scrollAreaWidgetContents_5)
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.gridLayout_21.setContentsMargins(0, 0, 0, 0)
        self.tableWidget_6 = QTableWidget(self.scrollAreaWidgetContents_5)
        if (self.tableWidget_6.columnCount() < 2):
            self.tableWidget_6.setColumnCount(2)
        __qtablewidgetitem141 = QTableWidgetItem()
        self.tableWidget_6.setHorizontalHeaderItem(0, __qtablewidgetitem141)
        __qtablewidgetitem142 = QTableWidgetItem()
        self.tableWidget_6.setHorizontalHeaderItem(1, __qtablewidgetitem142)
        if (self.tableWidget_6.rowCount() < 9):
            self.tableWidget_6.setRowCount(9)
        __qtablewidgetitem143 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(0, __qtablewidgetitem143)
        __qtablewidgetitem144 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(1, __qtablewidgetitem144)
        __qtablewidgetitem145 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(2, __qtablewidgetitem145)
        __qtablewidgetitem146 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(3, __qtablewidgetitem146)
        __qtablewidgetitem147 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(4, __qtablewidgetitem147)
        __qtablewidgetitem148 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(5, __qtablewidgetitem148)
        __qtablewidgetitem149 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(6, __qtablewidgetitem149)
        __qtablewidgetitem150 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(7, __qtablewidgetitem150)
        __qtablewidgetitem151 = QTableWidgetItem()
        self.tableWidget_6.setVerticalHeaderItem(8, __qtablewidgetitem151)
        __qtablewidgetitem152 = QTableWidgetItem()
        self.tableWidget_6.setItem(0, 0, __qtablewidgetitem152)
        __qtablewidgetitem153 = QTableWidgetItem()
        self.tableWidget_6.setItem(0, 1, __qtablewidgetitem153)
        __qtablewidgetitem154 = QTableWidgetItem()
        self.tableWidget_6.setItem(1, 0, __qtablewidgetitem154)
        __qtablewidgetitem155 = QTableWidgetItem()
        self.tableWidget_6.setItem(2, 0, __qtablewidgetitem155)
        __qtablewidgetitem156 = QTableWidgetItem()
        self.tableWidget_6.setItem(3, 0, __qtablewidgetitem156)
        __qtablewidgetitem157 = QTableWidgetItem()
        self.tableWidget_6.setItem(4, 0, __qtablewidgetitem157)
        __qtablewidgetitem158 = QTableWidgetItem()
        self.tableWidget_6.setItem(5, 0, __qtablewidgetitem158)
        __qtablewidgetitem159 = QTableWidgetItem()
        self.tableWidget_6.setItem(6, 0, __qtablewidgetitem159)
        __qtablewidgetitem160 = QTableWidgetItem()
        self.tableWidget_6.setItem(7, 0, __qtablewidgetitem160)
        __qtablewidgetitem161 = QTableWidgetItem()
        self.tableWidget_6.setItem(8, 0, __qtablewidgetitem161)
        self.tableWidget_6.setObjectName(u"tableWidget_6")
        self.tableWidget_6.setFont(font1)
        self.tableWidget_6.verticalHeader().setVisible(False)

        self.gridLayout_21.addWidget(self.tableWidget_6, 0, 0, 1, 1)

        self.scrollArea_5.setWidget(self.scrollAreaWidgetContents_5)

        self.gridLayout_20.addWidget(self.scrollArea_5, 0, 0, 1, 1)

        self.gridLayout_20.setRowStretch(0, 3)
        self.toolBox.addItem(self.page_10, u"\u5bfc\u51fa\u53c2\u6570")
        self.page_11 = QWidget()
        self.page_11.setObjectName(u"page_11")
        self.page_11.setGeometry(QRect(0, 0, 422, 388))
        self.gridLayout_2 = QGridLayout(self.page_11)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_6 = QScrollArea(self.page_11)
        self.scrollArea_6.setObjectName(u"scrollArea_6")
        self.scrollArea_6.setFrameShape(QFrame.NoFrame)
        self.scrollArea_6.setWidgetResizable(True)
        self.scrollAreaWidgetContents_6 = QWidget()
        self.scrollAreaWidgetContents_6.setObjectName(u"scrollAreaWidgetContents_6")
        self.scrollAreaWidgetContents_6.setGeometry(QRect(0, 0, 422, 388))
        self.gridLayout_23 = QGridLayout(self.scrollAreaWidgetContents_6)
        self.gridLayout_23.setObjectName(u"gridLayout_23")
        self.gridLayout_23.setContentsMargins(0, 0, 0, 0)
        self.tableWidget_7 = QTableWidget(self.scrollAreaWidgetContents_6)
        if (self.tableWidget_7.columnCount() < 2):
            self.tableWidget_7.setColumnCount(2)
        __qtablewidgetitem162 = QTableWidgetItem()
        self.tableWidget_7.setHorizontalHeaderItem(0, __qtablewidgetitem162)
        __qtablewidgetitem163 = QTableWidgetItem()
        self.tableWidget_7.setHorizontalHeaderItem(1, __qtablewidgetitem163)
        if (self.tableWidget_7.rowCount() < 30):
            self.tableWidget_7.setRowCount(30)
        __qtablewidgetitem164 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(0, __qtablewidgetitem164)
        __qtablewidgetitem165 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(1, __qtablewidgetitem165)
        __qtablewidgetitem166 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(2, __qtablewidgetitem166)
        __qtablewidgetitem167 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(3, __qtablewidgetitem167)
        __qtablewidgetitem168 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(4, __qtablewidgetitem168)
        __qtablewidgetitem169 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(5, __qtablewidgetitem169)
        __qtablewidgetitem170 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(6, __qtablewidgetitem170)
        __qtablewidgetitem171 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(7, __qtablewidgetitem171)
        __qtablewidgetitem172 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(8, __qtablewidgetitem172)
        __qtablewidgetitem173 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(9, __qtablewidgetitem173)
        __qtablewidgetitem174 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(10, __qtablewidgetitem174)
        __qtablewidgetitem175 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(11, __qtablewidgetitem175)
        __qtablewidgetitem176 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(12, __qtablewidgetitem176)
        __qtablewidgetitem177 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(13, __qtablewidgetitem177)
        __qtablewidgetitem178 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(14, __qtablewidgetitem178)
        __qtablewidgetitem179 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(15, __qtablewidgetitem179)
        __qtablewidgetitem180 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(16, __qtablewidgetitem180)
        __qtablewidgetitem181 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(17, __qtablewidgetitem181)
        __qtablewidgetitem182 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(18, __qtablewidgetitem182)
        __qtablewidgetitem183 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(19, __qtablewidgetitem183)
        __qtablewidgetitem184 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(20, __qtablewidgetitem184)
        __qtablewidgetitem185 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(21, __qtablewidgetitem185)
        __qtablewidgetitem186 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(22, __qtablewidgetitem186)
        __qtablewidgetitem187 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(23, __qtablewidgetitem187)
        __qtablewidgetitem188 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(24, __qtablewidgetitem188)
        __qtablewidgetitem189 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(25, __qtablewidgetitem189)
        __qtablewidgetitem190 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(26, __qtablewidgetitem190)
        __qtablewidgetitem191 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(27, __qtablewidgetitem191)
        __qtablewidgetitem192 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(28, __qtablewidgetitem192)
        __qtablewidgetitem193 = QTableWidgetItem()
        self.tableWidget_7.setVerticalHeaderItem(29, __qtablewidgetitem193)
        __qtablewidgetitem194 = QTableWidgetItem()
        self.tableWidget_7.setItem(0, 0, __qtablewidgetitem194)
        __qtablewidgetitem195 = QTableWidgetItem()
        self.tableWidget_7.setItem(0, 1, __qtablewidgetitem195)
        __qtablewidgetitem196 = QTableWidgetItem()
        self.tableWidget_7.setItem(1, 0, __qtablewidgetitem196)
        __qtablewidgetitem197 = QTableWidgetItem()
        self.tableWidget_7.setItem(2, 0, __qtablewidgetitem197)
        __qtablewidgetitem198 = QTableWidgetItem()
        self.tableWidget_7.setItem(3, 0, __qtablewidgetitem198)
        __qtablewidgetitem199 = QTableWidgetItem()
        self.tableWidget_7.setItem(4, 0, __qtablewidgetitem199)
        __qtablewidgetitem200 = QTableWidgetItem()
        self.tableWidget_7.setItem(5, 0, __qtablewidgetitem200)
        __qtablewidgetitem201 = QTableWidgetItem()
        self.tableWidget_7.setItem(6, 0, __qtablewidgetitem201)
        __qtablewidgetitem202 = QTableWidgetItem()
        self.tableWidget_7.setItem(7, 0, __qtablewidgetitem202)
        __qtablewidgetitem203 = QTableWidgetItem()
        self.tableWidget_7.setItem(8, 0, __qtablewidgetitem203)
        __qtablewidgetitem204 = QTableWidgetItem()
        self.tableWidget_7.setItem(9, 0, __qtablewidgetitem204)
        __qtablewidgetitem205 = QTableWidgetItem()
        self.tableWidget_7.setItem(10, 0, __qtablewidgetitem205)
        __qtablewidgetitem206 = QTableWidgetItem()
        self.tableWidget_7.setItem(11, 0, __qtablewidgetitem206)
        __qtablewidgetitem207 = QTableWidgetItem()
        self.tableWidget_7.setItem(12, 0, __qtablewidgetitem207)
        __qtablewidgetitem208 = QTableWidgetItem()
        self.tableWidget_7.setItem(13, 0, __qtablewidgetitem208)
        __qtablewidgetitem209 = QTableWidgetItem()
        self.tableWidget_7.setItem(14, 0, __qtablewidgetitem209)
        __qtablewidgetitem210 = QTableWidgetItem()
        self.tableWidget_7.setItem(15, 0, __qtablewidgetitem210)
        __qtablewidgetitem211 = QTableWidgetItem()
        self.tableWidget_7.setItem(16, 0, __qtablewidgetitem211)
        __qtablewidgetitem212 = QTableWidgetItem()
        self.tableWidget_7.setItem(17, 0, __qtablewidgetitem212)
        __qtablewidgetitem213 = QTableWidgetItem()
        self.tableWidget_7.setItem(18, 0, __qtablewidgetitem213)
        __qtablewidgetitem214 = QTableWidgetItem()
        self.tableWidget_7.setItem(19, 0, __qtablewidgetitem214)
        __qtablewidgetitem215 = QTableWidgetItem()
        self.tableWidget_7.setItem(20, 0, __qtablewidgetitem215)
        __qtablewidgetitem216 = QTableWidgetItem()
        self.tableWidget_7.setItem(21, 0, __qtablewidgetitem216)
        __qtablewidgetitem217 = QTableWidgetItem()
        self.tableWidget_7.setItem(22, 0, __qtablewidgetitem217)
        __qtablewidgetitem218 = QTableWidgetItem()
        self.tableWidget_7.setItem(23, 0, __qtablewidgetitem218)
        __qtablewidgetitem219 = QTableWidgetItem()
        self.tableWidget_7.setItem(24, 0, __qtablewidgetitem219)
        __qtablewidgetitem220 = QTableWidgetItem()
        self.tableWidget_7.setItem(25, 0, __qtablewidgetitem220)
        __qtablewidgetitem221 = QTableWidgetItem()
        self.tableWidget_7.setItem(26, 0, __qtablewidgetitem221)
        __qtablewidgetitem222 = QTableWidgetItem()
        self.tableWidget_7.setItem(27, 0, __qtablewidgetitem222)
        __qtablewidgetitem223 = QTableWidgetItem()
        self.tableWidget_7.setItem(28, 0, __qtablewidgetitem223)
        __qtablewidgetitem224 = QTableWidgetItem()
        self.tableWidget_7.setItem(29, 0, __qtablewidgetitem224)
        self.tableWidget_7.setObjectName(u"tableWidget_7")
        self.tableWidget_7.setFont(font1)
        self.tableWidget_7.verticalHeader().setVisible(False)

        self.gridLayout_23.addWidget(self.tableWidget_7, 0, 0, 1, 1)

        self.scrollArea_6.setWidget(self.scrollAreaWidgetContents_6)

        self.gridLayout_2.addWidget(self.scrollArea_6, 0, 0, 1, 1)

        self.toolBox.addItem(self.page_11, u"\u8d85\u53c2\u6570")

        self.gridLayout.addWidget(self.toolBox, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 440, 23))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.Save_a)
        self.menu.addAction(self.Update_a)

        self.retranslateUi(MainWindow)

        self.toolBox.setCurrentIndex(6)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.Save_a.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58", None))
#if QT_CONFIG(tooltip)
        self.Save_a.setToolTip(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.Save_a.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.Update_a.setText(QCoreApplication.translate("MainWindow", u"\u66f4\u65b0", None))
#if QT_CONFIG(tooltip)
        self.Update_a.setToolTip(QCoreApplication.translate("MainWindow", u"\u66f4\u65b0", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.Update_a.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+A", None))
#endif // QT_CONFIG(shortcut)
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem2 = self.tableWidget.verticalHeaderItem(0)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"task", None));
        ___qtablewidgetitem3 = self.tableWidget.verticalHeaderItem(1)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"mode", None));

        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        ___qtablewidgetitem4 = self.tableWidget.item(0, 0)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"\u4efb\u52a1", None));
        ___qtablewidgetitem5 = self.tableWidget.item(1, 0)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u5f0f", None));
        self.tableWidget.setSortingEnabled(__sortingEnabled)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page), QCoreApplication.translate("MainWindow", u"\u5168\u5c40\u53c2\u6570", None))
        ___qtablewidgetitem6 = self.tableWidget_2.horizontalHeaderItem(0)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem7 = self.tableWidget_2.horizontalHeaderItem(1)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem8 = self.tableWidget_2.verticalHeaderItem(0)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("MainWindow", u"model", None));
        ___qtablewidgetitem9 = self.tableWidget_2.verticalHeaderItem(1)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("MainWindow", u"data", None));
        ___qtablewidgetitem10 = self.tableWidget_2.verticalHeaderItem(2)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("MainWindow", u"epoches", None));
        ___qtablewidgetitem11 = self.tableWidget_2.verticalHeaderItem(3)
        ___qtablewidgetitem11.setText(QCoreApplication.translate("MainWindow", u"time", None));
        ___qtablewidgetitem12 = self.tableWidget_2.verticalHeaderItem(4)
        ___qtablewidgetitem12.setText(QCoreApplication.translate("MainWindow", u"patience", None));
        ___qtablewidgetitem13 = self.tableWidget_2.verticalHeaderItem(5)
        ___qtablewidgetitem13.setText(QCoreApplication.translate("MainWindow", u"batch", None));
        ___qtablewidgetitem14 = self.tableWidget_2.verticalHeaderItem(6)
        ___qtablewidgetitem14.setText(QCoreApplication.translate("MainWindow", u"imgsz", None));
        ___qtablewidgetitem15 = self.tableWidget_2.verticalHeaderItem(7)
        ___qtablewidgetitem15.setText(QCoreApplication.translate("MainWindow", u"save", None));
        ___qtablewidgetitem16 = self.tableWidget_2.verticalHeaderItem(8)
        ___qtablewidgetitem16.setText(QCoreApplication.translate("MainWindow", u"save_period", None));
        ___qtablewidgetitem17 = self.tableWidget_2.verticalHeaderItem(9)
        ___qtablewidgetitem17.setText(QCoreApplication.translate("MainWindow", u"cache", None));
        ___qtablewidgetitem18 = self.tableWidget_2.verticalHeaderItem(10)
        ___qtablewidgetitem18.setText(QCoreApplication.translate("MainWindow", u"device", None));
        ___qtablewidgetitem19 = self.tableWidget_2.verticalHeaderItem(11)
        ___qtablewidgetitem19.setText(QCoreApplication.translate("MainWindow", u"workers", None));
        ___qtablewidgetitem20 = self.tableWidget_2.verticalHeaderItem(12)
        ___qtablewidgetitem20.setText(QCoreApplication.translate("MainWindow", u"project", None));
        ___qtablewidgetitem21 = self.tableWidget_2.verticalHeaderItem(13)
        ___qtablewidgetitem21.setText(QCoreApplication.translate("MainWindow", u"name", None));
        ___qtablewidgetitem22 = self.tableWidget_2.verticalHeaderItem(14)
        ___qtablewidgetitem22.setText(QCoreApplication.translate("MainWindow", u"exist_ok", None));
        ___qtablewidgetitem23 = self.tableWidget_2.verticalHeaderItem(15)
        ___qtablewidgetitem23.setText(QCoreApplication.translate("MainWindow", u"pretrained", None));
        ___qtablewidgetitem24 = self.tableWidget_2.verticalHeaderItem(16)
        ___qtablewidgetitem24.setText(QCoreApplication.translate("MainWindow", u"optimizer", None));
        ___qtablewidgetitem25 = self.tableWidget_2.verticalHeaderItem(17)
        ___qtablewidgetitem25.setText(QCoreApplication.translate("MainWindow", u"verbose", None));
        ___qtablewidgetitem26 = self.tableWidget_2.verticalHeaderItem(18)
        ___qtablewidgetitem26.setText(QCoreApplication.translate("MainWindow", u"seed", None));
        ___qtablewidgetitem27 = self.tableWidget_2.verticalHeaderItem(19)
        ___qtablewidgetitem27.setText(QCoreApplication.translate("MainWindow", u"deterministic", None));
        ___qtablewidgetitem28 = self.tableWidget_2.verticalHeaderItem(20)
        ___qtablewidgetitem28.setText(QCoreApplication.translate("MainWindow", u"single_cls", None));
        ___qtablewidgetitem29 = self.tableWidget_2.verticalHeaderItem(21)
        ___qtablewidgetitem29.setText(QCoreApplication.translate("MainWindow", u"rect", None));
        ___qtablewidgetitem30 = self.tableWidget_2.verticalHeaderItem(22)
        ___qtablewidgetitem30.setText(QCoreApplication.translate("MainWindow", u"cos_lr", None));
        ___qtablewidgetitem31 = self.tableWidget_2.verticalHeaderItem(23)
        ___qtablewidgetitem31.setText(QCoreApplication.translate("MainWindow", u"close_mosaic", None));
        ___qtablewidgetitem32 = self.tableWidget_2.verticalHeaderItem(24)
        ___qtablewidgetitem32.setText(QCoreApplication.translate("MainWindow", u"resume", None));
        ___qtablewidgetitem33 = self.tableWidget_2.verticalHeaderItem(25)
        ___qtablewidgetitem33.setText(QCoreApplication.translate("MainWindow", u"amp", None));
        ___qtablewidgetitem34 = self.tableWidget_2.verticalHeaderItem(26)
        ___qtablewidgetitem34.setText(QCoreApplication.translate("MainWindow", u"fraction", None));
        ___qtablewidgetitem35 = self.tableWidget_2.verticalHeaderItem(27)
        ___qtablewidgetitem35.setText(QCoreApplication.translate("MainWindow", u"profile", None));
        ___qtablewidgetitem36 = self.tableWidget_2.verticalHeaderItem(28)
        ___qtablewidgetitem36.setText(QCoreApplication.translate("MainWindow", u"freeze", None));
        ___qtablewidgetitem37 = self.tableWidget_2.verticalHeaderItem(29)
        ___qtablewidgetitem37.setText(QCoreApplication.translate("MainWindow", u"multi_scale", None));
        ___qtablewidgetitem38 = self.tableWidget_2.verticalHeaderItem(30)
        ___qtablewidgetitem38.setText(QCoreApplication.translate("MainWindow", u"override", None));
        ___qtablewidgetitem39 = self.tableWidget_2.verticalHeaderItem(31)
        ___qtablewidgetitem39.setText(QCoreApplication.translate("MainWindow", u"mask_ratio", None));
        ___qtablewidgetitem40 = self.tableWidget_2.verticalHeaderItem(32)
        ___qtablewidgetitem40.setText(QCoreApplication.translate("MainWindow", u"dropout", None));

        __sortingEnabled1 = self.tableWidget_2.isSortingEnabled()
        self.tableWidget_2.setSortingEnabled(False)
        ___qtablewidgetitem41 = self.tableWidget_2.item(0, 0)
        ___qtablewidgetitem41.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u578b", None));
        ___qtablewidgetitem42 = self.tableWidget_2.item(1, 0)
        ___qtablewidgetitem42.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u96c6", None));
        ___qtablewidgetitem43 = self.tableWidget_2.item(2, 0)
        ___qtablewidgetitem43.setText(QCoreApplication.translate("MainWindow", u"\u5b66\u4e60\u5468\u671f", None));
        ___qtablewidgetitem44 = self.tableWidget_2.item(3, 0)
        ___qtablewidgetitem44.setText(QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u65f6\u95f4\uff08h\uff09", None));
        ___qtablewidgetitem45 = self.tableWidget_2.item(4, 0)
        ___qtablewidgetitem45.setText(QCoreApplication.translate("MainWindow", u"\u65e9\u505c\u5468\u671f\u6570", None));
        ___qtablewidgetitem46 = self.tableWidget_2.item(5, 0)
        ___qtablewidgetitem46.setText(QCoreApplication.translate("MainWindow", u"\u6279\u5927\u5c0f", None));
        ___qtablewidgetitem47 = self.tableWidget_2.item(6, 0)
        ___qtablewidgetitem47.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u5927\u5c0f", None));
        ___qtablewidgetitem48 = self.tableWidget_2.item(7, 0)
        ___qtablewidgetitem48.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58", None));
        ___qtablewidgetitem49 = self.tableWidget_2.item(8, 0)
        ___qtablewidgetitem49.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u5468\u671f", None));
        ___qtablewidgetitem50 = self.tableWidget_2.item(9, 0)
        ___qtablewidgetitem50.setText(QCoreApplication.translate("MainWindow", u"\u7f13\u5b58", None));
        ___qtablewidgetitem51 = self.tableWidget_2.item(10, 0)
        ___qtablewidgetitem51.setText(QCoreApplication.translate("MainWindow", u"\u9a71\u52a8", None));
        ___qtablewidgetitem52 = self.tableWidget_2.item(11, 0)
        ___qtablewidgetitem52.setText(QCoreApplication.translate("MainWindow", u"\u52a0\u8f7d\u7ebf\u7a0b\u6570", None));
        ___qtablewidgetitem53 = self.tableWidget_2.item(12, 0)
        ___qtablewidgetitem53.setText(QCoreApplication.translate("MainWindow", u"\u9879\u76ee\u540d\u79f0", None));
        ___qtablewidgetitem54 = self.tableWidget_2.item(13, 0)
        ___qtablewidgetitem54.setText(QCoreApplication.translate("MainWindow", u"\u5b9e\u9a8c\u540d\u79f0", None));
        ___qtablewidgetitem55 = self.tableWidget_2.item(14, 0)
        ___qtablewidgetitem55.setText(QCoreApplication.translate("MainWindow", u"\u8986\u76d6\u5df2\u6709\u5b9e\u9a8c", None));
        ___qtablewidgetitem56 = self.tableWidget_2.item(15, 0)
        ___qtablewidgetitem56.setText(QCoreApplication.translate("MainWindow", u"\u9884\u8bad\u7ec3", None));
        ___qtablewidgetitem57 = self.tableWidget_2.item(16, 0)
        ___qtablewidgetitem57.setText(QCoreApplication.translate("MainWindow", u"\u4f18\u5316\u5668", None));
        ___qtablewidgetitem58 = self.tableWidget_2.item(17, 0)
        ___qtablewidgetitem58.setText(QCoreApplication.translate("MainWindow", u"\u8be6\u7ec6\u8f93\u51fa", None));
        ___qtablewidgetitem59 = self.tableWidget_2.item(18, 0)
        ___qtablewidgetitem59.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u79cd\u5b50", None));
        ___qtablewidgetitem60 = self.tableWidget_2.item(19, 0)
        ___qtablewidgetitem60.setText(QCoreApplication.translate("MainWindow", u"\u786e\u5b9a\u6027", None));
        ___qtablewidgetitem61 = self.tableWidget_2.item(20, 0)
        ___qtablewidgetitem61.setText(QCoreApplication.translate("MainWindow", u"\u5355\u4e00\u79cd\u7c7b", None));
        ___qtablewidgetitem62 = self.tableWidget_2.item(21, 0)
        ___qtablewidgetitem62.setText(QCoreApplication.translate("MainWindow", u"\u7f29\u653e\u6539\u8fdb\u65b9\u6cd5", None));
        ___qtablewidgetitem63 = self.tableWidget_2.item(22, 0)
        ___qtablewidgetitem63.setText(QCoreApplication.translate("MainWindow", u"\u5b66\u4e60\u7387\u4f59\u5f26\u5316", None));
        ___qtablewidgetitem64 = self.tableWidget_2.item(23, 0)
        ___qtablewidgetitem64.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u589e\u5f3a\u5173\u95ed\u5468\u671f\u6570", None));
        ___qtablewidgetitem65 = self.tableWidget_2.item(24, 0)
        ___qtablewidgetitem65.setText(QCoreApplication.translate("MainWindow", u"\u6062\u590d\u8bad\u7ec3", None));
        ___qtablewidgetitem66 = self.tableWidget_2.item(25, 0)
        ___qtablewidgetitem66.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u52a8\u6df7\u5408\u7cbe\u5ea6", None));
        ___qtablewidgetitem67 = self.tableWidget_2.item(26, 0)
        ___qtablewidgetitem67.setText(QCoreApplication.translate("MainWindow", u"\u6570\u636e\u96c6\u4f7f\u7528\u6bd4\u4f8b", None));
        ___qtablewidgetitem68 = self.tableWidget_2.item(27, 0)
        ___qtablewidgetitem68.setText(QCoreApplication.translate("MainWindow", u"\u63a8\u7406\u5206\u6790", None));
        ___qtablewidgetitem69 = self.tableWidget_2.item(28, 0)
        ___qtablewidgetitem69.setText(QCoreApplication.translate("MainWindow", u"\u51bb\u7ed3", None));
        ___qtablewidgetitem70 = self.tableWidget_2.item(29, 0)
        ___qtablewidgetitem70.setText(QCoreApplication.translate("MainWindow", u"\u591a\u5c3a\u5ea6\u7f29\u653e", None));
        ___qtablewidgetitem71 = self.tableWidget_2.item(30, 0)
        ___qtablewidgetitem71.setText(QCoreApplication.translate("MainWindow", u"\u63a9\u819c\u53e0\u52a0\uff08seg\uff09", None));
        ___qtablewidgetitem72 = self.tableWidget_2.item(31, 0)
        ___qtablewidgetitem72.setText(QCoreApplication.translate("MainWindow", u"\u63a9\u819c\u4e0b\u91c7\u6837\uff08seg\uff09", None));
        ___qtablewidgetitem73 = self.tableWidget_2.item(32, 0)
        ___qtablewidgetitem73.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u5931\u6d3b\uff08cls\uff09", None));
        self.tableWidget_2.setSortingEnabled(__sortingEnabled1)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page_6), QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u53c2\u6570", None))
        ___qtablewidgetitem74 = self.tableWidget_3.horizontalHeaderItem(0)
        ___qtablewidgetitem74.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem75 = self.tableWidget_3.horizontalHeaderItem(1)
        ___qtablewidgetitem75.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem76 = self.tableWidget_3.verticalHeaderItem(0)
        ___qtablewidgetitem76.setText(QCoreApplication.translate("MainWindow", u"val", None));
        ___qtablewidgetitem77 = self.tableWidget_3.verticalHeaderItem(1)
        ___qtablewidgetitem77.setText(QCoreApplication.translate("MainWindow", u"split", None));
        ___qtablewidgetitem78 = self.tableWidget_3.verticalHeaderItem(2)
        ___qtablewidgetitem78.setText(QCoreApplication.translate("MainWindow", u"save_json", None));
        ___qtablewidgetitem79 = self.tableWidget_3.verticalHeaderItem(3)
        ___qtablewidgetitem79.setText(QCoreApplication.translate("MainWindow", u"save_hybrid", None));
        ___qtablewidgetitem80 = self.tableWidget_3.verticalHeaderItem(4)
        ___qtablewidgetitem80.setText(QCoreApplication.translate("MainWindow", u"conf", None));
        ___qtablewidgetitem81 = self.tableWidget_3.verticalHeaderItem(5)
        ___qtablewidgetitem81.setText(QCoreApplication.translate("MainWindow", u"iou", None));
        ___qtablewidgetitem82 = self.tableWidget_3.verticalHeaderItem(6)
        ___qtablewidgetitem82.setText(QCoreApplication.translate("MainWindow", u"max_det", None));
        ___qtablewidgetitem83 = self.tableWidget_3.verticalHeaderItem(7)
        ___qtablewidgetitem83.setText(QCoreApplication.translate("MainWindow", u"half", None));
        ___qtablewidgetitem84 = self.tableWidget_3.verticalHeaderItem(8)
        ___qtablewidgetitem84.setText(QCoreApplication.translate("MainWindow", u"dnn", None));
        ___qtablewidgetitem85 = self.tableWidget_3.verticalHeaderItem(9)
        ___qtablewidgetitem85.setText(QCoreApplication.translate("MainWindow", u"plots", None));

        __sortingEnabled2 = self.tableWidget_3.isSortingEnabled()
        self.tableWidget_3.setSortingEnabled(False)
        ___qtablewidgetitem86 = self.tableWidget_3.item(0, 0)
        ___qtablewidgetitem86.setText(QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1", None));
        ___qtablewidgetitem87 = self.tableWidget_3.item(1, 0)
        ___qtablewidgetitem87.setText(QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1\u96c6\u540d\u79f0", None));
        ___qtablewidgetitem88 = self.tableWidget_3.item(2, 0)
        ___qtablewidgetitem88.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u4e3aJSON\u6587\u4ef6", None));
        ___qtablewidgetitem89 = self.tableWidget_3.item(3, 0)
        ___qtablewidgetitem89.setText(QCoreApplication.translate("MainWindow", u"\u6df7\u5408\u6807\u7b7e", None));
        ___qtablewidgetitem90 = self.tableWidget_3.item(4, 0)
        ___qtablewidgetitem90.setText(QCoreApplication.translate("MainWindow", u"\u7f6e\u4fe1\u5ea6", None));
        ___qtablewidgetitem91 = self.tableWidget_3.item(5, 0)
        ___qtablewidgetitem91.setText(QCoreApplication.translate("MainWindow", u"Iou", None));
        ___qtablewidgetitem92 = self.tableWidget_3.item(6, 0)
        ___qtablewidgetitem92.setText(QCoreApplication.translate("MainWindow", u"\u6700\u5927\u68c0\u6d4b\u6570\u91cf", None));
        ___qtablewidgetitem93 = self.tableWidget_3.item(7, 0)
        ___qtablewidgetitem93.setText(QCoreApplication.translate("MainWindow", u"\u534a\u6d6e\u70b9\u7cbe\u5ea6", None));
        ___qtablewidgetitem94 = self.tableWidget_3.item(8, 0)
        ___qtablewidgetitem94.setText(QCoreApplication.translate("MainWindow", u"dnn\u63a8\u7406", None));
        ___qtablewidgetitem95 = self.tableWidget_3.item(9, 0)
        ___qtablewidgetitem95.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u7ed3\u679c\u66f2\u7ebf", None));
        self.tableWidget_3.setSortingEnabled(__sortingEnabled2)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page_7), QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1/\u6d4b\u8bd5\u53c2\u6570", None))
        ___qtablewidgetitem96 = self.tableWidget_4.horizontalHeaderItem(0)
        ___qtablewidgetitem96.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem97 = self.tableWidget_4.horizontalHeaderItem(1)
        ___qtablewidgetitem97.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem98 = self.tableWidget_4.verticalHeaderItem(0)
        ___qtablewidgetitem98.setText(QCoreApplication.translate("MainWindow", u"source", None));
        ___qtablewidgetitem99 = self.tableWidget_4.verticalHeaderItem(1)
        ___qtablewidgetitem99.setText(QCoreApplication.translate("MainWindow", u"vid_stride", None));
        ___qtablewidgetitem100 = self.tableWidget_4.verticalHeaderItem(2)
        ___qtablewidgetitem100.setText(QCoreApplication.translate("MainWindow", u"stream_buffer", None));
        ___qtablewidgetitem101 = self.tableWidget_4.verticalHeaderItem(3)
        ___qtablewidgetitem101.setText(QCoreApplication.translate("MainWindow", u"visualize", None));
        ___qtablewidgetitem102 = self.tableWidget_4.verticalHeaderItem(4)
        ___qtablewidgetitem102.setText(QCoreApplication.translate("MainWindow", u"augment", None));
        ___qtablewidgetitem103 = self.tableWidget_4.verticalHeaderItem(5)
        ___qtablewidgetitem103.setText(QCoreApplication.translate("MainWindow", u"agnostic_nms", None));
        ___qtablewidgetitem104 = self.tableWidget_4.verticalHeaderItem(6)
        ___qtablewidgetitem104.setText(QCoreApplication.translate("MainWindow", u"classes", None));
        ___qtablewidgetitem105 = self.tableWidget_4.verticalHeaderItem(7)
        ___qtablewidgetitem105.setText(QCoreApplication.translate("MainWindow", u"retina_masks", None));
        ___qtablewidgetitem106 = self.tableWidget_4.verticalHeaderItem(8)
        ___qtablewidgetitem106.setText(QCoreApplication.translate("MainWindow", u"embed", None));

        __sortingEnabled3 = self.tableWidget_4.isSortingEnabled()
        self.tableWidget_4.setSortingEnabled(False)
        ___qtablewidgetitem107 = self.tableWidget_4.item(0, 0)
        ___qtablewidgetitem107.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u6e90", None));
        ___qtablewidgetitem108 = self.tableWidget_4.item(1, 0)
        ___qtablewidgetitem108.setText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u5e27\u95f4\u9694", None));
        ___qtablewidgetitem109 = self.tableWidget_4.item(2, 0)
        ___qtablewidgetitem109.setText(QCoreApplication.translate("MainWindow", u"\u6d41\u7f13\u51b2", None));
        ___qtablewidgetitem110 = self.tableWidget_4.item(3, 0)
        ___qtablewidgetitem110.setText(QCoreApplication.translate("MainWindow", u"\u53ef\u89c6\u5316", None));
        ___qtablewidgetitem111 = self.tableWidget_4.item(4, 0)
        ___qtablewidgetitem111.setText(QCoreApplication.translate("MainWindow", u"\u589e\u5f3a", None));
        ___qtablewidgetitem112 = self.tableWidget_4.item(5, 0)
        ___qtablewidgetitem112.setText(QCoreApplication.translate("MainWindow", u"NMS\u65e0\u89c6\u79cd\u7c7b", None));
        ___qtablewidgetitem113 = self.tableWidget_4.item(6, 0)
        ___qtablewidgetitem113.setText(QCoreApplication.translate("MainWindow", u"\u8fc7\u6ee4\u79cd\u7c7b", None));
        ___qtablewidgetitem114 = self.tableWidget_4.item(7, 0)
        ___qtablewidgetitem114.setText(QCoreApplication.translate("MainWindow", u"\u9ad8\u7cbe\u5ea6\u63a9\u819c", None));
        ___qtablewidgetitem115 = self.tableWidget_4.item(8, 0)
        ___qtablewidgetitem115.setText(QCoreApplication.translate("MainWindow", u"\u6307\u5b9a\u5c42\u8f93\u51fa", None));
        self.tableWidget_4.setSortingEnabled(__sortingEnabled3)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page_8), QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u53c2\u6570", None))
        ___qtablewidgetitem116 = self.tableWidget_5.horizontalHeaderItem(0)
        ___qtablewidgetitem116.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem117 = self.tableWidget_5.horizontalHeaderItem(1)
        ___qtablewidgetitem117.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem118 = self.tableWidget_5.verticalHeaderItem(0)
        ___qtablewidgetitem118.setText(QCoreApplication.translate("MainWindow", u"show", None));
        ___qtablewidgetitem119 = self.tableWidget_5.verticalHeaderItem(1)
        ___qtablewidgetitem119.setText(QCoreApplication.translate("MainWindow", u"save_frames", None));
        ___qtablewidgetitem120 = self.tableWidget_5.verticalHeaderItem(2)
        ___qtablewidgetitem120.setText(QCoreApplication.translate("MainWindow", u"save_txt", None));
        ___qtablewidgetitem121 = self.tableWidget_5.verticalHeaderItem(3)
        ___qtablewidgetitem121.setText(QCoreApplication.translate("MainWindow", u"save_conf", None));
        ___qtablewidgetitem122 = self.tableWidget_5.verticalHeaderItem(4)
        ___qtablewidgetitem122.setText(QCoreApplication.translate("MainWindow", u"save_crop", None));
        ___qtablewidgetitem123 = self.tableWidget_5.verticalHeaderItem(5)
        ___qtablewidgetitem123.setText(QCoreApplication.translate("MainWindow", u"show_labels", None));
        ___qtablewidgetitem124 = self.tableWidget_5.verticalHeaderItem(6)
        ___qtablewidgetitem124.setText(QCoreApplication.translate("MainWindow", u"show_conf", None));
        ___qtablewidgetitem125 = self.tableWidget_5.verticalHeaderItem(7)
        ___qtablewidgetitem125.setText(QCoreApplication.translate("MainWindow", u"show_boxes", None));
        ___qtablewidgetitem126 = self.tableWidget_5.verticalHeaderItem(8)
        ___qtablewidgetitem126.setText(QCoreApplication.translate("MainWindow", u"line_width", None));

        __sortingEnabled4 = self.tableWidget_5.isSortingEnabled()
        self.tableWidget_5.setSortingEnabled(False)
        ___qtablewidgetitem127 = self.tableWidget_5.item(0, 0)
        ___qtablewidgetitem127.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a", None));
        ___qtablewidgetitem128 = self.tableWidget_5.item(1, 0)
        ___qtablewidgetitem128.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u6bcf\u5e27", None));
        ___qtablewidgetitem129 = self.tableWidget_5.item(2, 0)
        ___qtablewidgetitem129.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u7ed3\u679c\u6587\u672c", None));
        ___qtablewidgetitem130 = self.tableWidget_5.item(3, 0)
        ___qtablewidgetitem130.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u7f6e\u4fe1\u5ea6", None));
        ___qtablewidgetitem131 = self.tableWidget_5.item(4, 0)
        ___qtablewidgetitem131.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u76ee\u6807\u88c1\u56fe", None));
        ___qtablewidgetitem132 = self.tableWidget_5.item(5, 0)
        ___qtablewidgetitem132.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u4fe1\u606f\u6807\u7b7e", None));
        ___qtablewidgetitem133 = self.tableWidget_5.item(6, 0)
        ___qtablewidgetitem133.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u7f6e\u4fe1\u5ea6", None));
        ___qtablewidgetitem134 = self.tableWidget_5.item(7, 0)
        ___qtablewidgetitem134.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u76ee\u6807\u6846", None));
        ___qtablewidgetitem135 = self.tableWidget_5.item(8, 0)
        ___qtablewidgetitem135.setText(QCoreApplication.translate("MainWindow", u"\u7ebf\u5bbd", None));
        self.tableWidget_5.setSortingEnabled(__sortingEnabled4)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page_9), QCoreApplication.translate("MainWindow", u"\u53ef\u89c6\u5316\u53c2\u6570", None))
        ___qtablewidgetitem136 = self.tableWidget_6.horizontalHeaderItem(0)
        ___qtablewidgetitem136.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem137 = self.tableWidget_6.horizontalHeaderItem(1)
        ___qtablewidgetitem137.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem138 = self.tableWidget_6.verticalHeaderItem(0)
        ___qtablewidgetitem138.setText(QCoreApplication.translate("MainWindow", u"format", None));
        ___qtablewidgetitem139 = self.tableWidget_6.verticalHeaderItem(1)
        ___qtablewidgetitem139.setText(QCoreApplication.translate("MainWindow", u"keras", None));
        ___qtablewidgetitem140 = self.tableWidget_6.verticalHeaderItem(2)
        ___qtablewidgetitem140.setText(QCoreApplication.translate("MainWindow", u"optimize", None));
        ___qtablewidgetitem141 = self.tableWidget_6.verticalHeaderItem(3)
        ___qtablewidgetitem141.setText(QCoreApplication.translate("MainWindow", u"int8", None));
        ___qtablewidgetitem142 = self.tableWidget_6.verticalHeaderItem(4)
        ___qtablewidgetitem142.setText(QCoreApplication.translate("MainWindow", u"dynamic", None));
        ___qtablewidgetitem143 = self.tableWidget_6.verticalHeaderItem(5)
        ___qtablewidgetitem143.setText(QCoreApplication.translate("MainWindow", u"simplify", None));
        ___qtablewidgetitem144 = self.tableWidget_6.verticalHeaderItem(6)
        ___qtablewidgetitem144.setText(QCoreApplication.translate("MainWindow", u"opset", None));
        ___qtablewidgetitem145 = self.tableWidget_6.verticalHeaderItem(7)
        ___qtablewidgetitem145.setText(QCoreApplication.translate("MainWindow", u"workspace", None));
        ___qtablewidgetitem146 = self.tableWidget_6.verticalHeaderItem(8)
        ___qtablewidgetitem146.setText(QCoreApplication.translate("MainWindow", u"nms", None));

        __sortingEnabled5 = self.tableWidget_6.isSortingEnabled()
        self.tableWidget_6.setSortingEnabled(False)
        ___qtablewidgetitem147 = self.tableWidget_6.item(0, 0)
        ___qtablewidgetitem147.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u683c\u5f0f", None));
        ___qtablewidgetitem148 = self.tableWidget_6.item(1, 0)
        ___qtablewidgetitem148.setText(QCoreApplication.translate("MainWindow", u"keras", None));
        ___qtablewidgetitem149 = self.tableWidget_6.item(2, 0)
        ___qtablewidgetitem149.setText(QCoreApplication.translate("MainWindow", u"optimize", None));
        ___qtablewidgetitem150 = self.tableWidget_6.item(3, 0)
        ___qtablewidgetitem150.setText(QCoreApplication.translate("MainWindow", u"int8", None));
        ___qtablewidgetitem151 = self.tableWidget_6.item(4, 0)
        ___qtablewidgetitem151.setText(QCoreApplication.translate("MainWindow", u"dynamic", None));
        ___qtablewidgetitem152 = self.tableWidget_6.item(5, 0)
        ___qtablewidgetitem152.setText(QCoreApplication.translate("MainWindow", u"simplify", None));
        ___qtablewidgetitem153 = self.tableWidget_6.item(6, 0)
        ___qtablewidgetitem153.setText(QCoreApplication.translate("MainWindow", u"opset", None));
        ___qtablewidgetitem154 = self.tableWidget_6.item(7, 0)
        ___qtablewidgetitem154.setText(QCoreApplication.translate("MainWindow", u"workspace", None));
        ___qtablewidgetitem155 = self.tableWidget_6.item(8, 0)
        ___qtablewidgetitem155.setText(QCoreApplication.translate("MainWindow", u"nms", None));
        self.tableWidget_6.setSortingEnabled(__sortingEnabled5)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page_10), QCoreApplication.translate("MainWindow", u"\u5bfc\u51fa\u53c2\u6570", None))
        ___qtablewidgetitem156 = self.tableWidget_7.horizontalHeaderItem(0)
        ___qtablewidgetitem156.setText(QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        ___qtablewidgetitem157 = self.tableWidget_7.horizontalHeaderItem(1)
        ___qtablewidgetitem157.setText(QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtablewidgetitem158 = self.tableWidget_7.verticalHeaderItem(0)
        ___qtablewidgetitem158.setText(QCoreApplication.translate("MainWindow", u"lr0", None));
        ___qtablewidgetitem159 = self.tableWidget_7.verticalHeaderItem(1)
        ___qtablewidgetitem159.setText(QCoreApplication.translate("MainWindow", u"lrf", None));
        ___qtablewidgetitem160 = self.tableWidget_7.verticalHeaderItem(2)
        ___qtablewidgetitem160.setText(QCoreApplication.translate("MainWindow", u"momentum", None));
        ___qtablewidgetitem161 = self.tableWidget_7.verticalHeaderItem(3)
        ___qtablewidgetitem161.setText(QCoreApplication.translate("MainWindow", u"weight_decay", None));
        ___qtablewidgetitem162 = self.tableWidget_7.verticalHeaderItem(4)
        ___qtablewidgetitem162.setText(QCoreApplication.translate("MainWindow", u"warmup_epoches", None));
        ___qtablewidgetitem163 = self.tableWidget_7.verticalHeaderItem(5)
        ___qtablewidgetitem163.setText(QCoreApplication.translate("MainWindow", u"warmup_momentum", None));
        ___qtablewidgetitem164 = self.tableWidget_7.verticalHeaderItem(6)
        ___qtablewidgetitem164.setText(QCoreApplication.translate("MainWindow", u"warmup_bise_lr", None));
        ___qtablewidgetitem165 = self.tableWidget_7.verticalHeaderItem(7)
        ___qtablewidgetitem165.setText(QCoreApplication.translate("MainWindow", u"box", None));
        ___qtablewidgetitem166 = self.tableWidget_7.verticalHeaderItem(8)
        ___qtablewidgetitem166.setText(QCoreApplication.translate("MainWindow", u"cls", None));
        ___qtablewidgetitem167 = self.tableWidget_7.verticalHeaderItem(9)
        ___qtablewidgetitem167.setText(QCoreApplication.translate("MainWindow", u"dfl", None));
        ___qtablewidgetitem168 = self.tableWidget_7.verticalHeaderItem(10)
        ___qtablewidgetitem168.setText(QCoreApplication.translate("MainWindow", u"pose", None));
        ___qtablewidgetitem169 = self.tableWidget_7.verticalHeaderItem(11)
        ___qtablewidgetitem169.setText(QCoreApplication.translate("MainWindow", u"kobj", None));
        ___qtablewidgetitem170 = self.tableWidget_7.verticalHeaderItem(12)
        ___qtablewidgetitem170.setText(QCoreApplication.translate("MainWindow", u"label_smoothing", None));
        ___qtablewidgetitem171 = self.tableWidget_7.verticalHeaderItem(13)
        ___qtablewidgetitem171.setText(QCoreApplication.translate("MainWindow", u"nbs", None));
        ___qtablewidgetitem172 = self.tableWidget_7.verticalHeaderItem(14)
        ___qtablewidgetitem172.setText(QCoreApplication.translate("MainWindow", u"hsv_h", None));
        ___qtablewidgetitem173 = self.tableWidget_7.verticalHeaderItem(15)
        ___qtablewidgetitem173.setText(QCoreApplication.translate("MainWindow", u"hsv_s", None));
        ___qtablewidgetitem174 = self.tableWidget_7.verticalHeaderItem(16)
        ___qtablewidgetitem174.setText(QCoreApplication.translate("MainWindow", u"hsv_v", None));
        ___qtablewidgetitem175 = self.tableWidget_7.verticalHeaderItem(17)
        ___qtablewidgetitem175.setText(QCoreApplication.translate("MainWindow", u"degree", None));
        ___qtablewidgetitem176 = self.tableWidget_7.verticalHeaderItem(18)
        ___qtablewidgetitem176.setText(QCoreApplication.translate("MainWindow", u"translate", None));
        ___qtablewidgetitem177 = self.tableWidget_7.verticalHeaderItem(19)
        ___qtablewidgetitem177.setText(QCoreApplication.translate("MainWindow", u"scale", None));
        ___qtablewidgetitem178 = self.tableWidget_7.verticalHeaderItem(20)
        ___qtablewidgetitem178.setText(QCoreApplication.translate("MainWindow", u"shear", None));
        ___qtablewidgetitem179 = self.tableWidget_7.verticalHeaderItem(21)
        ___qtablewidgetitem179.setText(QCoreApplication.translate("MainWindow", u"perspective", None));
        ___qtablewidgetitem180 = self.tableWidget_7.verticalHeaderItem(22)
        ___qtablewidgetitem180.setText(QCoreApplication.translate("MainWindow", u"flipud", None));
        ___qtablewidgetitem181 = self.tableWidget_7.verticalHeaderItem(23)
        ___qtablewidgetitem181.setText(QCoreApplication.translate("MainWindow", u"fliplr", None));
        ___qtablewidgetitem182 = self.tableWidget_7.verticalHeaderItem(24)
        ___qtablewidgetitem182.setText(QCoreApplication.translate("MainWindow", u"mosaic", None));
        ___qtablewidgetitem183 = self.tableWidget_7.verticalHeaderItem(25)
        ___qtablewidgetitem183.setText(QCoreApplication.translate("MainWindow", u"mixup", None));
        ___qtablewidgetitem184 = self.tableWidget_7.verticalHeaderItem(26)
        ___qtablewidgetitem184.setText(QCoreApplication.translate("MainWindow", u"copy_paste", None));
        ___qtablewidgetitem185 = self.tableWidget_7.verticalHeaderItem(27)
        ___qtablewidgetitem185.setText(QCoreApplication.translate("MainWindow", u"auto_augment", None));
        ___qtablewidgetitem186 = self.tableWidget_7.verticalHeaderItem(28)
        ___qtablewidgetitem186.setText(QCoreApplication.translate("MainWindow", u"erasing", None));
        ___qtablewidgetitem187 = self.tableWidget_7.verticalHeaderItem(29)
        ___qtablewidgetitem187.setText(QCoreApplication.translate("MainWindow", u"crop_fraction", None));

        __sortingEnabled6 = self.tableWidget_7.isSortingEnabled()
        self.tableWidget_7.setSortingEnabled(False)
        ___qtablewidgetitem188 = self.tableWidget_7.item(0, 0)
        ___qtablewidgetitem188.setText(QCoreApplication.translate("MainWindow", u"\u521d\u59cb\u5b66\u4e60\u7387", None));
        ___qtablewidgetitem189 = self.tableWidget_7.item(1, 0)
        ___qtablewidgetitem189.setText(QCoreApplication.translate("MainWindow", u"\u6700\u7ec8\u5b66\u4e60\u7387", None));
        ___qtablewidgetitem190 = self.tableWidget_7.item(2, 0)
        ___qtablewidgetitem190.setText(QCoreApplication.translate("MainWindow", u"\u52a8\u91cf", None));
        ___qtablewidgetitem191 = self.tableWidget_7.item(3, 0)
        ___qtablewidgetitem191.setText(QCoreApplication.translate("MainWindow", u"\u6743\u91cd\u8870\u51cf", None));
        ___qtablewidgetitem192 = self.tableWidget_7.item(4, 0)
        ___qtablewidgetitem192.setText(QCoreApplication.translate("MainWindow", u"\u9884\u70ed\u5468\u671f", None));
        ___qtablewidgetitem193 = self.tableWidget_7.item(5, 0)
        ___qtablewidgetitem193.setText(QCoreApplication.translate("MainWindow", u"\u9884\u70ed\u52a8\u91cf", None));
        ___qtablewidgetitem194 = self.tableWidget_7.item(6, 0)
        ___qtablewidgetitem194.setText(QCoreApplication.translate("MainWindow", u"\u9884\u70edbias\u5b66\u4e60\u7387", None));
        ___qtablewidgetitem195 = self.tableWidget_7.item(7, 0)
        ___qtablewidgetitem195.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u6846Iou\u635f\u5931\u589e\u76ca", None));
        ___qtablewidgetitem196 = self.tableWidget_7.item(8, 0)
        ___qtablewidgetitem196.setText(QCoreApplication.translate("MainWindow", u"\u79cd\u7c7b\u635f\u5931\u589e\u76ca", None));
        ___qtablewidgetitem197 = self.tableWidget_7.item(9, 0)
        ___qtablewidgetitem197.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u6846dfl\u635f\u5931\u589e\u76ca", None));
        ___qtablewidgetitem198 = self.tableWidget_7.item(10, 0)
        ___qtablewidgetitem198.setText(QCoreApplication.translate("MainWindow", u"\u5173\u952e\u70b9\u4f4d\u7f6e\u635f\u5931\u589e\u76ca", None));
        ___qtablewidgetitem199 = self.tableWidget_7.item(11, 0)
        ___qtablewidgetitem199.setText(QCoreApplication.translate("MainWindow", u"\u5173\u952e\u70b9\u53ef\u89c1\u6027\u635f\u5931\u589e\u76ca", None));
        ___qtablewidgetitem200 = self.tableWidget_7.item(12, 0)
        ___qtablewidgetitem200.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e\u5e73\u6ed1", None));
        ___qtablewidgetitem201 = self.tableWidget_7.item(13, 0)
        ___qtablewidgetitem201.setText(QCoreApplication.translate("MainWindow", u"\u6807\u51c6\u6279\u6570\u91cf", None));
        ___qtablewidgetitem202 = self.tableWidget_7.item(14, 0)
        ___qtablewidgetitem202.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u8272\u76f8", None));
        ___qtablewidgetitem203 = self.tableWidget_7.item(15, 0)
        ___qtablewidgetitem203.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u9971\u548c\u5ea6", None));
        ___qtablewidgetitem204 = self.tableWidget_7.item(16, 0)
        ___qtablewidgetitem204.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u8272\u8c03", None));
        ___qtablewidgetitem205 = self.tableWidget_7.item(17, 0)
        ___qtablewidgetitem205.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u65cb\u8f6c\u89d2\u5ea6", None));
        ___qtablewidgetitem206 = self.tableWidget_7.item(18, 0)
        ___qtablewidgetitem206.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u5e73\u79fb\u6bd4\u4f8b", None));
        ___qtablewidgetitem207 = self.tableWidget_7.item(19, 0)
        ___qtablewidgetitem207.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u7f29\u653e\u6bd4\u4f8b", None));
        ___qtablewidgetitem208 = self.tableWidget_7.item(20, 0)
        ___qtablewidgetitem208.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u659c\u5207\u89d2\u5ea6", None));
        ___qtablewidgetitem209 = self.tableWidget_7.item(21, 0)
        ___qtablewidgetitem209.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u900f\u89c6\u53d8\u6362", None));
        ___qtablewidgetitem210 = self.tableWidget_7.item(22, 0)
        ___qtablewidgetitem210.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e0b\u7ffb\u8f6c\u6982\u7387", None));
        ___qtablewidgetitem211 = self.tableWidget_7.item(23, 0)
        ___qtablewidgetitem211.setText(QCoreApplication.translate("MainWindow", u"\u5de6\u53f3\u7ffb\u8f6c\u6982\u7387", None));
        ___qtablewidgetitem212 = self.tableWidget_7.item(24, 0)
        ___qtablewidgetitem212.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u62fc\u63a5\u6982\u7387", None));
        ___qtablewidgetitem213 = self.tableWidget_7.item(25, 0)
        ___qtablewidgetitem213.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u53e0\u52a0\u6982\u7387", None));
        ___qtablewidgetitem214 = self.tableWidget_7.item(26, 0)
        ___qtablewidgetitem214.setText(QCoreApplication.translate("MainWindow", u"\u8d4b\u503c\u9ecf\u8d34\u6982\u7387", None));
        ___qtablewidgetitem215 = self.tableWidget_7.item(27, 0)
        ___qtablewidgetitem215.setText(QCoreApplication.translate("MainWindow", u"\u5206\u7c7b\u81ea\u52a8\u589e\u5f3a", None));
        ___qtablewidgetitem216 = self.tableWidget_7.item(28, 0)
        ___qtablewidgetitem216.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u64e6\u9664\u6982\u7387", None));
        ___qtablewidgetitem217 = self.tableWidget_7.item(29, 0)
        ___qtablewidgetitem217.setText(QCoreApplication.translate("MainWindow", u"\u88c1\u526a\u6bd4\u4f8b", None));
        self.tableWidget_7.setSortingEnabled(__sortingEnabled6)

        self.toolBox.setItemText(self.toolBox.indexOf(self.page_11), QCoreApplication.translate("MainWindow", u"\u8d85\u53c2\u6570", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u6587\u4ef6", None))
    # retranslateUi

