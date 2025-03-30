# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'config.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QHeaderView, QLayout,
    QMainWindow, QSizePolicy, QStatusBar, QToolBar,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(450, 743)
        MainWindow.setStyleSheet(u"")
        self.Save_a = QAction(MainWindow)
        self.Save_a.setObjectName(u"Save_a")
        self.Update_a = QAction(MainWindow)
        self.Update_a.setObjectName(u"Update_a")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.Config_tw = QTreeWidget(self.centralwidget)
        brush = QBrush(QColor(0, 0, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setForeground(0, brush);
        self.Config_tw.setHeaderItem(__qtreewidgetitem)
        self.Config_tw.setObjectName(u"Config_tw")
        self.Config_tw.setStyleSheet(u"")
        self.Config_tw.setFrameShape(QFrame.StyledPanel)
        self.Config_tw.setDragEnabled(False)
        self.Config_tw.setDragDropOverwriteMode(False)
        self.Config_tw.setAllColumnsShowFocus(False)
        self.Config_tw.setWordWrap(False)

        self.verticalLayout.addWidget(self.Config_tw)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.Save_a)
        self.toolBar.addAction(self.Update_a)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        MainWindow.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        MainWindow.setStatusTip("")
#endif // QT_CONFIG(statustip)
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
        ___qtreewidgetitem = self.Config_tw.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("MainWindow", u"\u503c", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("MainWindow", u"\u5c5e\u6027", None));
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

