# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'makenet.ui'
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
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QSpacerItem, QStatusBar, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(974, 680)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.netLB = QLabel(self.centralwidget)
        self.netLB.setObjectName(u"netLB")
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(12)
        self.netLB.setFont(font)

        self.horizontalLayout_3.addWidget(self.netLB)

        self.netpathLE = QLineEdit(self.centralwidget)
        self.netpathLE.setObjectName(u"netpathLE")
        self.netpathLE.setFont(font)

        self.horizontalLayout_3.addWidget(self.netpathLE)

        self.licontxtPB = QPushButton(self.centralwidget)
        self.licontxtPB.setObjectName(u"licontxtPB")
        self.licontxtPB.setMaximumSize(QSize(30, 16777215))
        self.licontxtPB.setFont(font)

        self.horizontalLayout_3.addWidget(self.licontxtPB)

        self.savePB = QPushButton(self.centralwidget)
        self.savePB.setObjectName(u"savePB")
        self.savePB.setFont(font)

        self.horizontalLayout_3.addWidget(self.savePB)


        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 2)

        self.k_means_get_anchors_pb = QPushButton(self.centralwidget)
        self.k_means_get_anchors_pb.setObjectName(u"k_means_get_anchors_pb")
        self.k_means_get_anchors_pb.setFont(font)

        self.gridLayout.addWidget(self.k_means_get_anchors_pb, 0, 2, 1, 1)

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(0)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.treeWidget = QTreeWidget(self.frame)
        self.treeWidget.setObjectName(u"treeWidget")

        self.gridLayout_2.addWidget(self.treeWidget, 1, 1, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.delete_net_nodePB = QPushButton(self.frame)
        self.delete_net_nodePB.setObjectName(u"delete_net_nodePB")
        font1 = QFont()
        font1.setFamilies([u"\u5b8b\u4f53"])
        font1.setPointSize(10)
        self.delete_net_nodePB.setFont(font1)

        self.horizontalLayout_2.addWidget(self.delete_net_nodePB)

        self.rev_net_nodePB = QPushButton(self.frame)
        self.rev_net_nodePB.setObjectName(u"rev_net_nodePB")
        self.rev_net_nodePB.setFont(font1)

        self.horizontalLayout_2.addWidget(self.rev_net_nodePB)

        self.insert_net_nodePB = QPushButton(self.frame)
        self.insert_net_nodePB.setObjectName(u"insert_net_nodePB")
        self.insert_net_nodePB.setFont(font1)

        self.horizontalLayout_2.addWidget(self.insert_net_nodePB)

        self.add_net_nodePB = QPushButton(self.frame)
        self.add_net_nodePB.setObjectName(u"add_net_nodePB")
        self.add_net_nodePB.setFont(font1)

        self.horizontalLayout_2.addWidget(self.add_net_nodePB)

        self.loadPB = QPushButton(self.frame)
        self.loadPB.setObjectName(u"loadPB")

        self.horizontalLayout_2.addWidget(self.loadPB)

        self.clearPB = QPushButton(self.frame)
        self.clearPB.setObjectName(u"clearPB")

        self.horizontalLayout_2.addWidget(self.clearPB)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)


        self.gridLayout.addWidget(self.frame, 1, 1, 2, 2)

        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMaximumSize(QSize(400, 16777215))
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.select_nodeCBB = QComboBox(self.frame_2)
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.addItem("")
        self.select_nodeCBB.setObjectName(u"select_nodeCBB")
        self.select_nodeCBB.setMinimumSize(QSize(200, 0))
        self.select_nodeCBB.setMaximumSize(QSize(16777215, 16777215))
        self.select_nodeCBB.setFont(font)

        self.gridLayout_3.addWidget(self.select_nodeCBB, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(183, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.edit_nodeTX = QTextEdit(self.frame_2)
        self.edit_nodeTX.setObjectName(u"edit_nodeTX")
        self.edit_nodeTX.setFont(font)

        self.gridLayout_3.addWidget(self.edit_nodeTX, 1, 0, 1, 2)

        self.parse_net_label = QLabel(self.frame_2)
        self.parse_net_label.setObjectName(u"parse_net_label")
        self.parse_net_label.setFont(font)
        self.parse_net_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout_3.addWidget(self.parse_net_label, 2, 0, 1, 2)

        self.gridLayout_3.setRowStretch(1, 5)
        self.gridLayout_3.setRowStretch(2, 5)

        self.gridLayout.addWidget(self.frame_2, 1, 0, 2, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 974, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"ChildWindow-makenet", None))
        self.netLB.setText(QCoreApplication.translate("MainWindow", u"\u795e\u7ecf\u7f51\u7edc\u7ed3\u6784\u8def\u5f84\uff1a", None))
        self.licontxtPB.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.savePB.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58", None))
        self.k_means_get_anchors_pb.setText(QCoreApplication.translate("MainWindow", u"\u83b7\u53d6\u5148\u9a8c\u6846", None))
        ___qtreewidgetitem = self.treeWidget.headerItem()
        ___qtreewidgetitem.setText(3, QCoreApplication.translate("MainWindow", u"output size", None));
        ___qtreewidgetitem.setText(2, QCoreApplication.translate("MainWindow", u"input size", None));
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("MainWindow", u"net", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("MainWindow", u"num", None));
        self.delete_net_nodePB.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664\u6240\u9009\u7f51\u7edc\u8282\u70b9", None))
        self.rev_net_nodePB.setText(QCoreApplication.translate("MainWindow", u"\u4fee\u6539\u6240\u9009\u7f51\u7edc\u8282\u70b9", None))
        self.insert_net_nodePB.setText(QCoreApplication.translate("MainWindow", u"\u5411\u4e0b\u63d2\u5165\u7f51\u7edc\u8282\u70b9", None))
        self.add_net_nodePB.setText(QCoreApplication.translate("MainWindow", u"\u6dfb\u52a0\u7f51\u7edc\u8282\u70b9", None))
        self.loadPB.setText(QCoreApplication.translate("MainWindow", u"\u76f4\u63a5\u8f7d\u5165", None))
        self.clearPB.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u7a7a", None))
        self.select_nodeCBB.setItemText(0, QCoreApplication.translate("MainWindow", u"net", None))
        self.select_nodeCBB.setItemText(1, QCoreApplication.translate("MainWindow", u"Focus", None))
        self.select_nodeCBB.setItemText(2, QCoreApplication.translate("MainWindow", u"Conv", None))
        self.select_nodeCBB.setItemText(3, QCoreApplication.translate("MainWindow", u"nn.Conv", None))
        self.select_nodeCBB.setItemText(4, QCoreApplication.translate("MainWindow", u"CSP", None))
        self.select_nodeCBB.setItemText(5, QCoreApplication.translate("MainWindow", u"SPP", None))
        self.select_nodeCBB.setItemText(6, QCoreApplication.translate("MainWindow", u"nn.MaxPool2d", None))
        self.select_nodeCBB.setItemText(7, QCoreApplication.translate("MainWindow", u"nn.Upsample", None))
        self.select_nodeCBB.setItemText(8, QCoreApplication.translate("MainWindow", u"Concat", None))
        self.select_nodeCBB.setItemText(9, QCoreApplication.translate("MainWindow", u"Res", None))
        self.select_nodeCBB.setItemText(10, QCoreApplication.translate("MainWindow", u"Conn", None))

        self.parse_net_label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
    # retranslateUi

