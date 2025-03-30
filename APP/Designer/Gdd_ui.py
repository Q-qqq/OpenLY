# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Gdd.ui'
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
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSpacerItem, QStatusBar, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1242, 790)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_5 = QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setMaximumSize(QSize(500, 16777215))
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_4 = QGridLayout(self.frame)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        font = QFont()
        font.setFamilies([u"\u5b8b\u4f53"])
        font.setPointSize(12)
        self.label_2.setFont(font)

        self.gridLayout_4.addWidget(self.label_2, 0, 0, 1, 1)

        self.label_3 = QLabel(self.frame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)

        self.gridLayout_4.addWidget(self.label_3, 1, 0, 1, 1)

        self.add_cls_pb = QPushButton(self.frame)
        self.add_cls_pb.setObjectName(u"add_cls_pb")
        self.add_cls_pb.setMinimumSize(QSize(0, 30))
        self.add_cls_pb.setFont(font)

        self.gridLayout_4.addWidget(self.add_cls_pb, 2, 2, 1, 1)

        self.label_4 = QLabel(self.frame)
        self.label_4.setObjectName(u"label_4")
        font1 = QFont()
        font1.setFamilies([u"\u5b8b\u4f53"])
        font1.setPointSize(14)
        self.label_4.setFont(font1)

        self.gridLayout_4.addWidget(self.label_4, 5, 0, 1, 1)

        self.scrollArea_2 = QScrollArea(self.frame)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setMinimumSize(QSize(0, 0))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 286, 437))
        self.gridLayout_2 = QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.rect_LW = QListWidget(self.scrollAreaWidgetContents_2)
        self.rect_LW.setObjectName(u"rect_LW")

        self.gridLayout_2.addWidget(self.rect_LW, 0, 0, 1, 1)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.gridLayout_4.addWidget(self.scrollArea_2, 5, 1, 1, 2)

        self.del_cls_pb = QPushButton(self.frame)
        self.del_cls_pb.setObjectName(u"del_cls_pb")
        self.del_cls_pb.setMinimumSize(QSize(0, 30))
        self.del_cls_pb.setFont(font)

        self.gridLayout_4.addWidget(self.del_cls_pb, 3, 2, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.valildate_now_pb = QPushButton(self.frame)
        self.valildate_now_pb.setObjectName(u"valildate_now_pb")
        self.valildate_now_pb.setMinimumSize(QSize(0, 30))
        self.valildate_now_pb.setFont(font)

        self.horizontalLayout_3.addWidget(self.valildate_now_pb)

        self.clear_labelPB = QPushButton(self.frame)
        self.clear_labelPB.setObjectName(u"clear_labelPB")
        self.clear_labelPB.setMinimumSize(QSize(0, 30))
        self.clear_labelPB.setFont(font)

        self.horizontalLayout_3.addWidget(self.clear_labelPB)

        self.save_rect_pb = QPushButton(self.frame)
        self.save_rect_pb.setObjectName(u"save_rect_pb")
        self.save_rect_pb.setMinimumSize(QSize(0, 30))
        self.save_rect_pb.setFont(font)

        self.horizontalLayout_3.addWidget(self.save_rect_pb)


        self.gridLayout_4.addLayout(self.horizontalLayout_3, 6, 1, 1, 2)

        self.labelnameLE = QLineEdit(self.frame)
        self.labelnameLE.setObjectName(u"labelnameLE")
        self.labelnameLE.setMinimumSize(QSize(0, 30))
        self.labelnameLE.setFont(font)

        self.gridLayout_4.addWidget(self.labelnameLE, 1, 1, 1, 2)

        self.img_nameLE = QLineEdit(self.frame)
        self.img_nameLE.setObjectName(u"img_nameLE")
        self.img_nameLE.setMinimumSize(QSize(0, 30))
        self.img_nameLE.setFont(font)

        self.gridLayout_4.addWidget(self.img_nameLE, 0, 1, 1, 2)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer, 4, 2, 1, 1)

        self.sort_TW = QTableWidget(self.frame)
        if (self.sort_TW.columnCount() < 2):
            self.sort_TW.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.sort_TW.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.sort_TW.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        self.sort_TW.setObjectName(u"sort_TW")
        self.sort_TW.setMaximumSize(QSize(230, 150))
        self.sort_TW.setFrameShape(QFrame.Box)
        self.sort_TW.setTextElideMode(Qt.ElideMiddle)
        self.sort_TW.setSortingEnabled(False)

        self.gridLayout_4.addWidget(self.sort_TW, 2, 1, 3, 1)


        self.gridLayout_5.addWidget(self.frame, 0, 0, 1, 1)

        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Box)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.frame_2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.duHLayout = QHBoxLayout()
        self.duHLayout.setObjectName(u"duHLayout")
        self.upPB = QPushButton(self.frame_2)
        self.upPB.setObjectName(u"upPB")
        self.upPB.setMinimumSize(QSize(0, 30))
        self.upPB.setFont(font)

        self.duHLayout.addWidget(self.upPB)

        self.ok_get_rect_pb = QPushButton(self.frame_2)
        self.ok_get_rect_pb.setObjectName(u"ok_get_rect_pb")
        self.ok_get_rect_pb.setMinimumSize(QSize(0, 30))
        self.ok_get_rect_pb.setFont(font)

        self.duHLayout.addWidget(self.ok_get_rect_pb)

        self.delect_rect_pb = QPushButton(self.frame_2)
        self.delect_rect_pb.setObjectName(u"delect_rect_pb")
        self.delect_rect_pb.setMinimumSize(QSize(0, 30))
        self.delect_rect_pb.setFont(font)

        self.duHLayout.addWidget(self.delect_rect_pb)

        self.model_text_pb = QPushButton(self.frame_2)
        self.model_text_pb.setObjectName(u"model_text_pb")
        self.model_text_pb.setMinimumSize(QSize(0, 30))
        self.model_text_pb.setFont(font)

        self.duHLayout.addWidget(self.model_text_pb)

        self.downPB = QPushButton(self.frame_2)
        self.downPB.setObjectName(u"downPB")
        self.downPB.setMinimumSize(QSize(0, 30))
        self.downPB.setFont(font)

        self.duHLayout.addWidget(self.downPB)


        self.gridLayout.addLayout(self.duHLayout, 4, 0, 1, 2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.fileportpathLE = QLineEdit(self.frame_2)
        self.fileportpathLE.setObjectName(u"fileportpathLE")
        self.fileportpathLE.setMinimumSize(QSize(0, 30))

        self.horizontalLayout.addWidget(self.fileportpathLE)

        self.slepicfilePB = QPushButton(self.frame_2)
        self.slepicfilePB.setObjectName(u"slepicfilePB")
        self.slepicfilePB.setMinimumSize(QSize(0, 30))
        self.slepicfilePB.setFont(font)

        self.horizontalLayout.addWidget(self.slepicfilePB)


        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.trainRB = QRadioButton(self.frame_2)
        self.trainRB.setObjectName(u"trainRB")

        self.verticalLayout.addWidget(self.trainRB)

        self.valRB = QRadioButton(self.frame_2)
        self.valRB.setObjectName(u"valRB")

        self.verticalLayout.addWidget(self.valRB)


        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.paint_rectPB = QPushButton(self.frame_2)
        self.paint_rectPB.setObjectName(u"paint_rectPB")
        self.paint_rectPB.setMinimumSize(QSize(0, 30))
        self.paint_rectPB.setFont(font)

        self.horizontalLayout_2.addWidget(self.paint_rectPB)


        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.image_frame = QFrame(self.frame_2)
        self.image_frame.setObjectName(u"image_frame")
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_6 = QGridLayout(self.image_frame)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setHorizontalSpacing(0)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.image_scoll_are = QScrollArea(self.image_frame)
        self.image_scoll_are.setObjectName(u"image_scoll_are")
        self.image_scoll_are.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.image_scoll_are.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.image_scoll_are.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName(u"scroll_widget")
        self.scroll_widget.setGeometry(QRect(0, 0, 551, 562))
        self.scroll_widget.setMinimumSize(QSize(0, 0))
        self.gridLayout_7 = QGridLayout(self.scroll_widget)
        self.gridLayout_7.setSpacing(0)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.imageLB = QLabel(self.scroll_widget)
        self.imageLB.setObjectName(u"imageLB")
        self.imageLB.setFrameShape(QFrame.Box)
        self.imageLB.setAlignment(Qt.AlignCenter)

        self.gridLayout_7.addWidget(self.imageLB, 0, 0, 1, 1)

        self.image_scoll_are.setWidget(self.scroll_widget)

        self.gridLayout_6.addWidget(self.image_scoll_are, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.image_frame, 2, 0, 1, 2)

        self.show_num_img_L = QLabel(self.frame_2)
        self.show_num_img_L.setObjectName(u"show_num_img_L")
        self.show_num_img_L.setFont(font)

        self.gridLayout.addWidget(self.show_num_img_L, 1, 1, 1, 1)

        self.gridLayout.setRowStretch(2, 9)

        self.gridLayout_5.addWidget(self.frame_2, 0, 1, 1, 1)

        self.frame_3 = QFrame(self.centralwidget)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.Box)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.scrollArea = QScrollArea(self.frame_3)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setMinimumSize(QSize(200, 0))
        self.scrollArea.setMaximumSize(QSize(200, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 198, 557))
        self.verticalLayout_2 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.all_image_widget = QWidget(self.scrollAreaWidgetContents)
        self.all_image_widget.setObjectName(u"all_image_widget")

        self.verticalLayout_2.addWidget(self.all_image_widget)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout_3.addWidget(self.scrollArea, 3, 0, 1, 2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.load_all_img_pb = QPushButton(self.frame_3)
        self.load_all_img_pb.setObjectName(u"load_all_img_pb")
        self.load_all_img_pb.setMinimumSize(QSize(0, 30))
        self.load_all_img_pb.setFont(font)

        self.horizontalLayout_4.addWidget(self.load_all_img_pb)

        self.del_img_pb = QPushButton(self.frame_3)
        self.del_img_pb.setObjectName(u"del_img_pb")
        self.del_img_pb.setMinimumSize(QSize(0, 30))
        self.del_img_pb.setFont(font)

        self.horizontalLayout_4.addWidget(self.del_img_pb)


        self.gridLayout_3.addLayout(self.horizontalLayout_4, 4, 0, 1, 2)

        self.label_nolabel_cbb = QComboBox(self.frame_3)
        self.label_nolabel_cbb.addItem("")
        self.label_nolabel_cbb.addItem("")
        self.label_nolabel_cbb.addItem("")
        self.label_nolabel_cbb.setObjectName(u"label_nolabel_cbb")
        self.label_nolabel_cbb.setMinimumSize(QSize(0, 30))
        self.label_nolabel_cbb.setFont(font)

        self.gridLayout_3.addWidget(self.label_nolabel_cbb, 0, 0, 1, 1)

        self.del_all_pb = QPushButton(self.frame_3)
        self.del_all_pb.setObjectName(u"del_all_pb")
        self.del_all_pb.setMinimumSize(QSize(0, 30))
        self.del_all_pb.setFont(font)

        self.gridLayout_3.addWidget(self.del_all_pb, 1, 1, 1, 1)

        self.train_val_cbb = QComboBox(self.frame_3)
        self.train_val_cbb.addItem("")
        self.train_val_cbb.addItem("")
        self.train_val_cbb.addItem("")
        self.train_val_cbb.setObjectName(u"train_val_cbb")
        self.train_val_cbb.setMinimumSize(QSize(0, 30))
        self.train_val_cbb.setFont(font)

        self.gridLayout_3.addWidget(self.train_val_cbb, 0, 1, 1, 1)

        self.cls_cbb = QComboBox(self.frame_3)
        self.cls_cbb.addItem("")
        self.cls_cbb.setObjectName(u"cls_cbb")
        self.cls_cbb.setMinimumSize(QSize(0, 30))
        self.cls_cbb.setFont(font)

        self.gridLayout_3.addWidget(self.cls_cbb, 1, 0, 1, 1)

        self.finded_img_name_le = QLineEdit(self.frame_3)
        self.finded_img_name_le.setObjectName(u"finded_img_name_le")
        self.finded_img_name_le.setMinimumSize(QSize(0, 30))
        self.finded_img_name_le.setFont(font)

        self.gridLayout_3.addWidget(self.finded_img_name_le, 2, 0, 1, 2)


        self.gridLayout_5.addWidget(self.frame_3, 0, 2, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 2)
        self.gridLayout_5.setColumnStretch(1, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1242, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"ChildWindow-dataset", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf\u540d\u79f0\uff1a", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e\u540d\u79f0\uff1a", None))
        self.add_cls_pb.setText(QCoreApplication.translate("MainWindow", u"\u6dfb\u52a0\u7c7b", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u6807\u7b7e\uff1a", None))
        self.del_cls_pb.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664\u7c7b", None))
        self.valildate_now_pb.setText(QCoreApplication.translate("MainWindow", u"\u786e\u5b9a", None))
        self.clear_labelPB.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u7a7a", None))
        self.save_rect_pb.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58", None))
        ___qtablewidgetitem = self.sort_TW.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"\u79cd\u7c7b\u7d22\u5f15", None));
        ___qtablewidgetitem1 = self.sort_TW.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"\u79cd\u7c7b\u540d\u79f0", None));
        self.upPB.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e00\u5f20", None))
        self.ok_get_rect_pb.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u4e2a\u6846", None))
#if QT_CONFIG(shortcut)
        self.ok_get_rect_pb.setShortcut(QCoreApplication.translate("MainWindow", u"Return", None))
#endif // QT_CONFIG(shortcut)
        self.delect_rect_pb.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664\u6846", None))
#if QT_CONFIG(shortcut)
        self.delect_rect_pb.setShortcut(QCoreApplication.translate("MainWindow", u"Del", None))
#endif // QT_CONFIG(shortcut)
        self.model_text_pb.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u6d4b\u8bd5", None))
        self.downPB.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u5f20", None))
        self.fileportpathLE.setText("")
        self.slepicfilePB.setText(QCoreApplication.translate("MainWindow", u"\u6dfb\u52a0\u6570\u636e\u96c6", None))
        self.trainRB.setText(QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u96c6", None))
        self.valRB.setText(QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1\u96c6", None))
        self.paint_rectPB.setText(QCoreApplication.translate("MainWindow", u"\u753b\u6846", None))
        self.imageLB.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u50cf", None))
        self.show_num_img_L.setText(QCoreApplication.translate("MainWindow", u"n/m", None))
        self.load_all_img_pb.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165", None))
        self.del_img_pb.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664\u56fe\u50cf", None))
        self.label_nolabel_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"\u5168\u90e8", None))
        self.label_nolabel_cbb.setItemText(1, QCoreApplication.translate("MainWindow", u"\u5df2\u6807\u6ce8", None))
        self.label_nolabel_cbb.setItemText(2, QCoreApplication.translate("MainWindow", u"\u672a\u6807\u6ce8", None))

        self.del_all_pb.setText(QCoreApplication.translate("MainWindow", u"\u786e\u5b9a\u7b5b\u9009", None))
        self.train_val_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"\u5168\u90e8", None))
        self.train_val_cbb.setItemText(1, QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u96c6", None))
        self.train_val_cbb.setItemText(2, QCoreApplication.translate("MainWindow", u"\u9a8c\u8bc1\u96c6", None))

        self.cls_cbb.setItemText(0, QCoreApplication.translate("MainWindow", u"\u5168\u90e8", None))

    # retranslateUi

