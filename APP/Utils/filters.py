"""事件过滤器"""
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from APP.Utils import get_widget

class CentralWidgetFilter(QObject):
    def eventFilter(self, watched:QObject, event:QEvent) -> bool:
        if watched == self.parent():
            if event.type() == QEvent.Enter:
                self.enterAnimation()
            elif event.type() == QEvent.Leave:
                self.leaveAnimation()
        return super().eventFilter(watched, event)

    def enterAnimation(self):
        animal = QPropertyAnimation(self.parent().parent().Painter_tool_f, b"maximumHeight", self.parent().parent())
        animal.setStartValue(0)
        animal.setEndValue(40)
        animal.setDuration(50)
        animal.start()

    def leaveAnimation(self):
        animal = QPropertyAnimation(self.parent().parent().Painter_tool_f, b"maximumHeight", self.parent().parent())
        animal.setStartValue(40)
        animal.setEndValue(0)
        animal.setDuration(50)
        animal.start()

class CbbFilter(QObject):
    """parent： sift_dataset"""
    def eventFilter(self, watched, event):
        select_dataset_cbb = self.parent().Select_dataset_cbb
        select_class_cbb = self.parent().Select_class_cbb
        sift_dataset = self.parent().sift_dataset
        if event.type() == QEvent.MouseButtonPress:
            if watched == select_dataset_cbb:
                dataset_items = sift_dataset.getItems()
                items = [select_dataset_cbb.itemText(i) for i in range(select_dataset_cbb.count())]
                for item in dataset_items:
                    if item not in items:
                        select_dataset_cbb.addItem(item)
                for item in items:
                    if item not in dataset_items:
                        select_dataset_cbb.removeItem(items.index(item))
            elif watched == select_class_cbb:
                names = sift_dataset.getNames()
                class_items = ["all"] + list(names.values()) + ["ok"]
                items = [select_class_cbb.itemText(i) for i in range(select_class_cbb.count())]
                for item in class_items:
                    if item not in items:
                        select_class_cbb.addItem(item)
                for item in items:
                    if item not in class_items:
                        select_class_cbb.removeItem(items.index(item))
        return super().eventFilter(watched, event)

class MenuFilter(QObject):
    """parent menuTool"""
    def eventFilter(self, watched:QObject, event:QEvent) -> bool:
        if event.type() == QEvent.MouseButtonPress:
            if watched == self.parent().parent().menubar:
                self.parent().edit_showClasses()
        return super().eventFilter(watched, event)

