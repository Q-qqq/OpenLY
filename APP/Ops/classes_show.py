from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class ClassColorModel(QStandardItemModel):
    def __init__(self, img_label):
        super().__init__()
        self.img_label = img_label

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.BackgroundRole:
            colors = self.img_label.colors
            c = colors[index.row()]
            return QColor(*c)
        return super().data(index, role)
    

class ClassesView(QObject):
    """处理img_label和classes_lv的交互逻辑"""
    def __init__(self, classes_lv):
        super().__init__()
        self.img_label = None
        self.classes_lv = classes_lv
        self.eventConnect()

    def eventConnect(self):
        self.classes_lv.installEventFilter(self)
        self.classes_lv.clicked.connect(self.selectedClass)
    
    def setImgLabel(self, img_label):
        self.img_label = img_label
        self.img_label.Show_Classes_Signal.connect(self.show)
    
    def show(self):
        if self.img_label is None:
            return
        if self.img_label.label:
            model = ClassColorModel(self.img_label)
            ns = len(self.img_label.label["names"])
            for i in range(ns):
                model.appendRow(QStandardItem(self.img_label.label["names"][i]))
            self.classes_lv.setModel(model)
            # 获取第三行的模型索引 
            index = model.index(self.img_label.cls,  0)
            # 设置选中并滚动到可视区域 
            self.classes_lv.setCurrentIndex(index) 
            self.classes_lv.scrollTo(index,  QListView.PositionAtCenter)
        if self.classes_lv.maximumWidth() == 150:
            self.close_()
        elif 0 < self.classes_lv.maximumWidth() < 120:
            return
        else:
            animal = QPropertyAnimation(self.classes_lv, b"maximumWidth", self.classes_lv.parent())
            animal.setStartValue(self.classes_lv.maximumWidth())
            animal.setEndValue(150)
            animal.setDuration(500)   # 动画时长300ms
            animal.setEasingCurve(QEasingCurve.Type.OutCirc)   # 弹性效果
            animal.start()
        
    
    def close_(self):
        animal = QPropertyAnimation(self.classes_lv, b"maximumWidth", self.classes_lv.parent())
        animal.setStartValue(self.classes_lv.maximumWidth())
        animal.setEndValue(0)
        animal.setDuration(300)   # 动画时长300ms
        animal.start()
        

    def selectedClass(self, index:QModelIndex):
        self.img_label.cls = index.row()

    def eventFilter(self, obj, event):
        if obj == self.classes_lv  and event.type()  == QEvent.Leave and self.classes_lv.maximumWidth() > 120:
            self.close_()
        return super().eventFilter(obj, event)
    


