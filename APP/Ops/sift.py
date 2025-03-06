
import shutil
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


import os
from pathlib import Path
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import repeat

from ultralytics.data.utils import img2label_paths,IMG_FORMATS, verify_image
from ultralytics.utils import ThreadingLocked, threaded, yaml_load, NUM_THREADS, LOGGER




from APP  import PROJ_SETTINGS, getExperimentPath,EXPERIMENT_SETTINGS, debounce
from APP.Data.build import check_cls_dataset, check_detect_dataset
from APP.Data import getNoLabelPath, readLabelFile
from APP.Utils import get_widget
from APP.Utils.filters import CbbFilter



class SiftDataset(QObject):
    """筛选数据集"""
    def __init__(self,parent):
        super().__init__(parent)
        """parent:Sift_f"""
        self.task = ""
        self.args = None
        self.data = ""
        self.train_set = None
        self.val_set = None
        self.train_path = ""
        self.val_path = None
        self.base_items = ["总样本集","训练集", "验证集","未标注集", "结果集"]
        self.sift_tool = SiftTool(self)

    def build(self, data, task , args):
        if task == "classify":
            self.data, self.train_set, self.val_set, self.train_path, self.val_path = check_cls_dataset(data, args)
        else:
            self.data, self.train_set, self.val_set, self.train_path, self.val_path = check_detect_dataset(data, args)
        
        self.task = task
        self.args = args

    def initLoadDataset(self):
        """初始加载所有图像"""
        im_shapes = {**self.getValue("总样本集"), **self.getValue("结果集")}
        self.sift_tool.loadImages(im_shapes)
        if len(self.sift_tool.images_label.im_files) >= len(im_shapes):  #完全加载
            self.sift_tool.images_label.build = True
        self.sift_tool.siftImageSignal()
        self.sift_tool.image_scroll.resizeEvent(None)
    
    def addNolabels(self, im_files):
        """添加未标注图像"""
        no_label_path = getNoLabelPath()
        all_files_name = [Path(im_file).name for im_file in self.sift_tool.images_label.im_files]  #所有图像文件名
        exist_names = []
        new_im_files = []
        for im_file in im_files:
            if Path(im_file).exists():
                if Path(im_file).name not in all_files_name:
                    shutil.copy(im_file, no_label_path)
                    new_im_files.append(str(Path(no_label_path) / Path(im_file).name))
                else:
                    exist_names.append(Path(im_file).name)
        if new_im_files:
            im_shapes = self.getImShapes(new_im_files)
            self.sift_tool.loadImages(im_shapes)
            if self.sift_tool.select_dataset_cbb.currentText() in ("总样本集", "未标注集"):
                self.sift_tool.showImages(new_im_files+ self.sift_tool.images_label.show_files)
        return exist_names

    def getImShapes(self, im_files):
        """获取图像对应的大小，输出字典{im_file:shape}
        Args:
            im_files(list):图像文件列表
        Returns:
            im_shapes(dict):{im_file:shape}
        """
        im_shapes = {}
        if len(im_files):
            cls = [0 for i in im_files]
            pr = ["" for i in im_files]
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image,
                                    iterable=zip(zip(im_files, cls), pr))
                for (im_file, cls), nf, nc, msg, shape in results:
                    if msg != "":
                        LOGGER.warning("未标注图像损坏：" + msg)
                    else:
                        im_shapes[im_file] = list(reversed(shape))
        return im_shapes

    def get_no_label_files(self):
        """获取未标注的图像"""
        no_label_path = Path(PROJ_SETTINGS["name"]) / "data" / "no_label"
        im_files =  [str(path) for path in no_label_path.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
        return self.getImShapes(im_files)

    def tryGetConfusionResults(self):
        """尝试获取混淆矩阵结果"""
        path = Path(getExperimentPath()) / "Confusion_Matrix_Imfiles.yaml"
        if path.exists():
            return yaml_load(path)
        else:
            return {}

    def tryGetResults(self):
        """尝试获取验证结果图像"""
        results_path = Path(getExperimentPath())
        im_files = [str(path) for path in results_path.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
        return self.getImShapes(im_files)

    def getNames(self):
        """获取种类
        return(dict): {num:class}"""
        if self.train_set is not None:
            if self.task == "classify":
                names = self.train_set.names
            else:
                names = self.train_set.data["names"]
        else:
            names = {}
        return names

    def getItems(self):
        """
        获取数据集名称
        return(list):['总样本集','训练集','验证集', '未标注集','结果集',...]
        """
        confusions = self.tryGetConfusionResults()
        confu_items = list(confusions.keys())
        return self.base_items+ confu_items

    def getValue(self, item):
        """获取数据集名称对应的图像文件集"""
        confusions = self.tryGetConfusionResults()
        if item == "总样本集":
            im_shapes = {**self.train_set.getImShapes(), **self.val_set.getImShapes(), **self.get_no_label_files()}
        elif item == "训练集":
            im_shapes = self.train_set.getImShapes()
        elif item == "验证集":
            im_shapes = self.val_set.getImShapes()
        elif item == "未标注集":
            im_shapes =  self.get_no_label_files()
        elif item == "结果集":
            im_shapes = self.tryGetResults()
        elif item in confusions.keys():
            im_files = confusions[item]
            im_files = [im_file for im_file in im_files if Path(im_file).exists()]
            im_shapes = self.getImShapes(im_files)
        else:
            im_shapes = {}
        return im_shapes

    def searchAll(self,im_shapes, cls_name, name):
        """同时搜索种类和文件名称"""
        class_files = self.filterClass(im_shapes, cls_name)
        im_shapes = self.filterName(class_files, name)
        return im_shapes

    def filterClass(self,im_shapes, cls_name):
        """筛选当前数据集种类,
        Args:'all':所有种类, ‘ok’：已标注但没有标签的OK样本"""
        if not len(im_shapes):
            return {}
        if Path(list(im_shapes.keys())[0]).parent.name == EXPERIMENT_SETTINGS["name"]:
            return im_shapes
        if cls_name == "all":
            return im_shapes
        im_ss = {}
        if self.task == "classify":
            for file, shape in im_shapes.items():
                if Path(file).parent.name == cls_name:
                    im_ss.update({file: shape})
        else:
            names = list(self.train_set.data["names"].values())
            if cls_name not in names+["ok"]:
                return im_shapes
            label_files = img2label_paths(list(im_shapes.keys()))
            for label_file,(im_file, shape) in zip(label_files,im_shapes.items()):
                if Path(im_file).parent.name in ("no_label", EXPERIMENT_SETTINGS["name"]):
                    continue
                lb = readLabelFile(label_file)
                if len(lb):
                    for line in lb:
                        if line[0] == names.index(cls_name):
                            im_ss.update({im_file: shape})
                            break
                else:
                    if cls_name == "ok":
                        im_ss.update({im_file: shape})
        return im_ss

    def filterName(self,im_shapes, file_name):
        """筛选文件名称中包含file_name的文件"""
        if file_name == "":
            return im_shapes
        im_ss = {}
        for im_file, shape in im_shapes.items():
            if  file_name in Path(im_file).name.split(".")[0]:
                im_ss.update({im_file: shape})
        return im_ss


class SiftTool(QObject):
    def __init__(self,parent):
        """parent: SiftDataset"""
        super().__init__(parent)
        parent = self.parent().parent()   #Sift_f
        self.image_scroll = get_widget(parent, "image_scroll")
        self.images_label = get_widget(parent, "images_label")
        self.image_num_le = get_widget(parent, "Image_num_le")
        self.image_total_l = get_widget(parent, "Image_total_l")
        self.select_dataset_cbb = get_widget(parent, "Select_dataset_cbb")
        self.select_class_cbb = get_widget(parent, "Select_class_cbb")
        self.select_search_le = get_widget(parent, "Select_search_le")
        self.select_ops_cbb = get_widget(parent, "Select_ops_cbb")
        self.select_all_pb = get_widget(parent, "Select_all_pb")
        self.select_batch_ops_pb = get_widget(parent, "Select_batch_ops_pb")
        self.select_update_pb = get_widget(parent, "Select_update_pb")
        self.sift_image_dw = parent.parent().parent()  #sift_f -> dockWidgetContent-> dockWidget
        self.eventConnect()

    def eventConnect(self):
        self.select_dataset_cbb.currentTextChanged.connect(self.siftImageSignal)
        self.select_class_cbb.currentTextChanged.connect(self.siftImageSignal)
        self.select_search_le.textChanged.connect(self.siftImageSignal)
        self.select_batch_ops_pb.clicked.connect(self.batchOps)
        self.select_all_pb.clicked.connect(self.selectAll)
        self.select_dataset_cbb.installEventFilter(CbbFilter(self.parent()))
        self.select_class_cbb.installEventFilter(CbbFilter(self.parent()))
        self.select_ops_cbb.installEventFilter(CbbFilter(self.parent()))

    @debounce(500)
    @threaded
    def siftImageSignal(self, change_text=""):
        """筛选图像"""
        if self.select_class_cbb.currentText() == "":
            return
        items = self.parent().getItems()
        if change_text in items:
            self.updateOps()
        sift_dataset = self.parent()
        item_text = self.select_dataset_cbb.currentText()
        im_shapes = sift_dataset.getValue(item_text)
        self.images_label.dataset = item_text
        if not self.images_label.build:
            self.loadImages(im_shapes)
        im_shapes = sift_dataset.searchAll(im_shapes, self.select_class_cbb.currentText(), self.select_search_le.text())
        self.showImages(list(im_shapes.keys()))
        self.image_scroll.resizeEvent(None)

    def loadImages(self, im_shapes):
        """加载图像"""
        self.images_label.loadImages(im_shapes)
        self.sift_image_dw.show()
        self.sift_image_dw.raise_()

    def showImages(self, im_files):
        self.images_label.showImages(im_files)
        self.image_num_le.setText(str(1))
        self.image_total_l.setText(str(self.images_label.getShowLen()))
        self.sift_image_dw.show()
        self.sift_image_dw.raise_()

    def selectAll(self):
        self.images_label.selectAllShow()

    def batchOps(self):
        """批量处理图像"""
        im_files = self.images_label.getSelectedImgs()
        item = self.select_ops_cbb.currentText()
        if item == "删除":
            self.images_label.deleteImages(im_files)
        elif item == "转验证集":
            if self.select_dataset_cbb.currentText() == "训练集":
                self.images_label.train2Val(im_files)
            else:
                self.images_label.nolabel2Val(im_files)
        elif item == "转训练集":
            if self.select_dataset_cbb.currentText() == "验证集":
                self.images_label.val2Train(im_files)
            else:
                self.images_label.nolabel2Train(im_files)
        elif item == "转未标注集":
            self.images_label.toNolabel(im_files)

    def updateOps(self):
        """更新操作"""
        dataset = self.select_dataset_cbb.currentText()
        if dataset == "训练集":
            items = ["删除", "转验证集", "转未标注集"]
        elif dataset == "验证集":
            items = ["删除", "转训练集", "转未标注集"]
        elif dataset in ("未标注集"):
            items = ["删除", "转验证集", "转训练集"]
        else:
            items = []
        self.select_ops_cbb.clear()
        self.select_ops_cbb.addItems(items)
    
    def updateDataset(self):
        """更新数据集"""
        dataset_items = self.parent().getItems()
        items = [self.select_dataset_cbb.itemText(i) for i in range(self.select_dataset_cbb.count())]
        for item in dataset_items:
            if item not in items:
                self.select_dataset_cbb.addItem(item)
        for item in items:
            if item not in dataset_items:
                self.select_dataset_cbb.removeItem(items.index(item))

    def updateClass(self):
        """更新种类"""
        names = self.parent().getNames()
        class_items = ["all"] + list(names.values()) + ["ok"]
        items = [self.select_class_cbb.itemText(i) for i in range(self.select_class_cbb.count())]
        for item in class_items:
            if item not in items:
                self.select_class_cbb.addItem(item)
        for item in items:
            if item not in class_items:
                self.select_class_cbb.removeItem(items.index(item))

    def initSifter(self):
        """初始化筛选器"""
        #dataset
        self.updateDataset()

        #class
        self.updateClass()

        #ops 
        self.updateOps()

        #search
        self.select_search_le.setText("")


