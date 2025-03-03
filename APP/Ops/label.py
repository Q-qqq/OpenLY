import copy

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


import os
from pathlib import Path
import numpy as np

from ultralytics.data.utils import img2label_paths,IMG_FORMATS
from ultralytics.utils import yaml_load, ThreadingLocked
from ultralytics.data.augment import classify_transforms



from APP  import PROJ_SETTINGS
from APP.Data import readLabelFile, format_im_files, getNoLabelPath
from APP.Utils.base import QInstances
from APP.Utils import get_widget
from APP.Make.levelsaugmentM import LevelsAugment
from APP.Make.fastselectM import FastSelect
from APP.Make.pencilsetM import PencilSet



class LabelOps(QObject):
    """对数据集进行添加删除修改移动等操作"""
    def __init__(self,parent):
        """parent: trainM"""
        super().__init__(parent)
        self.img_label = None
        self.painter_tool = PainterTool(self)
        self.train_set = None
        self.val_set = None
        self.train_path = ""
        self.val_path = ""
        self.task = ""

    def updateDataset(self, train_set, val_set, train_path, val_path):
        self.train_set = train_set
        self.val_set = val_set
        self.train_path = train_path
        self.val_path = val_path

    def updateTask(self, task):
        self.task = task

    def updateImgLabel(self, img_label):
        """更新Img Label绘图widget"""
        self.img_label = img_label
        self.img_label.Change_Label_Signal.connect(self.save)
        self.painter_tool.setInitColor()

    def updateImageSize(self, img_sz):
        """更新输入网络的图像大小"""
        self.img_sz = img_sz
        if self.task == "classify":
            self.train_set.torch_transforms = classify_transforms(size=img_sz, crop_fraction=1)
            self.val_set.torch_transforms = classify_transforms(size=img_sz, crop_fraction=1)
        else:
            self.train_set.img_size = img_sz
            self.val_set.img_size = img_sz

    def getLabel(self, im_file=None, no_label_none=False):
        """获取图像对应的标签，如果no_label_none为True,当图像文件没有对应的标签时，返回None，否则返回一个空白label"""
        if not im_file:
            im_file = self.img_label.im_file
        if str(Path(im_file)) in self.train_set.im_files:
            label = self.train_set.getLabel(im_file)
        elif str(Path(im_file)) in self.val_set.im_files:
            label = self.val_set.getLabel(im_file)
        else:
            if no_label_none:
                label = None
            else:
                if self.task == "classify":
                    label = {"im_file": im_file, "cls": -1, "names": self.train_set.names, "ori_shape":[0,0]}
                elif self.task == "detect":
                    instance = QInstances(bboxes=np.array([], dtype=np.float32), bbox_format="xywh", normalized=False)
                    label = {"im_file": im_file, "cls": [], "names": self.train_set.data["names"],
                             "instances": instance, "ori_shape":[0,0]}
                elif self.task in ("segment", "obb"):
                    instance = QInstances(segments=[], normalized=False)
                    label = {"im_file": im_file, "cls": [], "names": self.train_set.data["names"],
                             "instances": instance, "ori_shape":[0,0]}
                elif self.task == "pose":
                    instance = QInstances(keypoints=np.array([], dtype=np.float32), normalized=False)
                    label = {"im_file": im_file, "cls": [], "names": self.train_set.data["names"],
                             "instances": instance, "ori_shape":[0,0]}
                else:
                    label = None
        if label:
            label["dataset"] = self.judgeDataset(label["im_file"])
        return label

    def judgeDataset(self, im_file):
        if im_file in self.train_set.im_files:
            dataset = "train"
        elif im_file in self.val_set.im_files:
            dataset = "val"
        elif Path(im_file).parent.name == "no_label":
            dataset = "no_label"
        else:
            dataset = "results"
        return dataset

    def getNames(self):
        """获取种类
        return(dict): {num:class}"""
        if self.task == "classify":
            names = self.train_set.names
        else:
            names = self.train_set.data["names"]
        return names



    def getTrainVal(self):
        """获取img_label当前图像的所属数据集：训练集、验证集、未标注集、结果集"""
        if not self.img_label:
            return ""
        train_val = "train" if self.img_label.im_file in self.train_set.im_files else None
        if not train_val:
            train_val = "val" if self.img_label.im_file in self.val_set.im_files else None
        if not train_val:
            train_val = "no_label" if Path(self.img_label.im_file).parent.name == "no_label" else "results"
        return train_val

    @ThreadingLocked()
    def save(self):
        """保存标签"""
        if not self.img_label:
            return
        im_file = self.img_label.im_file
        pix_img = self.img_label.pix
        label = self.img_label.label
        instance = copy.deepcopy(label.get("instances"))
        cls = label.get("cls")
        if self.painter_tool.results_rb.isChecked():  #结果集
            return

        if label:
            if self.painter_tool.train_rb.isChecked():
                set = self.train_set
                root = self.train_path
            elif self.painter_tool.val_rb.isChecked():
                set = self.val_set
                root = self.val_path
            else:
                return
            if Path(im_file).parent.name == "no_label":  #未标注图像
                if self.task == "classify":
                    new_im_files = set.addData(im_file, cls)
                else:
                    new_im_files = set.addData(im_file, root)
                    new_label_files = img2label_paths(new_im_files)
                    instance.save(new_label_files[0], cls, pix_img.width(), pix_img.height())
                self.img_label.im_file = new_im_files[0]
            else:    #已标注
                if self.task == "classify":
                    new_im_files = set.changeData(im_file, cls)  #转移至对应种类的文件夹
                    self.img_label.im_file = new_im_files[0]
                else:
                    label_file = img2label_paths(im_file)[0]
                    instance.save(label_file, cls, pix_img.width(), pix_img.height())
                    set.changeData(im_file)

    def deleteSamples(self, im_files=None, no_label_path=""):
        """删除样本，当no_label_path不为空时，将样本移动至未标注集"""
        if not im_files:
            im_files = self.img_label.im_file
        im_files = format_im_files(im_files)
        new_im_file = copy.deepcopy(im_files)
        for i, im_file in enumerate(im_files):
            if im_file in self.train_set.im_files:
                self.train_set.removeData(im_file, no_label_path)
                if no_label_path:
                    new_im_file[i] = Path(no_label_path) / Path(im_file).name
            elif im_file in self.val_set.im_files:
                self.val_set.removeData(im_file, no_label_path)
                if no_label_path:
                    new_im_file[i] = Path(no_label_path) / Path(im_file).name
            else:
                if Path(im_file).exists() and not no_label_path: #图像存在且no_label_path为空
                    os.remove(im_file)
            if im_file == self.img_label.im_file:
                self.img_label.init()
                self.painter_tool.setTrainVal()
        if no_label_path:
            return new_im_file
        else:
            return []

    def toNoLabel(self, im_files=None):
        no_label_path = getNoLabelPath()
        self.painter_tool.setTrainVal("no_label")
        return self.deleteSamples(im_files, no_label_path)

    def train2Val(self, im_files=None):
        if not im_files:
            im_files = self.img_label.im_file
        im_files = format_im_files(im_files)
        new_im_files = copy.deepcopy(im_files)
        for i, im_file in enumerate(im_files):
            if im_file not in self.train_set.im_files: #筛除验证集、未标注集和结果集
                continue
            if self.task == "classify":
                label = self.getLabel(im_file)
                cls = label["cls"]
                new_im_file = self.val_set.addData(im_file, cls)[0]
                self.train_set.removeData(im_file)
            else:
                new_im_file = self.val_set.addData(im_file, self.val_path)[0]
                self.train_set.removeData(im_file)
            new_im_files[i] = new_im_file
            if im_file == self.img_label.im_file:
                self.img_label.im_file = new_im_file
                self.painter_tool.setTrainVal("val")
        return new_im_files

    def val2Train(self, im_files=None):
        if not im_files:
            im_files = self.img_label.im_file
        im_files = format_im_files(im_files)
        new_im_files = copy.deepcopy(im_files)
        for i, im_file in enumerate(im_files):
            if im_file not in self.val_set.im_files:  #确保图像属于验证集
                continue
            if self.task == "classify":
                label = self.getLabel(im_file)
                new_im_file = self.train_set.addData(im_file, label["cls"])[0]
                self.val_set.removeData(im_file)
            else:
                new_im_file = self.train_set.addData(im_file, self.train_path)[0]
                self.val_set.removeData(im_file)
            new_im_files[i] = new_im_file
            if im_file == self.img_label.im_file:
                self.img_label.im_file = new_im_file
                self.painter_tool.setTrainVal("train")
        return new_im_files

    def addClass(self, cls_name):
        """添加种类cls_name
        Args:
            cls_name(str):新种类名称"""
        self.train_set.addClass(cls_name) #训练集和验证集同一个data

    def deleteClass(self, cls_name):
        """删除种类cls_name
        Args:
            cls_name(int | str): 种类索引或者种类名称"""
        no_label_path = Path(PROJ_SETTINGS["name"]) / "data" / "no_label"
        names = self.img_label.label["names"]
        cls = list(names.values()).index(cls_name) if isinstance(cls_name, str) else cls_name
        self.train_set.deleteClass(cls, no_label_path)
        self.val_set.deleteClass(cls, no_label_path)
        self.img_label.label["names"] = self.train_set.names if self.task == "classify" else self.train_set.data["names"]
        if self.img_label.cls >= len(self.img_label.label["names"]):
            self.img_label.cls = len(self.img_label.label["names"])-1

    def renameClass(self, cls, cls_name):
        if isinstance(cls, str):
            cls = list(self.train_set.data["names"].values()).index(cls) if self.task != "classify" else list(self.train_set.names.values()).index(cls)
        self.train_set.renameClass(cls, cls_name)

    def showNone(self):
        """将img_label的显示置为空白"""
        self.img_label.init()


class PainterTool(QObject):
    """绘制工具的设置"""
    def __init__(self, parent):
        """parent: LabelOps"""
        super().__init__(parent)
        self.label_ops = self.parent()
        self.show_pred_pb = get_widget(self.parent().parent(), "Tool_show_pred_pb")
        self.show_ture_pb = get_widget(self.parent().parent(), "Tool_show_true_pb")
        self.levels_augment_pb = get_widget(self.parent().parent(), "Tool_levels_augment_pb")
        self.train_rb = get_widget(self.parent().parent(), "Tool_train_rb")
        self.val_rb = get_widget(self.parent().parent(), "Tool_val_rb")
        self.no_label_rb = get_widget(self.parent().parent(), "Tool_no_label_rb")
        self.results_rb = get_widget(self.parent().parent(), "Tool_results_rb")
        self.pen_pb = get_widget(self.parent().parent(), "Tool_pen_pb")
        self.pencil_pb = get_widget(self.parent().parent(), "Tool_pencil_pb")
        self.fast_select_pb = get_widget(self.parent().parent(), "Tool_fast_select_pb")
        self.paint_pb = get_widget(self.parent().parent(), "Tool_paint_pb")
        self.white_color = u"background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 #C8A2C8, stop:1 #B19CD9);"
        self.blue_color = u"background-color: rgb(20, 46, 214);"
        self.levels_augment = LevelsAugment(self.parent().parent(), self.label_ops.img_label)
        self.fast_select = FastSelect(self.parent().parent(), self.label_ops.img_label)
        self.pencil_set = PencilSet(self.parent().parent(), self.label_ops.img_label)
        self.pen_pb.setStyleSheet(self.blue_color)
        self.eventConnect()

    def setInitColor(self):
        if self.label_ops.img_label:
            self.show_pred_pb.setStyleSheet(self.blue_color if self.label_ops.img_label.show_pred else self.white_color)
            self.show_ture_pb.setStyleSheet(self.blue_color if self.label_ops.img_label.show_true else self.white_color)

    def eventConnect(self):
        self.show_pred_pb.clicked.connect(self.showPredClicked)
        self.show_ture_pb.clicked.connect(self.showTrueClicked)
        self.levels_augment_pb.clicked.connect(self.levelsAugmentClicked)
        self.fast_select_pb.clicked.connect(self.fastSelectClicked)
        self.pen_pb.clicked.connect(self.penClicked)
        self.pencil_pb.clicked.connect(self.pencilClicked)
        self.paint_pb.clicked.connect(self.paintClicked)

    def paintClicked(self):
        if not self.label_ops.img_label.paint:
            self.label_ops.img_label.paint = True
            self.paint_pb.setStyleSheet(self.blue_color)
        else:
            self.label_ops.img_label.paint = False
            self.paint_pb.setStyleSheet(self.white_color)


    def showPredClicked(self):
        if self.label_ops.img_label:
            if self.label_ops.img_label.show_pred:
                self.label_ops.img_label.show_pred = False
            else:
                self.label_ops.img_label.show_pred = True
            self.show_pred_pb.setStyleSheet(self.blue_color if self.label_ops.img_label.show_pred else self.white_color)
            self.label_ops.img_label.update()

    def showTrueClicked(self):
        if self.label_ops.img_label:
            if self.label_ops.img_label.show_true:
                self.label_ops.img_label.show_true = False
            else:
                self.label_ops.img_label.show_true = True
            self.show_ture_pb.setStyleSheet(self.blue_color if self.label_ops.img_label.show_true else self.white_color)
            self.label_ops.img_label.update()

    def levelsAugmentClicked(self):
        if self.label_ops.img_label.im_file:
            self.levels_augment.img_label = self.label_ops.img_label
            self.levels_augment.show()

    def fastSelectClicked(self):
        if self.label_ops.img_label.im_file:
            self.fast_select.img_label = self.label_ops.img_label
            self.fast_select.show()

    def pencilClicked(self):
        if not  self.label_ops.img_label.use_pencil:
            self.label_ops.img_label.use_pencil = True
            self.label_ops.img_label.use_pen = False
            self.label_ops.img_label.openPencil()
            self.pencil_pb.setStyleSheet(self.blue_color)
            self.pen_pb.setStyleSheet(self.white_color)
        self.pencil_set.img_label = self.label_ops.img_label
        self.pencil_set.show()

    def penClicked(self):
        if not  self.label_ops.img_label.use_pen:
            self.label_ops.img_label.use_pencil = False
            self.label_ops.img_label.use_pen = True
            self.label_ops.img_label.openPen()
            self.pencil_pb.setStyleSheet(self.white_color)
            self.pen_pb.setStyleSheet(self.blue_color)
        self.pencil_set.close()

    def setTrainVal(self, dataset=""):
        if dataset == "":
            if self.label_ops.img_label.label is None:
                dataset = "no_label"
            else:
                dataset = self.label_ops.img_label.label["dataset"]
        train = dataset == "train"
        val = dataset == "val"
        no_label = dataset == "no_label"
        results = dataset == "results"
        self.train_rb.setChecked(train)
        self.val_rb.setChecked(val)
        self.no_label_rb.setChecked(no_label)
        self.results_rb.setChecked(results)
        self.train_rb.setEnabled(no_label)
        self.val_rb.setEnabled(no_label)
        self.label_ops.img_label.update()