import glob
from pathlib import Path

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from APP.Design import coco2yoloQT_ui, voc2yoloQT_ui, png2yoloQT_ui

from APP  import  PROJ_SETTINGS
from APP.Utils import getExistDirectory
from APP.Data.convert import COCO2YOLO, VOC2YOLO, PNG2YOLO


class CocoToYolo(QWidget,coco2yoloQT_ui.Ui_Form):
    def __init__(self, parent, f=Qt.Dialog):
        super().__init__(parent, f)
        self.setupUi(self)
        self.coco_yolo = COCO2YOLO()
        self.Train_img_dn_le.setText("train2017")
        self.Val_img_dn_le.setText("val2017")
        self.Annotations_dn_le.setText("annotations")
        self.Annotation_type_cbb.setCurrentText("instances")
        self.Annotation_suffix_le.setText(".json")
        self.Task_cbb.setCurrentText("detect")
        self.eventConnect()

    def eventConnect(self):
        self.Convert_pb.clicked.connect(self.convertPBClicked)
        self.Browse_coco_pb.clicked.connect(self.browseCocoPath)
        self.Browse_yolo_pb.clicked.connect(self.browseYoloPath)


    def convertPBClicked(self):
        try:
            coco_path = self.COCO_path_le.text()
            yolo_path = self.YOLO_path_le.text()
            train_img_dn = self.Train_img_dn_le.text()
            val_img_dn = self.Val_img_dn_le.text()
            annotation_dn = self.Annotation_dn_pb.text()
            annotation_type = self.Annotation_type_cbb.currentText()
            suffix = self.Annotation_suffix_le.text()
            task = self.Task_cbb.currentText()
            self.coco_yolo(coco_path, yolo_path, train_img_dn, val_img_dn, annotation_dn, annotation_type, task, suffix)
        except Exception as e:
            QMessageBox.warning(self, "报错", str(e))


    def browse(self):
        dir = getExistDirectory(self, "数据集路径")
        return dir

    def browseCocoPath(self):
        dir = self.browse()
        if dir != "":
            self.Coco_path_le.setText(dir)

    def browseYoloPath(self):
        dir = self.browse()
        if dir != "":
            fs = glob.glob(str(Path(dir) / "*.*"))
            if len(fs):
                QMessageBox.warning(self, "警告", "yolo数据集路径应该是一个空文件夹")
                return
            self.YOLO_path_le.setText(dir)


class VocToYolo(QWidget, voc2yoloQT_ui.Ui_Form):
    def __init__(self, parent, f=Qt.Dialog):
        super().__init__(parent, f)
        self.setupUi(self)
        self.voc_yolo = VOC2YOLO()
        self.Img_dn_le.setText("JPEGImages")
        self.Annotation_dn_le.setText("Annotations")
        self.Annotation_suffix_le.setText(".xml")
        self.Sets_name_le.setText("ImageSets//Main")
        self.eventConnect()

    def eventConnect(self):
        self.Convert_pb.clicked.connect(self.convertPBClicked)
        self.Browse_voc_pb.clicked.connect(self.browseVocPath)
        self.Browse_yolo_pb.clicked.connect(self.browseYoloPath)

    def convertPBClicked(self):
        try:
            voc_path = self.VOC_path_le.text()
            yolo_path = self.YOLO_path_le.text()
            img_dn = self.Img_dn_le.text()
            annotaion_dn = self.Annotation_dn_le.text()
            suffix = self.Annotation_suffix_le.text()
            sets_name = self.Sets_name_le.text()
            self.voc_yolo(voc_path, yolo_path, img_dn, annotaion_dn, suffix, sets_name)
        except Exception as e:
            QMessageBox.warning(self, "报错",str(e))

    def browse(self):
        dir = getExistDirectory(self, "数据集路径")
        return dir

    def browseVocPath(self):
        dir = self.browse()
        if dir != "":
            self.VOC_path_le.setText(dir)

    def browseYoloPath(self):
        dir = self.browse()
        if dir != "":
            fs = glob.glob(str(Path(dir) / "*.*"))
            if len(fs):
                QMessageBox.warning(self,"警告", "yolo数据集路径应该是一个空文件夹")
                return
            self.YOLO_path_le.setText(dir)


class PngToYolo(QWidget, png2yoloQT_ui.Ui_Form):
    def __init__(self, parent, f=Qt.Dialog):
        super().__init__(parent, f)
        self.setupUi(self)
        self.pngToYolo = PNG2YOLO()
        self.eventConnect()
        self.Yolo_p_le.setText(PROJ_SETTINGS["name"] + "//data")

    def eventConnect(self):
        self.Seg_train_p_browse_pb.clicked.connect(self.browseSegTrainPath)
        self.Seg_val_p_browse_pb.clicked.connect(self.browseSegValPath)
        self.Ori_train_p_browse_pb.clicked.connect(self.browseOriTrainPath)
        self.Ori_val_p_browse_pb.clicked.connect(self.browseOriValPath)
        self.Yolo_p_browse_pb.clicked.connect(self.browseYoloPath)
        self.Convert_pb.clicked.connect(self.convert)

    def browseSegTrainPath(self):
        dir = getExistDirectory(self, "分割训练集路径")
        if dir != "":
            self.Seg_train_p_le.setText(dir)

    def browseSegValPath(self):
        dir = getExistDirectory(self, "分割验证集路径")
        if dir != "":
            self.Seg_val_p_le.setText(dir)

    def browseOriTrainPath(self):
        dir = getExistDirectory(self,"原图像训练集路径")
        if dir != "":
            self.Ori_train_p_le.setText(dir)

    def browseOriValPath(self):
        dir = getExistDirectory(self, "原图像验证集路径")
        if dir != "":
            self.Ori_val_p_le.setText(dir)

    def browseYoloPath(self):
        dir = getExistDirectory(self, "Yolo保存路径")
        if dir != "":
            self.Yolo_p_le.setText(dir)

    def convert(self):
        seg_suffix = self.Seg_suffix_cbb.currentText()
        seg_tp = self.Seg_train_p_le.text()
        seg_vp = self.Seg_val_p_le.text()
        ori_suffix = self.Ori_suffix_cbb.currentText()
        ori_tp = self.Ori_train_p_le.text()
        ori_vp = self.Ori_val_p_le.text()
        yolo_p = self.Yolo_p_le.text()
        self.pngToYolo(seg_tp, seg_vp, seg_suffix, ori_tp, ori_vp, ori_suffix, yolo_p)

