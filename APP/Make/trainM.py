import copy
import os
import time

from PySide2.QtGui import *
from PySide2.QtWidgets import *
from APP.Designer.DesignerPy import trainUI

from pathlib import Path
import shutil
import torch
import numpy as np
import pandas as pd
import threading

from ultralytics.utils import yaml_save, DEFAULT_CFG_DICT, DEFAULT_CFG, LOGGER, yaml_load, PROGRESS_BAR


from APP import  PROJ_SETTINGS, getExperimentPath, getOpenFileName, EXPERIMENT_SETTINGS, APP_ROOT, loadQssStyleSheet
from APP.Make.startM import Start
from APP.Utils.base import QInstances, QTransformerLabel
from APP.Utils.label import (DetectTransformerLabel,
                             SegmentTransformerLabel,
                             KeypointsTransformerLabel,
                             ClassifyTransformerLabel,
                             ObbTransformerLabel,
                             ConfusionMatrixLabel,
                             ShowLabel)
from APP.Ops.cfgs import CfgsTreeWidget
from APP.Ops.classes_show import ClassesView
from APP.Utils.scroll import ImageScroll
from APP.Utils.plotting import PgPlotLossWidget
from APP.Ops import SiftDataset, LabelOps, MenuTool, RunMes
from APP.Utils.ultralytics_enginer import Yolo



class Train(QMainWindow, trainUI.Ui_MainWindow):
    def __init__(self, app):
        super(Train, self).__init__()
        self.setupUi(self)
        self.desktop = QApplication.desktop()
        self.resize(self.desktop.screenGeometry().width()*0.7, self.desktop.screenGeometry().height()*0.8)
        self.setStatusBar(QStatusBar())
        self.app = app

        self.cfg_path = ""
        self.pred_labels = {}
        self.img_label = None
        self.image_scroll=None
        self.label_ops = LabelOps(self)
        self.classes_view = ClassesView(self.Classes_lv)
        self.setUI()
        self.setPlot()
        self.sift_dataset = SiftDataset(self.Sift_f)
        self.eventConnect()
        self.start_w.show()


    def setUI(self):
        self.setLabel()
        self.setCfgsTree()
        self.menu_tool = MenuTool(self)
        self.start_w = Start(self)
        self.run = RunMes(self)
        self.Select_image_spliter.setStretchFactor(0,1)
        self.Select_image_spliter.setStretchFactor(1,0)
        self.tabifyDockWidget(self.Progress_dw, self.Sift_image_dw)
        self.Progress_dw.raise_()


    def setCfgsTree(self):
        self.cfgs_gl = QGridLayout(self.Args_dwc)
        self.cfgs_gl.setObjectName(u"cfgs_gl")
        self.cfgs_gl.setMargin(0)
        self.cfgs_widget = CfgsTreeWidget(self)
        self.cfgs_widget.setObjectName(u"cfgs_widget")
        self.cfgs_gl.addWidget(self.cfgs_widget, 0, 0, 1, 1)

    def setLabel(self):
        self.images_label = ShowLabel(self, self.label_ops)
        self.images_label.setObjectName(u"images_label")

        self.image_select_gl = QGridLayout(self.Image_select_f)
        self.image_select_gl.setObjectName(u"image_select_gl")
        self.image_select_gl.setMargin(0)
        self.image_scroll = ImageScroll(self.Image_select_f, self.images_label)
        self.image_scroll.setObjectName(u"image_scroll")
        self.image_select_gl.addWidget(self.image_scroll, 0, 0, 1, 1)

        self.show_img_gl = QGridLayout(self.Source_show_f)
        self.show_img_gl.setObjectName(u"show_img_gl")
        self.setImgLabel(QTransformerLabel)

        self.show_confusion_norm_gl = QGridLayout(self.Confusion_norm_w)
        self.show_confusion_norm_gl.setObjectName(u"show_confusion_norm_gl")
        self.confusion_norm_label = ConfusionMatrixLabel(self.Confusion_norm_w)
        self.confusion_norm_label.setObjectName(u"confusion_norm_label")
        #self.confusion_norm_label.setStyleSheet(u"background-color: rgb(249, 255, 253);")
        self.show_confusion_norm_gl.addWidget(self.confusion_norm_label, 0, 0, 1, 1)

        self.show_confusion_denorm_gl = QGridLayout(self.Confusion_denorm_w)
        self.show_confusion_denorm_gl.setObjectName(u"show_confusion_denorm_gl")
        self.confusion_denorm_label = ConfusionMatrixLabel(self.Confusion_denorm_w)
        self.confusion_denorm_label.setObjectName(u"confusion_denorm_label")
        #self.confusion_denorm_label.setStyleSheet(u"background-color: rgb(253, 255, 244);")
        self.show_confusion_denorm_gl.addWidget(self.confusion_denorm_label, 0, 0, 1, 1)





    def setImgLabel(self, transformerLabel):
        if self.img_label:
            self.img_label.deleteLater()
        self.img_label = transformerLabel(self.Source_show_f)
        self.img_label.setObjectName(u"image_label")
        self.img_label.Show_Status_Signal.connect(self.showStatusMessage)
        self.show_img_gl.addWidget(self.img_label, 0, 0, 1, 1)
        self.show_img_gl.setMargin(2)
        self.classes_view.setImgLabel(self.img_label)

        if self.label_ops:
            self.label_ops.updateImgLabel(self.img_label)
        if self.image_scroll:
            self.img_label.Next_Image_Signal.connect(self.image_scroll.nextImage)
            self.img_label.Last_Image_Signal.connect(self.image_scroll.lastImage)

    def setPlot(self):
        self.loss_plot = PgPlotLossWidget(self.Show_loss_f, background=QColor(255, 255, 243, 0))
        self.show_loss_gl = QGridLayout(self.Show_loss_f)
        self.show_loss_gl.setObjectName(u"show_loss_gl")
        self.show_loss_gl.setContentsMargins(0,0,5,0)
        self.show_loss_gl.addWidget(self.loss_plot, 0, 0)


    def eventConnect(self):
        self.start_w.Start_Signal.connect(self.openExperiment)
        self.cfgs_widget.Task_Change_Signal.connect(self.changeTaskSlot)
        

        self.Train_a.triggered.connect(self.startTrain)
        self.Val_a.triggered.connect(self.startVal)
        self.Predict_a.triggered.connect(self.startPredict)
        self.Export_a.triggered.connect(self.startExport)
        self.confusion_norm_label.Select_signal.connect(self.ConfusionImagesSlot)
        self.confusion_denorm_label.Select_signal.connect(self.ConfusionImagesSlot)

        self.images_label.Click_Signal.connect(self.selectImageSlot)




    def startTrain(self):
        self.Progress_dw.raise_()
        if self.Train_a.text() == "训练":
            LOGGER.stop = False
            self.cfgs_widget.save()
            model = Yolo(self.cfgs_widget.args["model"], self.cfgs_widget.args["task"])
            results = Path(getExperimentPath()) / "results.csv"
            if results.exists():
                data = pd.read_csv(results)
                x = data.values[:, 0]
                if model.ckpt is None or (model.ckpt is not None and model.ckpt["epoch"] != int(x[-1])-1):
                    req = QMessageBox.warning(self, "训练提示", "实验已存在训练结果，是否重新进行训练", QMessageBox.Yes, QMessageBox.No)
                    if req == QMessageBox.Yes:
                        os.remove(results)
                    else:
                        return

            DEFAULT_CFG.save_dir = getExperimentPath()
            if isinstance(self.cfgs_widget.args["pretrained"], str):
                model.load(self.cfgs_widget.args["pretrained"])
            self.cfgs_widget.args["exist_ok"] = False  #覆盖当前实验
            model.lyTrain(cfg=self.cfg_path, data=self.cfgs_widget.args["data"])
            self.Train_a.setText("停止")
            self.Train_a.setEnabled(False)
        else:
            LOGGER.stop = True
            self.Train_a.setText("训练")
            self.Train_a.setEnabled(False)


    def startVal(self):
        self.Progress_dw.raise_()
        self.cfgs_widget.save()
        yolo = Yolo(self.cfgs_widget.args["model"], self.cfgs_widget.args["task"])
        args = copy.deepcopy(self.cfgs_widget.args)
        args["save_dir"] = getExperimentPath()
        yolo.lyVal( **args)

    def startPredict(self):
        self.cfgs_widget.save()
        if self.cfgs_widget.args["source"] == "选中图像":
            source = self.images_label.getSelectedImgs()
        else:
            source = self.cfgs_widget.args["source"]
        yolo = Yolo(self.cfgs_widget.args["model"], self.cfgs_widget.args["task"])
        yolo.overrides = {**self.cfgs_widget.args, **yolo.overrides}
        self.pred_labels = yolo.lyPredict(source=source, stream=True, save_dir=getExperimentPath(),conf=self.cfgs_widget.args["conf"])
        if self.pred_labels is None:
            self.pred_labels = {}
        if self.img_label.im_file in self.pred_labels.keys():
            self.img_label.loadPredLabel(self.pred_labels[self.img_label.im_file])
    
    def startExport(self):
        yolo = Yolo(self.cfgs_widget.args["model"], self.cfgs_widget.args["task"])
        yolo.overrides = {**self.cfgs_widget.args, **yolo.overrides}
        yolo.lyExport()


    #SLOT 外部槽
    def changeTaskSlot(self, task):
        if task == "detect":
            transformerLabel = DetectTransformerLabel
        elif task == "segment":
            transformerLabel = SegmentTransformerLabel
        elif task == "obb":
            transformerLabel = ObbTransformerLabel
        elif task == "pose":
            transformerLabel = KeypointsTransformerLabel
        elif task == "classify":
            transformerLabel = ClassifyTransformerLabel
        else:
            transformerLabel = QTransformerLabel
        self.show_img_gl.removeWidget(self.img_label)
        self.setImgLabel(transformerLabel)
        self.label_ops.updateTask(task)


    def ConfusionImagesSlot(self, key):
        """往scoll加载图像集"""
        self.Select_dataset_cbb.setCurrentText(key)
        self.Select_class_cbb.setCurrentText("all")
        self.Select_search_le.setText("")

    def selectImageSlot(self, label):
        """选中图像槽"""
        im_file = label["im_file"]
        self.img_label.load_image(im_file, label)
        if im_file in self.pred_labels.keys():
            self.img_label.loadPredLabel(self.pred_labels[im_file])
        else:
            self.img_label.pred_label = None
        self.Image_num_le.setText(str(self.images_label.show_files.index(im_file) + 1))
        self.label_ops.painter_tool.setTrainVal()

    def showStatusMessage(self, str):
        self.statusBar().showMessage(str)

    def buildDataset(self, data):
        self.sift_dataset.build(data, self.cfgs_widget.args["task"], self.cfgs_widget.args)
        self.cfgs_widget.setValue("data", self.sift_dataset.data)
        self.cfgs_widget.save()
        self.label_ops.updateDataset(self.sift_dataset.train_set,
                                        self.sift_dataset.val_set,
                                        self.sift_dataset.train_path,
                                        self.sift_dataset.val_path)
        self.sift_dataset.initLoadDataset()
        self.sift_dataset.sift_tool.initSifter()  #初始化筛选器
        self.image_scroll.horBarValueChangedSlot()


    def openExperiment(self, experiment):
        experiment_path = Path(getExperimentPath(experiment))
        cfg_path = experiment_path / "cfgs" / "cfg.yaml"
        #检查实验是否存在,不存在则创建新实验
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True,exist_ok=True)
        if not (experiment_path / "cfgs"/ "cfg.yaml").exists():
            (experiment_path / "cfgs").mkdir(parents=True, exist_ok=True)
            if self.cfg_path != "": #复制前一个实验参数
                shutil.copy(self.cfg_path, experiment_path / "cfgs")
            else:  #新建默认参数
                cfg = copy.deepcopy(DEFAULT_CFG_DICT)
                yaml_save(cfg_path, DEFAULT_CFG_DICT)


        EXPERIMENT_SETTINGS.load(experiment)

        self.cfg_path = cfg_path
        self.cfgs_widget.updateTrees(self.cfg_path)
        self.changeTaskSlot(self.cfgs_widget.args["task"])
        self.setWindowTitle(str(experiment_path))
        self.buildDataset(self.cfgs_widget.args["data"])
        self.run.updateConfusion()
        self.run.updateLoss()

    
    def keyPressEvent(self, event):
        self.img_label.keyPressEvent(event)


    def closeEvent(self, event: QCloseEvent):
        if Path(getExperimentPath()).parent.name == "expcache":
            ans = QMessageBox.information(self, "关闭提示", "实验未保存，请问是否保存实验", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Yes:
                name, ok = QInputDialog.getText(self, "保存实验", "实验名称：",text=Path(getExperimentPath()).name)
                if ok and name != "":
                    shutil.copytree(getExperimentPath(), getExperimentPath(name))
                    shutil.rmtree(getExperimentPath())
                    PROJ_SETTINGS.update({"current_experiment":name})
                elif ok and name == "":
                    QMessageBox.information(self, "提示", "实验名称不能为空")
                    event.ignore()
                else:
                    event.ignore()
            elif ans == QMessageBox.Cancel:
                event.ignore()
            else:
                shutil.rmtree(getExperimentPath())
                event.accept()





