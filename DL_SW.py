import sys

import torch
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from APP.Make import makenet,trainM,start_project,addmodel,k_means,connx,Gdd,Mylabel
import os
import shutil
from ultralytics.train import Thread
import math
import numpy as np
import matplotlib
from ultralytics.others import public_method
from ultralytics.others.util import *
matplotlib.use("QT5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

__version__ = "DL_SW:2.0.0"

#region 训练窗口
class MainWindow_train(QMainWindow,train.Ui_MainWindow):
    def __init__(self):
        super(MainWindow_train,self).__init__()
        self.setupUi(self)

    #region 保存所有参数
    def save_project(self):
        try:
            # 保存训练信息
            items = ['', '']
            for i in range(2):
                for j in range(self.class_numSB.value()):
                    if j == self.class_numSB.value() - 1:
                        items[i] += self.sort_TW.item(j, i).text()
                    else:
                        items[i] += self.sort_TW.item(j, i).text() + ","

            with open(self.project_path + "\\config\\learning_config.config", "w") as f:
                f.write("model name=" + self.model_nameCBB.currentText() + "\n" +
                        "net name=" + self.net_nameCBB.currentText() + "\n" +
                        "batch size=" + str(self.batch_sizeSB.value()) + "\n" +
                        "learn rate init=" + str(self.learn_rate_initDSB.value()) + "\n" +
                        "learn rate final=" + str(self.learn_rate_finalDSB.value()) + "\n" +
                        "epoch num=" + str(self.epoch_numSB.value()) + "\n" +
                        "class num=" + str(self.class_numSB.value()) + "\n" +
                        "image size=" + str(self.img_sizeSB.value()) + "\n" +
                        "giou loss weight=" + str(self.giou_lossDSB.value()) + "\n" +
                        "obj loss weight=" + str(self.obj_lossDSB.value()) + "\n" +
                        "cls loss weight=" + str(self.cls_lossDSB.value()) + "\n" +
                        "gr=" + str(self.gr_DSB.value()) + "\n" +
                        "anchors thres=" + str(self.anchor_tDSB.value()) + "\n" +
                        "val conf thres=" + str(self.val_conf_thres_DSB.value()) + "\n" +
                        "val iou thres=" + str(self.val_iou_thres_DSB.value()) + "\n" +
                        "test conf thres=" + str(self.test_conf_thres_DSB.value()) + "\n" +
                        "test iou thres=" + str(self.test_iou_thres_DSB.value()) + "\n" +
                        "device=" + str(self.device_CBB.currentIndex()) + "\n" +
                        "learning mode=" + str(self.learning_mode_CBB.currentIndex()) + "\n" +
                        "sort index=" + items[0] + "\n" +
                        "sort name=" + items[1] + "\n" +
                        "image type=" + str(self.image_type_CBB.currentIndex()) + "\n" +
                        "cache img=" + str(self.cache_img_cb.isChecked()) + "\n" +
                        "updata cache label=" + str(self.updata_cache_label_cb.isChecked()) + "\n" +
                        "extract bounding boxes=" + str(self.extract_bounding_boxes_cb.isChecked()) + "\n" +
                        "single cls=" + str(self.single_cls_cb.isChecked()) + "\n" +
                        "rect=" + str(self.rect_cb.isChecked()) + "\n" +
                        "rect size=" + str(self.rect_size_SB.value()) + "\n" +
                        "augment=" + str(self.augment_cb.isChecked()) + "\n" +
                        "border=" + str(self.border_cb.isChecked()) + "\n" +
                        "augment hsv=" + str(self.augment_hsv_cb.isChecked()) + "\n" +
                        "lr flip=" + str(self.lr_flip_cb.isChecked()) + "\n" +
                        "ud flip=" + str(self.ud_flip_cb.isChecked()) + "\n" +
                        "degrees=" + str(self.degrees_DSB.value()) + "\n" +
                        "translate=" + str(self.translate_DSB.value()) + "\n" +
                        "scale=" + str(self.scale_DSB.value()) + "\n" +
                        "shear=" + str(self.shear_DSB.value()) + "\n" +
                        "hsv h=" + str(self.hsv_h_DSB.value()) + "\n" +
                        "hsv s=" + str(self.hsv_s_DSB.value()) + "\n" +
                        "hsv v=" + str(self.hsv_v_DSB.value()) + "\n" +
                        "iou=" + str(self.iou_cbb.currentIndex()) + "\n" +
                        "auto anchors=" + str(self.auto_anchor_cb.isChecked()) + "\n" +
                        "auto batch size=" + str(self.auto_batch_size_cb.isChecked()) + "\n" +
                        "learning rate mode=" + str(self.learning_rate_mode_cbb.currentIndex()) + "\n" +
                        "warmup epochs=" + str(self.warmup_epochsSB.value()) + "\n" +
                        "warmup bias lr=" + str(self.warmup_bias_lrDSB.value()) + "\n" +
                        "warmup momentum=" + str(self.warmup_momentumDSB.value()) + "\n" +
                        "optimizers=" + str(self.optimizers_cbb.currentIndex()) + "\n" +
                        "momentum=" + str(self.momentumDSB.value()) + "\n" +
                        "weight decay=" + str(self.weight_decayDSB.value()) + "\n" +
                        "val able=" + str(self.val_able_cb.isChecked()) + "\n" +
                        "multi scale able=" + str(self.multi_scale_able_cb.isChecked()) + "\n" +
                        "multi scale=" + str(self.multi_scale_DSB.value()) + "\n" +
                        "fl gamma=" + str(self.fl_gamma_DSB.value()) + "\n" +
                        "cls smooth=" + str(self.cls_smooth_SB.value()) + "\n" +
                        "cls pw=" + str(self.cls_pwDSB.value()) + "\n" +
                        "obj pw=" + str(self.obj_pwDSB.value())
                        )
            QMessageBox.information(self,"提示","保存成功")
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion

    #region 关闭事件
    def closeEvent(self, a0: QCloseEvent) -> None:
        try:
            result = QMessageBox.information(self,"退出","退出系统！\n是否保存更改？",QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,QMessageBox.Cancel)
            if result == QMessageBox.Yes:
                a0.accept()
                self.save_project()
            elif result == QMessageBox.No:
                a0.accept()
            else:
                a0.ignore()
            gdd_ui.close()
            makenet_ui.close()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion

    #region 退出
    def Exit(self):
        self.close()

    #endregion

    #region 返回开始界面
    def bact_start(self):
        self.close()
        start_project_ui.show()
    #endregion

    # region 新增project
    def add_new_project_action(self):
        try:
            init_project_path = "./project_init/"
            project_dir = QFileDialog.getExistingDirectory(self, "选择新项目路径", "C://")
            try:
                os.removedirs(project_dir)
            except Exception as ex:
                QMessageBox.warning(self, "error", ex.__str__())
                return
            shutil.copytree(init_project_path, project_dir)
            with open(start_project_ui.path, 'a') as f:  # 保存新建project
                f.write(project_dir + "\n")
            start_project_ui.load_all_project()
            QMessageBox.information(None, "提示", "创建成功")
            self.project_path = project_dir
            self.load_project()
            self.setWindowTitle(self.project_path)
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    # endregion
    #region 打开project
    def open_project_action(self):
        try:
            project_dir = QFileDialog.getExistingDirectory(self, "选择项目", "C://")
            self.project_path = project_dir
            self.load_project()
            self.setWindowTitle(self.project_path)
            QMessageBox.information(self, "提示", "打开成功")
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())

    #endregion
    #region 打开已存在的项目
    def open_exist_project(self):
        try:
            self.project_path = start_project_ui.project_path
            self.load_project()
            self.setWindowTitle(self.project_path)
            start_project_ui.close()
            self.show()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())

    #endregion
    #region 创建新的项目
    def open_new_project(self):
        try:
            dir = start_project_ui.new_project_dir_LE.text() + "/" + start_project_ui.new_project_name_LE.text()
            b = os.path.exists(dir)
            assert not b, "项目已存在，请更改名称或路径"

            init_project_path = "./project_init/"
            shutil.copytree(init_project_path, dir)
            with open(start_project_ui.path, 'a') as f:   #保存新建project
                f.write(dir + "\n")
            start_project_ui.load_all_project()
            QMessageBox.information(None, "提示", "创建成功")
            start_project_ui.project_path = dir
            self.project_path = dir
            self.load_project()
            self.setWindowTitle(self.project_path)
            start_project_ui.close()
            self.show()
        except Exception as e:
            QMessageBox.warning(None, "Error", e.__str__())

    #endregion
    #region 打开已添加的项目
    def open_add_project(self):
        try:
            self.project_path = start_project_ui.exist_project_dir_LE.text()
            with open(start_project_ui.path, 'a') as f:  # 保存新建project
                f.write(self.project_path + "\n")
            self.load_project()
            self.setWindowTitle(self.project_path)
            start_project_ui.close()
            self.show()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())


    #endregion
    # region 开始训练
    def train(self):
        try:
            d = "" if self.device_CBB.currentText() == "GPU" else "cpu"
            self.save_project()  #保存参数
            if self.trainPB.text() == "训练":
                self.trainPB.setText("训练中断")

                self.hyp = {
                    "project_path": self.project_path,
                    "batch_size": self.batch_sizeSB.value(),
                    "epoch_num": self.epoch_numSB.value(),
                    "learn_rate_finaly": self.learn_rate_finalDSB.value(),
                    "learn_rate_init":self.learn_rate_initDSB.value(),
                    "class_num": self.class_numSB.value(),
                    "img_size": self.img_sizeSB.value(),
                    "net_name": self.net_nameCBB.currentText(),
                    "model_name": self.model_nameCBB.currentText(),
                    "giou": self.giou_lossDSB.value(),
                    "obj": self.obj_lossDSB.value(),
                    "cls": self.cls_lossDSB.value(),
                    "gr": self.gr_DSB.value(),
                    "anchors_t": self.anchor_tDSB.value(),
                    "val_conf_thres": self.val_conf_thres_DSB.value(),
                    "val_iou_thres": self.val_iou_thres_DSB.value(),
                    "test_conf_thres": self.test_conf_thres_DSB.value(),
                    "test_iou_thres": self.test_iou_thres_DSB.value(),
                    "device": d,
                    "img_type": self.image_type_CBB.currentText(),
                    "cache_img": self.cache_img_cb.isChecked(),
                    "extract_bounding_boxes": self.extract_bounding_boxes_cb.isChecked(),
                    "single_cls": self.single_cls_cb.isChecked(),
                    "rect": self.rect_cb.isChecked(),
                    "rect_size": self.rect_size_SB.value(),
                    "augment": self.augment_cb.isChecked(),
                    "border": self.border_cb.isChecked(),
                    "augment_hsv": self.augment_hsv_cb.isChecked(),
                    "lr_flip": self.lr_flip_cb.isChecked(),
                    "ud_flip": self.ud_flip_cb.isChecked(),
                    "degrees": self.degrees_DSB.value(),
                    "translate": self.translate_DSB.value(),
                    "scale": self.scale_DSB.value(),
                    "shear": self.shear_DSB.value(),
                    "hsv_h": self.hsv_h_DSB.value(),
                    "hsv_s": self.hsv_s_DSB.value(),
                    "hsv_v": self.hsv_v_DSB.value(),
                    "iou": self.iou_cbb.currentText(),
                    "auto_anchors":self.auto_anchor_cb.isChecked(),        #自动聚类预选框
                    "auto_batch":self.auto_batch_size_cb.isChecked(),          #自动选择合适的batchsize
                    "lf":self.learning_rate_mode_cbb.currentText(),       #学习率变化规则
                    "warmup_epochs":self.warmup_epochsSB.value(),        #预热学习迭代次数
                    "warmup_bias_lr":self.warmup_bias_lrDSB.value(),    #预热学习初始学习率
                    "warmup_momentum":self.warmup_momentumDSB.value(),      #预热学习初始动量
                    "optimizer":self.optimizers_cbb.currentText(),          #优化器类型
                    "momentum":self.momentumDSB.value(),             #动量（防止局部最小值）
                    "weight_decay":self.weight_decayDSB.value(),        #衰减系数
                    "val_able":self.val_able_cb.isChecked(),            #是否使用验证集
                    "amp":False,                 #是否使用amp训练
                    "multi_scale_able":self.multi_scale_able_cb.isChecked(),   #对网络的输入图像进行随机上采样或下采样
                    "multi_scale":self.multi_scale_DSB.value(),           #缩放正负倍数
                    "fl_gamma":self.fl_gamma_DSB.value(),               #focal_loss<0
                    "label_smooth":self.cls_smooth_SB.value(),
                    "cls_pw" : self.cls_pwDSB.value(),  # 类别BCE损失正权值
                    "obj_pw" : self.obj_pwDSB.value(),   # 置信度BCE损失正权值
                    "load_new_dataset":self.updata_cache_label_cb.isChecked()  #是否作为新的数据集加载/更新缓存
                }

                # 训练线程
                self.thread_train = Thread.Thread_train_yolo(self.hyp)
                self.thread_train._train_num_Signal.connect(self.receice_train_num)
                self.thread_train._train_run_epoch_Signal.connect(self.receive_epoch_train_loss_val_accuracy)
                self.thread_train._show_Signal.connect(self.receive_toshow)
                self.thread_train._train_run_Signal.connect(self.receive_batch_train_loss)
                self.thread_train.start()

            elif self.trainPB.text() == "训练中断":
                self.thread_train.terminate()
                self.thread_train.quit()
                del self.thread_train.md
                self.thread_train = None
                self.trainPB.setText("训练")
        except Exception as e:
            QMessageBox.warning(None, "Error", e.__str__())

    #endregion

    #region 训练后台信号接收-显示信息
    #region接收训练次数
    def receice_train_num(self,L):
        self.progressB.setRange(0, L)
        self.progressB.setValue(0)
    #endregion
    #region 接收训练每一迭代的平均损失与准确率
    def receive_epoch_train_loss_val_accuracy(self,S):
        self.outTE.append(S)
    #endregion
    #region 接收每一批次的训练损失
    def receive_batch_train_loss(self,S):
        self.outTE.append(S)
        self.progressB.setValue(self.progressB.value() + 1)
    #endregion

    #region 接收复杂信息，判断-显示特定信息
    def receive_toshow(self,S):
        try:
            if S == "replace":
                QMessageBox.information(None, "提示", "模型文件已存在，将继续上次训练结果继续训练模型！")
                return
            elif S.startswith("start train!"):
                self.progressB.setValue(int(S.split("$$")[-1]))
                self.outTE.append(S.split("$$")[0])
            elif S.startswith("time"):
                self.progressB.setFormat("剩余时间："+S.split("-")[-1]+"      百分比：%p%")
            elif S.startswith("plot"):
                self.updata_RMP_PB_clicked()
                self.updata_per_class_RMP_PB_clicked()
            elif S.startswith("error:"):
                QMessageBox.warning(None, "Error", S)
                return
            else:
                self.outTE.append(S)
        except Exception as ex:
            QMessageBox.warning(None,"Error",ex.__str__())

    #endregion

    #region 刷新总RMP显示
    def updata_RMP_PB_clicked(self):
        try:
            path_loss = self.project_path.replace("//",
                                                  "\\") + f"\\runs\\results\\{self.model_nameCBB.currentText()}_train_loss.txt"
            y_loss = np.loadtxt(path_loss, usecols=[1, 2, 3, 4])
            y_loss = y_loss / y_loss.max()
            if self.val_able_cb.isChecked():
                path_map = self.project_path.replace("//",
                                                     "\\") + f"\\runs\\results\\{self.model_nameCBB.currentText()}_val_map.txt"
                y_map = np.loadtxt(path_map, usecols=[2, 3, 4, 5])
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
            return
        leny = len(y_loss)
        if leny == 0:
            return
        if (leny > 100):
            x = np.append(np.linspace(0, leny - 11, 50).round(), np.linspace(leny - 10, leny - 1, 10))
            y_loss = y_loss[x.astype(np.int32)]
            if self.val_able_cb.isChecked():
                y_map = y_map[x.astype(np.int32)]
        else:
            x = np.linspace(0, leny - 1, leny)
        self.cls_name = []
        for j in range(self.class_numSB.value()):
            self.cls_name.append(self.sort_TW.item(j, 1).text())
        self.plot_loss(x, y_loss)
        if self.val_able_cb.isChecked():
            self.plot_map(x, y_map)
    #endregion
    #region 刷新每一个种类的RMP显示
    def updata_per_class_RMP_PB_clicked(self):
        if self.val_able_cb.isChecked():
            try:
                path_per_p = self.project_path.replace("//",
                                                       "\\") + f"\\runs\\results\\{self.model_nameCBB.currentText()}_val_per_p.txt"
                path_per_r = self.project_path.replace("//",
                                                       "\\") + f"\\runs\\results\\{self.model_nameCBB.currentText()}_val_per_r.txt"
                path_per_ap50 = self.project_path.replace("//",
                                                          "\\") + f"\\runs/results\\{self.model_nameCBB.currentText()}_val_per_ap50.txt"
                path_per_ap = self.project_path.replace("//",
                                                        "\\") + f"\\runs\\results\\{self.model_nameCBB.currentText()}_val_per_ap.txt"
                y_per_p = np.loadtxt(path_per_p, usecols=np.arange(1, self.class_numSB.value() + 1))
                y_per_r = np.loadtxt(path_per_r, usecols=np.arange(1, self.class_numSB.value() + 1))
                y_per_ap50 = np.loadtxt(path_per_ap50, usecols=np.arange(1, self.class_numSB.value() + 1))
                y_per_ap = np.loadtxt(path_per_ap, usecols=np.arange(1, self.class_numSB.value() + 1))
            except Exception as ex:
                QMessageBox.warning(None, "Error", ex.__str__())
                return

            leny = len(y_per_p)
            if leny == 0:
                return
            if (leny > 100):
                x = np.append(np.linspace(0, leny - 11, 50).round(), np.linspace(leny - 10, leny - 1, 10))
                y_per_p = y_per_p[x.astype(np.int32)]
                y_per_r = y_per_r[x.astype(np.int32)]
                y_per_ap50 = y_per_ap50[x.astype(np.int32)]
                y_per_ap = y_per_ap[x.astype(np.int32)]
            else:
                x = np.linspace(0, leny - 1, leny)
            self.cls_name = []
            for j in range(self.class_numSB.value()):
                self.cls_name.append(self.sort_TW.item(j, 1).text())
            self.plot_per_p(x, y_per_p)
            self.plot_per_r(x, y_per_r)
            self.plot_per_ap50(x, y_per_ap50)
            self.plot_per_ap(x, y_per_ap)

    #endregion
    #region plot
    def plot_map(self,x,y):
        # map_plot
        F_map = Myfigure(8, 2, 80)
        F_map.fig.suptitle("map")
        F_map.axs0 = F_map.fig.add_subplot(111)
        line0 = F_map.axs0.plot(x, y)
        F_map.axs0.legend(line0, ["mp", "mr", "map50", "map"])
        F_map.axs0.grid()
        self.RMP_GLY.addWidget(F_map, 0, 0)

    def plot_loss(self,x,y):
        # loss_plot
        F_loss = Myfigure(8, 2, 80)
        F_loss.fig.suptitle("loss")
        F_loss.axs1 = F_loss.fig.add_subplot(111)
        line1 = F_loss.axs1.plot(x, y)
        F_loss.axs1.legend(line1, ["loss", "lbox", "lobj", "lcls"])
        F_loss.axs1.grid()
        self.RMP_GLY.addWidget(F_loss, 1, 0)

    def plot_per_p(self,x,y):
        # per_p_plot
        F_per_p = Myfigure(4, 2, 80)
        F_per_p.fig.suptitle("precision")
        F_per_p.axs2 = F_per_p.fig.add_subplot(111)
        line2 = F_per_p.axs2.plot(x, y)
        F_per_p.axs2.legend(line2, self.cls_name)
        F_per_p.axs2.grid()
        self.per_class_RMP_GLY.addWidget(F_per_p, 0, 0)

    def plot_per_r(self,x,y):
        # per_r_plot
        F_per_r = Myfigure(4, 2, 80)
        F_per_r.fig.suptitle("recall")
        F_per_r.axs3 = F_per_r.fig.add_subplot(111)
        line3 = F_per_r.axs3.plot(x, y)
        F_per_r.axs3.legend(line3, self.cls_name)
        F_per_r.axs3.grid()
        self.per_class_RMP_GLY.addWidget(F_per_r, 0, 1)

    def plot_per_ap50(self,x,y):
        # per_ap50_plot
        F_per_ap50 = Myfigure(4, 2, 80)
        F_per_ap50.fig.suptitle("ap50")
        F_per_ap50.axs4 = F_per_ap50.fig.add_subplot(111)
        line4 = F_per_ap50.axs4.plot(x, y)
        F_per_ap50.axs4.legend(line4, self.cls_name)
        F_per_ap50.axs4.grid()
        self.per_class_RMP_GLY.addWidget(F_per_ap50, 1, 0)

    def plot_per_ap(self,x,y):
        # per_ap_plot
        F_per_ap = Myfigure(4, 2, 80)
        F_per_ap.fig.suptitle("ap")
        F_per_ap.axs5 = F_per_ap.fig.add_subplot(111)
        line5 = F_per_ap.axs5.plot(x, y)
        F_per_ap.axs5.legend(line5, self.cls_name)
        F_per_ap.axs5.grid()
        self.per_class_RMP_GLY.addWidget(F_per_ap,1, 1)
    #endregion
    #endregion



#endregion

#region 画图类-将plot图嵌入layout
class Myfigure(FigureCanvas):
    def __init__(self,width =5,height = 4,dpi =100):
        self.fig = Figure(figsize=(width,height),dpi=dpi)
        super(Myfigure,self).__init__(self.fig)
#endregion


#region 制作数据集窗口
class ChildWindow1_Gdd(QMainWindow,Gdd.Ui_MainWindow):
    def __init__(self):
        super(ChildWindow1_Gdd,self).__init__()
        self.setupUi(self)

    #region 打开
    def open(self):
        try:
            self.project_path = train_ui.project_path     #项目路径
            self.updata_project_path()                #更新项目数据集路径
            self.setWindowTitle(self.project_path)
            self.sort_TW.setRowCount(train_ui.class_numSB.value())  #设置种类，
            self.cls_name = []
            for i in range(train_ui.class_numSB.value()):
                twr = QTableWidgetItem()
                twr.setText(train_ui.sort_TW.item(i,1).text())  #i行，1列
                twr.setTextAlignment(Qt.AlignCenter)           #居中
                self.sort_TW.setItem(i,1,twr)
                self.cls_name.append(train_ui.sort_TW.item(i,1).text())     #加载种类名
                twl = QTableWidgetItem()
                twl.setText(str(i))
                twl.setTextAlignment(Qt.AlignCenter)
                self.sort_TW.setItem(i,0,twl)                   #i行0列
            self.sort_TW.setColumnWidth(0,70)
            self.sort_TW.setColumnWidth(1, 120)
            self.img_type = train_ui.image_type_CBB.currentText()    #图像类型：灰度图或彩色图
            self.trainRB.setChecked(True)
            self.show()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion

    #region 种类改变触发
    def sort_TW_changed(self):
        try:
            if self.sort_TW.currentItem()!= None:
                #同步改变train窗口种类
                twr = QTableWidgetItem()
                twr.setText(self.sort_TW.currentItem().text())
                twr.setTextAlignment(Qt.AlignCenter)
                train_ui.sort_TW.setItem(self.sort_TW.currentRow(), self.sort_TW.currentColumn(), twr)
                #改变imageLB中的rect种类
                if self.imageLB.rect != {} and self.cls_name[self.sort_TW.currentRow()] != '':
                    if self.cls_name[self.sort_TW.currentRow()] not in self.imageLB.rect.keys():   #旧的不存在就赋值[]
                        self.imageLB.rect[self.cls_name[self.sort_TW.currentRow()]] = []
                    rect_cls = self.imageLB.rect[self.cls_name[self.sort_TW.currentRow()]]   #旧rect
                    self.imageLB.rect.pop(self.cls_name[self.sort_TW.currentRow()])
                    self.imageLB.rect[self.sort_TW.item(self.sort_TW.currentRow(),1).text()] = rect_cls
                    if self.imageLB.current_state["cls"] == self.cls_name[self.sort_TW.currentRow()]:
                        self.imageLB.current_state["cls"] = self.sort_TW.item(self.sort_TW.currentRow(), 1).text()
                #同步改变种类名集合
                self.cls_name[self.sort_TW.currentRow()] = self.sort_TW.item(self.sort_TW.currentRow(),1).text()

                #同步改变radio按钮的text
                if self.cls_rbs != []:
                    self.cls_rbs[self.sort_TW.currentRow()].setText(self.sort_TW.item(self.sort_TW.currentRow(),1).text())
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion
    #region 增加种类
    def add_cls_pb_clicked(self):
        try:
            if not self.imageLB.paint_rect_flag:
                self.cls_name.append('')  # 增加一个种类
                # item1 种类索引
                item1 = QTableWidgetItem()
                item1.setTextAlignment(Qt.AlignCenter)
                item1.setText(str(self.sort_TW.rowCount()))
                # item2 种类名
                item2 = QTableWidgetItem()
                item2.setTextAlignment(Qt.AlignCenter)
                item2.setText("")
                # train窗口的种类名
                item3 = QTableWidgetItem()
                item3.setTextAlignment(Qt.AlignCenter)
                item3.setText(str(self.sort_TW.rowCount()))
                self.sort_TW.setRowCount(self.sort_TW.rowCount() + 1)
                self.sort_TW.setItem(self.sort_TW.rowCount() - 1, 0, item1)
                self.sort_TW.setItem(self.sort_TW.rowCount() - 1, 1, item2)
                train_ui.sort_TW.setItem(self.sort_TW.rowCount() - 1, 0, item3)
                train_ui.class_numSB.setValue(train_ui.class_numSB.value() + 1)  # 种类数加1
            else:
                QMessageBox.information(self,"提示","请关闭打标签模式")
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion
    #region 删除最后一个种类
    def del_cls_pb_clicked(self):
        try:
            if self.imageLB.paint_rect_flag:
                #删除radio
                self.cls_bg.removeButton(self.cls_rbs[self.sort_TW.rowCount()-1])
                self.cls_rbs.pop(self.sort_TW.rowCount()-1)
                self.horizontalLayout_2.itemAt(self.sort_TW.rowCount()).widget().deleteLater()
                #删除rect
                self.imageLB.rect.pop(self.sort_TW.item(self.sort_TW.rowCount()-1, 1).text())
            #删除种类集合中的种类
            self.cls_name.remove(self.sort_TW.item(self.sort_TW.rowCount()-1,1).text())
            #删除sort_TW中的种类
            self.sort_TW.removeRow(self.sort_TW.rowCount()-1)
            train_ui.sort_TW.removeRow(self.sort_TW.rowCount()-1)
            train_ui.class_numSB.setValue(train_ui.class_numSB.value() - 1)  #种类数减1
            self.update_rect_list()
            self.imageLB.update()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion

    #region 半自动标注
    def run_model_test(self):
        try:
            if not self.imageLB.img.shape[0]:
                return
            if train_ui.model_nameCBB.currentText() == "":
                QMessageBox.information(self,"提示","请先在train界面选择已训练模型")
                return
            d = "" if train_ui.device_CBB.currentText() == "GPU" else "cpu"
            device, _ = select_device(d)
            if self.md_name !=train_ui.model_nameCBB.currentText():
                self.md_name = train_ui.model_nameCBB.currentText()
                self.conf_th = train_ui.test_conf_thres_DSB.value()

                model_path = self.project_path + "//runs//models//" + train_ui.model_nameCBB.currentText() + ".pt"
                if not os.path.exists(model_path):
                    QMessageBox.information(self, "提示", "模型不存在，请训练")
                    return
                self.ckpt = torch.load(model_path, map_location=device)
            self.model = self.ckpt["model"]
            self.model.eval()
            self.model.model[-1].conf_thres = train_ui.test_conf_thres_DSB.value()
            self.model.model[-1].iou_thres = train_ui.test_iou_thres_DSB.value()

            img = self.imageLB.img
            if train_ui.rect_cb.isChecked():
                h,w = img.shape[:2]
                s = h/w
                if s > 1:
                    shape = [1,1/s]
                else:
                    shape = [s,1]
                shape = np.ceil(np.array(shape) * train_ui.img_sizeSB.value() / float(train_ui.rect_size_SB.value())).astype(np.int32) * train_ui.rect_size_SB.value()
            else:
                shape = [train_ui.img_sizeSB.value(),train_ui.img_sizeSB.value()]
            img_train, r0,radio, dwh = public_method.pad_img(img,
                                                          shape,
                                                          train_ui.image_type_CBB.currentText())
            h1, w1 = img_train.shape[0:2]
            img_train = img_train.reshape(h1, w1, 1) if train_ui.image_type_CBB.currentText() != "color" else img_train
            img_train = img_train[:, :, ::-1].transpose(2, 0,
                                                        1) if train_ui.image_type_CBB.currentText() == "color" else img_train.transpose(
                2, 0, 1)  # BGR to RGB to 3*h*w
            img_train = torch.from_numpy(np.ascontiguousarray(img_train))
            pre_out, _ = self.model(img_train.unsqueeze(0).float().to(device) / 255.0)
            output = pre_out
            box = None
            conf = None
            cls = None
            if output != [None] and len(output[0]) > 0:
                h, w = img.shape[:2]
                clip_coords(output[0], shape)
                op = torch.zeros((len(output[0]), 6))
                for i, o in enumerate(output[0]):
                    op[i] = o
                box = xyxy2xywh(op[:, :4])
                box[:, 0] = (box[:, 0] - dwh[0]) / (radio[0]*r0)/w
                box[:, 1] = (box[:, 1] - dwh[1]) / (radio[1]*r0)/h
                box[:, 2] = box[:, 2] / (radio[0]*r0)/w
                box[:, 3] = box[:, 3] / (radio[1]*r0)/h
                conf = op[:, 4]
                cls = op[:, 5]
                # 将模型输出转移到标注上
                for i,b in enumerate(box):
                    re = Mylabel.myrect()
                    c = self.cls_name[int(cls[i])]
                    re.rect = [r.item() for r in b]
                    if c not in self.imageLB.rect.keys():
                        self.imageLB.rect[c] = []
                    re.conf = conf[i].item()
                    re.cls = int(cls[i])
                    self.imageLB.rect[c].append(re)
                self.imageLB.scale_wheel = 1
                self.imageLB.update()
                self.update_rect_list()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
    #endregion
#endregion


#region 制作网络结构窗口
class ChildWindow2_makenet(QMainWindow,makenet.Ui_MainWindow):
    def __init__(self):
        super(ChildWindow2_makenet,self).__init__()
        self.setupUi(self)
    def open(self):
        self.net_path = train_ui.project_path+"//nets"
        self.netpathLE.setText(train_ui.project_path + "/nets/" + train_ui.net_nameCBB.currentText() + ".yaml")
        self.show()
#endregion

#region 开始选择项目窗口
class ChildWindow0_start_project(QMainWindow,start_project.Ui_MainWindow):
    def __init__(self):
        super(ChildWindow0_start_project,self).__init__()
        self.setupUi(self)
#endregion

#region 增加模型小窗口
class ChildWindow3_add_model(QMainWindow,addmodel.Ui_MainWindow):
    def __init__(self):
        super(ChildWindow3_add_model,self).__init__()
        self.setupUi(self)
        self.ok_PB.clicked.connect(self.ok_PB_clicked)

    def open(self):
        self.show()

    def ok_PB_clicked(self):
        flag = False
        for index in range(train_ui.model_nameCBB.count()):
            if self.model_name_LE.text() == train_ui.model_nameCBB.itemText(index):
                flag = True
        if flag:
            QMessageBox.information(self,"提示","模型已存在，请重新命名!")
            return
        else:
            train_ui.model_nameCBB.addItem(self.model_name_LE.text())
            train_ui.model_nameCBB.setCurrentIndex(train_ui.model_nameCBB.count()-1)
            QMessageBox.information(self, "提示", "模型建立成功")
            self.close()
#endregion

#region k_means聚类获取预选框
class ChildWindow4_k_means(QMainWindow,k_means.Ui_MainWindow):
    def __init__(self):
        super(ChildWindow4_k_means,self).__init__()
        self.setupUi(self)
    def open(self):
        try:
            self.project_path = train_ui.project_path
            configs = self.parse_config(self.project_path + "//config//k_means_config.config")
            self.labels_pathLE.setText(self.project_path + "//data//path_train.txt")
            self.img_widthSB.setValue(int(configs["img_width"]))
            self.img_heightSB.setValue(int(configs["img_height"]))
            self.epoch_numSB.setValue(int(configs["epoch_num"]))
            self.boxes_numSB.setValue(int(configs["boxes_num"]))
            self.img_size = train_ui.img_sizeSB.value()
            self.show()
        except Exception as ex:
            QMessageBox.warning(None, "Error", ex.__str__())
#endregion

class ChildWindow5_connx(QMainWindow,connx.Ui_MainWindow):
    def __init__(self):
        super(ChildWindow5_connx,self).__init__()
        self.setupUi(self)
    def open(self):
        try:
            self.project_path = train_ui.project_path
            self.init_pt_path = self.project_path + "//runs//models//"
            self.pt_model_iconLE.setText(self.init_pt_path + train_ui.model_nameCBB.currentText() + ".pt")
            self.conf_threDSB.setValue(0.0001)
            self.iou_threDSB.setValue(0.1)
            self.img_size = train_ui.img_sizeSB.value()
            self.rect_flag = train_ui.rect_cb.isChecked()
            self.rect_v = train_ui.rect_size_SB.value()
            self.img_channelsSB.setValue(1 if train_ui.image_type_CBB.currentText() == "gray" else 3)
            self.show()
        except Exception as ex:
            QMessageBox.warning(None,"Error",ex.__str__())

ui = MainWindow_train()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)

        train_ui = MainWindow_train()
        gdd_ui = ChildWindow1_Gdd()
        makenet_ui = ChildWindow2_makenet()
        start_project_ui = ChildWindow0_start_project()
        addmodel_ui = ChildWindow3_add_model()
        k_means_ui = ChildWindow4_k_means()
        connx_ui = ChildWindow5_connx()



        train_ui.makedatasetPB.clicked.connect(gdd_ui.open)
        train_ui.makenetPB.clicked.connect(makenet_ui.open)
        train_ui.add_modelPB.clicked.connect(addmodel_ui.open)
        train_ui.save_as_onnxPB.clicked.connect(connx_ui.open)
        train_ui.actionback_start.triggered.connect(train_ui.bact_start)
        train_ui.actionexit.triggered.connect(train_ui.Exit)
        train_ui.actionnew_file.triggered.connect(train_ui.add_new_project_action)
        train_ui.actionopen_project.triggered.connect(train_ui.open_project_action)
        train_ui.actionsave.triggered.connect(train_ui.save_project)

        start_project_ui.project_path_LW.doubleClicked.connect(train_ui.open_exist_project)
        start_project_ui.create_new_project.clicked.connect(train_ui.open_new_project)
        start_project_ui.Add_exist_project_pb.clicked.connect(train_ui.open_add_project)

        gdd_ui.sort_TW.itemChanged.connect(gdd_ui.sort_TW_changed)
        gdd_ui.add_cls_pb.clicked.connect(gdd_ui.add_cls_pb_clicked)
        gdd_ui.del_cls_pb.clicked.connect(gdd_ui.del_cls_pb_clicked)
        gdd_ui.model_text_pb.clicked.connect(gdd_ui.run_model_test)
        gdd_ui.model_text_pb.setShortcut("space")

        makenet_ui.k_means_get_anchors_pb.clicked.connect(k_means_ui.open)

        start_project_ui.show()
        sys.exit(app.exec_())
    except Exception as ex:
        QMessageBox.warning(None,"警告",ex.__str__())
