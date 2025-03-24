from pathlib import Path
import math
import time

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ultralytics.utils import LOGGER

from APP.Utils import get_widget, append_formatted_text
from APP import PROJ_SETTINGS, getExperimentPath


class RunMes(QObject):
    def __init__(self, parent):
        """paren： trainM"""
        super().__init__(parent)
        self.run_mes_te = get_widget(parent, "Run_mes_te")
        self.batch_mes_le = get_widget(parent, "Batch_mes_le")
        self.tip_te = get_widget(parent, "Tip_te")
        self.train_progressbar = get_widget(parent, "Train_progressbar")
        self.confusion_norm_label = get_widget(parent, "confusion_norm_label")
        self.confusion_denorm_label = get_widget(parent, "confusion_denorm_label")
        self.train_a = get_widget(parent, "Train_a")
        self.start_epoch = -1
        self.end_epoch = -1
        self.current_epoch = -1
        self.epoch_time = 0
        self.epoch_start_time = 0
        self.eventConnect()

    def eventConnect(self):
        LOGGER.Start_Train_Signal.connect(self.startTrainSlot)
        LOGGER.Batch_Finish_Signal.connect(self.batchFinishSlot)
        LOGGER.Epoch_Finish_Signal.connect(self.epochFinishSlot)
        LOGGER.Train_Finish_Signal.connect(self.trainFinishSlot)
        LOGGER.Start_Val_Signal.connect(self.startValSlot)
        LOGGER.Val_Finish_Signal.connect(self.valFinishSlot)
        LOGGER.Show_Mes_Signal.connect(self.showMesSlot)
        LOGGER.Error_Signal.connect(self.errorShowSlot)
        LOGGER.Interrupted_Error_Signal.connect(self.interruptSlot)

    def infoAppend(self,msg):
        append_formatted_text(self.tip_te, msg, 11)
    
    def warningAppend(self, msg):
        append_formatted_text(self.tip_te, msg, 13, QColor(255, 165, 0))
    
    def errorAppend(self, msg):
        append_formatted_text(self.tip_te, msg, 15, QColor(Qt.GlobalColor.red), True)

    def showMesSlot(self, level, msg):
        if level == "info":
            self.infoAppend(msg)
        elif level == "warning":
            self.warningAppend(msg)
        else:
            self.errorAppend(msg)

    def startTrainSlot(self, mes_epochs):
        self.train_a.setEnabled(True)
        mes = mes_epochs[0]
        self.start_epoch = mes_epochs[1]
        self.end_epoch = mes_epochs[2]
        self.run_mes_te.clear()
        append_formatted_text(self.run_mes_te, mes, 12)
        self.train_progressbar.setRange(self.start_epoch, self.end_epoch)
        self.epoch_start_time = time.time()

    def batchFinishSlot(self, mes):
        self.batch_mes_le.setText(mes)

    def epochFinishSlot(self, mes_epoch):
        mes = mes_epoch[0]
        self.current_epoch = mes_epoch[1]
        self.batch_mes_le.clear()
        append_formatted_text(self.run_mes_te, mes, 12)
        self.updateLoss()
        t = time.time()
        self.epoch_time = t - self.epoch_start_time
        self.epoch_start_time = t
        left_epochs = self.end_epoch - self.current_epoch   #剩下的epochs
        T = left_epochs * self.epoch_time
        h = math.floor(T / 3600)
        m = math.floor((T - h * 3600) / 60)
        s = math.floor(T - h * 3600 - m * 60)
        left_time = f"{h}:{m}:{s}"    #剩下的时间
        self.train_progressbar.setValue(self.current_epoch)
        self.train_progressbar.setFormat(f"%p%   {left_time}")


    def startValSlot(self, mes):
        self.infoAppend(mes)

    def valFinishSlot(self, mes):
        self.updateConfusion()

    def trainFinishSlot(self, mes):
        LOGGER.stop = True
        self.train_a.setText("训练")
        self.train_a.setEnabled(True)
        self.run_mes_te.append(mes)
        self.run_mes_te.moveCursor(QTextCursor.End)

    def interruptSlot(self, msg):
        """训练、验证、预测、加载等中断信息显示"""
        QMessageBox.information(self.parent(), "Interrupt", msg)
        if msg.startswith("训练中断"):
            self.train_a.setEnabled(True)
            LOGGER.stop = True
            self.train_a.setText("训练")
            self.train_a.setEnabled(True)


    def updateConfusion(self):
        title_norm = "Confusion Matrix" + " Normalized"
        title_denorm = "Confusion Matrix"
        p_norm = Path(getExperimentPath()) / f"{title_norm.lower().replace(' ', '_')}.png"
        p_denorm = Path(getExperimentPath()) / f"{title_denorm.lower().replace(' ', '_')}.png"
        self.confusion_norm_label.load_image(p_norm)
        self.confusion_denorm_label.load_image(p_denorm)

    def updateLoss(self):
        self.parent().loss_plot.lossPlot()

    def errorShowSlot(self, msg):
        QMessageBox.critical(self.parent(), "运行错误", msg)
