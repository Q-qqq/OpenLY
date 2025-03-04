
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from pathlib import Path
import numpy as np
import pyqtgraph as pg
import cv2

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from ultralytics.utils import plt_settings,threaded
from APP  import PROJ_SETTINGS


class QFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


class LossFigure(QFigure):

    @threaded
    @plt_settings( backend="QtAgg")
    def plot(self, metrics_csv=f"results.csv"):
        metrics_csv = Path(PROJ_SETTINGS["current_experiment"]) / metrics_csv
        if not metrics_csv.exists():
            return
        data = pd.read_csv(metrics_csv)
        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]
        train_loss_y = np.zeros(len(x), dtype=np.float32)
        val_loss_y = np.zeros(len(x), dtype=np.float32)
        len_t = 0
        len_v = 0
        for i, s in enumerate(s[1:]):
            if s.endswith("loss"):
                y = np.array(data.values[:, i+1].astype("float"), dtype=np.float32)
                if s.startswith("train"):
                    train_loss_y += y
                    len_t += 1
                elif s.startswith("val"):
                    val_loss_y += y
                    len_v += 1
        #损失均值
        train_loss_y /= len_t
        val_loss_y /= len_v

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.patch.set_facecolor("#eeffee")  # 设置ax区域背景颜色
        #self.ax.patch.set_alpha(0.5)  # 设置ax区域背景颜色透明度
        self.fig.patch.set_facecolor('#fefff2')  # 设置绘图区域颜色
        #self.ax.spines['bottom'].set_color('r')  # 设置下边界颜色
        self.ax.spines['top'].set_visible(False)  # 顶边界不可见
        self.ax.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        self.ax.set_xlim((1,int(x[-1])+2))
        y_max = max(train_loss_y.max(), val_loss_y.max())
        self.ax.set_ylim((0, y_max + y_max/5))
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax.set_title("Loss of train&val")
        self.ax.grid()
        lines = []
        for y, label in zip([train_loss_y, val_loss_y],["train", "val"]):
            line1, = self.ax.plot(x, y.tolist(), marker=".", label=label, linewidth=2, markersize=5)  # actual results
            line2, = self.ax.plot(x, gaussian_filter1d(y.tolist(), sigma=3), ":", label=label+"_smooth", linewidth=2)  # smoothing line
            lines += [line1]

        self.ax.legend(handles=lines, loc="best")


        self.fig.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.fig.canvas.flush_events()  # 画布刷新self.figs.canvas



class PgPlotLossWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_line = None
        self.val_line = None

        font = QFont("Times", 12)
        self.getAxis("left").setTickFont(font)
        self.getAxis("bottom").setTickFont(font)

        styles = {'color': 'k', 'font-size': '15px'}
        self.setLabel("bottom", "epoch", **styles, blod=True)
        self.setLabel("left", "loss", **styles, blod=True)
        self.setTitle("Average Loss of train/val",color="k", size="20px", bold=True)

        pg.setConfigOption("leftButtonPan", False)
        pg.setConfigOption("antialias", False)
        self.setMouseEnabled(x=False, y=True)
        self.showGrid(x=True, y=True)
        self.addLegend()


    def lossPlot(self, metrics_csv=f"results.csv"):
        metrics_csv = Path(PROJ_SETTINGS["current_experiment"]) / metrics_csv
        if not metrics_csv.exists():
            return
        data = pd.read_csv(metrics_csv)
        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]
        train_loss_y = np.zeros(len(x), dtype=np.float32)
        val_loss_y = np.zeros(len(x), dtype=np.float32)
        len_t = 0
        len_v = 0
        for i, s in enumerate(s[1:]):
            s = s.lower()
            if s.endswith("loss"):
                y = np.array(data.values[:, i+1].astype("float"), dtype=np.float32)
                if s.startswith("train"):
                    train_loss_y += y
                    len_t += 1
                elif s.startswith("val"):
                    val_loss_y += y
                    len_v += 1
        #损失均值
        train_loss_y /= len_t
        val_loss_y /= len_v
        if self.train_line is None:
            self.train_line = self.plot(x, train_loss_y.tolist(),pen=pg.mkPen((255,0,0), width=2), symbol="s", symbolSize=2,name="train")
        else:
            self.train_line.setData(x, train_loss_y.tolist())
        if self.val_line is None:
            self.val_line = self.plot(x, val_loss_y.tolist(), pen=pg.mkPen((0,255,0), width=2), symbol="s", symbolSize=2, name="val")
        else:
            self.val_line.setData(x, val_loss_y.tolist())
        #self.setXRange(1, x[-1]+2)
        #self.setYRange(0, max(train_loss_y.max(), val_loss_y.max()))



class HistFigure(QFigure):

    @threaded
    @plt_settings( backend="QtAgg")
    def plot(self, img, c):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        #self.ax.spines['bottom'].set_color('r')  # 设置下边界颜色
        self.ax.spines['top'].set_visible(False)  # 顶边界不可见
        self.ax.spines['right'].set_visible(False)  # 右边界不可见
        self.ax.spines['left'].set_visible(False)  # 左边界不可见
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1,hspace=0.1,wspace=0.1)
        # 设置左、下边界在（0，0）处相交
        self.ax.set_xlim((0,255))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        if c == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            color = "black"
        elif c == "R":
            img = img[:,:,2]
            color = "red"
        elif c == "G":
            img = img[:,:,1]
            color = "green"
        elif c == "B":
            img = img[:,:,0]
            color = "blue"
        self.ax.hist(img.ravel(),256, [0,256],color=color)

        self.fig.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.fig.canvas.flush_events()  # 画布刷新self.figs.canvas