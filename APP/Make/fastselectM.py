import math

import cv2

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import numpy as np

from APP.Design import fast_selectQT_ui


def clip(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n

class FastSelect(QWidget, fast_selectQT_ui.Ui_Form):
    def __init__( self, parent, img_label, f=Qt.Tool):
        super().__init__(parent, f)
        self.setupUi(self)
        self.img_label = img_label
        self.Finish_pb.setVisible(False)
        self.Cancel_pb.setVisible(False)
        # 初始化GrabCut参数
        self.bgd_model = np.zeros((1, 65), np.float64)
        self.fgd_model = np.zeros((1, 65), np.float64)
        self.mask = None
        #初始化FF参数
        self.rgb_lo = 0
        self.rgb_up = 0
        self.r_lo = 0
        self.r_up = 0
        self.b_lo = 0
        self.b_up = 0
        self.g_lo = 0
        self.g_up = 0
        self.eventConnect()



    def eventConnect(self):
        self.Thre_cre_sel_pb.clicked.connect(self.createSelectSpace)
        self.GC_cre_sel_pb.clicked.connect(self.createSelectSpace)
        self.FF_cre_sel_pb.clicked.connect(self.createSelectSpace)
        self.FF_lo_diff_hs.valueChanged.connect(self.ffValueChangedSlot)
        self.FF_up_diff_hs.valueChanged.connect(self.ffValueChangedSlot)
        self.FF_sel_seed_pb.clicked.connect(self.findSeedClicked)
        self.Fast_sel_methods_cbb.currentIndexChanged.connect(self.methodSelectedSlot)
        self.Finish_pb.clicked.connect(self.finish)
        self.Cancel_pb.clicked.connect(self.cancel)

    def show(self):
        self.img_label.Fast_Sel_Cop_Signal.connect(self.compute)
        super().show()



    def methodSelectedSlot(self):
        self.Finish_pb.setVisible(False)
        self.Cancel_pb.setVisible(False)
        if self.Fast_sel_methods_cbb.currentText() == "threshold":
            self.FF_color_ch_cbb.setVisible(False)
        elif self.Fast_sel_methods_cbb.currentText() == "grabcut":
            self.FF_color_ch_cbb.setVisible(False)
        elif self.Fast_sel_methods_cbb.currentText() == "floodfill":
            self.FF_color_ch_cbb.setVisible(True)
            self.ffValueChangedSlot()
        self.img_label.fast_method = self.Fast_sel_methods_cbb.currentText()



    def ffValueChangedSlot(self):
        if self.FF_color_ch_cbb.currentText() == "RGB":
            self.rgb_lo = self.FF_lo_diff_hs.value()
            self.rgb_up = self.FF_up_diff_hs.value()
        elif self.FF_color_ch_cbb.currentText() == "R":
            self.r_lo = self.FF_lo_diff_hs.value()
            self.r_up = self.FF_up_diff_hs.value()
        elif self.FF_color_ch_cbb.currentText() == "G":
            self.g_lo = self.FF_lo_diff_hs.value()
            self.g_up = self.FF_up_diff_hs.value()
        elif self.FF_color_ch_cbb.currentText() == "B":
            self.b_lo = self.FF_lo_diff_hs.value()
            self.b_up = self.FF_up_diff_hs.value()


    def createSelectSpace(self):
        """开始创建选区"""
        self.img_label.fast_cre_sel = True
        self.img_label.fast_seed_searching = False
        self.Fast_sel_methods_cbb.setEnabled(False)
        self.Finish_pb.setVisible(True)
        self.Cancel_pb.setVisible(True)

    def findSeedClicked(self):
        self.Finish_pb.setVisible(True)
        self.Cancel_pb.setVisible(True)
        self.img_label.fast_seed_searching = True


    def finish(self):
        self.Finish_pb.setVisible(False)
        self.Cancel_pb.setVisible(False)
        self.img_label.fast_cre_sel = False
        self.img_label.fast_rect = None
        self.img_label.fast_seed_searching = False
        if self.img_label.fast_segment is not None and isinstance(self.img_label.fast_segment,np.ndarray):
            if self.img_label.use_pencil:
                if self.img_label.pencil_mode == "add":
                    self.img_label.addSegment(self.img_label.fast_segment, self.img_label.cls)
                else:
                    self.img_label.removeSegment(self.img_label.fast_segment)
            else:  # 钢笔只有添加功能
                self.img_label.addSegment(self.img_label.fast_segment, self.img_label.cls)
            self.img_label.Change_Label_Signal.emit()
        self.img_label.fast_segment = None
        self.Fast_sel_methods_cbb.setEnabled(True)
        self.img_label.update()

    def cancel(self):
        self.Finish_pb.setVisible(False)
        self.Cancel_pb.setVisible(False)
        self.img_label.fast_cre_sel = False
        self.img_label.fast_rect = None
        self.img_label.fast_seed_searching = False
        self.img_label.fast_segment = None
        self.Fast_sel_methods_cbb.setEnabled(True)
        self.img_label.update()


    def bgraTomask(self, mask_bgra):
        if self.Fast_sel_methods_cbb.currentText() != "grabcut":
            mask = mask_bgra[:,:,2]
            mask = mask / 255
            mask = mask.astype(np.uint8)
        else:
            #红色r 前景1 , 绿色g 可能的前景2， 蓝色b 可能的背景3, 黑色0 背景0
            mb,mg,mr,_ = cv2.split(mask_bgra)
            mb = (mb / 255).astype(np.uint8) * 3
            mg = (mg / 255).astype(np.uint8) * 2
            mr = (mr / 255).astype(np.uint8) * 1
            mask = mb+mg+mr
        return mask

    def getMask(self, h ,w):
        if self.mask is None:
            self.mask = np.zeros((h,w), dtype=np.uint8)
        else:
            mh,mw = self.mask.shape[:2]
            if h != mh or w != mw:
                self.mask = np.zeros((h,w), dtype=np.uint8)
        return self.mask.copy()

    def rectOpt(self, rect, h, w):
        """处理矩形框"""
        rect[0] = clip(math.floor(rect[0]),0, w)
        rect[1] = clip(math.ceil(rect[1]), 0, h)
        rect[2] = clip(math.floor(rect[2]), 0, w)
        rect[3] = clip(math.ceil(rect[3]), 0, h)
        area = (rect[3] - rect[1]) * (rect[2]-rect[0])
        return rect, area


    def thresholdGetObject(self):
        """阈值法计算区域内目标,只获取最大的目标
        Returns:
            c(list|nd.array): 若找到目标，返回nd,array(n,2), 若未找到目标，返回空白list[]"""
        th_lo = self.Thre_lo_th_sb.value()
        th_up = self.Thre_up_th_sb.value()
        if self.img_label.levels_img is not None:
            img = self.img_label.levels_img.astype(np.uint8).copy()
        else:
            img = self.img_label.img.copy()
        h, w = img.shape[:2]
        rect,area = self.rectOpt(self.img_label.fast_rect, h, w)
        if area ==0:
            return []
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t_img = img[rect[1]:rect[3], rect[0]:rect[2]]
        _, t_mask_lo = cv2.threshold(t_img,th_lo,255,cv2.THRESH_BINARY)
        _, t_mask_up = cv2.threshold(t_img, th_up, 255, cv2.THRESH_BINARY_INV)
        t_mask = cv2.bitwise_and(t_mask_lo, t_mask_up)
        #优化
        kernel = np.ones((3, 3), np.uint8)
        t_mask = cv2.morphologyEx(t_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
        c = cv2.findContours(t_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(c):
            # 坐标系转换为原图坐标系
            c_ = []
            for mc in c:
                if len(mc) > 2:
                    mc = mc.squeeze(1).astype(np.float32)
                    mc[:, 0] = mc[:, 0] + rect[0]
                    mc[:, 1] = mc[:, 1] + rect[1]
                    c_.append(mc)
            if len(c_):
                lc = [len(mc) for mc in c_]
                c = c_[lc.index(max(lc))]
        return c


    def grabcutGetObject(self):
        """grabcut法计算区域内目标,只获取最大的目标
        Returns:
            c(list|nd.array): 若找到目标，返回nd,array(n,2), 若未找到目标，返回空白list[]"""
        if self.img_label.levels_img is not None:
            img = self.img_label.levels_img.copy()
        else:
            img = self.img_label.img.copy()
        h, w = img.shape[:2]

        rect, area = self.rectOpt(self.img_label.fast_rect, h, w)
        if area == 0:
            return []


        temp_mask = self.getMask(h, w)
        cv2.grabCut(img, temp_mask, rect, self.bgd_model, self.fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        # 生成最终蒙版（前景和可能的前景）
        final_mask = np.where((temp_mask == 1) | (temp_mask == 3), 255, 0).astype(np.uint8)
        _, final_mask = cv2.threshold(final_mask,128,255,cv2.THRESH_BINARY)
        # 优化
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
        c = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(c):
            c = [mc.squeeze(1).astype(np.float32) for mc in c if len(mc) > 2]
            if len(c):
                lc = [len(mc) for mc in c]
                c = c[lc.index(max(lc))]
        return c



    def floodfillGetObject(self, seed):
        """floodfill法计算区域内目标,只获取最大的目标
        Args:
            seed(list|nd.array):种子选取点（x,y）
        Returns:
            c(list|nd.array): 若找到目标，返回nd,array(n,2), 若未找到目标，返回空白list[]"""
        if self.img_label.levels_img is not None:
            img = self.img_label.levels_img.copy()
        else:
            img = self.img_label.img.copy()
        img = img.astype(np.int32)
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        #check seed
        seed[0] = clip(int(seed[0]), 0 ,w)
        seed[1] = clip(int(seed[1]), 0, h)
        gray = ((img[:,:,0] == img[:,:,1]) &(img[:,:,1] ==img[:,:,2])).all()
        if gray:
            lo_diff = [self.rgb_lo] * 3
            up_diff = [self.rgb_up] * 3
        else:
            lo_diff = [self.b_lo, self.g_lo, self.r_lo]
            up_diff = [self.b_up, self.g_up, self.r_up]
        final_mask = cv2.floodFill(img, mask,seed,[0,0,0], lo_diff, up_diff,flags=cv2.FLOODFILL_FIXED_RANGE |cv2.FLOODFILL_MASK_ONLY|8)[2]
        # check rect
        mask_rect = np.zeros((h+2, w+2), np.uint8)
        if self.img_label.fast_rect is not None:
            rect, area = self.rectOpt(self.img_label.fast_rect, h, w)
            if area == 0:
                return []
            if rect:
                mask_rect[(rect[1] + 1):(rect[3] + 1), (rect[0] + 1):(rect[2] + 1)] = 1
                final_mask = ((final_mask == 1) & (mask_rect == 1)).astype(np.uint8)

        # 优化
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
        final_mask = final_mask[1:-1, 1:-1]
        c = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(c):
            c = [mc.squeeze(1).astype(np.float32) for mc in c if len(mc) > 2]
            if len(c):
                lc = [len(mc) for mc in c]
                c = c[lc.index(max(lc))]
        return c


    def compute(self,point=[0,0]):
        if self.Fast_sel_methods_cbb.currentText() == "threshold":
            obj = self.thresholdGetObject()
        elif self.Fast_sel_methods_cbb.currentText() == "grabcut":
            obj = self.grabcutGetObject()
        else:
            obj = self.floodfillGetObject(point)
        if len(obj):
            self.img_label.fast_segment = obj
            self.img_label.update()





