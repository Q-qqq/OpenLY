import time
from typing import Union
import cv2
import numpy as np

import torch

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


import copy
from pathlib import Path
import threading
from multiprocessing.pool import ThreadPool

from ultralytics.utils import threaded, yaml_load,NUM_THREADS, PROGRESS_BAR, LOGGER
from ultralytics.data.utils import verify_image

from APP  import  FILL_RULE
from APP.Utils import debounce, getcat
from APP.Label.base import QBboxes, QInstances, QTransformerLabel, QSizeLabel, QFastSelectLabel
from APP.Label.utils import *
from APP.Data import format_im_files


class DetectTransformerLabel(QFastSelectLabel):
    """目标检测标注label"""
    def __init__(self, parent):
        super().__init__(parent)
        self.bounding_rect = True
        self.ori_x = -1
        self.ori_y = -1
        self.task = "detect"

    def drawLabel(self, painter, label, pred=False):
        super().drawLabel(painter, label, pred)
        if self.painting and label["instances"]._bboxes is not None and len(label["instances"]._bboxes) == self.index:
            painter.setPen(QPen(Qt.GlobalColor.green, 10, Qt.PenStyle.SolidLine))
            painter.drawPoint(self.mouse_point)


    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if self.pix == None or self.fastCreate or self.resizing(event):
            return
        if self.painting and event.button() == Qt.MouseButton.LeftButton:
            if len(self.label["instances"]._bboxes) - 1 == self.index:
                self.painting = False
        if not self.painting and self.paint and event.button() == Qt.MouseButton.LeftButton and self.cursor().shape() != Qt.CursorShape.CrossCursor:
            self.getOri()

    def getOri(self):
        """获取矩形框起始点"""
        instance = copy.deepcopy(self.label["instances"])
        self.getLabelSizeInstance(instance)
        instance.convert_bbox("xyxy")
        x1, y1, x2, y2 = instance.bboxes[self.index]
        if self.point_ind1 == self.point_ind2 == 0:
            self.ori_x = x2
            self.ori_y = y2
        elif self.point_ind1 == self.point_ind2 == 1:
            self.ori_x = x1
            self.ori_y = y2
        elif self.point_ind1 == self.point_ind2 == 2:
            self.ori_x = x2
            self.ori_y = y1
        elif self.point_ind1 == self.point_ind2 == 3:
            self.ori_x = x1
            self.ori_y = y1
        elif self.point_ind1 == self.point_ind2 == 4:
            self.ori_x = x1
            self.ori_y = x1
        elif self.point_ind1 == 0 and self.point_ind2 == 1:
            self.ori_x = x1
            self.ori_y = y2
        elif self.point_ind1 == 2 and self.point_ind2 == 3:
            self.ori_x = x1
            self.ori_y = y1
        elif self.point_ind1 == 0 and self.point_ind2 == 2:
            self.ori_x = x2
            self.ori_y = y1
        elif self.point_ind1 == 1 and self.point_ind2 == 3:
            self.ori_x = x1
            self.ori_y = y1


    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if self.pix == None or self.fastCreate or self.resizing(event):
            return
        if self.painting and event.buttons() == Qt.LeftButton and len(self.label["instances"]._bboxes) == self.index: #未添加框
            p = self.getPixSizePoint(event.x(), event.y())
            self.addBox([p[0], p[1], 0, 0], self.cls)   #添加框
            self.ori_x = p[0]
            self.ori_y = p[1]
        if self.painting and len(self.label["instances"]._bboxes)-1 == self.index:  #绘制中且已添加新框
            p = self.getPixSizePoint(event.x(), event.y())
            p1 = [p[0].item(), p[1].item()]
            p2 = [self.ori_x, self.ori_y]
            box = twoPoints2box(p1, p2, self.label["instances"]._bboxes.format)
            self.label["instances"]._bboxes.setItem(self.index,box, self.label["instances"]._bboxes.format)
            self.Change_Label_Signal.emit()
        elif self.paint and not self.painting:
            #操作标签
            if event.buttons() == Qt.LeftButton and event.modifiers() != Qt.KeyboardModifier.ControlModifier:
                tx = event.x() - self.start.x()
                ty = event.y() - self.start.y()
                self.start = QPoint(event.x(), event.y())
                if self.cursor().shape() != Qt.CursorShape.CrossCursor:
                    instance = self.label["instances"][self.index]
                    if instance.bboxes is not None:  # 移动矩形框
                        self.getLabelSizeInstance(instance)
                        if self.cursor().shape() == Qt.CursorShape.SizeVerCursor:
                            instance._bboxes.translateVer(0, self.ori_y, event.y())  # 上下移动
                        elif self.cursor().shape() == Qt.CursorShape.SizeHorCursor:
                            instance._bboxes.translateHor(0, self.ori_x, event.x())  # 左右移动
                        elif self.cursor().shape() == Qt.CursorShape.SizeFDiagCursor or self.cursor().shape() == Qt.CursorShape.SizeBDiagCursor:
                            instance._bboxes.translatePoint(0, self.ori_x, self.ori_y, event.x(), event.y())
                        elif self.cursor().shape() == Qt.CursorShape.SizeAllCursor:
                            instance._bboxes.moveBox(0, event.x(), event.y())
                        self.getPixSizeInstance(instance)
                        self.label["instances"].setItem(self.index, instance)
                        self.Change_Label_Signal.emit()
        self.update()

    def addBox(self, box, cls:int):
        """添加矩形标签"""
        if isinstance(box, list):
            assert len(box) == 4
            box = np.array(box,dtype=np.float32)
        elif isinstance(box, QRect):
            box = qrect2box(box, self.label["instances"]._bboxes.format if self.label["instances"]._bboxes is not None else "xywh")
        box = box if box.ndim==3 else box[None]
        if self.label["instances"]._bboxes is not None:
            cat = getcat(box)
            self.label["instances"]._bboxes.bboxes = cat(
                (self.label["instances"]._bboxes.bboxes, box), 0) if len(self.label["instances"]._bboxes) else box
        else:
            self.label["instances"]._bboxes = QBboxes(box,format="xywh")
        self.label["cls"].append(cls)
        self.Change_Label_Signal.emit()

    def removeBox(self, index):
        cat = getcat(self.label["instances"].bboxes)
        self.label["instances"]._bboxes.bboxes = cat(
            (self.label["instances"].bboxes[:index], self.label["instances"].bboxes[index+1 :]), 0)
        self.label["cls"].pop(index)

    def clearLabel(self):
        self.label["instances"].clear()
        self.label["cls"].clear()
        self.index = -1
        self.point_ind1 = -1
        self.point_ind2 = -1


    def setInstanceCursor(self, instance, pos):
        """
        设置光标落在标签上的样式
        Args:
            instance(QInstances):标签实例
            pos(QPoint): 光标位置"""
        rect = box2qrect(instance.bboxes, instance._bboxes.format)
        r = min(self.width(), self.height()) / 40
        point_lu = QPoint(rect.x(), rect.y())  # 左上角 0
        point_ru = QPoint(rect.x() + rect.width(), rect.y())  # 右上角 1
        point_ld = QPoint(rect.x(), rect.y() + rect.height())  # 左下角 2
        point_rd = QPoint(rect.x() + rect.width(), rect.y() + rect.height())  # 右下角 3
        point_c = rect.center()  # 中心点 4
        if judgePointUpLine(point_lu, point_ru, pos, r):
            self.point_ind1 = 0
            self.point_ind2 = 1
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif judgePointUpLine(point_ld, point_rd, pos, r):
            self.point_ind1 = 2
            self.point_ind2 = 3
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif judgePointUpLine(point_lu, point_ld, pos, r):
            self.point_ind1 = 0
            self.point_ind2 = 2
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif judgePointUpLine(point_ru, point_rd, pos, r):
            self.point_ind1 = 1
            self.point_ind2 = 3
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif judgePointInCircle(point_lu, pos, r):
            self.point_ind1 = 0
            self.point_ind2 = 0
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif judgePointInCircle(point_rd, pos, r):
            self.point_ind1 = 3
            self.point_ind2 = 3
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif judgePointInCircle(point_ru, pos, r):
            self.point_ind1 = 1
            self.point_ind2 = 1
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif judgePointInCircle(point_ld, pos, r):
            self.point_ind1 = 2
            self.point_ind2 = 2
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif judgePointInCircle(point_c, pos, r):
            self.point_ind1 = 4
            self.point_ind2 = 4
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def keyPressEvent(self, ev:QKeyEvent) -> None:
        super().keyPressEvent(ev)
        if self.pix == None or self.fastCreate or self.resizing(ev):
            return
        if ev.text() == "\r" and self.paint and not self.painting:  #绘制下一个标签
            self.index = len(self.label["instances"]._bboxes)   #下一个标签索引，未添加
            self.painting = True
            self.update()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self.painting and len(self.label["instances"]._bboxes) - 1 == self.index and self.label["instances"]._bboxes.areas()[-1] > 10 and ev.button() == Qt.MouseButton.LeftButton:
            self.painting = False

    def cancelPaint(self):
        self.painting = False
        self.Change_Label_Signal.emit()
        self.update()

    def contextMenuEvent(self, ev:QContextMenuEvent) -> None:
        if self.painting:
            self.cancelPaint()
            return
        if self.cursor().shape() == Qt.CursorShape.CrossCursor:
            super().contextMenuEvent(ev)
        else:
            main_menu = QMenu(self)
            main_menu.setObjectName("right_menu")
            delete_a = main_menu.addAction("删除")
            req = main_menu.exec_(self.mapToGlobal(ev.pos()))
            if req == delete_a:
                self.removeBox(self.index)
                self.Change_Label_Signal.emit()

class SegmentTransformerLabel(QFastSelectLabel):
    """分割标注label"""
    Fast_Sel_Cop_Signal = Signal(list)

    def __init__(self, parent):
        super().__init__(parent)
        self.bounding_rect = False
        self.task = "segment"
        self.show_area = False

        self.mask = None  # 掩膜
        self.mask_pixmap = None

        # 画笔工具
        self.use_pen = True
        self.use_pencil = False
        self.pencil_mode = "add"  #add or sub
        self.line_sp = [0, 0]  #line start
        self.line_ep = [0, 0]  #line end
        self.line_width = 5

    def load_image(self, image: Union[str, Path], label=None):
        super().load_image(image,label)
        if self.use_pencil:
            self.mask = self.polygonsToMask(self.label["instances"].segments, self.label["cls"])
            self.mask, self.mask_pixmap = self.setOpencvMask(self.mask)


    def drawLabel(self, painter, label, pred=False):
        """
            绘制分割多边形
        Args:
            painter（QPainter）:画板上的画笔类
            instance(QInstances): 标签实例
            pred(bool): 是否为预测标签
        """
        super().drawLabel(painter, label, pred)
        instance = label["instances"]
        classes = label["cls"]
        areas = instance.segments_area(self.pix.width(), self.pix.height())
        if self.use_pencil:
            mask = self.mask.copy()
            if self.fast_segment is not None and len(self.fast_segment):
                color = self.getPencilColor(self.cls)
                fast_segment = copy.deepcopy(self.fast_segment)
                cv2.fillPoly(mask,[fast_segment.astype(np.int32)], color)
            _, mask_pixmap = self.setOpencvMask(mask)
            painter.drawPixmap(self.image_rect,mask_pixmap)
            if not self.fast_rect:
                painter.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.SolidLine))
                color = QColor(self.colors[self.cls][0], self.colors[self.cls][1], self.colors[self.cls][2])
                color.setAlpha(100)
                painter.setBrush(QBrush(color,Qt.BrushStyle.SolidPattern))
                h, w = self.img.shape[:2]
                scale = self.image_rect.width() / w
                painter.drawEllipse(self.mouse_point, self.line_width*scale/2, self.line_width*scale/2)

        if pred or self.use_pen:
            segments = instance.segments  #label size  shape： list(n, [mi, 2])   xyxyxyxyxy....
            if not segments:
                return

            brush = QBrush(Qt.BrushStyle.SolidPattern)
            for i, (segment, area, c) in enumerate(zip(segments, areas, classes)):
                color = QColor(self.colors[c][0], self.colors[c][1], self.colors[c][2])
                color.setAlpha(100)
                brush.setColor(color if not pred else self.red)
                painter.setBrush(brush)
                points = []
                for p in segment:
                    points.append(QPoint(int(p[0]), int(p[1])))
                if len(segment) < 3:  # 小于3个点 只绘制点
                    painter.setPen(QPen(Qt.GlobalColor.green if not pred else self.red, 1, Qt.PenStyle.SolidLine))
                    painter.drawPoints(points)
                    continue
                painter.setPen(QPen(Qt.GlobalColor.green if not pred else self.red, 1, Qt.PenStyle.SolidLine))
                painter.drawPolygon(points, FILL_RULE)
                if self.show_area or self.show_cls:
                    lu, rd = get_segment_diagnol_point(segment)
                    mes = f"{self.label['names'][int(c)]}  " * self.show_cls + f"{area:3.2f}px" * self.show_area
                    if pred:
                        self.drawText(painter, QPoint(int(rd[0]), int(rd[1])),  mes + f" {self.pred_label['conf'][i]:3.2f}", 12, Qt.GlobalColor.white)
                    else:
                        self.drawText(painter, QPoint(int(lu[0]), int(lu[1])),mes, 12, Qt.GlobalColor.green)



    def setOpencvMask(self, cv_mask: np.ndarray):
        """将OpenCV图像设置为掩膜"""
        if cv_mask.ndim == 2:
            cv2.merge([cv_mask, cv_mask, cv_mask], cv_mask)
        if cv_mask.shape[2] == 3:
            b, g, r = cv2.split(cv_mask)
            k = ((b==g) & (g==r)) & (b==r)
            a = np.where(k.astype(np.uint8)==1,0,100).astype(np.uint8)
            cv_mask = cv2.merge([b, g, r, a])
        qimage = QImage(
            cv_mask.data,
            cv_mask.shape[1],
            cv_mask.shape[0],
            cv_mask.strides[0],
            QImage.Format_ARGB32
        )
        return cv_mask[:,:,0:3].copy(), QPixmap.fromImage(qimage)


    def getOpencvMask(self) -> np.ndarray:
        """获取当前掩膜的OpenCV格式（BGRA四通道）"""
        qimage = self.mask_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape(
            qimage.height(),
            qimage.width(),
            4
        )  # shape (H, W, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)  # 转换为OpenCV的BGRA格式


    def polygonsToMask(self, segments, classes):
        h,w = self.img.shape[:2]
        mask = np.zeros((h,w,3), dtype=np.uint8)
        for seg, cls in zip(segments, classes):
            color = copy.copy(self.colors[cls])
            color.reverse()
            cv2.fillPoly(mask, [seg.astype(np.int32)], color)
        return mask

    def maskToPolygons(self,mask):
        """将指定种类的掩膜生成为多边形数据
        Args:
            mask(np.array): 铅笔绘制掩膜
        Returns:
            classes（lsit）:新种类列表
            segments（list）：新分割标签"""
        classes = []
        segments = []
        for cls,color in enumerate(self.colors):
            color = copy.copy(color)
            color.reverse()
            cls_mask = cv2.inRange(self.mask[:,:,0:3],np.array(color), np.array(color))
            c = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            c = [mc.squeeze(1).astype(np.float32) for mc in c if len(mc) > 2]
            if len(c):
                segments = segments + c
            classes = classes + [cls]*len(c)
        return classes, segments

    def saveMask(self):
        self.label["cls"], self.label["instances"].segments = self.maskToPolygons(self.mask)
        self.Change_Label_Signal.emit()

    def getPencilColor(self):
        if self.pencil_mode == "add":
            color = copy.copy(self.colors[self.cls])
            color.reverse()
        else:
            color = [0,0,0]
        return color


    def openPencil(self, mode="add"):
        """打开铅笔绘制
        Args:
            mode(str):铅笔绘制模式 绘制add, 擦除sub
        """
        self.use_pen = False
        self.use_pencil = True
        self.mask = self.polygonsToMask(self.label["instances"].segments, self.label["cls"])
        self.mask, self.mask_pixmap = self.setOpencvMask(self.mask)
        self.pencil_mode = mode
        self.update()

    def openPen(self):
        """打开钢笔绘制"""
        self.use_pen = True
        self.use_pencil = False
        self.saveMask()
        self.update()

    def pencilPaint(self, pos):
        """铅笔绘制"""
        self.line_ep = self.getPixSizePoint(pos.x(), pos.y())
        color = self.getPencilColor()
        cv2.line(self.mask, self.line_sp.astype(np.int32), self.line_ep.astype(np.int32), color, self.line_width)
        self.mask, self.mask_pixmap = self.setOpencvMask(self.mask)
        self.line_sp = self.line_ep.astype(np.int32).copy()
        self.saveMask()
        self.update()


    def removePoint(self, seg_i, point_i):
        """删除指定点
        Args:
            seg_i(int):指定分割标签索引
            point_i(int): 指定分割标签中点索引"""
        self.label["instances"].segments[seg_i] = np.concatenate((self.label["instances"].segments[seg_i][:point_i], self.label["instances"].segments[seg_i][point_i + 1:]), 0)

    def addPoint(self,seg_i, point):
        """添加点
        Args:
            seg_i(int):指定分割标签索引
            point(np.array): 需要添加的点，shape(2,)"""
        if len(self.label["instances"].segments) == seg_i:
            self.addSegment(np.array([[0,0]],dtype=np.float32), self.cls)
            self.label["instances"].segments[seg_i][0] = point
        else:
            self.label["instances"].segments[seg_i] = np.concatenate(
                (self.label["instances"].segments[seg_i], point[None]), 0)

    def setPoint(self,seg_i, point_i, point):
        """
        设置点的值
        Args:
            seg_i(int):指定分割标签索引
            point_i(int): 指定分割标签中点索引
            point(np.array): 需要重新设值的点，shape(2,)"""
        if len(self.label["instances"].segments) == seg_i:
            self.addSegment(np.array([[0,0]],dtype=np.float32), self.cls)
            point_i = -1
        self.label["instances"].segments[seg_i][point_i] = point

    def insertPoint(self, seg_i, point_i, point):
        """
        插入点
        Args:
            seg_i(int):指定分割标签索引
            point_i(int): 指定分割标签中点索引
            point(np.array): 要插入的点，shape(2,)"""
        segment = self.label["instances"].segments[seg_i].tolist()
        segment.insert(point_i, point)
        self.label["instances"].segments[seg_i] = np.array(segment, dtype=np.float32)

    def addSegment(self,segment, cls):
        if self.use_pencil:
            color = self.getPencilColor(cls)
            cv2.fillPoly(self.mask, [segment.astype(np.int32)], color)
            self.mask, self.mask_pixmap = self.setOpencvMask(self.mask)
            self.saveMask()
        else:
            if self.label["instances"].segments is None:
                self.label["instances"].segments = []
            self.label["instances"].segments.append(segment)
            if len(self.label["cls"]) != len(self.label["instances"].segments):
                self.label["cls"].append(cls)
            if len(segment) > 2:
                self.Change_Label_Signal.emit()



    def removeSegment(self, seg):
        """移除某个分割实例，铅笔绘制时，传入分割数据；钢笔绘制时，传入分割实例索引"""
        if self.use_pencil:
            cv2.fillPoly(self.mask, [seg.astype(np.int32)], [0,0,0])
            self.mask, self.mask_pixmap = self.setOpencvMask(self.mask)
            self.saveMask()
        else:
            self.label["instances"].segments.pop(seg)
            self.label["cls"].pop(seg)
            self.label["instances"].getBoundingRect()
            self.Change_Label_Signal.emit()
            self.update()

    def clearLabel(self):
        self.label["instances"].segments.clear()
        self.label["cls"].clear()
        self.label["instances"].getBoundingRect()
        self.index = -1
        self.point_ind1 = -1
        self.point_ind2 = -1
        self.Change_Label_Signal.emit()


    def mousePressEvent(self, event:QMouseEvent):
        super().mousePressEvent(event)
        if self.pix == None or self.fastCreate or self.resizing(event):
            return
        if event.button() == Qt.MouseButton.LeftButton and self.paint:
            point  = self.getPixSizePoint(event.x(), event.y())
            if self.use_pencil:    #铅笔绘制
                self.line_sp = point
                self.line_ep = point
                self.pencilPaint(event.pos())
            elif self.painting:  #下一个点
                self.setPoint(self.index, -1, point)
                self.addPoint(self.index, point)
            elif self.cursor().shape() == Qt.CursorShape.PointingHandCursor:  #插入点
                self.insertPoint(self.index, self.point_ind2, point)
                self.Change_Label_Signal.emit()
            self.update()


    def setInstanceCursor(self, instance, pos):
        """
        设置光标落在标签上的样式
        Args:
            instance(QInstances):标签实例
            pos(QPoint): 光标位置"""
        if self.use_pencil:
            return
        segment = points2qpolygon(instance.segments[0])
        r = min(self.width(), self.height()) / 50
        for i in range(segment.count()):
            p = segment[i]
            next_i = i + 1 if i < segment.count() - 1 else 0
            next_p = segment[next_i]
            if judgePointUpLine(p, next_p, pos, r):
                self.point_ind1 = i
                self.point_ind2 = next_i
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                break
            elif judgePointInCircle(p, pos, r):
                self.point_ind1 = i
                self.point_ind2 = i
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                break
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if self.pix == None or self.fastCreate or self.resizing(event):
            return
        if self.paint:
            if self.use_pencil and event.buttons() == Qt.MouseButton.LeftButton:  # 铅笔绘制
                self.pencilPaint(event.pos())

            elif self.painting:  # 钢笔绘制中
                point = self.getPixSizePoint(event.x(), event.y())
                if event.buttons() == Qt.MouseButton.LeftButton:
                    self.addPoint(self.index, point)
                else:
                    self.setPoint(self.index, -1, point)
                self.update()
            else:
                if event.buttons() == Qt.MouseButton.LeftButton and event.modifiers() != Qt.KeyboardModifier.ControlModifier:
                    tx = event.x() - self.start.x()  # 鼠标位置移动X距离
                    ty = event.y() - self.start.y()  # 鼠标位置移动Y距离
                    self.start = QPoint(event.x(), event.y())
                    if self.cursor().shape() != Qt.CursorShape.CrossCursor:
                        instance = self.label["instances"][self.index]
                        if instance.segments is not None:
                            self.getLabelSizeInstance(instance)
                            if self.cursor().shape() == Qt.CursorShape.SizeAllCursor:  # 移动选中的点
                                instance.segments[0][self.point_ind1, 0] += tx
                                instance.segments[0][self.point_ind1, 1] += ty
                            self.getPixSizeInstance(instance)
                            self.label["instances"].setItem(self.index, instance)
                            self.Change_Label_Signal.emit()
            self.update()


    def mouseReleaseEvent(self, ev:QMouseEvent):
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev:QKeyEvent):
        super().keyPressEvent(ev)
        if self.pix == None or self.fastCreate or self.resizing(ev):
            return
        if ev.text() == "\r" and self.paint  and self.use_pen and not self.painting:  #绘制下一个标签
            self.addSegment(np.array([[0,0]], dtype=np.float32), self.cls)
            self.index = len(self.label["instances"].segments) - 1
            self.painting = True
        if ev.text() == "\x1b" and self.use_pen and self.painting and self.paint:
            self.painting = False
            self.removeSegment(-1)
        self.update()

    def cancelPaint(self):
        self.painting = False
        self.removePoint(self.index, len(self.label["instances"].segments[self.index]) - 1)
        self.label["instances"].remove_same_point_segments()
        if len(self.label["instances"].segments[self.index]) < 3: #小于3个点
            self.removeSegment(-1)
            self.update()
            return
        self.label["instances"].getBoundingRect()
        self.Change_Label_Signal.emit()
        self.update()

    def contextMenuEvent(self, ev: QContextMenuEvent) -> None:
        if self.painting:   #右键结束标签绘制
            self.cancelPaint()
            return
        if self.cursor().shape() == Qt.CursorShape.SizeAllCursor:
            main_menu = QMenu(self)
            main_menu.setObjectName("right_menu")
            delete_point_a = QAction(text="删除点", parent=main_menu)
            delete_seg_a = QAction(text="删除标签", parent=main_menu)
            main_menu.addActions([delete_point_a, delete_seg_a])
            req = main_menu.exec_(self.mapToGlobal(ev.pos()))
            if req == delete_point_a:
                self.removePoint(self.index, self.point_ind1)
                self.Change_Label_Signal.emit()
            elif req == delete_seg_a:
                self.removeSegment(self.index)
                self.Change_Label_Signal.emit()
            self.update()
        else:
            super().contextMenuEvent(ev)

class KeypointsTransformerLabel(QTransformerLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.task = "keypoint"

    """关键点标注label"""
    def drawLabel(self, painter, label, pred=False):
        """
            绘制关键点
        Args:
            painter（QPainter）:画板上的画笔类
            instance(QInstances): 标签实例
        """
        instance = label["instances"]
        if instance.keypoints is None: return
        keypoints = instance.keypoints  #(n,m,d) xyv xyv xyv...
        cls = label["cls"]
        s = [1] * (max(cls)+ 1)   #色调权重
        for i, (keypoint, c) in enumerate(zip(keypoints, cls)):
            points = []
            pL = self.point_ind1 if self.painting and self.index == i else len(keypoint)
            for pi in range(pL):
                p = keypoint[pi]
                if isinstance(p, torch.Tensor):
                    p = p.tolist()
                painter.setFont(QFont("宋体",15))
                painter.drawText(QPoint(p[0]-8, p[1]-5), f"{pi}:{int(p[2])}" if self.label['ndim'] ==3 else f"{pi}")
                points.append(QPoint(p[0], p[1]))
            color = QColor(self.colors[int(c)][0], self.colors[int(c)][1], self.colors[int(c)][2])
            color.setHsv(color.hue(), min(color.saturation() * s[c]/4 + 40,250), color.value(), 255)
            s[c] += 1
            painter.setPen(QPen(color if not pred else self.red, 10, Qt.PenStyle.SolidLine))
            painter.drawPoints(points)


    def addKeypoints(self, keypoints, cls):
        """
        增加一组关键点
        Args:
            keypoints(torch.Tensor): shape(1, nkpt, ndim）
            cls(int): 种类
        """
        cat = getcat(self.label["instances"].keypoints)
        self.label["instances"].keypoints = cat((self.label["instances"].keypoints, keypoints), 0) if self.label["instances"].keypoints is not None else keypoints
        self.label["cls"].append(cls)

    def setKeypoint(self, key_i, point_i, point):
        """
        设置第key_i组关键点的第point_i个关键点为p
        """
        if point_i == 5: return
        p = self.getPixSizePoint(point.x(), point.y())
        self.label["instances"].keypoints[key_i, point_i, 0] = p[0].item()
        self.label["instances"].keypoints[key_i, point_i, 1] = p[1].item()

    def removeKeypoints(self, key_i):
        """移除第key_i组关键点"""
        cat = getcat(self.label["instances"].keypoints)
        self.label["instances"].keypoints = cat((self.label["instances"].keypoints[:key_i], self.label["instances"].keypoints[key_i+1 :]), 0)
        self.label["cls"].pop(key_i)

    def clearLabel(self):
        self.label["instances"].keypoints = None
        self.label["cls"] = []
        self.index = -1
        self.point_ind1 = -1
        self.point_ind2 = -1

    def setInstanceCursor(self, instance, pos):
        """
        设置光标落在标签上的样式
        Args:
            instance(QInstances):标签实例
            pos(QPoint): 光标位置"""
        if instance.keypoints is None: return
        keypoint = instance.keypoints
        r = min(self.width(), self.height()) / 20
        for i, p in enumerate(keypoint):
            if judgePointInCircle(QPoint(p[0].item(), p[1].item()), pos, r):
                self.point_ind1 = i
                self.point_ind2 = i
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                break
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if self.pix == None or self.resizing(event):
            return
        if self.painting:
            if event.button() == Qt.MouseButton.LeftButton:
                self.point_ind1 += 1
            elif event.modifiers() == Qt.Key.Key_Alt and event.buttons() == Qt.MouseButton.LeftButton and self.label["ndim"] == 3:
                self.label["instances"].keypoints[self.index, self.point_ind1,2] = 1
                self.point_ind1 += 1
            if self.point_ind1 == self.label["nkpt"]:
                self.painting = False
            self.Change_Label_Signal.emit()


    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        super().mouseMoveEvent(event)
        if self.pix == None or self.resizing(event):
            return
        if self.painting:
            self.setKeypoint(self.index, self.point_ind1, self.mouse_point)

        if self.paint and not self.painting:
            if event.buttons() == Qt.MouseButton.LeftButton and event.modifiers() != Qt.KeyboardModifier.ControlModifier:
                if self.cursor().shape() != Qt.CursorShape.CrossCursor:
                    if self.label["instances"].keypoints is not None:
                        if self.cursor().shape() == Qt.CursorShape.SizeAllCursor:
                            self.setKeypoint(self.index, self.point_ind1, event.pos())
                            self.Change_Label_Signal.emit()
        self.update()

    def keyPressEvent(self, ev:QKeyEvent):
        super().keyPressEvent(ev)
        if self.pix == None or self.resizing(ev):
            return
        if ev.text() == "\r" and self.paint and not self.painting:  # 绘制下一个标签
            keypoints = torch.zeros((1, self.label["nkpt"], self.label["ndim"]), dtype=torch.float32)
            self.addKeypoints(keypoints, self.cls)
            self.point_ind1 = self.point_ind2 = 0
            self.index = len(self.label["instances"].keypoints) - 1
            self.setKeypoint(self.index, self.point_ind1, self.mouse_point)
            self.painting = True

    def contextMenuEvent(self, ev: QContextMenuEvent) -> None:
        if self.painting:   #右键结束标签绘制
            self.painting = False
            self.removeKeypoints(self.index)
            self.Change_Label_Signal.emit()
            self.update()
            return
        if self.cursor().shape() == Qt.CursorShape.SizeAllCursor:
            main_menu = QMenu(self)
            main_menu.setObjectName("right_menu")
            visible_a = QAction(text="可见", parent=main_menu)
            visible_a.setCheckable(True)
            visible_a.setChecked(bool(self.label["instances"].keypoints[self.index, self.point_ind1, 2]) if len(self.label["instances"].keypoints) else False)
            delete_seg_a = QAction(text="删除标签", parent=main_menu)
            main_menu.addAction(visible_a) if self.label["ndim"] == 3 else None
            main_menu.addAction(delete_seg_a)
            req = main_menu.exec_(self.mapToGlobal(ev.pos()))
            if req == visible_a:
                self.label["instances"].keypoints[self.index, self.point_ind1, 2] = int(visible_a.isChecked())
                self.Change_Label_Signal.emit()
            elif req == delete_seg_a:
                self.removeKeypoints(self.index)
                self.Change_Label_Signal.emit()
            self.update()
        else:
            super().contextMenuEvent(ev)

class ObbTransformerLabel(QTransformerLabel):
    """定向框标注label"""
    def __init__(self, parent):
        super().__init__(parent)
        self.task = "obb"


    def drawLabel(self, painter, label, pred=False):
        """
        绘制矩形定向框
        Args:
            painter（QPainter）:画板上的画笔类
            instance(QInstances): 标签实例
            cls(list | torch.Tensor): 目标对应的种类，一一对应，len(n)
        """
        instance = label["instances"]
        obbs = instance.segments   #(n,m,8) xyxyxyxy
        cls = self.label["cls"]

        for i,(obb, c) in enumerate(zip(obbs, cls)):
            points = []
            for i in range(len(obb)):
                points.append(QPoint(obb[i, 0].item(), obb[i, 1].item()))
            if len(points) == 4:
                painter.setPen(QPen(Qt.GlobalColor.green if not pred else self.red, 2, Qt.PenStyle.SolidLine))
                color = QColor(self.colors[int(c)][0], self.colors[int(c)][1], self.colors[int(c)][2])
                painter.setBrush(QBrush(color if not pred else self.red, Qt.BrushStyle.SolidPattern))
                painter.drawPolygon(points, FILL_RULE)
                c_point = self.getObbCenter(obb, True)
                r_point = self.getObbRotate(obb, True)
                points.append(c_point)
                points.append(r_point)
            painter.setPen(QPen(Qt.GlobalColor.green if not pred else self.red, 10, Qt.PenStyle.SolidLine))
            painter.drawPoints(points)

    def getObbCenter(self, obb, qpoint=False):
        """获取定向框的中心点"""
        line02 = LineTool(obb[0, 0], obb[0, 1], obb[2, 0], obb[2, 1])
        line13 = LineTool(obb[1, 0], obb[1, 1], obb[3, 0], obb[3, 1])
        c = line02.insertLineAndLine(line13)
        return c if not qpoint else QPoint(int(c[0]), int(c[1]))

    def getObbRotate(self, obb, qpoint=False):
        """获取定向框旋转点"""
        r = ((obb[2, 0] + obb[3, 0]) / 2, (obb[2, 1] + obb[3, 1]) / 2)  # 旋转点
        return r if not qpoint else QPoint(int(r[0]), int(r[1]))



    def setInstanceCursor(self, instance, pos):
        """
        设置光标落在标签上的样式
        Args:
            instance(QInstances):单个标签实例
            pos(QPoint): 光标位置"""
        obb = instance.segments[0]  #xy xy xy xy
        r = min(self.width(), self.height()) / 15
        r_point = self.getObbRotate(obb, True)
        c = self.getObbCenter(obb)
        c_point = QPoint(c[0], c[1])   #中心点
        if judgePointInCircle(r_point, pos, r):   #旋转点
            self.point_ind1 = 5
            self.point_ind2 = 5
            self.setCursor(Qt.CursorShape.WaitCursor)
        elif judgePointInCircle(c_point, pos, r):  #中心点
            self.point_ind1 = 4
            self.point_ind2 = 4
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            for i in range(4):
                p = QPoint(obb[i,0].item(), obb[i,1].item())
                if judgePointInCircle(p, pos, r):
                    self.point_ind1 = i
                    self.point_ind2 = i
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    break
                else:
                    self.setCursor(Qt.CursorShape.CrossCursor)

    def addObb(self, obb, cls):
        self.label["instances"].segments.append(obb)
        if len(self.label["instances"].segments) != len(self.label["cls"]):
            self.label["cls"].append(cls)

    def removeObb(self, obb_i):
        self.label["instances"].segments.pop(obb_i)
        self.label["cls"].pop(obb_i)

    def setPoint(self, obb_i, point_i, point):
        self.label["instances"].segments[obb_i][point_i] = point

    def addPoint(self, obb_i, point):
        self.label["instances"].segments[obb_i] = np.concatenate(
            (self.label["instances"].segments[obb_i], point[None]), 0)

    def movePoint(self,obb_i, point_i, point):
        """移动某个点，相邻两个点一起被带动，其位置由线与线的交点获得"""
        p = self.getPixSizePoint(point.x(), point.y())
        obb = self.label["instances"].segments[obb_i]
        #原先定向框的4条边线
        line01 = LineTool(obb[0,0], obb[0,1], obb[1,0], obb[1,1])
        line12 = LineTool(obb[1,0], obb[1,1], obb[2,0], obb[2,1])
        line23 = LineTool(obb[2,0], obb[2,1], obb[3,0], obb[3,1])
        line30 = LineTool(obb[3,0], obb[3,1], obb[0,0], obb[0,1])
        lines = [line01, line12, line23, line30]
        last_i = 3 if point_i==0 else point_i - 1   #移动点的上一个点
        next_i = 0 if point_i==3 else point_i + 1   #移动点的下一个点
        line_last = LineTool(p[0], p[1], lines[point_i-1].k)  #移动点前一条直线， 与原直线平行
        line_next = LineTool(p[0], p[1], lines[point_i].k)    #移动点后一条直线， 与原直线平行
        last_p = np.array(line_last.insertLineAndLine(lines[point_i -2]),dtype=np.float32)
        next_p = np.array(line_next.insertLineAndLine(lines[next_i]), dtype=np.float32)
        flag = [0 < pp[0] < self.pix.width() and 0 < pp[1] < self.pix.height() for pp in [last_p,next_p,p]]
        if all(flag):
            self.label["instances"].segments[obb_i][last_i] = last_p # 线与线交点
            self.label["instances"].segments[obb_i][next_i] =next_p
            self.label["instances"].segments[obb_i][point_i] = np.array(p, dtype=np.float32)

    def rotateObb(self, obb_i,pos):
        obb = self.label["instances"].segments[obb_i]
        p = self.getPixSizePoint(pos.x(), pos.y())                #光标位置
        c = self.getObbCenter(obb)                                #中心点
        r = self.getObbRotate(obb, False)                         #旋转点
        line_pc = LineTool(p[0], p[1], c[0], c[1])
        line_rc = LineTool(r[0], r[1], c[0], c[1])
        d_angle = math.atan(line_pc.k) - math.atan(line_rc.k)  #旋转角度
        p0 = np.array(P2P_rot_angle(obb[0,0], obb[0,1], c[0], c[1],d_angle), dtype=np.float32)
        p1 = np.array(P2P_rot_angle(obb[1,0], obb[1,1], c[0], c[1],d_angle), dtype=np.float32)
        p2 = np.array(P2P_rot_angle(obb[2,0], obb[2,1], c[0], c[1],d_angle), dtype=np.float32)
        p3 = np.array(P2P_rot_angle(obb[3,0], obb[3,1], c[0], c[1],d_angle), dtype=np.float32)
        flag = [0<p[0]<self.pix.width() and 0<p[1]<self.pix.height() for p in [p0,p1,p2,p3]]
        if all(flag):
            for i, p in enumerate([p0, p1, p2, p3]):
                self.label["instances"].segments[obb_i][i] = p

    def getPoint23(self, line01,pos):
        """
        已知定向框的边01和23的直线，和点0，1，求点2，3
        Args:
            line01(LineTool): 定向框的第一条边线, 起始点为0点，结束点为1点
            pos(tuple | list): 光标位置
        """
        line12 = LineTool(line01.x2, line01.y2, -1/line01.k)
        line23 = LineTool(pos[0], pos[1], line01.k)
        line30 = LineTool(line01.x1, line01.y1, -1/line01.k)
        p2 = line23.insertLineAndLine(line12)
        p3 = line23.insertLineAndLine(line30)
        return p2,p3


    def setPoint23(self, obb_i, pos):
        obb = self.label["instances"].segments[obb_i]
        line01 = LineTool(obb[0,0], obb[0,1], obb[1,0], obb[1,1])
        p = self.getPixSizePoint(pos.x(), pos.y())
        p2, p3 = self.getPoint23(line01,  p)
        L = len(self.label["instances"].segments[obb_i])
        flag = [0<pp[0]<self.pix.width() and 0<pp[1]<self.pix.height() for pp in [p2,p3]]
        if all(flag):
            if L == 2:
                self.addPoint(obb_i, np.array(p2, dtype=np.float32))
                self.addPoint(obb_i, np.array(p3, dtype=np.float32))
            else:
                self.setPoint(obb_i, 2, np.array(p2, dtype=np.float32))
                self.setPoint(obb_i, 3, np.array(p3, dtype=np.float32))

    def moveObb(self, obb_i, last_pos, pos):
        p0 = self.getPixSizePoint(last_pos.x(), last_pos.y())
        p1 = self.getPixSizePoint(pos.x(), pos.y())
        tx = p1[0] - p0[0]
        ty = p1[1] - p0[1]
        obb = self.label["instances"].segments[obb_i]
        flag = all(obb[...,0]+tx>0) and all(obb[..., 0] + tx < self.pix.width()) and all(obb[...,1]+ty>0) and all(obb[..., 1]+ty < self.pix.height())
        if flag:
            self.label["instances"].segments[obb_i][..., 0] += tx
            self.label["instances"].segments[obb_i][..., 1] += ty


    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if self.painting:
            if self.point_ind1 == self.point_ind2 == 0:
                p = self.getPixSizePoint(event.x(), event.y())
                self.addPoint(self.index, p)
                self.point_ind1 += 1
                self.point_ind2 += 1
            elif self.point_ind1 == self.point_ind2 == 1:
                self.point_ind1 += 1
                self.point_ind2 += 1
                self.setPoint23(self.index, event.pos())
            elif self.point_ind1 == self.point_ind2 == 2:
                self.painting = False


    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if self.painting:
            if self.point_ind1 == self.point_ind2 in (0, 1):
                p = self.getPixSizePoint(event.x(), event.y())
                self.setPoint(self.index, self.point_ind1, p)
            elif self.point_ind1 == self.point_ind2 == 2:
                self.setPoint23(self.index,event.pos())
        elif self.paint and event.buttons() == Qt.MouseButton.LeftButton and event.modifiers() is not Qt.MouseButton.ControlModifier:
            if self.cursor().shape() == Qt.CursorShape.SizeAllCursor:
                if self.point_ind1 == self.point_ind2  in [0, 1, 2, 3]:
                    self.movePoint(self.index, self.point_ind1, event.pos())   #移动定向框某一点
                elif self.point_ind1 == self.point_ind2 == 4:
                    self.moveObb(self.index,self.start, event.pos()) #移动定向框
                    self.start = event.pos()
                self.Change_Label_Signal.emit()
            elif self.cursor().shape() == Qt.CursorShape.WaitCursor:
                self.rotateObb(self.index, event.pos())   #旋转定向框
                self.Change_Label_Signal.emit()
        self.update()


    def keyPressEvent(self, ev:QKeyEvent):
        super().keyPressEvent(ev)
        if ev.text() == "\r" and not self.painting:
            self.addObb(np.array([[0, 0]], dtype=np.float32), self.cls)
            self.index = len(self.label["instances"].segments) - 1
            self.point_ind1 = self.point_ind2 = 0
            self.painting = True

    def clearLabel(self):
        self.label["instances"].segments.clear()
        self.label["cls"].clear()

    def contextMenuEvent(self, ev: QContextMenuEvent):
        if self.painting:   #右键结束标签绘制
            if self.point_ind1 in [0, 1]:
                self.removeObb(self.index)
            self.painting = False
            self.Change_Label_Signal.emit()
            self.update()
            return
        if self.cursor().shape() == Qt.CursorShape.SizeAllCursor or self.cursor().shape() == Qt.CursorShape.WaitCursor:
            main_menu = QMenu(self)
            main_menu.setObjectName("right_menu")

            delete_seg_a = QAction(text="删除标签", parent=main_menu)

            main_menu.addActions([ delete_seg_a])
            req = main_menu.exec_(self.mapToGlobal(ev.pos()))
            if req == delete_seg_a:
                self.removeObb(self.index)
                self.Change_Label_Signal.emit()
            self.update()
        else:
            super().contextMenuEvent(ev)

class ClassifyTransformerLabel(QTransformerLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.task = "classify"

    """分类标注label"""
    def drawLabel(self, painter, label=None, pred=False):
        """
            绘制矩形定向框
            Args:
                painter（QPainter）:画板上的画笔类
                nstance(QInstances): None
                cls(int): 图像对应的种类
        """
        cls = label["cls"]
        if cls == -1: return
        name = self.label["names"][cls]
        painter.setPen(QPen(Qt.GlobalColor.green if not pred else self.red, 5, Qt.PenStyle.SolidLine))
        painter.setFont(QFont("宋体", min(self.width(),self.height())/20))
        ty = self.image_rect.y()+min(self.width(),self.height())/20
        painter.drawText(QPoint(self.image_rect.x(), ty if not pred else ty+min(self.width(),self.height())/20), f"{cls}:{name}")

    def keyPressEvent(self, ev:QKeyEvent):
        super().keyPressEvent(ev)
        if self.pix == None or self.resizing(ev):
            return
        if self.paint:
            if ev.text() == "\r":
                self.label["cls"] = self.cls
            elif ev.key() == Qt.Key.Key_Escape:
                self.label["cls"] = -1
            self.update()

class ConfusionMatrixLabel(QSizeLabel):
    Select_signal = Signal(str)
    def __init__(self,parent):
        super().__init__(parent)
        self.select_rect = None
        self.rects = None
        self.keys = None
        self.pred_i = -1
        self.gt_i = -1
        self.nc = 0
        self.ls = 130/600
        self.rs = 130/600
        self.ts = 40/450
        self.ds = 70/450


    def load_image(self, image,label=None):
        super().load_image(image, label)
        im_paths = Path(image).parent / "Confusion_Matrix_Imfiles.yaml"
        if im_paths.exists():
            im_paths = yaml_load(im_paths)
            self.nc = int(math.sqrt(len(im_paths.keys())))
            self.rects = [[None for i in range(self.nc)] for j in range(self.nc)]
            self.keys = [[None for i in range(self.nc)] for j in range(self.nc)]
            for key, value in im_paths.items():
                spl = key.split("$")[1].split(",")
                pred_i = int(spl[0])
                gt_i = int(spl[1])
                self.keys[pred_i][gt_i] = key
        else:
            self.nc = 0
            self.keys = None

        self.select_rect = None
        self.pred_i = -1
        self.gt_i = -1
        self.update()


    def mouseMoveEvent(self, ev:QMouseEvent):
        super().mouseMoveEvent(ev)
        self.setRectFocus(ev.pos())
        self.update()

    def setRectFocus(self, pos):
        for pred_i in range(self.nc):
            for gt_i in range(self.nc):
                if self.rects[pred_i][gt_i] and self.rects[pred_i][gt_i].contains(pos):
                    self.select_rect = self.rects[pred_i][gt_i]
                    self.pred_i = pred_i
                    self.gt_i = gt_i
                    return
        self.select_rect = None
        self.pred_i = -1
        self.gt_i = -1

    def mouseDoubleClickEvent(self, event):
        if self.pred_i == self.gt_i == -1 or self.keys is None:
            return
        self.Select_signal.emit(self.keys[self.pred_i][self.gt_i])


    def getClsRects(self):
        if self.nc == 0:
            return
        L = self.ls * self.image_rect.width()
        R = self.rs * self.image_rect.width()
        T = self.ts * self.image_rect.height()
        D = self.ds * self.image_rect.height()
        rw = (self.image_rect.width() - L - R)/self.nc  #格子宽
        rh = (self.image_rect.height() - T - D)/self.nc  #格子高
        lu_point = QPoint(int(L)+self.image_rect.x(),int(T)+self.image_rect.y()) #所有格子的左上角点

        for pred_i in range(self.nc):
            for gt_i in range(self.nc):
                self.rects[pred_i][gt_i] = QRect(lu_point.x() + rw*gt_i, lu_point.y() + rh*pred_i, rw, rh)
        
    def draw(self, painter):
        self.getClsRects()
        if self.select_rect is not None:
            painter.setPen(QPen(Qt.GlobalColor.green, 5, Qt.PenStyle.SolidLine))
            color = QColor(Qt.GlobalColor.darkBlue)
            color.setAlpha(100)
            painter.setBrush(QBrush(color,Qt.BrushStyle.SolidPattern))
            painter.drawRect(self.select_rect)
            painter.setFont(QFont("幼圆", 14))
            painter.setPen(QPen(Qt.GlobalColor.darkBlue))
            pred = self.keys[self.pred_i][self.gt_i].split("$")[0].split(",")[0]
            gt = self.keys[self.pred_i][self.gt_i].split("$")[0].split(",")[1]
            point = copy.deepcopy(self.mouse_point)
            y = point.y()
            point.setY(y - 24)
            painter.drawText(point, gt)
            point.setY(y -8)
            painter.drawText(point, pred)


class ShowLabel(QLabel):
    """显示图像供选择
    attributes:
        label_ops(LabelOps): 对显示在主窗口的图像的操作类
        im_files(List): 所有图像
        show_file(List): 显示出来的图像
        pixes(List): 图像的Qpixmap类
        selecteds(List): 只存储False/True， 指示对应图像是否被选中
        labels(List): 存储对应图像的标签数据（“im_file", "img", "names", instances", "cls")
        current_selected(int): 当前选中图像的索引（所有图像）
        dataset(str): 当前筛选的数据集的名称："总样本集", "训练集", "验证集"， "未标注集", "结果集"等
        widths(list): 存储每个图像显示出来的宽度
        ori_shapes(lsit): 存储每个图像的原尺寸（w, h）
        spacing(int): 显示图像之间的间隔
        build(bool): 是否已生成数据集
    """
    Click_Signal = Signal(dict)
    def __init__(self, parent, label_ops):
        super(ShowLabel, self).__init__(parent)
        self.setMouseTracking(True)  # 鼠标在窗口内时刻触发鼠标移动事件
        self.label_ops = label_ops
        self.im_files = []
        self.pixes = []
        self.selecteds = []
        self.current_selected = -1
        self.show_files = []
        self.dataset = ""
        self.spacing=6
        self.widths = []
        self.ori_shapes = []
        self.build = False


    def paintEvent(self, arg__1:QPaintEvent) -> None:
        ws = 0
        painter = QPainter(self)
        for show_file in self.show_files:
            all_ind = self.im_files.index(show_file)
            pix = self.pixes[all_ind]
            h = self.height()
            w = h * (self.ori_shapes[all_ind][0] / self.ori_shapes[all_ind][1])
            self.widths[all_ind] = w
            if pix is not None:
                img_rect = QRect(ws+self.spacing, 0, w, h)
                self.drawImage(painter, img_rect, pix, self.selecteds[all_ind], show_file)
            ws += w + self.spacing
        painter.end()


    def drawImage(self, painter, rect, img, selected, im_file):
        painter.drawPixmap(rect, img)
        if selected:
            color = QColor(Qt.GlobalColor.darkBlue)
            color.setAlpha(50)
            painter.setPen(QPen(Qt.GlobalColor.blue, 3, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(color, Qt.BrushStyle.SolidPattern))
            painter.drawRect(rect)
        painter.setPen(QPen(Qt.GlobalColor.green))
        painter.setFont(QFont("宋体", 20))
        painter.drawText(QPoint(rect.x(), rect.y()+20), f"{self.label_ops.judgeDataset(im_file)}:")
        painter.setFont(QFont("宋体", 20))
        painter.drawText(QPoint(rect.x()+15, rect.y()+50), f"{Path(im_file).name}")

    def loadImages(self, im_shapes):
        """加载所有图像"""
        total = len(im_shapes)
        PROGRESS_BAR.start("图像生成", "开始生成", [0 ,total], False)
        for i, (im_file, shape) in enumerate(im_shapes.items()):
            new_width = self.height() * (shape[0]/ shape[1])
            if im_file not in self.im_files:
                self.im_files.append(im_file)
                self.widths.append(new_width)
                self.selecteds.append(False)
                self.pixes.append(None)
                self.ori_shapes.append(shape)
            PROGRESS_BAR.setValue(i + 1, f"添加...{im_file}")
        PROGRESS_BAR.close()
        self.update()


    def showImages(self, show_files):
        """显示指定图像"""
        self.show_files.clear()
        for im_file in show_files:
            if im_file in self.im_files:
                self.show_files.append(im_file)
                ind = self.im_files.index(im_file)
                self.selecteds[ind] = False
        if len(self.show_files):
            all_ind = self.im_files.index(self.show_files[0])
            self.clickImage(all_ind)
        self.update()


    def loadPix(self, scoll_value, scoll_width):
        ws = 0
        show_ind = 0
        for i, show_file in enumerate(self.show_files):
            all_ind = self.im_files.index(show_file)
            w = self.widths[all_ind]
            if ws <= scoll_value <= ws+w+self.spacing:
                show_ind = i
                break
            ws = ws + w + self.spacing

        #显示图像往后n个pix
        back_num = 0
        ws = 0
        for i in reversed(range(show_ind)):
            all_ind = self.im_files.index(self.show_files[i])
            ws = ws + self.widths[all_ind] + self.spacing
            back_num += 1
            if ws > 1.2 * scoll_width:
                break
        #显示图像往后n个pix
        font_num = 0
        ws = 0
        for i in range(show_ind, len(self.show_files)):
            all_ind = self.im_files.index(self.show_files[i])
            ws = ws + self.widths[all_ind] + self.spacing
            font_num += 1
            if ws > 1.2 * scoll_width:
                break
        #删除多余pixmap 添加往下的pixmap
        for all_ind, im_file in enumerate(self.im_files):
            if im_file in self.show_files:
                s_ind = self.show_files.index(im_file)
                if show_ind-back_num <= s_ind < show_ind+font_num:
                    if self.pixes[all_ind] is None:
                        label = self.label_ops.getLabel(im_file, True)
                        if label is None:
                            self.pixes[all_ind] = self.getTinyImg(im_file)
                        else:
                            self.pixes[all_ind] = cvImg2Qpix(label["img"])
                else:
                    if self.pixes[all_ind] is not None:
                        self.pixes[all_ind] = None
        self.update()

    def getTinyImg(self, im_file):
        """获取缩略图"""
        image = QImage(im_file)
        thumbnail = image.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return QPixmap.fromImage(thumbnail)


    def getSelectedInd(self, point):
        """获取当前鼠标所在的label索引"""
        x = point.x()
        ws = 0
        for show_file in self.show_files:
            ind = self.im_files.index(show_file)
            w = self.widths[ind]
            ws += self.spacing
            if ws < x < w+ws:
                return ind
            ws += w
        return -1

    def getSelectedImgs(self):
        im_files = []
        for im_file in self.show_files:
            all_ind = self.im_files.index(im_file)
            if self.selecteds[all_ind]:
                im_files.append(im_file)
        return im_files

    def getWidth(self):
        """获取画布宽度"""
        ws = 0
        for show_file in self.show_files:
            ind = self.im_files.index(show_file)
            w = self.widths[ind]
            ws += w + self.spacing
        return ws

    def getShowLen(self):
        return len(self.show_files)

    def getShowInd(self, im_file):
        return self.show_files.index(im_file)

    def clearSelect(self):
        self.selecteds = [False for i in self.selecteds]
        self.current_selected = -1

    @threaded
    def train2Val(self, im_files):
        new_im_files = self.label_ops.train2Val(im_files)
        self.label_ops.img_label.label["dataset"] = "val"
        self.updateImagesFile(im_files, new_im_files)  # 更新label对应的图像路径
        #if self.dataset not in ("总样本集", "验证集"):         # 只有总样本集同时包含训练集、验证集、未标注集
        #    self.removeImages(new_im_files)                #移除显示列表

    @threaded
    def val2Train(self, im_files):
        new_im_files = self.label_ops.val2Train(im_files)
        self.label_ops.img_label.label["dataset"] = "train"
        self.updateImagesFile(im_files, new_im_files)
        #if self.dataset not in ("总样本集", "训练集"):
        #    self.removeImages(new_im_files)

    @threaded
    def toNolabel(self, im_files):
        new_im_files = self.label_ops.toNoLabel(im_files)
        self.label_ops.img_label.label["dataset"] = "no_label"
        self.updateImagesFile(im_files, new_im_files)
        #if self.dataset not in ("总样本集", "未标注集"):
        #    self.removeImages(new_im_files)

    @threaded
    def nolabel2Train(self, im_files):
        new_im_files = self.label_ops.nolabel2Train(im_files)
        self.label_ops.img_label.label["dataset"] = "train"
        self.updateImagesFile(im_files, new_im_files)
        #if self.dataset not in ("总样本集", "训练集"):
        #    self.removeImages(new_im_files)
    
    @threaded
    def nolabel2Val(self, im_files):
        new_im_files = self.label_ops.nolabel2Val(im_files)
        self.label_ops.img_label.label["dataset"] = "val"
        self.updateImagesFile(im_files, new_im_files)
        #if self.dataset not in ("总样本集", "验证集"):
        #    self.removeImages(new_im_files)

    @threaded
    def deleteImages(self, im_files):
        """删除指定的图像"""
        im_files = format_im_files(im_files)
        ind = -1
        self.label_ops.deleteSamples(im_files)
        for im_file in im_files:
            all_ind = self.im_files.index(im_file)
            self.im_files.pop(all_ind)
            self.selecteds.pop(all_ind)
            self.pixes.pop(all_ind)
            self.widths.pop(all_ind)
            self.ori_shapes.pop(all_ind)

            if im_file in self.show_files:
                show_ind = self.show_files.index(im_file)
                self.show_files.pop(show_ind)
                if len(self.show_files):
                    ind = show_ind if show_ind < len(self.show_files) else len(self.show_files) - 1
                else:
                    ind = -1
        if ind != -1:
            show_file = self.show_files[ind]
            all_ind = self.im_files.index(show_file)
            self.clickImage(all_ind)   #选中新的图像
        else:
            self.label_ops.showNone()   #图像集为空，不显示图像
        self.setSize()
        self.update()


    def removeImages(self, im_files):
        """将图像从显示集中移除"""
        im_files = format_im_files(im_files)
        ind =  -1
        for im_file in im_files:
            if im_file not in self.im_files:
                continue
            if im_file not in self.show_files:
                continue
            all_ind = self.im_files.index(im_file)
            show_ind = self.show_files.index(im_file)
            self.show_files.pop(show_ind)   #从显示集中移除
            self.selecteds[all_ind] = False
            self.pixes[all_ind] = None
            if len(self.show_files):
                ind = show_ind if show_ind < len(self.show_files) else len(self.show_files) - 1
            else:
                ind = -1
        if ind != -1:
            show_file = self.show_files[ind]
            all_ind = self.im_files.index(show_file)
            self.clickImage(all_ind)   #选中新的图像
        else:
            self.label_ops.showNone()   #图像集为空，不显示图像
        self.setSize()
        self.update()

    def updateImagesFile(self, im_files, new_im_files):
        """更新指定图像的文件路径和所属数据集信息"""
        im_files = format_im_files(im_files)
        new_im_files = format_im_files(new_im_files)
        for im_file, new_im_file in zip(im_files, new_im_files):
            new_im_file = str(new_im_file)
            all_ind = self.im_files.index(im_file)
            #(new_im_file, cls), nd, nc, msg, shape = verify_image(((new_im_file, 0), ""))
            #if msg != "":
            #    LOGGER.warning(f"更新{new_im_file}图像信息失败：{msg}")
            #    continue
            self.im_files[all_ind] = new_im_file
            if im_file in self.show_files:
                show_ind = self.show_files.index(im_file)
                self.show_files[show_ind] = new_im_file
            #self.ori_shapes[all_ind] = list(reversed(shape))   # h, w
            #self.widths[all_ind] = self.height() * (shape[1] / shape[0])
        self.label_ops.painter_tool.setTrainVal()
        self.update()

    def selectAllShow(self):
        """选中所有显示的图像"""
        self.selecteds = [False for i in self.im_files]
        for im_file in self.show_files:
            all_ind = self.im_files.index(im_file)
            self.selecteds[all_ind] = True
        self.update()
        
    def clickImage(self,all_ind):
        self.selecteds.clear()
        self.selecteds = [False for f in self.im_files]
        self.selecteds[all_ind] = True
        self.current_selected = all_ind
        self.Click_Signal.emit(self.label_ops.getLabel(self.im_files[all_ind], False))
        self.update()

    def setSize(self):
        w = self.getWidth()
        self.setMinimumWidth(w)
        self.setMaximumWidth(w)


    def mousePressEvent(self, ev:QMouseEvent) -> None:
        if ev.buttons() == Qt.MouseButton.LeftButton or ev.buttons() == Qt.MouseButton.RightButton:
            all_ind = self.getSelectedInd(ev.pos())
            if all_ind != -1:
                if ev.modifiers() != Qt.KeyboardModifier.ControlModifier:
                    self.clickImage(all_ind)
                else:
                    if self.selecteds[all_ind]:
                        if self.selecteds.count(True) == 1:
                            return
                        self.selecteds[all_ind] = False
                        self.current_selected = self.selecteds.index(True)
                    else:
                        self.selecteds[all_ind] = True
                        self.current_selected = all_ind
                    self.Click_Signal.emit(self.label_ops.getLabel(self.im_files[self.current_selected], False))
                    self.update()

    def mouseMoveEvent(self, ev:QMouseEvent) -> None:
        all_ind = self.getSelectedInd(ev.pos())
        im_file =self.im_files[all_ind]
        self.setToolTip(im_file)

    def keyPressEvent(self, ev):
        self.label_ops.img_label.keyPressEvent(ev)


    def contextMenuEvent(self, ev:QContextMenuEvent) -> None:
        all_ind = self.getSelectedInd(ev.pos())
        main_menu = QMenu(self)
        main_menu.setObjectName("right_menu")
        delete_a = QAction(text="删除", parent=main_menu)
        to_val_a = QAction(text="转验证集", parent=main_menu)
        to_train_a = QAction(text="转训练集", parent=main_menu)
        to_noLabel_a = QAction(text="转未标注", parent=main_menu)

        main_menu.addActions([delete_a])
        dataset = self.label_ops.judgeDataset(self.im_files[all_ind])
        if dataset == "train":
            main_menu.addActions([to_val_a,to_noLabel_a])
        elif dataset == "val":
            main_menu.addActions([to_train_a, to_noLabel_a])
        elif dataset == "no_label":
            main_menu.addActions([to_train_a, to_val_a])
        req = main_menu.exec_(self.mapToGlobal(ev.pos()))
        if req == delete_a:
            self.deleteImages(self.im_files[all_ind])
        elif req == to_val_a:
            if dataset == "train":
                self.train2Val(self.im_files[all_ind])
            else:
                self.nolabel2Val(self.im_files[all_ind])
        elif req == to_train_a:
            if dataset == "val":
                self.val2Train(self.im_files[all_ind])
            else:
                self.nolabel2Train(self.im_files[all_ind])
        elif req == to_noLabel_a:
            self.toNolabel(self.im_files[all_ind])







