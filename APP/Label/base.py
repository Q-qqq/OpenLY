import shutil
from typing import Union

import cv2

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from pathlib import Path
from APP.Utils import getcat
from ultralytics.utils.instance import Bboxes,Instances,_formats
from ultralytics.utils.ops import segments2boxes, xyxy2xywh
from ultralytics.data.utils import img2label_paths, IMG_FORMATS
import numpy as np
import copy
import torch
from APP.Label.utils import cvImg2Qpix, generate_distinct_colors, segment2Box,segmentArea, twoPoints2box




class QBboxes(Bboxes):
    def __init__(self, bboxes, format="xyxy") -> None:
        if bboxes is None:
            self.bboxes = torch.tensor([], dtype=torch.float32)
            self.format = format
        else:
            super().__init__(bboxes, format)


    def translateVer(self, box_i, ori_y, y):
        """移动上边或下边
              0-----1
              |  4  |
              2-----3"""
        x1, y1, x2, y2 = self.bboxes[box_i]
        self.setItem(box_i, x1, min(ori_y, y), x2, max(ori_y, y), "xyxy")


    def translateHor(self, box_i, ori_x, x):
        """移动左边或右边"""
        x1, y1, x2, y2 = self.bboxes[box_i]
        self.setItem(box_i, min(ori_x, x), y1, max(ori_x, x), y2, "xyxy")

    def translatePoint(self, box_i, ori_x, ori_y, x, y):
        """移动点"""
        self.setItem(box_i, min(ori_x, x), min(ori_y, y), max(ori_x, x),  max(ori_y, y), "xyxy")

    def moveBox(self,box_i, center_x, center_y):
        if self.format == "xywh":
            x, y, w, h = self.bboxes[box_i]
        else:
            x,y,w,h = xyxy2xywh(self.bboxes[box_i])
        self.setItem(box_i, center_x, center_y, w, h, "xywh")
    
    def addBox(self, box, format):
        old_format = self.format
        self.convert(format)
        cat = getcat(box)
        self.bboxes = cat(
            (self.bboxes, box), 0) if len(self.bboxes) else box
        self.convert(old_format)

    def __getitem__(self, index) -> "Bboxes":
        """
        取出一个指定的检测框的BBoxes类或者一个集合的检测框BBoxes类
        :param index（int,slice or np.ndarray）: 指定检测框的索引
        :return: 指定检测框的BBoxes类
        """
        if isinstance(index, int):
            return QBboxes(self.bboxes[index].view(1,-1))

        b = self.bboxes[index]
        assert b.ndim == 2, f"对于指定的Bboxes{index},无法返回一个正常的二维矩阵"
        return QBboxes(b)


    def setItem(self, *args):
        if len(args) == 3:
            box_i, box, format = args
            assert len(box) == 1
            old_format = self.format
            if self.format != format:
                self.convert(format)
            self.bboxes[box_i] = box[0]
            if self.format != old_format:
                self.convert(old_format)
        elif len(args) == 5:
            i, x1, y1, x2, y2 = args
            self.bboxes[i, 0] = x1
            self.bboxes[i, 1] = y1
            self.bboxes[i, 2] = x2
            self.bboxes[i, 3] = y2
        elif len(args) == 6:
            i,x1,y1,x2,y2,format = args
            if self.format == format:
                self.bboxes[i, 0] = x1
                self.bboxes[i, 1] = y1
                self.bboxes[i, 2] = x2
                self.bboxes[i, 3] = y2
            elif self.format != format:
                old_format = self.format
                if format == "xyxy":
                    self.convert("xyxy")
                elif format == "xywh":
                    self.convert("xywh")
                self.bboxes[i, 0] = x1
                self.bboxes[i, 1] = y1
                self.bboxes[i, 2] = x2
                self.bboxes[i, 3] = y2
                if self.format != old_format:
                    self.convert(old_format)


class QInstances(Instances):
    def __init__(self, bboxes=None, segments=None, keypoints=None, bbox_format="xywh", normalized=True):
        """
        :param bboxes(ndarray):  目标检测框[N,4]
        :param segments(list|ndarray): 分割坐标[N,m,2]
        :param keypoints(ndarray): 关键点坐标[N,17,3]
        :param bbox_format:目标检测框格式
        :param normalized:是否归一化
        """
        self.normalized = normalized
        self.segments = segments
        self.keypoints = keypoints
        self._bboxes = None
        if bboxes is None or not len(bboxes):
            self.getBoundingRect()
            if not self._bboxes:
                self._bboxes = QBboxes(bboxes=None, format=bbox_format)
        else:
            self._bboxes = QBboxes(bboxes=bboxes, format=bbox_format)

    def scale(self, scale_w, scale_h, bbox_only=False):
        '''对数据集进行缩放'''
        self._bboxes.mul(scale = (scale_w,scale_h,scale_w,scale_h))
        if bbox_only:
            return
        #分割数据
        if self.segments is not None:
            for i in range(len(self.segments)):
                for j in range(len(self.segments[i])):
                    self.segments[i][j, 0] *= scale_w
                    self.segments[i][j, 1] *= scale_h
        #关键点数据
        if self.keypoints is not None:
            self.keypoints[...,0] *= scale_w
            self.keypoints[...,1] *= scale_h

    def denormalize(self,w,h):
        """去除归一化"""
        if not self.normalized:
            return
        if len(self._bboxes):
            self._bboxes.mul(scale=(w, h, w, h))
        if self. segments is not None:
            for i in range(len(self.segments)):
                for j in range(len(self.segments[i])):
                    self.segments[i][j, 0] *= w
                    self.segments[i][j, 1] *= h
        if self.keypoints is not None:
            self.keypoints[...,0] *= w
            self.keypoints[...,1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """归一化"""
        if self.normalized:
            return
        if len(self._bboxes):
            self._bboxes.mul(scale=(1/w,1/h,1/w,1/h))
        if self.segments is not None:
            for i in range(len(self.segments)):
                for j in range(len(self.segments[i])):
                    self.segments[i][j, 0] /= w
                    self.segments[i][j, 1] /= h
        if self.keypoints is not None:
            self.keypoints[...,0] /= w
            self.keypoints[...,1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        assert not self.normalized,"需要在绝对坐标上进行填充"
        assert self._bboxes.format == "xyxy",f"目标检测框格式必须为xyxy"
        self._bboxes.add(offset=(padw,padh,padw,padh))
        if self.segments is not None:
            for i in range(len(self.segments)):
                for j in range(len(self.segments[i])):
                    self.segments[i][j, 0] += padw
                    self.segments[i][j, 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def setItem(self, i, instance):
        if instance.bboxes is not None:
            self._bboxes.setItem(i, instance.bboxes, instance._bboxes.format)

        if instance.segments is not None:
            self.segments[i] = instance.segments[0]

        if instance.keypoints is not None:
            if instance.keypoints.ndim == 2:
                instance.keypoints = instance.keypoints[None, :]
            self.keypoints[i] = instance.keypoints[0]

    def clip_num(self, num, mi, ma):
        """限制一个数字的上下限"""
        if num < mi:
            return mi
        elif num > ma:
            return ma
        else:
            return num

    def segments_area(self, w, h):
        old_norm = self.normalized

        area = []
        if self.segments:
            if self.normalized:
                self.denormalize(w, h)
            for segment in self.segments:
                area.append(segmentArea(segment))
        if old_norm != self.normalized:
            self.normalize(w,  h)
        return area

    def clip(self,w,h):
        """限制数据集坐标在图像范围内"""
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:,[0,2]] = self.bboxes[:,[0,2]].clip(0,w)
        self.bboxes[:,[1,3]] = self.bboxes[:,[1,3]].clip(0,h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        if self.segments is not None:
            for i in range(len(self.segments)):
                for j in range(len(self.segments[i])):
                        self.segments[i][j, 0] = self.clip_num(self.segments[i][j][0] , 0, w)
                        self.segments[i][j, 1] = self.clip_num(self.segments[i][j][1] , 0, h)
        if self.keypoints is not None:
            self.keypoints[...,0] = self.keypoints[..., 0].clip(0,w)
            self.keypoints[...,1] = self.keypoints[..., 1].clip(0,h)

    def remove_zero_area_boxes(self):
        """移除面积为0的标签"""
        self.getBoundingRect()
        if self._bboxes is not None:
            good = self.bbox_areas > 0
            if not all(good):
                self._bboxes = self._bboxes[good]
                if self.segments is not None and len(self.segments) > 0:
                    for i in range(len(good)):
                        if not good[i]:
                            if 0 < len(self.segments[i]) < 3:
                                continue
                            self.segments.pop(i)
                if self.keypoints is not None:
                    self.keypoints = self.keypoints[good]
            return good

    def remove_same_point_segments(self):
        for i in range(len(self.segments)):
            uni, ind = np.unique(self.segments[i].sum(1), 1)
            self.segments[i] = self.segments[i][np.sort(ind)]


    def getBoundingRect(self):
        if self.segments is not None:
            if len(self.segments) > 0:
                bboxes = segments2boxes(self.segments)   #numpy.array (n, 4) xywh
                self._bboxes = QBboxes(bboxes,"xywh")
            else:
                self._bboxes = QBboxes(None, "xywh")
        elif self.keypoints is not None:
            if len(self.keypoints) > 0:
                bboxes = segments2boxes(self.keypoints[..., :2])
                self._bboxes = QBboxes(bboxes, "xywh")
            else:
                self._bboxes = QBboxes(None, "xywh")


    def __len__(self):
        self.getBoundingRect()
        return len(self._bboxes) if self._bboxes is not None else 0



    def __getitem__(self, index) -> "QInstances":
        """
        取出指定的instance
        :param index:（int，slice or np.array）
        :return:
        """
        if self.segments is not None:
            segments = [self.segments[index]] if len(self.segments) else self.segments
        else:
            segments = None
        keypoints = self.keypoints[index] if self.keypoints is not None else None

        bboxes = self.bboxes[index] if self._bboxes is not None else None  #属性方法property
        bbox_format = self._bboxes.format if self._bboxes is not None else "xyxy"
        return QInstances(bboxes=bboxes,
                         segments=segments,
                         keypoints=keypoints,
                         bbox_format=bbox_format,
                         normalized=self.normalized)

    def save(self, label_file, cls, w, h):
        self.getBoundingRect()  #获取最新矩形框
        if self._bboxes and len(self._bboxes):
            self.clip(w, h)
            old_norm = self.normalized
            if not self.normalized:
                self.normalize(w, h)  # 归一化
            self._bboxes.convert("xywh")
            texts = []
            for i in range(len(self._bboxes)):
                line = (cls[i], *(
                    self.bboxes[i].view(-1) if isinstance(self.bboxes[i], torch.Tensor) else self.bboxes[i].reshape(
                        -1)))
                if self.segments:
                    line = (cls[i], *(
                        self.segments[i].view(-1) if isinstance(self.segments[i], torch.Tensor) else self.segments[
                            i].reshape(-1)))
                elif self.keypoints:
                    line = (cls[i], *(self.keypoints[i].view(-1) if isinstance(self.keypoints[i], torch.Tensor) else
                                        self.keypoints[i].reshape(-1)))
                texts.append(("%g " * len(line)).rstrip() % line)
            Path(label_file).parent.mkdir(parents=True, exist_ok=True)
            with open(label_file, "w") as f:
                f.writelines(text + "\n" for text in texts)
            if old_norm:
                self.normalize(w, h)
            else:
                self.denormalize(w, h)
        else:
            with open(label_file, "w") as f:
                f.write("")    #空白标签





class QSizeLabel(QLabel):
    def __init__(self, parent):
        """
        Attriabute:
        pix(QPixmap): 原图像
        pix_half_height(float):显示图像的高度的一半
        pix_half_weight(flaot):显示图像的宽度的一半
        pix_scale_x(float): 原图像变换至显示图像的宽度比例
        pix_scale_y(float): 原图像变换至显示图像的高度比例
        translate_x(int): 移动显示图像的x距离
        translate_y(int): 移动显示图像的y距离
        center(tuple): 显示图像中心点
        start(QPoint): 鼠标点击的开始点
        scale_left_w(flaot): 显示图像以鼠标位置为基准进行缩放后，其左边距离中心的距离
        scale_right_w(float): 显示图像以鼠标位置为基准进行缩放后，其右边距离中心的距离
        scale_down_h(float): 显示图像以鼠标位置为基准进行缩放后，其底边距离中心的距离
        scale_up_h(float): 显示图像以鼠标位置为基准进行缩放后，其顶边距离中心的距离
        image_rect(QRect): 显示图像矩形区域
        mouse_point(QPoint): 实时鼠标位置
        zoom_zone(list): 指定缩放区域[x1,y1,x2,y2]
        zoom_finish(bool): 放大区域绘制完成信号
        """
        super().__init__(parent)
        self.setMouseTracking(True)  # 鼠标在窗口内时刻触发鼠标移动事件
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setCursor(Qt.ArrowCursor)
        self.im_file = ""
        self.pix = None
        self.pix_half_height = 0
        self.pix_half_width = 0
        self.pix_scale_x = 0
        self.pix_scale_y = 0
        self.translate_x = 0
        self.translate_y = 0
        self.center = (0, 0)
        self.start = QPoint(0, 0)
        self.scale_left_w = 0
        self.scale_right_w = 0
        self.scale_down_h = 0
        self.scale_up_h = 0
        self.image_rect = None
        self.mouse_point = QPoint(0,0)
        self.zoom_zone = []
        self.zoom_finish = QRhiTextureSubresourceUploadDescription


    # region 焦点事件
    def focusInEvent(self, ev):
        self.grabKeyboard()

    def focusOutEvent(self, ev):
        self.releaseKeyboard()
    # endregion


    def setImageSize(self, width, height):
        if self.pix:
            self.center = (width / 2, height / 2)  # 图像中心点
            s = min(width / self.pix.width(), height / self.pix.height())
            self.scale_left_w = (self.pix.width() * s) / 2  # 图像中心左侧宽度
            self.scale_right_w = (self.pix.width() * s) / 2  # 图像中心右侧宽度
            self.scale_down_h = (self.pix.height() * s) / 2  # 图像中心下侧高度
            self.scale_up_h = (self.pix.height() * s) / 2  # 图像中心上侧高度
            self.translate_x = 0  # 鼠标拖动距离x
            self.translate_y = 0  # 鼠标拖动距离y
            self.pix_half_height = (self.pix.height() * s) / 2
            self.pix_half_width = (self.pix.width() * s) / 2
            self.start = QPoint(0, 0)  # 鼠标单击位置
            self.update()

    def load_image(self, image: Union[str, Path], label=None):
        self.im_file = str(Path(image))
        self.pix = QPixmap(self.im_file)
        self.pix_scale_x = self.pix.width() / (self.pix.width() + self.height())  # 宽度缩放比例
        self.pix_scale_y = self.pix.height() / (self.pix.width() + self.height())  # 高度缩放比例
        self.fit()
        self.update()

    def fit(self):
        """自适应图像"""
        self.setImageSize(self.width(), self.height())


    def getLabelSizePoint(self, x, y):
        """将pix坐标系的点转换为Label坐标系的点"""
        x *= self.image_rect.width()/ self.pix.width()
        y *= self.image_rect.height() / self.pix.height()
        x += self.image_rect.x()
        y += self.image_rect.y()
        return np.array([x, y], dtype=np.float32)

    def getPixSizePoint(self, x, y):
        """将label坐标系的点转换为pix坐标系的点"""
        x -= self.image_rect.x()
        y -= self.image_rect.y()
        x *= self.pix.width() / self.image_rect.width()
        y *= self.pix.height() / self.image_rect.height()
        return np.array([x, y], dtype=np.float32)

    def resizeEvent(self, event):
        self.fit()

    def paintEvent(self, event):
        self.center = (self.center[0] + self.translate_x, self.center[1] + self.translate_y)
        point_lu = QPoint(self.center[0] - self.scale_left_w, self.center[1] - self.scale_up_h)  # 左上角坐标
        point_rd = QPoint(self.center[0] + self.scale_right_w, self.center[1] + self.scale_down_h)  # 右下角坐标
        self.image_rect = QRect(point_lu, point_rd)
        w = point_rd.x() - point_lu.x()
        h = point_rd.y() - point_lu.y()
        self.center = (point_lu.x() + w / 2, point_lu.y() + h / 2)
        self.scale_left_w = w / 2
        self.scale_right_w = w / 2
        self.scale_down_h = h / 2
        self.scale_up_h = h / 2

        if self.pix is None:
            return
        painter = QPainter(self)
        painter.drawPixmap(self.image_rect, self.pix)  # 指定矩形范围rect绘制图像
        if not self.zoom_finish and self.zoom_zone:
            zone_l = min(self.zoom_zone[0], self.zoom_zone[2])
            zone_u = min(self.zoom_zone[1], self.zoom_zone[3])
            zone_r = max(self.zoom_zone[0], self.zoom_zone[2])
            zone_d = max(self.zoom_zone[1], self.zoom_zone[3])
            zoom_rect = QRect(QPoint(int(zone_l), int(zone_u)), QPoint(int(zone_r), int(zone_d)))
            painter.setPen(QPen(Qt.GlobalColor.blue,3, Qt.PenStyle.DashDotLine))
            painter.drawRect(QRect(zoom_rect))
        self.draw(painter)
        painter.end()


    def draw(self, painter):
        """绘制"""
        pass

    def resizing(self, event: QMouseEvent):
        """处于指定放大范围状态"""
        return (event.modifiers() == Qt.KeyboardModifier.AltModifier and not self.zoom_finish) or event.modifiers() == Qt.KeyboardModifier.ControlModifier

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.position()
            if event.modifiers() == Qt.KeyboardModifier.AltModifier:
                self.zoom_zone = [self.start.x(), self.start.y(), self.start.x(), self.start.y()]
                self.zoom_finish = False

    def mouseMoveEvent(self, event: QMouseEvent):
        x = event.position().x()
        y = event.position().y()
        if event.modifiers() == Qt.KeyboardModifier.AltModifier and event.buttons() == Qt.MouseButton.LeftButton and not self.zoom_finish:
            self.zoom_zone[2] = x
            self.zoom_zone[3] = y
            self.update()
        elif event.modifiers() != Qt.KeyboardModifier.AltModifier and event.buttons() == Qt.MouseButton.LeftButton and not self.zoom_finish:
            self.finishZoom()
        elif event.buttons() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.translate_x = x - self.start.x()
            self.translate_y = y - self.start.y()
            self.start = QPoint(x, y)
            self.update()
        else:
            self.translate_x = 0
            self.translate_y = 0
        self.mouse_point = QPoint(x, y)

    def mouseReleaseEvent(self, ev:QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and ev.modifiers() == Qt.KeyboardModifier.AltModifier:
            self.finishZoom()


    def finishZoom(self):
        center_x = (self.zoom_zone[0] + self.zoom_zone[2]) / 2
        center_y = (self.zoom_zone[1] + self.zoom_zone[3]) / 2
        w = abs(self.zoom_zone[2] - self.zoom_zone[0])
        h = abs(self.zoom_zone[3] - self.zoom_zone[1])
        if w < 0.01 or h < 0.01:
            self.zoom_finish = True
            self.update()
            return
        scale = max(w/self.width(), h/self.height())
        half_w = self.pix_half_width / scale 
        x = (half_w-self.pix_half_width)/self.pix_scale_x   #更新公式： pix_half_width + x*self.pix_scale_x = pix_half_width  逆推
        half_h = self.pix_half_height + x * self.pix_scale_y # pix_half_height = pix_half_height + x *self.pix_scale_y
        self.scalePix(half_w, half_h, center_x, center_y)
        self.zoom_finish =True
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        if self.pix == None:
            return

        if (self.pix_half_height > 24 and self.pix_half_width > 24) or event.angleDelta().y() > 0:  # 120/5=24，不能使得pix_half_width和pix_halg_height为负数
            half_w = self.pix_half_width + event.angleDelta().y() / 5 * self.pix_scale_x  #放大后的一半宽
            half_h = self.pix_half_height + event.angleDelta().y() / 5 * self.pix_scale_y  #放大后的一半高
            x = event.position().x() if event.angleDelta().y() > 0  else self.width()/2  #放大以鼠标位置为缩放中心，缩小以label中心为缩放中心
            y = event.position().y() if event.angleDelta().y() > 0  else self.height()/2
            self.scalePix(half_w, half_h, x, y)


    def scalePix(self, half_w, half_h, center_x, center_y):
        """对显示图像进行缩放
        Args:
            w: 缩放后的宽的一半
            h： 缩放后的高的一半
            center_x：缩放中心x
            center_y：缩放中心y"""
        old_w = self.pix_half_width
        old_h = self.pix_half_height
        self.pix_half_width = half_w if half_w != 0 else self.pix_half_width
        self.pix_half_height = half_h if half_h != 0 else self.pix_half_height
        new_w = self.pix_half_width
        new_h = self.pix_half_height
        x = center_x  # 放大以鼠标位置为缩放中心，缩小以label中心为缩放中心
        y = center_y
        x1 = self.center[0] - x
        y1 = self.center[1] - y
        sx = new_w / old_w
        sy = new_h / old_h
        x2 = (old_w - x1) * sx
        y2 = (old_h - y1) * sy
        self.scale_left_w = int(x1 + x2)
        self.scale_right_w = self.pix_half_width * 2 - self.scale_left_w
        self.scale_up_h = int(y1 + y2)
        self.scale_down_h = self.pix_half_height * 2 - self.scale_up_h
        self.update()


    def contextMenuEvent(self, ev: QContextMenuEvent) -> None:
        main_menu = QMenu(self)
        main_menu.setObjectName("right_menu")
        auto_fit_a = QAction(text="自适应", parent=main_menu)
        main_menu.addActions([auto_fit_a])
        req = main_menu.exec(self.mapToGlobal(ev.pos()))
        if req == auto_fit_a:
            self.fit()
        self.update()


class QTransformerLabel(QSizeLabel):
    """自定义的图像和标签显示，可新型数据集的绘制"""
    Change_Label_Signal = Signal()  #标签改变信号
    Next_Image_Signal = Signal()   #下一张图像
    Last_Image_Signal = Signal()   #上一张图像
    Create_Select_Signal = Signal()  #创建选区完成信号
    Create_Mask_Signal = Signal()  #创建掩膜完成信号
    Show_Status_Signal = Signal(str) #显示图像坐标信号
    Show_Classes_Signal = Signal()  #显示种类信号
    def __init__(self, parent):
        """
        Attriabute:
        label(dict): 标签字典，存储{"cls": [],"instances": QInstance, "nkpt": int, "ndim": int, "names": []}
        pred_label(dict):预测的标签字典， {"cls":[], "instances":QInstance, "names":[], "conf",[]}
        image_rect(QRect): 显示图像矩形区域
        index(int): 当前标签索引
        point_ind1: 当前选中点1
        point_inde2: 当前选中点2，当点1和点2不同时，表示选中直线
        paint(bool): 是否允许绘制标签
        painting(bool): 是否在绘制标签中
        cls(int): 当前种类
        mouse_point(QPoint): 实时鼠标位置
        cross_cursor(bool): 是否显示十字线
        bounding_rect(bool): 是否显示标签外接矩形
        old_size(float): 上一次窗口缩放前的最短边
        show_pred(bool): 显示预测的标签
        show_true(bool): 显示真实的标签
        levels_img(np.ndarray): 色阶增强后的图像
        """
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.label = None
        self.pred_label = None
        self.image_rect = None
        self.index = -1
        self.point_ind1 = -1
        self.point_ind2 = -1
        self.paint = False
        self.painting = False
        self.cls = 0
        self.cross_cursor = True
        self.bounding_rect = False
        self.show_cls = False
        self.old_size = min(self.width(), self.height())
        self.show_pred = False
        self.show_true = True
        self.red = QColor(Qt.red)
        self.red.setAlpha(150)
        self.task = ""
        self.img = None
        self.levels_img = None
        self.colors = []



    def init(self):
        """初始化"""
        self.img =None
        self.levels_img = None
        self.label = None
        self.im_file = ""
        self.pix = None
        self.paint = False
        self.painting = False
        self.update()

    def generateColors(self):
        """生成颜色"""
        self.colors = generate_distinct_colors(len(self.label["names"])+1)[1:]


    def load_image(self, image: Union[str, Path], label=None):
        """加载图像"""
        if Path(image).exists() and Path(image).suffix[1:].lower() in IMG_FORMATS:
            self.label = label
            if self.label and self.label.get("instances") and self.label["instances"].keypoints:
                n, nkpt, ndim = self.label["instances"].keypoints.shape
                self.label["nkpt"] = nkpt
                self.label["ndim"] = ndim
            self.label["instances"].denormalize(self.label["ori_shape"][1], self.label["ori_shape"][0])
            self.label["cls"] = [int(c) for c in self.label["cls"]]
            self.generateColors()
            self.index = -1
            self.point_ind1 = -1
            self.point_ind2 = -1
            self.painting = False
            self.img = cv2.imread(image)
            self.levels_img = None
            if self.cls >= len(self.label["names"]):
                self.cls = self.cls - 1 if len(self.label["names"]) > 0 else 0
            super().load_image(image, label)
        else:
            self.init()

    def loadPredLabel(self, pred_label):
        """加载预测标签"""
        pred_label["cls"] = [int(c) for c in pred_label["cls"]]
        self.pred_label = pred_label

    def getLabelSizeInstance(self, instance: Instances):
        if len(instance):
            if instance.normalized:
                instance.denormalize(self.pix.width(), self.pix.height())
            old_format = instance._bboxes.format
            instance.scale(self.image_rect.width() / self.pix.width(),
                           self.image_rect.height() / self.pix.height())  # 缩放
            instance.convert_bbox("xyxy")
            instance.add_padding(self.image_rect.x(), self.image_rect.y())
            instance.convert_bbox(old_format)

    def getPixSizeInstance(self, instance: Instances):
        if len(instance):
            if instance.normalized:
                instance.denormalize(self.pix.width(), self.pix.height())
            old_format = instance._bboxes.format
            instance.convert_bbox("xyxy")
            instance.add_padding(-self.image_rect.x(), -self.image_rect.y())
            instance.convert_bbox(old_format)
            instance.scale(self.pix.width() / self.image_rect.width(),
                           self.pix.height() / self.image_rect.height())  # 缩放



    def resizeEvent(self, event):
        if abs(min(self.width(), self.height()) - self.old_size) > 200:
            self.old_size = min(self.width(), self.height())
            self.fit()

    def clearLabel(self):
        """清空标签"""
        pass

    def draw(self, painter):
        if self.levels_img is not None:
            pix = cvImg2Qpix(self.levels_img)
            painter.drawPixmap(self.image_rect, pix)
        if self.cross_cursor:
            self.drawCrossLine(painter)
        if self.show_true and self.label:   #真实标签
            true_label = copy.deepcopy(self.label)
            if true_label.get("instances"):
                true_label["instances"] = self.getInstance(self.label)
            self.drawLabel(painter, true_label)
            if self.bounding_rect and true_label:
                self.drawBoxes(painter, true_label)
        if self.show_pred and self.pred_label:  #预测标签
            pred_label = copy.deepcopy(self.pred_label)
            if pred_label.get("instances"):
                pred_label["instances"] = self.getInstance(self.pred_label)
                self.drawLabel(painter, pred_label, True)
                if self.bounding_rect:
                    self.drawBoxes(painter, pred_label, True)

    def getInstance(self, label):
        label["instances"].getBoundingRect()
        instance = copy.deepcopy(label["instances"])
        if instance._bboxes is not None and len(instance._bboxes) > 0:  #存在标签
            if not self.painting and self.cursor().shape() == Qt.CrossCursor:
                label["instances"].remove_zero_area_boxes()
            label["instances"].clip(self.pix.width(), self.pix.height())
            instance.convert_bbox("xyxy")
            self.getLabelSizeInstance(instance)  # 将标签缩放移动至显示大小位置
        return instance


    def drawLabel(self, painter, label, pred=False):
        """
        绘制标签
        Args:
            painter(QPainter): 绘制画笔类
            instance(QInstance): 标签实例
        """
        pass

    def drawCrossLine(self, painter):
        """
        绘制矩形
        Args:
            painter(QPainter): 绘制画笔类
            instance(QInstance): 标签实例
        """
        pen = QPen(QColor(198, 247, 111), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(QLine(self.mouse_point.x(), 0, self.mouse_point.x(), self.height()))
        painter.drawLine(QLine(0, self.mouse_point.y(), self.width(), self.mouse_point.y()))

    def drawBoxes(self, painter, label, pred=False):
        """
        绘制矩形
        Args:
            painter(QPainter): 绘制画笔类
            instance(QInstance): 标签实例
            pred(bool): 是否预测标签
        """
        instance = label["instances"]
        if instance._bboxes is not None and len(instance._bboxes):
            if instance._bboxes.format != "xyxy":
                instance.convert_bbox("xyxy")
            boxes = instance.bboxes
            cls = self.label["cls"]
            for i,(box,c) in enumerate(zip(boxes,cls)):
                box = box.tolist()
                box = [int(b) for b in box]
                lu = QPoint(box[0], box[1])
                ru = QPoint(box[2], box[1])
                ld = QPoint(box[0], box[3])
                rd = QPoint(box[2], box[3])
                rect = QRect(lu, rd)
                painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                color = QColor(self.colors[c][0], self.colors[c][1], self.colors[c][2])
                painter.setPen(QPen(color if not pred else self.red, 3, Qt.PenStyle.SolidLine))
                painter.drawRect(rect)
                painter.setPen(QPen(Qt.GlobalColor.green if not pred else self.red, 5, Qt.PenStyle.SolidLine))
                painter.drawPoints([lu, ru, ld, rd])
                if self.show_cls and self.task == "detect":
                    if not pred:
                        self.drawText(painter, QPoint(box[0],box[1]-4), self.label["names"][int(c)], 12, color=Qt.GlobalColor.green)
                    else:
                        self.drawText(painter, QPoint(box[2], box[3] + 12), self.pred_label["names"][int(c)] + f" {self.pred_label['conf'][i]:3.2f}", 12, color=Qt.GlobalColor.white)

    def drawText(self, painter,point, text, font_size=8, color=Qt.GlobalColor.green):
        font = QFont("幼圆", font_size)
        # 计算文本宽度
        font_metrics = QFontMetrics(font)
        text_w= font_metrics.horizontalAdvance(text)
        text_h = font_metrics.height()
        brush = QBrush(Qt.BrushStyle.SolidPattern)
        brush.setColor(QColor(255, 0, 0, 150))
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRect(point.x()-2, point.y() -text_h, text_w+4, text_h+2))
        painter.setPen(QPen(color))
        painter.setFont(font)
        painter.drawText(point, text)




    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        #显示种类
        if self.label is not None:
            rect = QRect(self.width()- 1, 0, 1, self.height())
            if rect.contains(event.pos()):
                self.Show_Classes_Signal.emit()
        #显示像素值
        if self.img is not None:
            x = event.x()
            y = event.y()
            p = self.getPixSizePoint(x, y)
            h, w = self.img.shape[:2]
            if 0<=p[0]<w and 0<=p[1]<h:
                if self.levels_img is not None:
                    color = self.levels_img[int(p[1]),int(p[0]),:]
                else:
                    color = self.img[int(p[1]),int(p[0]),:]
                tip = f"x:{p[0]:0.2f} y:{p[1]:0.2f}, BGR:({int(color[0]), int(color[1]), int(color[2])})"
            else:
                tip = ""
            self.Show_Status_Signal.emit(tip)
        # 当前选中标签
        if self.paint and not self.painting and event.buttons() != Qt.MouseButton.LeftButton:
            if self.label.get("instances", None) and len(self.label["instances"]):
                for i, inst in enumerate(self.label["instances"]):
                    instance = copy.deepcopy(inst)
                    self.getLabelSizeInstance(instance)
                    self.setInstanceCursor(instance, self.mouse_point)
                    if self.cursor().shape() != Qt.CursorShape.CrossCursor:
                        if self.index != i:
                            self.index = i
                        break
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def setInstanceCursor(self, instance, pos):
        """
        设置光标落在标签上的样式
        Args:
            instance(QInstances):单个标签实例
            pos(QPoint): 光标位置"""
        pass

    def keyPressEvent(self, ev:QKeyEvent) -> None:
        if ev.key() == Qt.Key.Key_Down:
            self.Next_Image_Signal.emit()
        elif ev.key() == Qt.Key.Key_Up:
            self.Last_Image_Signal.emit()
        elif ev.key() in (Qt.Key.Key_0, Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5, Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9):
            if int(ev.text()) < len(self.label["names"]):
                self.cls = int(ev.text())
        elif ev.key() == Qt.Key.Key_Return and self.paint and not self.painting:
            self.setCursor(Qt.CursorShape.CrossCursor)


    def contextMenuEvent(self, ev: QContextMenuEvent) -> None:
        if not self.pix:
            return
        main_menu = QMenu(self)
        main_menu.setObjectName("right_menu")
        clear_all_a = QAction(text="清空", parent=main_menu)
        auto_fit_a = QAction(text="自适应", parent=main_menu)
        paint_a = QAction(text="标注", parent=main_menu)
        paint_a.setCheckable(True)
        paint_a.setChecked(self.paint)
        bounding_rect_a = QAction(text="矩形框", parent=main_menu)
        bounding_rect_a.setCheckable(True)
        bounding_rect_a.setChecked(self.bounding_rect)

        show_cls_a = QAction(text="显示种类", parent=main_menu)
        show_cls_a.setCheckable(True)
        show_cls_a.setChecked(self.show_cls)

        show_area_a = QAction(text="显示面积", parent=main_menu)
        show_area_a.setCheckable(True)
        


        cross_a = QAction(text="十字线", parent=main_menu)
        cross_a.setCheckable(True)
        cross_a.setChecked(self.cross_cursor)
        main_menu.addActions([auto_fit_a, cross_a])

        # 种类
        if self.label:
            class_menu = QMenu(title="种类", parent=main_menu)
            for cls, name in self.label["names"].items():
                class_a = class_menu.addAction(name)
                class_a.setCheckable(True)
                class_a.setChecked(True if name == self.label["names"][self.cls] else False)
            main_menu.addMenu(class_menu)
            main_menu.addActions([paint_a, bounding_rect_a, clear_all_a])
            if self.task != "classify":
                main_menu.addAction(show_cls_a)
                if self.task == "segment":
                    show_area_a.setChecked(self.show_area)
                    main_menu.addAction(show_area_a)

        req = main_menu.exec_(self.mapToGlobal(ev.pos()))
        if req == auto_fit_a:
            self.fit()
        elif req == paint_a:
            self.paint = paint_a.isChecked()
        elif req == bounding_rect_a:
            self.bounding_rect = bounding_rect_a.isChecked()
        elif req == show_cls_a:
            self.show_cls = show_cls_a.isChecked()
        elif req == show_area_a:
            self.show_area = show_area_a.isChecked()
        elif req == clear_all_a:
            self.clearLabel()
            self.Change_Label_Signal.emit()
        elif req == cross_a:
            self.cross_cursor = cross_a.isChecked()
        if self.label:
            for i in range(len(class_menu.actions())):
                if req == class_menu.actions()[i]:
                    self.cls = i
        self.update()


class QFastSelectLabel(QTransformerLabel):
    Fast_Sel_Cop_Signal = Signal(list)
    def __init__(self, parent):
        super().__init__(parent)
         # 快速选择
        self.fast_cre_sel = False  # 创建选区
        self.fast_rect = None   #快速选择的选区
        self.fast_segment = None #存储快速选择的分割实例
        self.fast_method = "threshold"  #快速选择方法
        self.fast_seed_searching = False
        self.seed = None

    def init(self):
        super().init()
        self.fast_cre_sel = False 
        self.fast_rect = None 
        self.fast_segment = None 
        self.fast_seed_searching = False
        self.seed = None
        self.update()
        self.Fast_Sel_Cop_Signal.emit()


    def drawLabel(self, painter, label, pred=False):
        if pred:
            return
        #显示快速选择搜索区域
        if self.fast_cre_sel and self.fast_rect:
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            p1 = self.getLabelSizePoint(self.fast_rect[0], self.fast_rect[1])
            p2 = self.getLabelSizePoint(self.fast_rect[2], self.fast_rect[3])
            painter.drawRect(QRect(int(p1[0]), int(p1[1]), int(p2[0]-p1[0]), int(p2[1]-p1[1])))
        if self.fast_seed_searching:
            painter.setPen(QPen(Qt.green, 8, Qt.SolidLine))
            painter.drawPoint(self.mouse_point)

        instance = label["instances"]
        fast_segment = copy.deepcopy(self.fast_segment)
        if self.fast_segment is not None and len(self.fast_segment) and (self.fast_cre_sel or self.fast_seed_searching):
            self.getPixSizeInstance(instance)
            if self.task == "segment":
                if instance.segments is None:
                    instance.segments = []
                instance.segments.append(fast_segment)
                instance.getBoundingRect()
            else:
                format = instance._bboxes.format if instance._bboxes else "xyxy"
                box = segment2Box(fast_segment, format)
                if instance._bboxes:
                    instance._bboxes.addBox(box, format)
                else:
                    instance._bboxes = QBboxes(box, format=format)
            self.getLabelSizeInstance(instance)
            label["cls"].append(self.cls)

    @property
    def fastCreate(self):
        """创建快速选区中"""
        return self.fast_cre_sel or self.fast_seed_searching

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.paint and event.buttons() == Qt.MouseButton.LeftButton and self.fast_cre_sel and not self.fast_seed_searching and not self.resizing(event):# 快速选择创建选区
            p1 = self.getPixSizePoint(self.start.x(), self.start.y())
            p2 = self.getPixSizePoint(event.x(), event.y())
            self.fast_rect = twoPoints2box(p1, p2, "xyxy")[0].tolist()
            self.update()

    
    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self.paint and ev.button() == Qt.MouseButton.LeftButton and ((self.fast_cre_sel and self.fast_rect is not None) or (self.fast_method == "floodfill" and self.fast_seed_searching)) and not self.resizing(ev): 
            self.seed  = self.getPixSizePoint(ev.x(), ev.y()).tolist()
            self.Fast_Sel_Cop_Signal.emit(self.seed)
