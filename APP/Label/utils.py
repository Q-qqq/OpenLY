
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import math
import torch
import numpy as np
import cv2
from PIL import Image

from ultralytics.utils.ops import xywhr2xyxyxyxy

def distPointToLine(point1, point2, point):
    """点到直线的距离"""
    k = (point2[1] - point1[1]) / (point2[0] - point1[0] + 1e-12)
    A = k
    B = -1
    C = point1[1] - k * point1[0]
    d = (A * point[0] + B * point[1] + C) / math.sqrt(A ** 2 + B ** 2)  # 点到线的距离
    return d

def judgePointUpLine(point1, point2, point, r=1):
    """判断point是否在point1和point2的连接线段上，r限制线段与两点的距离"""
    #line
    line = LineTool(point1.x(), point1.y(), point2.x(), point2.y())
    if abs(line.k) > 100000:  #垂直线
        d = abs(point.x() - line.x1)
        if d < r and point.y() > min(line.y1, line.y2)+r and point.y() < max(line.y1, line.y2)-r:
            return True
        else:
            return False
    elif abs(line.k) < 0.1:  #水平线
        d = abs(point.y() - line.y1)
        if d<r and point.x() > min(line.x1, line.x2)+r and point.x() < max(line.x1, line.x2)-r:
            return True
        else:
            return False
    else:  #斜线
        d = abs(line.disPointToLine(point.x(), point.y()))  #点到线的距离
        dp1 = math.sqrt((point.x() - line.x1) ** 2 + (point.y() - line.y1) ** 2)  #点到端点的距离
        dp2 = math.sqrt((point.x() - line.x2) ** 2 + (point.y() - line.y2) ** 2)

        if d<1 and min(line.x1, line.x2) - r < point.x() < max(line.x1, line.x2) + r \
                and min(line.y1, line.y2) - r < point.y() < max(line.y1, line.y2) + r\
                and dp1 > r and dp2 > r:
            return True
        else:
            return False

def judgePointInCircle(center, point, r):
    """判断点point是否在以center为圆心，半径为r的圆内"""
    d = (point.x() - center.x())**2 + (point.y() - center.y())**2 - r**2
    if d < 0:
        return True
    else:
        return False

def getVerPoint(p0, p1, d):
    """根据点（x,y）到直线（p0, p1）的距离d，求解过点p0和p1的直线（p0,p1）的垂直线上两点p2,p3，到直线（p0，p1）的距离为d"""
    k = (p1[1] - p0[1]) / (p1[0] - p0[0])
    A = k
    B = -1
    C = p0[1] - k * p0[0]
    x2 = (d*math.sqrt(A**2 + B**2) - k*B*p0[0] - B*p0[1] - C)/ (A - k*B)
    y2 = (d*math.sqrt(A**2 + B**2)-A*x2 -C) / B
    x3 = (d * math.sqrt(A ** 2 + B ** 2) - k * B * p1[0] - B * p1[1] - C) / (A - k * B)
    y3 = (d*math.sqrt(A ** 2 + B ** 2) - A * x3 - C) / B
    return (x2, y2), (x3, y3)

def qrect2box(rect:QRect, format= "xywh"):
    """将矩形类转换为指定的矩形数据Tensot(1,4)"""
    assert format in ["xywh", "xyxy"]
    if format=="xywh":
        x = rect.x() + rect.width() / 2
        y = rect.y() + rect.width() / 2
        w = rect.width()
        h = rect.height()
        return torch.Tensor([x, y, w, h])[None, :]
    elif format == "xyxy":
        x1 = rect.x()
        y1 = rect.y()
        x2 = rect.x() + rect.width()
        y2 = rect.y() + rect.height()
        return torch.Tensor([x1,y1,x2,y2])[None, :]


def qpolygon2points(polygon: QPolygon):
    """将多边形类转换为点数据Tensor(1,n,2)"""
    ps = []
    for point in polygon:
        ps.append([point.x(), point.y()])
    return torch.Tensor(ps)[None, :]

def box2qrect(box, format= "xywh"):
    """将指定的矩形数据转换为矩形类"""
    if box.ndim == 2:
        box = box[0,:]
    if isinstance(box, torch.Tensor):
        box = box.tolist()
    assert format in ["xywh", "xyxy"]
    if format == "xywh":
        point1 = QPoint(box[0] - box[2]/2, box[1] - box[3]/2)
        point2 = QPoint(box[0] + box[2]/2, box[1] + box[3]/2)
    elif format == "xyxy":
        point1 = QPoint(box[0], box[1])
        point2 = QPoint(box[2], box[3])
    return QRect(point1, point2)

def points2qpolygon(points):
    """将输入的点集（n,2）转换为多边形类"""
    poly = QPolygon()
    for point in points:
        poly.append(QPoint(point[0], point[1]))
    return poly

def segment2obb(segment, format):
    """将输入的分割数据(n,M,2)，转换为obb(n, 8)"""
    if isinstance(segment, torch.Tensor):
        segment = segment.numpy()
    (x,y), (w,h), angle =  cv2.minAreaRect(segment)
    obb = np.array([x, y, w, h, angle/180 *math.pi])[None,:]
    return obb if format == "xywhr" else xywhr2xyxyxyxy(obb)

def twoPoints2box(point1, point2, format):
    """将两个点转换为矩形框
    Args:
        point1(list):点1
        point2(list):点2
        format(string):输出矩形框格式xywh or xyxy
    Returns:
        (torch.Tensor):矩形框[[*, *, *, *]]"""
    if format == "xywh":
        w = max(point1[0], point2[0]) - min(point1[0], point2[0])
        h = max(point1[1], point2[1]) - min(point1[1], point2[1])
        x = min(point1[0], point2[0]) + w/2
        y = min(point1[1], point2[1]) + h/2
        return torch.Tensor([x,y,w,h])[None]
    elif format == "xyxy":
        x1 = min(point1[0], point2[0])
        y1 = min(point1[1], point2[1])
        x2 = max(point1[0], point2[0])
        y2 = max(point1[1], point2[1])
        return torch.Tensor([x1,y1,x2,y2])[None]


class LineTool:
    def __init__(self,*args, **kwargs):
        if len(args) == 3:
            self.x1, self.y1, self.k = args
            self.format = "xyk"
        elif len(args) == 4:
            self.x1, self.y1, self.x2, self.y2 = args
            self.format = "xyxy"
        if self.format == "xyxy":
            self.k = (self.y2 - self.y1) / (self.x2 - self.x1 + 1e-12)
        self.A = self.k
        self.B = -1
        self.C = self.y1 - self.k * self.x1

    def disPointToLine(self, x, y):
        """点到直线的距离"""
        return (self.A * x + self.B * y + self.C) / math.sqrt(self.A ** 2 + self.B ** 2)  # 点到线的距离

    def insertLineAndLine(self, line):
        """
        线与线的交点
        Args:
            line(LineTool): 另一条线"""
        A1, B1, C1 = self.A, self.B, self.C
        A2, B2, C2 = line.A, line.B, line.C
        x = (C1 * B2 - C2 *B1) / (B1* A2 - B2* A1)
        y = (-A1 * x - C1)/B1
        return (x, y)



def P2P_rot_angle(x, y, c_x, c_y, angle):
    """
    点(x,y)绕旋转中心(c_x, c_y)旋转角度angle后的位置
    Args:
        x(float): 旋转前点x
        y(float)：旋转前点y
        c_x(float): 旋转中心x
        c_y(float): 旋转中心y
        angle(float): 旋转角度 格式：rad 0-2*pi
    """
    result_x = (x - c_x) * math.cos(angle) - (y - c_y) * math.sin(angle) + c_x
    result_y = (y - c_y) * math.cos(angle) + (x - c_x) * math.sin(angle) + c_y
    return (result_x, result_y)

def get_segment_diagnol_point(segment):
    "获取分割多边形的左上点和右下点"
    def distance(p1, p2):
        return math.sqrt((p1[0]- p2[0])**2 + (p1[1]-p2[1])**2)
    l = segment[..., 0].min()  #left
    r = segment[..., 0].max()  #right
    u = segment[...,1].min()   #up
    d = segment[...,1].max()   #down
    lu = (l, u)
    rd = (r, d)
    lu_ds = []
    rd_ds = []
    for p in segment:
        lu_ds.append(distance(lu, p))
        rd_ds.append(distance(rd, p))
    return segment[np.argmin(lu_ds)], segment[np.argmin(rd_ds)]


def cvImg2Qpix(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = im.astype("uint8")
    h,w,c = im.shape
    image = QImage(im.data, w, h, w*3, QImage.Format_RGB888)
    pix =QPixmap.fromImage(image)
    return pix


import colorsys


def generate_distinct_colors(n, saturation=0.9, value=0.9):
    """
    生成n个视觉可区分的颜色（HEX格式）

    Args：
    n : int - 需要生成的颜色数量
    saturation : float (0-1) - 饱和度，默认90%
    value : float (0-1) - 明度，默认90%

    Returns：
    list - 包含n个HEX颜色代码的列表
    """
    colors = []
    hue_step = 360 / n  # 色相间隔

    for i in range(n):
        # 计算HSV值（色相转换为角度制）
        hue = i * hue_step
        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)

        # 转换为HEX格式
        hex_code = [int(r * 255),
            int(g * 255),
            int(b * 255)]
        colors.append(hex_code)

    return colors

def segmentArea(segment):
    "计算分割多边形面积"
    A1, A2 = 0, 0
    for i in range(len(segment)):
        if i == len(segment) - 1:
            A1 += segment[i][0] * segment[0][1]  # x_i * y_0 首尾相接
            A2 += segment[0][0] * segment[i][1]  # x_0 * y_i
        else:
            A1 += segment[i][0] * segment[i + 1][1]  # x_i * y_i+1
            A2 += segment[i + 1][0] * segment[i][1]  # x_i+1 * y_i
    return abs((A1 - A2) / 2)













