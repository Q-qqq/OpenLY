from ultralytics.utils.ops import xyxy2xywh,xywh2xyxy,xyxy2ltwh,ltwh2xyxy,ltwh2xywh,xywh2ltwh, resample_segments
from numbers import Number
from itertools import repeat
from collections import abc
from typing import List
import numpy as np
import torch

_formats = ["xyxy", "xywh", "ltwh"]

def _ntuple(n):
    """复制n个x，变成一个tuple"""
    def parse(x):
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x,n))
    return parse

to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

class Bboxes:
    """处理目标检测框的类
    支持目标检测框数据格式：xyxy，xywh，ltwh
    框数据需提供ndarray格式"""
    def __init__(self, bboxes, format="xyxy") -> None:
        assert  format in _formats, f"无效的检测框格式{format},必须是{_formats}其中之一"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self,format):
        assert format in _formats, f"无效的检测框数据格式：{format}, 检测框数据格式必须是{_formats}的其中之一"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """获取每个框的面积，使用该函数将会把bboxes的格式转换为xyxy"""
        self.convert("xyxy")
        return (self.bboxes[:,2] - self.bboxes[:,0])*(self.bboxes[:,3] - self.bboxes[:,1]) #w*h

    def mul(self, scale):
        """将所有目标检测框乘以scale
        :arg scale (tuple| list | int): 缩放比例[1，4]/1"""
        if isinstance(scale,Number):
            scale = to_4tuple(scale)
        assert isinstance(scale,(tuple, list))
        assert len(scale) == 4
        self.bboxes[:,0] *= scale[0]
        self.bboxes[:,1] *= scale[1]
        self.bboxes[:,2] *= scale[2]
        self.bboxes[:,3] *= scale[3]

    def add(self, offset):
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:,0] += offset[0]
        self.bboxes[:,1] += offset[1]
        self.bboxes[:,2] += offset[2]
        self.bboxes[:,3] += offset[3]




    def __len__(self):
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """将传入函数的boxes_list合并为新的bboxes，并以这个新的bboxes初始化Bboxes类"""
        assert  isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box,Bboxes) for box in boxes_list)
        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis = axis))

    def __getitem__(self, index) -> "Bboxes":
        """
        取出一个指定的检测框的BBoxes类或者一个集合的检测框BBoxes类
        :param index（int,slice or np.ndarray）: 指定检测框的索引
        :return: 指定检测框的BBoxes类
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].view(1,-1))

        b = self.bboxes[index]
        assert b.ndim == 2, f"对于指定的Bboxes{index},无法返回一个正常的二维矩阵"
        return Bboxes(b)



class Instances:
    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xywh", normalized=True):
        """
        :param bboxes(ndarray):  目标检测框[N,4]
        :param segments(list|ndarray): 分割坐标[N,m,2]
        :param keypoints(ndarray): 关键点坐标[N,17,3]
        :param bbox_format:目标检测框格式
        :param normalized:是否归一化
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments

    @property
    def bboxes(self):
        return self._bboxes.bboxes


    def convert_bbox(self, format):
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        return  self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        '''对数据集进行缩放'''
        self._bboxes.mul(scale = (scale_w,scale_h,scale_w,scale_h))
        if bbox_only:
            return
        #分割数据
        if self.segments is not None:
            self.segments[...,0] *= scale_w
            self.segments[...,1] *= scale_h
        #关键点数据
        if self.keypoints is not None:
            self.keypoints[...,0] *= scale_w
            self.keypoints[...,1] *= scale_h

    def denormalize(self,w,h):
        """去除归一化"""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        if self. segments is not None:
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[...,0] *= w
            self.keypoints[...,1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """归一化"""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1/w,1/h,1/w,1/h))
        if self.segments is not None:
            self.segments[...,0] /= w
            self.segments[...,1] /= h
        if self.keypoints is not None:
            self.keypoints[...,0] /= w
            self.keypoints[...,1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        assert not self.normalized,"需要在绝对坐标上进行填充"
        assert self._bboxes.format == "xyxy",f"目标检测框格式必须为xyxy"
        self._bboxes.add(offset=(padw,padh,padw,padh))
        if self.segments is not None:
            self.segments[...,0] += padw
            self.segments[...,1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) -> "Instances":
        """
        取出指定的instance
        :param index:（int，slice or np.array）
        :return:
        """
        if self.segments is not None:
            segments = self.segments[index] if len(self.segments) else self.segments
        else:
            segments = None
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]   #属性方法property
        bbox_format = self._bboxes.format
        return Instances(bboxes=bboxes,
                         segments=segments,
                         keypoints=keypoints,
                         bbox_format=bbox_format,
                         normalized=self.normalized)

    def flipud(self,h):
        """上下翻转数据集坐标
        :arg h 图像高度"""
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:,1].copy()
            y2 = self.bboxes[:,3].copy()
            self.bboxes[:,1] = h - y2
            self.bboxes[:,3] = h - y1
        else:
            self.bboxes[:,1] = h - self.bboxes[:,1]
        if self.segments is not None:
            self.segments[...,1] = h - self.segments[...,1]
        if self.keypoints is not None:
            self.keypoints[...,1] = h - self.keypoints[...,1]

    def fliplr(self,w):
        """
        左右翻转数据集坐标
        :param w:图像宽度
        """
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:,0].copy()
            x2 = self.bboxes[:,2].copy()
            self.bboxes[:,0] = w - x2
            self.bboxes[:,2] = w - x1
        else:
            self.bboxes[:,0] = w - self.bboxes[:,0]
        if self.segments is not None:
            self.segments[...,0] = w - self.segments[...,0]
        if self.keypoints is not None:
            self.keypoints[...,0] = w - self.keypoints[...,0]

    def clip(self,w,h):
        """限制数据集坐标在图像范围内"""
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:,[0,2]] = self.bboxes[:,[0,2]].clip(0,w)
        self.bboxes[:,[1,3]] = self.bboxes[:,[1,3]].clip(0,h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        if self.segments is not None:
            self.segments[...,0] = self.segments[..., 0].clip(0,w)
            self.segments[...,1] = self.segments[..., 1].clip(0,h)
        if self.keypoints is not None:
            self.keypoints[...,0] = self.keypoints[..., 0].clip(0,w)
            self.keypoints[...,1] = self.keypoints[..., 1].clip(0,h)

    def remove_zero_area_boxes(self):
        """移除面积为0的标签"""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments is not None and len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes, segments=None, keypoints=None):
        """更新数据集"""
        self._bboxes = Bboxes(bboxes,format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis = axis)
        cat_segments = np.concatenate([ins.segments for ins in instances_list], axis = axis)
        cat_keypoints = np.concatenate([ins.keypoints for ins in instances_list], axis = axis) if use_keypoint else None

        return  cls(cat_boxes,cat_segments,cat_keypoints,bbox_format,normalized)

    def clear(self):
        self._bboxes.bboxes = torch.tensor([], dtype=torch.float32)
        if self.segments is not None:
            self.segments = []
        if self.keypoints is not None:
            self.keypoints = torch.tensor([], dtype=torch.float32)


