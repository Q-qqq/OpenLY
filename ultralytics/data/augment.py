import math
import random

import numpy as np
from ultralytics.utils.instance import Instances
from ultralytics.utils import (LOGGER)
from ultralytics.utils.checks import check_version
from copy import deepcopy
from ultralytics.utils.ops import segment2box, segments2boxes
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13
import cv2
import torch
import torchvision.transforms as T

DEFAULT_MEAN = (0.0,0.0,0.0)
DEFAULT_STD = (1.0,1.0,1.0)
DEFAULT_CROP_FTACTION = 1.0

def v8_transforms(dataset,imgsz, hyp, stretch=False):
    pre_transform = Compose(
        [
            Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            CopyPaste(p=hyp.copy_paste),
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox( new_shape=(imgsz, imgsz)),
            )
        ]
    )
    flip_idx = dataset.data.get("flip_idx", [])   #用于点云上下翻转
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.flipllr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("未定义flip_idx，设置fliplr为0")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            LOGGER.error(f"flip_idx的长度{len(flip_idx)}必须等于kpt_shape[0]={kpt_shape[0]}")
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p = hyp.mixup),
            Albumentations(p = 1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx)
        ]
    )


#region 运行数据增强
class Compose:
    def __init__(self,transforms):
        """transforms 为list"""
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)    #data -> labels
        return  data

    def append(self, transform):
        self.transforms.append(transform)

    def tolist(self):
        return self.transforms

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
#endregion
#region 混合
class BaseMixTransform:
    '''混合标签图像'''
    def __init__(self, dataset, pre_transform=None, p=0.0):
        """
        Args:
        :param dataset: 数据集
        :param pre_transform: 对混合的图像进行预处理函数
        :param p: 进行混合的概率
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        if random.uniform(0,1) > self.p:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)   #图像自适应+填充
        labels["mix_labels"] = mix_labels

        labels = self._mix_transform(labels)
        labels.pop("mix_labels",None)
        return labels


    def get_indexes(self):
        """获取用于数据增强的一个随机索引列表"""
        raise NotImplementedError

    def _mix_transform(self,labels):
        """对labels进行数据增强"""
        raise NotImplementedError
#endregion
#region 拼接图像
class Mosaic(BaseMixTransform):
    #拼接图像
    def __init__(self,dataset, imgsz=640, p=1.0, n=4):
        assert  0 <= p <= 1.0, f"概率值应为0-1，现为{p}"
        assert n in (3, 4, 9), "拼接数必需为3、4或9"
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  #宽度，高度
        self.n = n

    def get_indexes(self,buffer=True):
        """返回一个属于数据集的随机索引列表，索引数为n"""
        if buffer:   #从缓存区选择图像
            return random.choices(list(self.dataset.buffer),k = self.n-1)
        else: #选择任意图像
            return [random.randint(0,len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self,labels):
        assert labels.get("rect_shape",None) is None, "图像适应改进法rect和图像拼接是冲突的"
        assert len(labels.get("mix_labels",[])),"没有图像应用于拼接"
        if self.n==3:
            return (self._mosaic3(labels))
        elif self.n==4:
            return (self._mosaic4(labels))
        else:
            return (self._mosaic9(labels))



    def _mosaic3(self, labels):
        """3个图像拼接"""
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i-1]
            #load image
            img = labels_patch["img"]
            h,w = labels_patch.pop("resized_shape")
            if i == 0: #中间
                img3 = np.full((s*3, s*3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h   #中间图像位置（xyxy）
            elif i == 1: #右边
                c = s + w0, s, s+ w0 + w, s + h  #右边图像位置
            elif i == 2: #左边
                c = s - w, s + h0 -h, s, s + h0  #左边图像位置

            padw, padh = c[:2]  # x1 y1
            x1, y1, x2, y2 = (max(x,0) for x in c)   #保证图像在img3内

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  #将img3对应位置c的数据替换为img

            #标签去归一化 添加填充
            labels_patch = self._update_label(labels_patch,padw+self.border[0], padh+self.border[1])  #后续将截图[-border[0]:border[0].-border[1]:border[1]]，所以padw和padh需加上border
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0]:self.border[0], - self.border[1]:self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """4个图像拼接"""
        mosaic_labels = []
        s = self.imgsz
        xc, yc = (int(random.uniform(-x, 2*s + x)) for x in self.border)
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i-1]
            #加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resize_shape")

            #将图像放置入拼接图像中
            if i==0:   #左上角
                img4 = np.full((s*2, s*2, img.shape[2]), 114, dtype=np.uint8)  #2x2的基础图像
                x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc  # xmin, ymin, xmax, ymax  图像在拼接图像中的位置
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h #xmin, ymin, xmax, ymax 图像本身裁图位置
            elif i == 1:  #右上角
                x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, s*2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a-y1a), min(w, x2a-x1a),h
            elif i == 2:  #左下角
                x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(s*2, yc+h)
                x1b, y1b, x2b, y2b = w-(x2a - x1a), 0, w, min(y2a -y1a, h)
            elif i==3:   #右下角
                x1a, y1a, x2a, y2a = xc, yc, min(xc+w, s*2), min(s*2, yc+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a-x1a), min(y2a-y1a,h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]    #img4[ymin:ymax,xmin:xmax] 赋值拼接图像
            padw = x1a - x1b  #图像转入拼接图像后x坐标变化值
            padh = y1a - y1b  #图像转入拼接图像后y坐标变化值

            #更新label
            labels_patch = self._update_label(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)  #标签合并
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels):
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i-1]
            #加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            #将图像放入拼接图像中
            if i == 0: #中间
                img9 = np.full(s*3, s*3, img.shape[2], 114, dtype=np.uint8)   #3x3的拼接图像
                h0, w0 = h, w
                c = s, s, s+w, s+h
            elif i == 1: #上
                c = s, s-h, s+w, s
            elif i == 2: #右上角
                c = s+wp, s-h, s+wp+w, s
            elif i == 3: #右
                c = s+w0, s, s+w0+w, s+h
            elif i == 4: #右下角
                c = s+w0, s+hp, s+w0+w, s+hp+h
            elif i == 5: #下
                c = s+w0-w, s+h0, s+w0, s+hp+h
            elif i == 6: #左下
                c = s+w0-wp-w, s+hp, s+w0-wp, s+h0+h
            elif i == 7: #左
                c = s-w, s+h0-h, s, s+h0
            elif i == 8: #左上
                c = s-w, s+h0-hp-h, s, s+h0-hp

            padw, padh = c[:2]   #x1,y1
            x1, y1, x2, y2 = (max(x,0) for x in c)  #拼接图像内小图像坐标

            img9[y1:y2, x1:x2] = img[(y1-padh) :, (x1-padw) :] #小图像赋值进拼接图像
            hp, wp = h, w   # h0 w0

            #拼接图像imgsz*3X3转imgsz*2x2
            labels_patch = self._update_label(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]   #[imgsz:-imgsz, imgsz:-imgsz]
        return final_labels


    @staticmethod
    def _update_label(labels, padw, padh):
        """更新label"""
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy") #使add_padding后只挪动位置不影响尺寸
        labels["instances"].denormalize(nw,nh)
        labels["instances"].add_padding(padw, padh)  #挪动标签位置至拼接后图像的标签位置
        return labels

    def _cat_labels(self, mosaic_labels):
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz *2 #拼接图像大小
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])

        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),  #更新新尺寸
            "cls":np.concatenate(cls,0),  #拼接图像标签合并
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        return final_labels
#endregion
#region 随机复制标签
class CopyPaste:
    #labels左右翻转并随机复制p%个重合率0.3以内的标签与分割出的img至原图像
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, labels):
        im = labels["img"]
        cls = labels["cls"]
        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape   # h w c
            im_new = np.zeros(im.shape, np.uint8)

            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)  #标签左右翻转

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)   #交集面积与instance的box面积比值 重合度
            indexes = np.nonzero((ioa < 0.30).all(1))[0]   #重合度 小于0.3的bboes索引
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p*n)): #随机k个list样本进行循环
                cls = np.concatenate((cls,cls[[j]]), axis=0)              #添加复制的种类
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)   #将要添加的示例添加至instances
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1,1,1), cv2.FILLED) # 绘制mask

            result = cv2.flip(im,1)
            i = cv2.flip(im_new,1).astype(bool)   #标签分割像素索引
            im[i] = result[i]    #将翻转后的标签对应img复制至新图像
        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        return labels
#endregion
#region 旋转移动缩放畸变斜切变换
class RandomPerspective:
    '''图像标签：方框、分割、关键点应用随机的几何变换-旋转、移动、缩放、斜切等
    Attributes:
        degrees(float): 随机旋转角度范围0-180
        translate(float):随机移动百分比0-1
        scale(float): 随机缩放图像比例0-1
        shear：随机斜切0-180
        perspective(float):透视畸变系数0-0.001
        border(tuple):指定填充边缘的元素
        pre_transform(callable):在开始自由变换前对图像进行的函数/变换处理
    '''
    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0,0),
                 pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.per_transform = pre_transform

    def affine_transform(self, img, border):
        '''仿射变换'''

        #中心
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  #x-w
        C[1, 2] = -img.shape[0] / 2  #y-h

        #透视变换
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform( -self.perspective, self.perspective)
        P[2, 1] = random.uniform( -self.perspective, self.perspective)

        #旋转和缩放
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees);
        s = random.uniform(1 - self.scale, 1+self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        #斜切
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tanh(random.uniform(-self.shear, self.shear) * math.pi / 180)  #x 斜切角度（deg）
        S[1, 0] = math.tanh(random.uniform(-self.shear, self.shear) * math.pi / 180)  #y 斜切角度（deg）

        #移动
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]

        #合并变换矩阵
        M = T @ S @ R @ P @ C
        #变换图像
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114,114,114))  #透视变换
            else:
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114,114,114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        '''将变换矩阵应用于目标检测框'''
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n*4, 3), dtype = bboxes.dtype)
        xy[:, :2] = bboxes[:, [0,1,2,3,0,3,2,1]].reshape(n * 4, 2)   #0-n行x1y1, n-2n行x2y2, 2n-3n行x1y2, 3n-4n行x2y1   方框四角点
        xy = xy @ M.T     #对每一行的xy坐标进行变换
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8) #透视变换或突通变换

        #new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1),y.max(1)), dtype= bboxes.dtype).reshape(4, n).T  #将四角点转化为两对角点格式

    def apply_segments(self, segments, M):
        '''将变换矩阵应用于分割数据并生成新的分割方框'''
        n, num = segments.shape[:2]   #分割数据集已经过上采样得到相同数量坐标点的分割轮廓
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T
        xy = xy[:,:2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])  #确保分割数据集不会超出框选方位也即图像范围
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        '''将变换矩阵应用于点云数据集'''
        n, nkpt = keypoints.shape[:2]
        if n ==0:
            return keypoints
        xy = np.ones((n*nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n*nkpt, 1)
        xy[:, :2] = keypoints[..., 2].reshape(n*nkpt, 2)
        xy = xy @ M.T
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0]> self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        '''变换图像和标签'''
        if self.per_transform and "mosaic_border" not in labels:
            labels = self.per_transform(labels)   #预处理进行图像自适应填充
        labels.pop("radio_pad", None)

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")

        #确保坐标格式正确
        instances.convert_bbox(format = "xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1]+border[1]*2, img.shape[0]+border[0]*2

        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints

        #如果有分割数据集，则更新目标检测框
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        new_instances.clip(*self.size)

        #筛选instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)  #对box进行scale比例缩放 使变换前后的标签box比例一致
        i = self.box_candidates(box1=bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10)

        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resize_shape"] = img.shape[:2]
        return labels




    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        '''对box通过变换后的长度、宽度、长宽比和变换前后的面积比进行阈值筛选
        Args:
            box1(numpy.ndarray):变换前的目标检测框[x1, y1, x2, y2]
            box2(numpy.ndarray):变换后的目标检测框[x1, y1, x2, y2]
            wh_thr(float):变换后的目标检测框长宽阈值，默认2 - >
            ar_thr(float):变换后的目标检测框长宽比阈值， 默认100 - <
            area_thr(float):变换后的目标检测框面积/变换前的目标检测框面积的阈值， 默认0.1- >
        Returns:
             (numpy.ndarray):一个boolean array, 满足条件的box索引
         '''
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar =np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))   #长宽比
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  #筛选
#endregion
#region 图像标签缩放填充
class LetterBox:
    '''对图像大小进行适应resize并填充
    Args:
        new_shapr((int,int)):新的图像大小
        auto(bool): 是否自动适应填充大小
        scaleFill(bool): 是否直接缩放不填充
        scaleup(bool):是否允许放大图像
        center（bool）：是否左右上下等比填充或只填充右下
        stride(int):神经网络适应的图像stride'''
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
    def __call__(self, labels=None, image=None):
        '''返回更新后的图像个标签'''
        if labels is None:
            labels = {}
        img =  labels.get("img") if image is None else image
        shape = img.shape[:2] #h, w
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        #计算缩放比（new/old）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  #缩小不放大
            r = min(r, 1.0)

        #计算填充大小
        ratio = r, r # w, h ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  #旧图像长宽等比例缩放后未填充 w， h
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]       #将new_unpad填充至new_shape需要的长宽元素大小
        if self.auto:    #适应神经网络stride的最小填充
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:  #不填充，直接缩放至new_shape
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2
        if shape[::-1] != new_unpad:  #需要resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int (round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int (round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  #add border

        if labels.get("radio_pad"):
            labels["radio_pad"] = (labels["radio_pad"], (left, top))  #用于评估

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)   #将图像上的缩放和填充应用于标签
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img


    def _update_labels(self, labels, ratio, padw, padh):
        '''更新labels，对标签进行缩放、填充'''
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels
#endregion

#region 图像重叠
class MixUp(BaseMixTransform):
    '''将两个图像按随机比例相加'''
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        '''获取一个随机的数据集索引'''
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self,labels):
        r = np.random.beta(32.0, 32.0)  #混合比例(0-1)beta概率分布 alpha=beta=32
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"]* (1-r)).astype(np.uint8)
        labels["instances"] =  Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels["cls"]], 0)
        return labels
#endregion
#region 图像预处理-模糊、亮度、对比度、压缩、灰度
class Albumentations:
    def __init__(self,p =1.0):
        self.p = p
        self.transform = None
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)

            #transforms
            T = [
                A.Blur(p=0.01),       #图像模糊
                A.MedianBlur(p=0.01), #中值滤波模糊
                A.ToGray(p=0.01),     #灰度图像返回原图， 3通道彩色图像返回3通道灰度图像
                A.CLAHE(p=0.01),       #限制对比度直方图均衡化
                A.RandomBrightnessContrast(p=0.0),  #随机改变亮度和对比度
                A.RandomGamma(p=0.0),    #随机伽马校正 将过亮过暗的部分校正
                A.ImageCompression(quality_lower=25, p=0.0),  #图像压缩
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            LOGGER.info("".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:
            pass
        except Exception as e:
            LOGGER.error(f"ERROR {e}")

    def __call__(self, labels):
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes,class_labels=cls)
                if len(new["class_labels"]) > 0:  #如果在新图像里没有box，则跳过
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes = bboxes)
        return labels
#endregion

#region 随机HSV增强
class RandomHSV:
    def __init__(self, hgain=0.5, sgain=0.1, vgain=0.5) -> None:
        '''
        初始化HSV三通道的增益
        :param hgain: 色相(hue)增益，默认 0.5
        :param sgain: 饱和度(saturation)， 默认0.5
        :param vgain: 色调(value), 默认0.5
        '''
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        '''对图像应用随机的hsv三通道增强'''
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) *[self.hgain, self.sgain, self.vgain]
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  #uint8

            x = np.arange(0, 256, dtype = r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  #同内存，不需要赋值labels["img"]
        return labels
#endregion
#region 随机翻转
class RandomFlip:
    '''随机左右或上下翻转'''
    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        '''

        :param p(float):  翻转概率
        :param derection(str):  翻转方向  "horizontal" or "vertical"
        :param flip_idx(array-Like):  翻转点云标签的index mapping
        '''
        assert direction in ["horizontal", "vertical"], f"支持的方向有“horizontal”或“vertical”，但现在是{direction}"
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        #翻转
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            #keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)  #使内存连续
        labels["instances"] = instances
        return labels
#endregion


#Classification augmentations ------------------------------
def classify_transforms(
        size=224,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        interpolation:T.InterpolationMode = T.InterpolationMode.BILINEAR,
        crop_fraction: float=DEFAULT_CROP_FTACTION,
):
    """
    用于评估或者推理的分类转换, 先缩放后裁剪再归一化
    Args:
        size(int): 图像大小
        mean(tuple): RGB通道的平均值
        std(tuple): RGB通道的标准差
        interpolation(T.InterpolationMode): 插入模式
        crop_fraction(float):裁切图像分数
    Returns:
        (T.Compose): torchvision transforms
    """
    if isinstance(size, (tuple, list)):
        assert len(size) == 2
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    if scale_size[0] == scale_size[1]:
        tfl = [T.Resize(scale_size[0], interpolation=interpolation)]   #resize  将图像短边缩放至scale_size, 长宽比不变
    else:
        tfl = [T.Resize(scale_size)]   #将图像缩放至scale size, 长宽比改变

    #tfl += [T.CenterCrop(size)]   #图像从中心向四周裁剪size大小的图像
    tfl += [
        T.ToTensor(),
        T.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std),
        ),
    ]
    return T.Compose(tfl)


def classify_augmentations(size=224,
                           mean=DEFAULT_MEAN,
                           std=DEFAULT_STD,
                           degree=None,
                           translate=None,
                           shear=None,
                           scale=None,
                           ratio=None,
                           hflip=0.5,
                           vflip=0.0,
                           auto_augment=None,
                           hsv_h=0.015,
                           hsv_s=0.4,
                           hsv_v=0.4,
                           force_color_jitter=False,
                           erasing=0.0,
                           interpolation:T.InterpolationMode =T.InterpolationMode.BILINEAR):
        """
        用于训练增强的分类变换
        Args:
            size(int): 图像大小
            scale(tuple): 图像随机缩放范围, 默认（0.08， 1，0）
            ratio(tuple): 图像纵横比范围，默认（3/4， 4/3）
            mean(tuple): RGB通道的平均值
            std(tuple): RGB通道的标准差
            hflip(float): 水平翻转的概率
            vflip(float): 垂直翻转的概率
            auto_augment(str): 自动增强策略，i.e 'randaugment', 'augmix', 'autoaugment', 'None'
            hsv_h(float): 色调增强（0-1）
            hsv_s(float): 饱和度增强（0-1）
            hsv_v(float): 明度增强（0-1）
            force_color_jitter(bool):即使使能了auto augment也要应用color jitter
            erasing(float): 随机擦除的概率
            interpolation(T.InterpolationMode):插入模式，默认线性
        Returns:
            (T.Compose):torchvision transforms
        """
        if not isinstance(size, int):
            raise TypeError(f"'classify_transforms()'的输入参数size={size}必须是整数")

        scale = tuple(scale or (0.08, 1.0))
        ratio = tuple(ratio or (3.0/4.0, 4.0/3.0))
        if auto_augment:
            primary_tfl =[T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
        else:
            primary_tfl = [T.Resize(size, interpolation=interpolation)]
        if hflip > 0.0:
            primary_tfl += [T.RandomHorizontalFlip(p=hflip)]
        if vflip > 0.0:
            primary_tfl += [T.RandomVerticalFlip(p=vflip)]


        secondary_tfl = []
        disable_color_jitter = False
        
        if auto_augment:
            assert isinstance(auto_augment, str)
            disable_color_jitter = not force_color_jitter

            if auto_augment == "randaugment":
                if TORCHVISION_0_11:
                    secondary_tfl += [T.RandAugment(interpolation=interpolation)]
                else:
                    LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

            elif auto_augment == "augmix":
                if TORCHVISION_0_13:
                    secondary_tfl += [T.AugMix(interpolation=interpolation)]
                else:
                    LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

            elif auto_augment == "autoaugment":
                if TORCHVISION_0_10:
                    secondary_tfl += [T.AutoAugment(interpolation=interpolation)]
                else:
                    LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')
            else:
                raise ValueError(
                    f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                    f'"augmix", "autoaugment" or None'
                )
        if not disable_color_jitter:
            secondary_tfl = [T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h)]

        final_tfl = [T.ToTensor(),
                     T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
                     T.RandomErasing(p=erasing,inplace=True)]

        return T.Compose(primary_tfl + secondary_tfl + final_tfl)