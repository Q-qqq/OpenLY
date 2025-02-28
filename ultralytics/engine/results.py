from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode

class BaseTensor(SimpleClass):
    """具有其他方法的基础张量类，便于简单操作和驱动处理"""
    def __init__(self, data, orig_shape):
        """data(torch.Tensor | np.ndarray): 预测结果， 例如bboxes, masks, keypoints"""
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        """返回一个在CPU内存上的复制对象"""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """转换device/dtype"""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)



class Boxes(BaseTensor):
    """
    存储和操作目标框
    Attributes:
        xyxy(torch.Tensor | numpy.ndarray): xyxy格式的boxes
        conf(torch.Tensor | numpy.ndarray): boxes的置信度
        cls(torch.Tensor | numpy.ndarray): boxes的种类
        id(torch.Tensor | numpy.ndarray): boxes的跟踪id
        xywh(torch.Tensor | numpy.ndarray): xywh格式的boxes
        xyxyn(torch.Tensor | numpy.ndarray): 归一化的xyxy格式boxes
        xywhn(torch.Tensor | numpy.ndarray): 归一化的xywh格式boxes
        data(torch.Tensor): 原始的bboxes张量
        """

    def __init__(self, boxes, orig_shape):
        if boxes.ndim == 1: #(N,)
            boxes = boxes[None, :]  #(1,N)
        n = boxes.shape[-1]
        assert n in (6, 7), f"boxes的列数只有6或7，现在是{n}" #xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    @lru_cache(maxsize=2)  #缓存装饰器，缓存最近2次结果，如果相同的输入则返回同样的结果
    def xywh(self):
        return ops.xyxy2xywh(self.xyxy)

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0,2]] /= self.orig_shape[1] #x/w
        xyxy[..., [1,3]] /= self.orig_shape[0] #y/h
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh

class Masks(BaseTensor):
    """
        存储和操作目标掩膜
    Attributes:
        xy(list): 分割像素坐标
        xyn(list): 归一化的分割像素坐标
    Methods:
        cpu(): 返回cpu内存的Masks对象
        numpy(): 返回numpy格式的Masks对象
        cuda(): 返回GPU内存的Masks对象
    """
    def __init__(self, masks, orig_shape):
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """返回故意话的分割数据"""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """返回分割数据"""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]

class Keypoints(BaseTensor):
    """
    存储和操作关键点
    Attributes:
        xy(torch.Tensor):shape(n,nkpt, 2) 关键点坐标
        xyn(torch.Tensor): 归一化的关键点坐标
        conf(toch.Tensor): 每个关键点的可视化置信度，如果没有，为None
        """
    @smart_inference_mode()
    def __init__(self, keypoints, orig_shape):
        if keypoints.ndim == 2: #只有一个目标
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3: #没有置信度
            mask = keypoints[..., 2] < 0.5   #conf < 0.5的点 ，不可视
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3  #存在可视化置信度

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """返回点坐标"""
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """返回归一化的点坐标"""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """关键点可见性"""
        return self.data[..., 2] if self.has_visible else None

class Probs(BaseTensor):
    """
    存储和操作分类预测
    Attributes:
        top1(int): 预测分数第一的种类索引
        top5(list(int)): 预测分数前五的种类索引
        top1conf(torch.Tensor): 第一的预测分数
        top5conf(torch.Tensor): 前五的预测分数
    """
    def __init__(self, probs, orig_shape=None):
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        return (-self.data).argsort(0)[:5].tolist()

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        return self.data[self.top5]

class OBB(BaseTensor):
    """
    存储和操作定向框
    Attributes：
        xywhr(torch.Tensor | numpy.ndarray):shape(n,5) xywhr
        conf(torch.Tensor | numpy.ndarray):每个框的置信度
        cls(torch.Tensor | numpy.ndarray): 每个框的种类
        id(torch.Tensor | numpy.ndarray): 每个框的跟踪id
        xyxyxyxyn(torch.Tensor | numpy.ndarray): 旋转框的归一化xyxyxyxy格式
        xyxyxyxy(torch.Tensor | numpy.ndarray): 旋转框的xyxyxyxy格式
        xyxy(torch.Tensor | numpy.ndarray): 包围旋转框的水平框xyxy格式
        data(torch.Tensor): OBBtensor
        """
    def __init__(self, boxes, orig_shape):
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (7, 8), f"OBB Tensor每一行应该有7或8个值，但是现在只有{n}个值"
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        return self.data[:, :5]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]  #x/w
        xyxyxyxyn[..., 1] /= self.orig_shape[0]  #y/h
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        x1 = self.xyxyxyxy[..., 0].min(1).values
        x2 = self.xyxyxyxy[..., 0].max(1).values
        y1 = self.xyxyxyxy[..., 1].min(1).values
        y2 = self.xyxyxyxy[..., 1].max(1).values
        xyxy = [x1, y1, x2, y2]
        return np.stack(xyxy, axis=-1) if isinstance(self.data, np.ndarray) else torch.stack(xyxy, dim=-1)



class Results(SimpleClass):
    """
    存储绘制推理结果
    Args:
        orin_img(numpy.ndarray): 原图
        path(str): 图像文件的路径
        names(dict): 种类名称的字典
        boxes(torch.tensor, optional): 2维tensor，检测目标框
        masks(torch.tensor, optional): 3维tensor，分割掩膜，每一个掩膜是一个二值图像
        probs(torch.tensor, optional): 1维tensor，分类任务的种类概率
        keypoints(List[List[float]], optional): 列表，关键点
    """
    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2] # h ,w
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None

        self.speed = {"preprocess": None, "inference": None, "postprocess": None} #ms
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """获取指定索引数据的子Results"""
        return self._apply("__getitem__", idx)

    def _apply(self, fn, *args, **kwargs):
        """创建一个新的self，从self._keys中不为None的属性中获取属性方法fn，并输入参数运行，返回结果赋值给新的self，返回这个新的子self"""
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def new(self):
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    def __len__(self):
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb. self.orig_shape)

    def cpu(self):
        return self._apply("cpu")

    def numpy(self):
        return self._apply("numpy")

    def cuda(self):
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        return self._apply("to", *args, **kwargs)

    def plot(self,
             conf=True,
             line_width=None,
             font_size=None,
             font="Arial.ttf",
             pil=False,
             img=None,
             im_gpu=None,
             kpt_radius=5,
             kpt_line=True,
             labels=True,
             boxes=True,
             masks=True,
             probs=True):
        """
        将预测目标绘制在输入图像上
        Args:
            conf(bool): 是否绘制置信度
            line_width(float,optional): 目标框的线宽度，如果none,则自适应
            font_size(float, optional): 文本字体大小，如果None，则自适应
            font(str): 字体格式
            pil(bool): 是否返回图像为PIL
            img(numpy.ndarray): 绘制目标图像，如果没有，则绘制在原始目标图像上
            im_gpu（torch.Tensor）: 存储在gpu上的归一化图像，shape(1,3,h,w),为了更快的掩膜绘制
            kpt_radius(int, optional): 关键点绘制半径大小
            kpt_line(bool): 是否绘制关键点连接线
            labels(bool)：是否绘制目标框标签
            boxes(bool): 是否绘制目标框
            masks(bool): 是否绘制掩膜
            probs(bool): 是否绘制分类概率
        Returns:
            (numpy.ndarray): 绘制后的图像
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()   #h, w, 3

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),   #分类任务默认pil
            example=names
        )

        #绘制分割结果
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1).flip(0).contiguous() / 255
                )  #tensor  3,h,w  归一化
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))  #种类 - 对应颜色
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)  #绘制masks

        #绘制目标检测结果
        if pred_boxes is not None and show_boxes:
            for box in reversed(pred_boxes):
                c, conf, id = int(box.cls), float(box.conf) if conf else None, int(box.id.item()) if box.id is not None else None
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = box.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else box.xyxy.squeeze()  #(n,4,2) / (n,4)
                annotator.box_label(box, label, color=colors(c,True), rotated=is_obb)  #绘制

        #绘制分类结果
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255,255,255))

        #绘制关键点结果
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):  #反向迭代
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result()

    def verbose(self):
        """返回每一任务的日志文本"""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections),"
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)},"
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  #单类数量
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}"
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """保存预测结果到文本文件"""
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            #Detect/Segment/Pose
            for j, box in enumerate(boxes):
                c, conf, id = int(box.cls), float(box.conf), int(box.id.item()) if box.id is not None else None
                line = (c, *(box.xyxyxyxyn.view(-1) if is_obb else box.xywhn.view(-1)))   #(n,4,2) -> (n*4*2) / (n, 4) -> (n*4)
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  #(n,2) - (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,)*save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True,exist_ok=True)
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """将预测目标裁剪出来进行保存"""
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ 分类任务不支持'save_crop'")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ 定向框任务不支持 'save_crop'")
            return
        for box in self.boxes:
            save_one_box(
                box.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir)/self.names[int(box.cls)] / f"{Path(file_name)}.jpg",
                BGR=True
            )

    def tojson(self, normalize=False):
        """转换目标为json格式"""
        if self.probs is not None:
            LOGGER.warning(f"WARNING ⚠️ 分类任务不支持'tojson'")
            return

        import json

        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            box = {"x" :row[0] / w, "y1": row[1]/h, "x2": row[2]/w, "y2": row[3]/h}
            conf = row[-2]
            class_id = int(row[-1])
            name = self.names[class_id]
            result = {"name": name, "class": class_id, "confidence": conf, "box": box}
            if self.boxes.is_track:
                result["track_id"] = int(row[-3])
            if self.masks:
                x,y = self.mask.xy[i][:,0], self.masks.xy[i][:, 1]
                result["segments"] = {"x": (x/w).tolist(), "y":(y/h).tolist()}
            if self.keypoints is not None:
                x,y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  #(nkpt,3)
                result["keypoints"] = {"x": (x/w).tolist(), "y": (y/h).tolist(), "visible":visible.tolist()}
            results.append(result)

        return json.dumps(results,indent=2)