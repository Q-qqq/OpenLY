import re
import math
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import time
import contextlib
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils import LOGGER

#region 目标检测框格式转换
def xyxy2xywh(x):
    """
    转换检测框从格式（x1,y1,x2,y2）到（x,y,w,h）
    :param x(np.ndarray|torch.Tensor):
    :return y(np.ndarray|torch.Tensor):
    """
    assert x.shape[-1] == 4, f"输入shape最后一个维度应为4，实际为{x.shape}"
    y = torch.empty_like(x) if isinstance(x,torch.Tensor) else np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    """转换检测框从格式（x,y,w,h）到（x1,y1,x2,y2）
    :param x(np.ndarray|torch.Tensor):
    :return y(np.ndarray|torch.Tensor):"""
    assert x.shape[-1] == 4, f"输入尺寸的最后一个维度应为4，现为{x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    dw = x[...,2] / 2  #宽的一半
    dh = x[...,3] / 2  #高的一半
    y[...,0] = x[...,0] - dw   #左上角点 x
    y[...,1] = x[...,1] - dh   #左上角点 y
    y[...,2] = x[...,0] + dw   #右上角点 x
    y[...,3] = x[...,1] + dh   #右上角点 y
    return y

def xyxy2ltwh(x):
    """
        转换检测框从格式（x1,y1,x2,y2）到（x1,y1,w,h）
        :param x(np.ndarray|torch.Tensor):
        :return y(np.ndarray|torch.Tensor):
        """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,2] = x[...,2] - x[...,0]  #width
    y[...,3] = x[...,3] - x[...,1]  #height
    return x

def ltwh2xyxy(x):
    """
        转换检测框从格式（x1,y1,w,h）到（x1,y1,x2,y2）
        :param x(np.ndarray|torch.Tensor):
        :return y(np.ndarray|torch.Tensor):
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,2] = x[..., 0] + x[...,2] #x2
    y[...,3] = x[..., 1] + x[...,3] #y2
    return y

def xywh2ltwh(x):
    """
         转换检测框从格式（x,y,w,h）到（x1,y1,w,h）
         :param x(np.ndarray|torch.Tensor):[N,4]
         :return y(np.ndarray|torch.Tensor):[N,4]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,0] = x[..., 0] - x[..., 2] / 2 #左上角x
    y[...,1] = x[..., 1] - x[..., 3] / 2 #左上角y
    return y

def ltwh2xywh(x):
    """
         转换检测框从格式（x1,y1,w,h）到（x,y,w,h）
         :param x(np.ndarray|torch.Tensor):[N,4]
         :return y(np.ndarray|torch.Tensor):[N,4]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,0] = x[...,0] + x[...,2]/2 #中心x
    y[...,1] = x[...,1] + x[...,3]/2 #中心y
    return y
#endregion

def xyxyxyxy2xywhr(corners):
    '''
    将Oriented Bouding Boxes(OBB）格式[xy1,xy2,xy3,xy4]的目标检测框 转换为[x,y,w,h，rotation]的带角度目标检测框格式，rotation 0-90
    :param corners(numpy.ndarray|torch.Tensor): n个框，每个框四个角点（x1, y1, x2, y2, x3, y3, x4, y4）  shape-(n,8)
    :return(numpy.ndarray|torch.Tensor): n个框，一点(x,y)，一宽（w），一高（h），一角度（rotation） shape-(n,5)
    '''
    is_torch = isinstance(corners, torch.Tensor)
    points = corners.cpu().numpy() if is_torch else corners
    points = points.reshape(len(corners), -1, 2)
    rboxes = []
    for pts in points:
        #NOTE: 使用cv2.minAreaRect（返回点集的最小矩形） 获取准确的xywhr
        #需要注意的是一些目标已经被数据增强剪切了
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle/180 * np.pi])
    return (
        torch.tensor(rboxes, device=corners.device, dtype=corners.dtype) if is_torch else
        np.asarray(rboxes, dtype=points.dtype)
    )

def xywhr2xyxyxyxy(rboxes):
    """
    将带角度的OBB目标框转换成四个角点格式
    Args:
        rboxes(numpy.ndarray | torch.Tensor): shape(n, 5) / (b, n, 5)
    Returns:
        (numpy.ndarray | torch.Tensor): (n, 4, 2) or (b, n, 4, 2)
    """
    is_numpy = isinstance(rboxes, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = rboxes[..., :2]    #xy
    w, h, angle = (rboxes[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2* cos_value, w / 2 * sin_value]
    vec2 = [-h /2 *sin_value, h / 2*cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2   #右下/左下
    pt2 = ctr + vec1 - vec2   #右上/左上
    pt3 = ctr - vec1 - vec2   #左上/右上
    pt4 = ctr - vec1 + vec2   #左下/右下
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)



def segments2boxes(segments):
    """
    将分割标签转换为目标检测框标签
    :param segment(torch.Tensor):分割标签 n*m*2
    :return: 目标检测框标签
    """
    boxes = []
    for s in segments:
        x, y = s.T   #m*2  ->   2*m
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))   #n, 4

def segment2box(segment, width=640, height=640):
    '''将一个分割标签转换呈1个目标检测狂'''
    x,y = segment.T #分割标签的x、y坐标
    inside = (x >= 0) & (y >= 0)  & (x < width) & (y < height) #在图像内部的x、y值
    x = x[inside]
    y = y[inside]
    return (np.array([x.min(), y.min(), x.max(), y.max()], dtype = segment.dtype) if any(x) else np.zeros(4, dtype=segment.dtype))

def resample_segments(segments, n=1000):
    """输入一个分割数据集列表(m，2)，对其进行上取样（密集取样），返回一个（n，2）的分割数据集列表，n默认1000"""
    for i,s in enumerate(segments):
        s = np.concatenate((s,s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (np.concatenate([np.interp(x,xp,s[:,i]) for i in range(2)], dtype=np.float32).reshape(2,-1).T)
    return segments

def crop_mask(masks, boxes):
    """
    剪裁masks在boxes内
    Args:
        masks(torch.Tensor): shape(n ,h ,w)
        boxes(torch.Tensor): shape(n,4)   xyxy
    Returns:
        (torch.Tensor): shape(n, h, w) 剪裁在boxes内的新masks
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)   #x1 shape(n, 1, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]    #rows (1, 1, w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]    #cols (1, h, 1)
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps

class Profile(contextlib.ContextDecorator):
    """YOLOv8计时类"""
    def __init__(self, t=0.0, device:torch.device=None):
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """开始计时"""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """停止计时"""
        self.dt = self.time() - self.start
        self.t += self.dt #累加

    def __str__(self):
        return f"Elpsed time is {self.t}s"

    def time(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()

def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for obbs
    Args:
        boxes(torch.Tensor):shape（N，5）  xywhr
        scores(torch.Tensor): shape(N, )   预测分数
        threshold(float): IoU阈值"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)  #从大到小
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)   #只取对角线以上的部分。其余为0  因为输入的两个boxes相同，所以去除重复的部分
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)   #每一个框只跟比自己分数大的框做iou比较， 小于阈值则保留，大于阈值不保留
    return sorted_idx[pick]

def v5_non_max_suppression(
        prediction,
        conf_thres = 0.35,
        iou_thres = 0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0):
    #prediction (bs, h*w*nl, 4+nc+nm)
    assert 0<=conf_thres<=1, "无效的置信度阈值"
    assert 0<=iou_thres<=1, "无效的IoU阈值"
    if isinstance(prediction, (list, tuple)):  #YOLOv8模型在验证时的输出为（inference_out, loss_out）
        prediction = prediction[0]  # 只选推理输出
    bs = prediction.shape[0]  #batch size
    nc = nc or prediction.shape[2] - 5  #种类数量
    nm = prediction.shape[2] - nc - 5
    xc = prediction[...,4] > conf_thres   #置信度大于阈值的索引

    max_wh = 7680       #最大的图像长宽
    max_nms = 30000   #计算nms时一张图像内最大检测目标数目
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6+nm), device=prediction.device)] * bs
    for img_i,x in enumerate(prediction):   #image index,  pred in a image
        x = x[xc[img_i]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[img_i]):
            lb = labels[img_i]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)  # append labels
        
        if not x.shape[0]:   #图像内无检测到框，下一张图像
            continue

        x[:,5:] *= x[:,4:5]  #类别概率乘以置信度
        

        box = xywh2xyxy(x[:,0:4])        #xywh  to xyxy
        mask = x[:, mi:]   #分割掩膜

        #[box conf cls]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5+j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf,j = x[:, 5:mi].max(1,keepdim = True)    #最大的置信度   类别索引
            x = torch.cat((box, conf, j.float(), mask),1)[conf.view(-1) > conf_thres]      #置信度大于阈值的[box conf cls]  box - xyxy

        #Filter by class
        if classes is not None:
            x = x[(x[:,5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:        #无目标，下一张图像
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        #NMS
        c = x[:,5:6] * (0 if agnostic else max_wh)  #类别 * 4096 放大类别差
        boxes, scores = x[:,:4] + c, x[:,4]     #将不同类别的框加上不同的偏差，进行区分，scores为各个框的置信度分数
        i = torchvision.ops.boxes.nms(boxes,scores,iou_thres)       #去除相同类别相近（iou > iou_thres)的框，并按置信度排序输出
        i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[img_i] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output # (bs, 6) xywh conf cls

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7600,
        rotated=False,
    ):
    """
    在一个boxes集上进行非最大值抑制， 支持掩膜和每个box有多个标签
    Args:
        prediction(torch.Tensor): shape(batch_size, 4+num_classes+num_masks, num_boxes) 包含预测框，种类和masks，
        conf_thres(float): 0-1，置信度阈值，在阈值下的boxes将舍弃
        iou_thres(float): 0-1，IoU阈值，在NMS期间，低于IoU阈值的boxes将舍弃
        classes(List[int]): 种类索引列表，如果为None，将考虑所有种类
        agnostic(bool): 如果为True，将忽略种类，将所有种类认为同一种类
        multi_label(bool): 如果为True，那么每一个box可能拥有多个cls
        labels(List[List[Union[int, float, torch.Tensor]]]): 自动贴的标签是一个包含列表的列表，每一个内部列表应该包含所给图像的先验标签，
            列表的格式应该是一个dataloader输出格式（class_index, x1, y1, x2, y2）
        max_det(int): 在NMS后最大的检测boxes数
        nc(int, optional):模型输出的种类数量， 在其之后的任何指数都将被认为masks
        max_time_img(float): 处理一张图像的最大时间（seconds）
        max_nms(int): 输入torchvision.ops.nms()的最大boxes数量
        max_wh(int): 最大box的宽度和高度
    Returns:
        (List[torch.Tensor]): 一个长度batch_size的列表，每一个Tensor shape(num_boxes, 6 + num_masks)
            包含了（x1, y1, x2, y2, condidence, class, mask1, maks2...）
    """
    #Checks
    assert 0 <= conf_thres <= 1, f"无效的置信度阈值{conf_thres}, 请将其设置在0-1"
    assert 0 <= iou_thres <= 1, f"无效的IoU阈值{iou_thres}，请将其设置在0-1"
    if isinstance(prediction, (list, tuple)):  #YOLOv8模型在验证时的输出为（inference_out, loss_out）
        prediction = prediction[0]  # 只选推理输出

    bs = prediction.shape[0]  #batch size
    nc = nc or (prediction.shape[1] - 4) #种类数量
    nm = prediction.shape[1] - nc - 4   #分割masks数量  / rotate
    mi = 4 + nc #mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # 候选 每一个预测的最大种类分数大于置信度阈值

    #Settings
    time_limit = 2.0 + max_time_img * bs   #在time_limit后停止运行
    multi_label &= nc > 1 #每个box对应多个labels。 每张图像增加0.5ms运行时间

    prediction = prediction.transpose(-1, -2) #shape(b, 4+nc+nm, h*w*nl) to (b, h*w*nl, 4+nc+nm)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  #xywh -> xyxy

    t = time.time()
    output = [torch.zeros((0, 6+nm), device=prediction.device)] * bs   #list(Tensor(x1,y1,x2,y2,conf,cls,m1,m2...))
    for xi, x in enumerate(prediction):  #按图像循环
        x = x[xc[xi]]  #第xi张图像里置信度满足要求的预测目标

        #当save_hybrid为True时，会将真实目标框和预测框同时添加至lb，传入的labels已包含真实目标框
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm +4), device=x.device)
            v[:, 4] = xywh2xyxy(lb[:, 1:5])   #bbox
            v[range(len(lb)), lb[:, 0].long + 4] = 1.0   #cls
            x = torch.cat((x, v), 0)   #真实目标框+预测框

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)  #(xyxy, conf, cls, mask1,mask2...）

        if multi_label:
            i, j = torch.where(cls > conf_thres)  #取分数大于阈值的种类 i 第几个，j 第几类(多个)
            x = torch.cat((box[i], x[i, 4+j, None], j[:, None].float(), mask[i]),1)
        else:
            conf, j = cls.max(1, keepdim=True)  #取分数最大的一个种类
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        #过滤种类
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        #Check shape
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]   #排序从大到小 取前max_nms个

        #Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  #种类要么忽略，全为0，要么放大对比。同乘以7600
        scores = x[:, 4]   #分数
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:,-1:]), dim=-1)  #xywhr   r在x的最后    xy+c将各种类框分开
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c #按种类分开
            i = torchvision.ops.nms(boxes, scores, iou_thres)  #NMS
        i = i[:max_det]  #限制检测数量
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS超时，用时{time.time()-t}, 限时{time_limit:.3f}s ")
            break  # time limit exceeded
    return output   #bs, n, 6+m

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    将适应于img1的boxes缩放至适应img0, 会进行填充适应
    Args:
        img1_shape(tuple): (h, w) 输入边界框对应的图像大小
        boxes(torch.Tensor): format(x1, y1, x2, y2)
        img0_shape(tuple): （h, w）边界框要去适应的图像大小
        ratio_pad(tuple): (ratio, pad)用于缩放boxes，如果未提供，则通过两个图像shape计算得出
        padding(bool): 如果为真，boxes通过填充方式的图像缩放方式进行缩放，否则，通过传统的直接缩放进行缩放
        xywh()bool: 输入boxes是否format为xywh
    Returns:
        new_boxes(torch.Tensor)：(x1, y1, x2, y2)"""
    if ratio_pad is None:
        scale = min(img1_shape[0] / img0_shape[0], img1_shape[1]/ img0_shape[1])  #scale = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1]*scale) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0]*scale) / 2 - 0.1)
        )   #wh padding
    else:
        scale = ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        boxes[..., 0] -= pad[0]    # x - pad_w
        boxes[..., 1] -= pad[1]    # y - pad_h
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]
    boxes[..., :4] /= scale    #缩放到适应img0
    return clip_boxes(boxes, img0_shape)

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  #x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  #y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  #x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  #y2
    else: #np.array
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  #x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  #y1, y2
    return boxes

def scale_image(masks, im0_shape, ratio_pad=None):
    """输入一个mask，并降至缩放至原图像大小"""
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:
        scale = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # old / new
        pad = (im1_shape[1] - im0_shape[1]*scale) / 2, (im1_shape[0] - im0_shape[0]*scale) / 2  #wh padding
    else:
        pad = ratio_pad[1]
    top,left = int(pad[1]), int(pad[0]) #y x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f"masks的shape长度应该是2或者3，但是现在masks.shape={len(masks.shape)}")
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def clean_str(s):
    """通过使用下划线替换特殊字符串的方式清理字符串"""
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def masks2segments(masks, strategy="largest"):
    """
    处理一个masks(n,h,w)列表，返回分割点数据segments(n, 2(xy))
    Args:
        masks(list(torch.Tensor)): 分割模型的输出，一个Tensor的shape(barch_Size, imgh, imgw)
        strategy(str): 'concat' or 'largest'， 默认largest
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":   #连接所有segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  #最大的segments
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1,2)
        else:
            c = np.zeros((0,2))
        segments.append(c.astype("float32"))
    return segments

def clip_coords(coords, shape):
    """限制坐标点在图像范围内"""
    if isinstance(coords, torch.Tensor):
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  #x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])   #y
    else:
        coords[..., 0] = coords[..., 0].clip(0, shape[1])   #x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])   #y
    return coords

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    将属于img1_shape的点坐标缩放至img0_shape
    Args:
        img1_shape(tuple): 初始图像shape(h1,w1)
        coords(torch.Tensor): 分割坐标(n,2)
        img0_shape(tuple): 将要转换过去的图像shape(h0,w0)
        ratio_pad(tuple): 忽略img0_shape，直接应用ratio_pad(ratio, pad)
        normalize(bool): 是否对坐标进行归一化，默认False
        padding(bool): 是否使用yolo风格的缩放填充，不然则直接缩放
    Returns:
        coords(torch.Tensor): 缩放过的点坐标
    """
    if ratio_pad is None:
        scale = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  #scale = old / new
        pad = (img1_shape[1] - img0_shape[1]*scale) / 2, (img1_shape[0] - img0_shape[0]*scale) / 2
    else:
        scale = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]   #x padding
        coords[..., 1] -= pad[1]   #y padding
    coords[..., 0] /= scale
    coords[..., 1] /= scale
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  #w
        coords[..., 1] /= img0_shape[0]  #h
    return coords

def convert_torch2numpy_batch(batch: torch.Tensor):
    """将一批次的FP32（0.0-1.0）的Tensor转换为uint8（0-255）的array，且BCHW->BHWC"""
    return (batch.permute(0,2,3,1).contiguous()*255).clamp(0,255).to(torch.uint8).cpu().numpy()


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    接收分割模型检测头输出的protos掩膜，并将掩膜应用于检测框，获取更高品质的掩膜，但速度较慢
    Args:
        protos(troch.Tensor): [mask_num, mask_h, mask_w] 预测掩膜
        masks_in(torch.Tensor): [n,mask_num] 经过nms后的掩膜系数,n为nms后的目标数量
        bboxes(torch.Tensor): [n,4],经过nms后的目标框
        shape(tuple): 输入图像大小（h, w）
    Return:
        (torch.Tensor): 上采样的masks  shape（n, h ,w）
    """
    c, mh ,mw = protos.shape # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)   #矩阵乘积获取通过了nms的protos，在经过激活层，获取真正的masks
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]   #masks-hw上采样至shape
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.5)  #分数在0.5以上的为真

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    使用分割模型检测头的输出protos应用掩膜到目标框
    Args:
        protos(troch.Tensor): [mask_num, mask_h, mask_w]预测掩膜
        masks_in(torch.Tensor): [n,mask_num] 经过nms后的掩膜系数,n为nms后的目标数量
        bboxes(torch.Tensor): [n,4],经过nms后的目标框
        shape(tuple): 输入图像大小（h, w）
        upsample(bool): 是否将masks上采样到输入图像大小
    Return:
        (torch.Tensor): 经过nms的二值掩膜，如果upsample，则输出掩膜长宽等同shape，如果upsample为False，则输出掩膜长宽等于protos长宽
    """
    c, mh ,mw = protos.shape #CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)

    #适应掩膜大小的box
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw  #x1
    downsampled_bboxes[:, 2] *= mw / iw  #x2
    downsampled_bboxes[:, 3] *= mh / ih  #y1
    downsampled_bboxes[:, 1] *= mh / ih  #y2

    masks = crop_mask(masks, downsampled_bboxes)  #c,mh,mw
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  #c, h, w   #上采样至输入图像大小
    return masks.gt_(0.5)

def process_mask_native(protos, masks_in, bboxes, shape):
    """接收分割模型检测头的输出，并在mask上采样后，对其进行适应box剪切"""
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh,mw)
    masks = scale_masks(masks[None], shape)[0]  #CHW
    masks=crop_mask(masks, bboxes)  #CHW
    return masks.gt_(0.5)


def scale_masks(masks, shape, padding=True):
    """缩放分割掩膜到shape"""
    mh, mw = masks.shape[2:]  #b,c,h,w
    scale = min(mh / shape[0], mw / shape[1])
    pad = [mw - shape[1] * scale, mh - shape[0] * scale]
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)
    return masks

def regularize_rboxes(rboxes):
    """规范化角度在[0,pi/2]范围内的旋转框，使之长边为w，短边为h"""
    x, y, w, h, r = rboxes.unbind(dim=-1)
    #如果h>=w 角度逆时针增加90°，w和h边名称互换
    w_ = torch.where(h>w, h, w)
    h_ = torch.where(h>w, w, h)
    r = torch.where(h>w, r+math.pi/2, r) % math.pi
    return torch.stack([x, y, w_, h_, r], dim=-1)
