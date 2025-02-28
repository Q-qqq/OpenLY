import contextlib

from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.files import increment_path
import torch
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils.checks import is_ascii, check_font, check_version


mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class Colors:
    """
    调色板
    Attribute:
        palette(list of tuple): RGB 颜色值
        n(int): 调色板中颜色数量
        pose_palette(np.array): 指定的调色板矩阵np.uint8
    """
    def __init__(self):
        hexes = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexes]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ], dtype=np.uint8
        )

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """转换16进制颜色代码为RGB值，默认PIL排序"""
        return  tuple(int(h[1+i : 1+i+2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'

class Annotator:
    """
    对拼接图像进行预测
    Attributes:
        im(Image.Image | numpy.array): 需要注释的图像
        pil(bool):是否使用PIL或者cv2绘制注释
        font(ImageFont.truetype | ImageFont.Load_default): 注释的字体
        lw(font): 绘制线宽度
        skeleton(List[List[int]): 关键点的骨骼结构（各点连接）
        limb_color(List[int]):四肢的调色板
        kpt_color(List[int]): 关键点的调色板"""

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        assert im.data.contiguous, "图像存储不连续，使用'np.ascontiguousarray(im)'去处理'Annotator()'的输入图像"
        non_ascii = not is_ascii(example)   # non-latin labels i.e asian, arabic, cyrillic
        self.pil = pil or non_ascii
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  #line width
        if self.pil:
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font("simsun.ttc" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 *0.005), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  #text width, height
        else: #cv2
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw-1, 1)  #font thickness
            self.sf = self.lw / 3  # font scale

        #Pose
        self.skeleton=[
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self,box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """增加一个带label的xyxybox到image"""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                #NOTE:PIL绘制多边形需要tuple类型
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  #box
            if label:
                w, h = self.font.getsize(label)
                outside = p1[1] - h >= 0  #标签是否超出图像外
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1],  p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else: #cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                #NOTE: cv2绘制多边形需要np.asarray类型
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1,p2,color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  #text width height
                outside = p1[1] - h >= 3  #out image
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  #填充标签范围
                cv2.putText(
                    self.im,
                    label,
                    (p1[0],p1[1]-2 if outside else p1[1] + h + 2),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA
                )

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """在图像上绘制masks
        Args:
            masks(tensor):shape(n, h, w)预测masks
            colors(List[List[Int]]): 预测masks的颜色[[r,g,b] *n]
            im_gpu(tensor): 在cuda上的图像，shape(3, h, w) range[0,1]
            alpha(float): 透明度，0全透明，1全覆盖
            retina_masks(bool):是否使用高分辨率masks"""
        if self.pil:
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  #shape(n, 3)  #颜色归一化
        colors = colors[:, None, None]  #shape (n,1,1,3)
        masks = masks.unsqueeze(3) #shape(n,h,w,1)
        masks_color = masks * (colors * alpha) #shape(n,h,w,3)

        inv_alpha_masks = (1 - masks * alpha).cumprod(0)  #shape(n,h,w,1) 越往上颜色越深，或者相反
        mcs = masks_color.max(dim=0).values  #shape(h,w,3)   #masks在各个像素点的最大数值

        im_gpu = im_gpu.flip(dims=[0])   #flip channel 上下翻转 bgr -> rgb
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  #shape(h, w, 3)
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs  #图像*最深的透明度 + 最大数值的mask颜色
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)
        if self.pil:
            self.fromarray(self.im)

    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True):
        """
        Args:
            kpts(tensor): 预测的keypoints。 shape(17, 3), (x, y, confidence)
            shape(tuple): image shape (h, w)
            radius(int, optional): 绘制关键点的半径大小，默认5
            kpt_line(bool,optional): 是否绘制关键点连线，针对人体位姿
        """
        if self.pil:
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}  #人体位姿点
        kpt_line &= is_pose
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)   #关键点颜色
            x_coord, y_coord = k[0], k[1]   #关键点坐标
            if x_coord % shape[1] != 0 and y_coord & shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))   #起点
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))   #终点
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
        if self.pil:
            self.fromarray(self.im)



    def fromarray(self, im):
        """从一个numpy.array更新self.im"""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255,255,255), anchor="top", box_style=False):
        if anchor == "bottom":
            w, h = self.font.getsize(text)
            xy[1] += 1-h   #往下沉字体高度
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0]+w+1, xy[1]+h+1), fill=txt_color)
                txt_color = (255,255,255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                outside = xy[1] - h >= 3
                p2 = xy[0] + w, xy[1] - h - 3 if outside else xy[1] + h + 3
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def result(self):
        """返回np.ndarray格式的注释图像"""
        return self.im

@threaded
def plot_images(
        images,
        batch_idx,
        cls,
        bboxes=np.zeros(0, dtype=np.float32),
        confs=None,
        masks=np.zeros(0, dtype=np.uint8),
        kpts=np.zeros((0, 51), dtype=np.float32),
        paths=None,
        fname="image.jpg",
        names=None,
        on_plot=None,
        max_subplots=16,
        save=True,
):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920   #最大图像大小
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)  #限制绘制图像数量
    ns = np.ceil(bs**0.5)  #subplots的数量
    if np.max(images[0]) <= 1:
        images *= 255    #去归一化

    #Build image
    mosaic = np.full((int(ns*h), int(ns*w), 3), 255, dtype=np.uint8)  #底图
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))   #块原点
        mosaic[y : y+h, x : x+w, :] = images[i].transpose(1, 2, 0)  #覆盖底图

    #Resize
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale*h)
        w = math.ceil(scale*w)
        mosaic = cv2.resize(mosaic, tuple(int(x*ns) for x in (w, h)))

    #Annotate
    fs = int((h+w) * ns * 0.01) #字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):  #循环绘制块
        x, y = int(w * (i // ns)), int(h * (i%ns))  #绘制块图像原点
        annotator.rectangle([x, y, x+w, y+h], None, (255,255,255), width=2)
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220,220,200))  #文件名称
        if len(cls) > 0:
            idx = batch_idx == i  #一张图像目标框索引
            classes = cls[idx].astype("int")
            labels = confs is None
            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None
                is_obb = boxes.shape[-1] == 5  #xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  #是否归一化
                        boxes[..., 0::2] *= w
                        boxes[..., 1::2] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale   #缩放至适应图像大小
                boxes[..., 0::2] += x   #将boxes挪移至图像块处
                boxes[..., 1::2] += y
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]  #cls
                    color = colors(c)  #种类颜色
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > 0.25:  #没有置信度或者置信度达标
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box,label,color=color,rotated=is_obb)  #绘制目标框
            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            #keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() < 1.01:  #归一化
                        kpts_[..., 0] *= w
                        kpts_[..., 1] *= h
                    elif scale < 1:
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > 0.25:
                        annotator.kpts(kpts_[j])
            #masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:   #重叠masks=False
                    image_masks = masks[idx]
                else:  #overlap_masks=True
                    image_masks = masks[[i]]   #(1, h, w)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)  #(nl, h, w) 将重叠masks复制nl层
                    image_masks = np.where(image_masks == index, 1.0, 0.0)  #将每一层恢复至对应1，0状态，1表示目标，0表示背景

                im = np.array(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > 0.25:
                        color = colors(classes[j])  #种类颜色
                        mh ,mw = image_masks[j].shape
                        if mh != h or mw != w:   #mask.shape != image.shape
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y: y + h, x: x + w, :][mask] = (
                                    im[y: y + h, x: x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    if on_plot:
        on_plot(fname)

def output_to_target(output, max_det=300):
    """转换模型输出为目标格式（batch_id,class_id,x,y,w,h,conf）"""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4,1,1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf),1))   #batch_id, cls, x, y, w, h, conf
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]   #batch_id, cls, box, conf

def output_to_rotated_target(output, max_det=300):
    """转换obb模型预测输出格式"""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, angle = o[:max_det].cpu().split((4,1,1,1),1)
        j = torch.full((conf.shape[0],1), i)
        targets.append(torch.cat((j,cls,box, angle, conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]  #batch_id, class_id, box(x,y,w,h,r), conf

@plt_settings()
def plot_results(file="path/to/results.csv", segment=False, pose=False, classify=False, on_plot=None):
    """从一个csv文件中绘制训练结果"""
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 4, 2, 3]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1,2,3,4,5,6,9,10,13,14,15,16,7,8,11,12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1,2,3,4,5,6,7,10,11,14,15,16,17,18,8,9,12,13]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1,2,3,4,5,8,9,10,6,7]
    ax = ax.ravel()
    assert Path(file).exists(), f"未发现训练结果csv文件{file}"
    try:
        data = pd.read_csv(file)
        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]
        for i, j in enumerate(index):
            y = data.values[:, j].astype("float")
            ax[i].plot(x, y, marker=".", label=Path(file).stem, linewidth=2, markersize=8)   #actual results
            ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  #smoothing line
            ax[i].set_title(s[j], fontsize=12)
    except Exception as e:
        LOGGER.warning(f"WARNING: Plotting Error {file} : {e}")
    ax[1].legend()
    fname = Path(file).with_suffix(".png")
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)

def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    保存图像的剪裁区域
    Args:
        xyxy(torch.Tensor or list): boxes xyxy
        im(numpy.ndarray): 输入图像
        file(Path, optional): 裁剪图像保存路径
        gain(float, optional): 一个增大box大小的比例
        pad(int, optional): 增大box长宽大小的像素数
        square(bool, optional): box是否转换为正方形，默认False
        BGR(bool, optional): 裁剪图像是否保存为RGB图像，默认False
        save(bool, optional): 是否保存到存储盘内，默认True
    Returns:
        (numpy.ndarray): 裁剪图像
    """
    if not isinstance(xyxy, torch.Tensor):
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1,4))
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)   #适应长边
    b[:, 2:] = b[:, 2:] * gain + pad  # new box
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0,0]) : int(xyxy[0,2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)
        f = str(increment_path(file).with_suffix(".jpg"))
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)
    return crop

def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """可视化一个给定模型模块的特征图
    Args:
        x(torch.Tensor):需要可视化的模块运行输出向量
        module_type(str):模块类型
        stage(int):在模型中，该模块排第几阶段
        n(int, optional):需要取绘制的最大特征图的数量
        save_dir(Path, optional): 保存路径"""
    for m in ["Detect","Pose", "Segment"]:
        if m in module_type:
            return

    batch, channels, height, width = x.shape
    if height > 1 and width > 1:
        f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  #filename
        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  #选择batch[0]的输出，并将各通道分开
        n = min(n, channels)  #绘制数量
        fig, ax = plt.subplots(math.ceil(n/8), 8, tight_layout=True)  #图像矩阵：8行  n.8列
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  #显示
            ax[i].axis("off")
        LOGGER.info(f"Saving {f}...({n}/{channels})")
        plt.savefig(f, dpi=300, bbox_inches="tight")#保存图像
        plt.close()
        np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())   #缓存



