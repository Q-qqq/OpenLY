import glob
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from ultralytics.utils import (LOGGER, NUM_THREADS, DEFAULT_CFG, cv2_readimg,PROGRESS_BAR)
from ultralytics.data.utils import IMG_FORMATS
from typing import Optional
import math
import psutil
from pathlib import Path
from multiprocessing.pool import ThreadPool
from copy import deepcopy





class BaseDataset(Dataset):
    def __init__(self,
                 img_path,
                 img_size=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix="",
                 rect=False,
                 batch_size=4,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        super().__init__()
        self.img_path = img_path
        self.img_size = img_size
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels() if self.im_files else []
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.hyp = hyp
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        self.buffer = [] #buffer size = batch size
        self.max_buffer_length = min((self.ni,self.batch_size * 8, 1000)) if self.augment else 0

        #缓存图像
        if cache == "ram" and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None]*self.ni, [None]*self.ni, [None]*self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images(cache)
        #数据增强
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """从所有图像路径文件中获取所有图像路径"""
        try:
            f = [] #image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.suffix.split(".")[-1].lower() in IMG_FORMATS:
                    f += p
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep  #上级路径
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]   #local to global path
                else:
                    raise FileNotFoundError(f"{self.prefix}'{p}'路径不存在")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert  im_files, f"{self.prefix}在{img_path}内未发现图像"
        except Exception as e:
            LOGGER.error(f"{self.prefix}从{img_path}错误加载数据: {str(e)}")
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        im_files = [str(Path(f)) for f in im_files]
        return im_files

    def get_labels(self):
        """
        自定义
        Note：
            该函数输出label字典
            dict(
                img_file:图像文件,
                shape:图像大小（height，width）,
                cls:种类,
                bboxes:xywh/xyxy,
                bbox_format:标注框格式"xyxy","xywh","ltwh",
                segments:分割点xy,
                keypoint:位姿点xy,
                normalized:归一化True/False
                )
                """
        raise NotImplementedError

    def update_labels(self,include_class:Optional[list]):
        #使标签只包含指定的种类
        include_class_array = np.array(include_class).reshape(1,-1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si,idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:,0] = 0

    def set_rectangle(self):
        """自适应缩放"""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  #batch index
        nb = bi[-1] + 1 #number of batches

        #排序
        s = np.array([x.pop("shape") for x in self.labels]) #hw
        ar = s[:,0] / s[:,1]   #aspect ratio(h/w)
        irect = ar.argsort()  #排序索引
        self.im_files = [self.im_files[i] for i in irect]   #重新排序
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        #设置shapes
        shapes = [[1,1]]*nb
        for i in range(nb):
            ari = ar[bi == i]  #属于批次i的h/w
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def cache_images_to_disk(self, i):
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2_readimg(self.im_files[i]), allow_pickle=False)


    def check_cache_ram(self,safety_margin=0.5):
        b, gb = 0, 1 << 30     #0,1073741824
        n = min(self.ni,30)   #随机抽取图像数
        for _ in range(n):
            im = cv2_readimg(random.choice(self.im_files))
            ratio = self.img_size/ max(im.shape[0], im.shape[1])    #max(h,w)  radio
            b += im.nbytes *ratio ** 2
        mem_required = b *self.ni / n *(1+ safety_margin)   #GB 请求缓存至RAN的数据集的内存大小
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available    #是否满足缓存条件
        if not cache:
            LOGGER.info(f"{self.prefix}{mem_required/gb:.1f}GB RAM 请求缓存图像；\n"
                         f"需要{int(safety_margin * 100)}% 安全裕度;\n"
                         f"{mem.available / gb:.1f}/{mem.total/gb:.1f}GB可用;\n"
                         f"满足缓存图像条件✅" if cache else "不满足缓存图像条件⚠️\n")
        return cache



    def load_image(self, i, rect_mode=True):
        """根据索引获取图像"""
        im,f,fn = self.ims[i], self.im_files[i],self.npy_files[i]
        if im is None:
            if fn.exists():  #加载npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}因为: {e}，将移除*.npy图像文件{fn} ")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2_readimg(f)  #BGR
            else:
                im = cv2_readimg(f) #BGR
            if im is None:
                LOGGER.error(f"未发现图像{f}")

            h0,w0 = im.shape[:2] #原图像hw

            if rect_mode:   #适应长边
                r = self.img_size / max(h0,w0)
                if r != 1:
                    w, h = (min(math.ceil(w0*r),self.img_size), min(math.ceil(h0*r),self.img_size))
                    im = cv2.resize(im,(w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.img_size):
                im = cv2.resize(im,(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            if self.augment:
                self.ims[i], self.im_hw0[i],self.im_hw[i] = im ,(h0,w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self,cache):
        b, gb = 0, 1<<30
        PROGRESS_BAR.show()
        PROGRESS_BAR.start(0, self.ni)
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            for i, x in enumerate(results):
                if cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  #ram
                    self.ims[i],self.im_hw0[i],self.im_hw[i] = x  #load_image
                    b += self.ims[i].nbytes
                PROGRESS_BAR.setValue(i+1, f"{self.prefix}缓存图像中...{b / gb:.1f}GB {cache}")
        PROGRESS_BAR.close()

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        label.pop("shape", None)   #原图像shape 移除
        label["img"], label["ori_shape"], label["resize_shape"] = self.load_image(index)
        label["radio_pad"] = (
            label["resize_shape"][0]/label["ori_shape"][0],
            label["resize_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def update_labels_info(self, label):
        return label

    def build_transforms(self, hyp=None):
        """数据增强"""
        raise NotImplementedError

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))





