import contextlib
import gc
import os
from pathlib import Path
import random
import math

import torch
import torchvision.datasets
from torch.utils.data import Dataset
from PIL import Image,ExifTags
import numpy as np

from ultralytics.utils import (NUM_THREADS, LOCAL_RANK)
import cv2
from ultralytics.utils import colorstr,is_dir_writeable,cv2_readimg, LOGGER, PROGRESS_BAR
from ultralytics.data.base import BaseDataset
from ultralytics.utils.ops import resample_segments
from ultralytics.data.utils import (verify_image_label,
                                     get_hash,
                                    img2label_paths,
                                    Format,
                                    verify_image,
                                    )
from ultralytics.utils.instance import Instances
from ultralytics.data.augment import v8_transforms,Compose,LetterBox,classify_augmentations,classify_transforms
from multiprocessing.pool import ThreadPool
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    def __init__(self,*args, data=None,task="detect",**kwargs):
        """data: 参数字典"""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        super().__init__(*args,**kwargs)

    def get_labels(self):
        """加载标签"""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")   #缓存文件
        try:
            cache = load_dataset_cache_file(cache_path)
            exists = True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  #确保数据集数量未变化
        except(FileNotFoundError, AssertionError, AttributeError):
            cache,exists = self.cache_labels(cache_path, self.im_files), False   #主动缓存加载

        #显示
        nf,nm,ne,nc,npc,n = cache.pop("results")  # found,missing,empty,corrupt,per class,total
        if exists and LOCAL_RANK in (-1,0):
            LOGGER.info(f"读取路径{cache_path}...找到{nf}个带标签图像,{nm}个无标签图像 ,{ne}个空白图像,{nc}个读取失败图像\n")
            LOGGER.info("各种类特征数量：")
            for i in range(len(self.data["names"])):
                name = self.data["names"][i]
                LOGGER.info(f"{name}-{npc[i]}")
            LOGGER.info("\n")

        #读取 缓存
        [cache.pop(k) for k in ("hash","version","msgs")]  #去除没用的项
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"在{cache_path}上未找到训练图像，训练无法正常运行")
        self.im_files = [lb["im_file"] for lb in labels]
        self.shapes = [lb["shape"] for lb in labels]
        self.bboxes = [lb["bboxes"] for lb in labels]
        #检查数据集是全目标检测或者全像素分割
        lengths = ((len(lb["cls"]),len(lb["bboxes"]),len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(f"目标检测和分割标签数量应相等或全目标检测或全分割，但现在len(segment)={len_segments},len(boxes)={len_boxes}。\n"
                           f"将移除分割标签，避免混合数据集")
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"{cache_path}中未发现标签，训练将无法正常进行")
        return labels



    def cache_labels(self,path, im_files, progress=True):
        path = Path(path)
        x = {"labels":[]}
        nm,nf,ne,nc,npc,msgs = 0,0,0,0,np.array([0]*len(self.data["names"])),[] #miss found empty corrupt messages
        total = len(im_files)
        nkpt,ndim = self.data.get("kpt_shape",(0,0))
        label_files = img2label_paths(im_files)
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2,3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        if progress:
            PROGRESS_BAR.show("数据集加载", "开始加载")
            PROGRESS_BAR.start(0, total, False)
        with ThreadPool(NUM_THREADS ) as pool:
            results = pool.imap(
                verify_image_label,
                zip(im_files,
                    label_files,
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim))
            )
            for i, (im_file, lb, shape, segments, keypoint, nm_f,nf_f, ne_f,nc_f,npc_f,msg) in enumerate(results):
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                npc += npc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file = im_file,
                            shape=shape,
                            cls=lb[:,0:1],    #[n,1]
                            bboxes=lb[:,1:],  #[n,4]
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh"
                        )
                    )
                if msg:
                    msgs.append(f"{im_file}:{msg}")
                if progress:
                    PROGRESS_BAR.setValue(i+1, f"数据集加载中...{im_file if im_file else msg}")
        if msgs:
            LOGGER.info("\n".join(msgs))
        if progress:
            PROGRESS_BAR.close()
        assert nf != 0, f"在路径{path}上未找到标签"
        x["hash"] = get_hash(im_files + label_files)
        x["results"] = nf, nm, ne, nc, npc, len(im_files)
        x["msgs"] = msgs
        if len(im_files) != len(self.im_files):
            x["version"] = DATASET_CACHE_VERSION
        else:
            save_dataset_cache_file(path, x)
        return x

    def update_labels_info(self, label):
        """label : 单个图像标签"""
        bboxes = label.pop("bboxes")
        segments = label.pop("segments",[])
        keypoints = label.pop("keypoints",None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:  #存在分割数据集
            #上采样(N,m,2)->(N,segment_resmaples,2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0,segment_resamples,2),dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    def build_transforms(self, hyp=None):
        if self.augment:  #数据增强
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0  #图像拼接不与rect兼容
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0   #随机混合图像不予rect兼容
            transforms = v8_transforms(self,self.img_size,hyp)   #数据增强函数
        else:
            transforms = Compose([LetterBox(new_shape=(self.img_size, self.img_size),scaleup=False)])   #只自适应填充，不增强
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


    @staticmethod
    def collate_fn(batch):    #batch->batch个labels 合并
        new_batch = {}
        keys = batch[0].keys()   #单个labels
        values = list(zip(*[list(b.values()) for b in batch]))   #将labels字典里的元素进行一一对应归类，例如batch[0]和batch[1]的“img”都存放到一个tuple内
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)   #图像合并
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)    #标签合并
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])    #每个图像对一个一个batch_idx   batch_idx是一个Tensor.zeros， 元素个数为bbox的数量
        for i in range(len(new_batch["batch_idx"])):    #遍历每个图像的batch_idx
            new_batch["batch_idx"][i] += i      #为所有batch_idx按图像顺序赋值图像索引，用于bulid_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    分类数据集
    Attributes:
        root(str):数据集路径
        cache_ram(bool): 图像是否缓存于RAM
        cache_disk(bool): 图像是否缓存于硬盘
        samples(list): 样本列表，包含file, index, npy, im
        torch_transforms(callable): torchvision转换，对数据集进行数据增强
        album_transforms(callable, optional): Albumentations转换，对数据集进行数据增强，需要augment为True
        """
    def __init__(self, root, args, augment=False, cache=False, prefix=""):
        super().__init__(root=root)    #通过所给分类数据集路径，获取图像，图像路径，标签，分类类别名称等数据
        if augment and args.fraction > 1.0:
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]  #获取数据集fraction比例数据
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples,self.shapes = self.verify_images(self.samples)  #过滤损坏图像
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  #file, cls_index, npy, im
        scale = (1.0 - args.scale, 1.0)  #(0.08, 1.0)
        self.torch_transforms= (classify_augmentations(size=args.imgsz,
                                                       degree=args.degrees,
                                                       translate=args.translate,
                                                       shear=args.shear,
                                                       scale=scale,
                                                       hflip=args.fliplr,
                                                       vflip=args.flipud,
                                                       erasing=args.erasing,
                                                       auto_augment=args.auto_augment,
                                                       hsv_h=args.hsv_h,
                                                       hsv_s=args.hsv_s,
                                                       hsv_v=args.hsv_v) if augment
                                else
                                classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
                                )


    def verify_images(self, im_cls, progress=True):
        """验证所有图像是否损坏"""
        desc = f"{self.prefix}扫描{self.root}..."
        path = Path(self.root).with_suffix(".cache")

        #读取缓存
        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)   #尝试加载缓存文件
            assert  cache["version"] == DATASET_CACHE_VERSION   #验证版本
            assert cache["hash"] == get_hash(x[0] for x in im_cls)   #验证图像路径
            nf, nc, n, samples = cache.pop("results")  #found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f"{desc} 找到{nf}张图像, {nc}张损坏"
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))
            return samples

        #主动加载
        if progress:
            PROGRESS_BAR.show("数据集加载", "开始加载")
            PROGRESS_BAR.start(0,len(im_cls), False)
        nf, nc, msgs, samples, x,shapes = 0, 0, [], [], {}, []
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(im_cls, repeat(self.prefix)))
            for i,(sample, nf_f, nc_f, msg, shape) in enumerate(results):  #sample:[im_file,cls]
                if nf_f:
                    samples.append(sample)
                    shapes.append(shape)
                    if msg:
                        msgs.append(sample)
                    nf += nf_f
                    nc += nc_f
                if progress:
                    PROGRESS_BAR.setValue(i+1, f"数据集加载中...{sample[0]}, {sample[1]}")
        if msgs:
            LOGGER.info("\n".join(msgs))
        if progress:
            PROGRESS_BAR.close()
        x["hash"] = get_hash([x[0] for x in im_cls])
        x["results"] = nf, nc, len(samples), samples
        x["msgs"] = msgs
        save_dataset_cache_file(path, x)
        return samples,shapes

    def __getitem__(self, i):
        f, j, fn, im = self.samples[i]  #filename, cls_index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2_readimg(f)
        elif self.cache_disk:
            if not fn.exists(): #load npy
                np.save(fn.as_posix(), cv2_readimg(f), allow_pickle=False)
            im = np.load(fn)
        else: #read image
            im = cv2_readimg(f)  #BGR
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  #PIL Image
        sample = self.torch_transforms(im)  #Tensor
        return {"img":sample, "cls":j, "im_file":f}

    def __len__(self):
        return len(self.samples)



def load_dataset_cache_file(path):
    """加载标签缓存文件"""
    import gc
    gc.disable()  #关闭gc回收，减少载入时间
    cache = np.load(str(path),allow_pickle=True).item()
    gc.enable()
    return cache

def save_dataset_cache_file(path,x):
    x["version"] = DATASET_CACHE_VERSION
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  #移除旧缓存
        np.save(str(path),x)
        path.with_suffix(".cache.npy").rename(path)  #移除 .npy suffix
    else:
        LOGGER.warning(f"缓存路径{path.parent}不可被写入,缓存失败")