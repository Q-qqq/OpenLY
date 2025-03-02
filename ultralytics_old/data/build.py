import os
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS,PIN_MEMORY
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImages,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list
    )


class InfiniteDataLoader(dataloader.DataLoader):
    """重复循环的Dataloader"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """永远重复的采样器
    Args:
        sampler(Dataset.sampler): 要重复的样本"""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)  #不断重复sampler

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() %2 **32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    return YOLODataset(
        img_path=img_path,
        img_size=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode=="train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode=="train" else 1.0,
    )

def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), workers])
    sampler = None if rank==-1 else distributed.DistributedSampler(dataset, shuffle=shuffle)  #DDP样本分配
    generator = torch.Generator()   #随机数生成器
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,   #多卡训练必须False
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,  #随机数生成器 确定的随机种子
    )

def check_source(source):
    """检测源的类型并返回对应的标志"""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  #int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".stream") or (is_url and not is_file)
        screenshot = source.lower() =="screen"
        if is_url and is_file:
            source = check_file(source)
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # 混合所有源
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("不支持的图像源格式")

    return source, webcam, screenshot, from_img, in_memory, tensor




def load_inference_source(source=None, vid_stride=1, buffer=False):
    """加载用于目标检测的推理源并引用必要的转换
    Args:
        source(str, Path, Tensor, PIL.Image, np.ndarray): 推理源
        vid_stride(int, optional): 视频源的帧间隔，默认1
        buffer(bool,optional):决定是否对流帧进行缓冲，默认False
    Returns:
        dataset(Dataset): 指定输入源的数据集对象"""
    source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

    #Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif webcam:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImages(source, vid_stride=vid_stride)

    setattr(dataset, "source_type", source_type)   #添加源类型到dataset
    return dataset