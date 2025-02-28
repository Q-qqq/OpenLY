import math
import random
from copy import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils  import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.models.yolo.detect.val import DetectionValidator

class DetectionTrainer(BaseTrainer):
    """目标检测训练对象"""
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        生成YOLO数据集
        Args:
            img_path(str): 图像路径文件的路径
            mode(str):'train' or 'val'
            batch(int, optional): batch的大小，默认None"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  #最大的stride
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode=="val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert  mode in ["train", "val"]
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode=="train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True'不兼容'shuffle=True',设置'shuffle=False'")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size,workers, shuffle, rank), dataset  #dataloader

    def preprocess_batch(self, batch):
        """图像值归一化并随机缩放"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:   #输入图像随机缩放
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride) // self.stride * self.stride
            )  #图像最长边新的长度
            sf = sz / max(imgs.shape[2:]) #最长边缩放比例
            if sf != 1:  #有缩放
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]  #new shape  长宽皆缩放
                imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  #缩放
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]     #number of classes
        self.model.names = self.data["names"]  #name of classes
        self.model.args = self.args   #hyperparanmeters

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取目标检测模型"""
        model = DetectionModel(cfg, nc=self. data["nc"], verbose=verbose and RANK==-1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix="train"):
        """损失的名称：损失值"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]   #5位数浮点数
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (5 + len(self.loss_names))) %(
            "Epoch","Batch", "GPU_mem", *self.loss_names, "Instances", "Size"
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot)

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)

    #TODO: 忽略，不绘制
    def plot_training_labels(self):
        pass
        '''
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_dict = {"boxes":boxes,
                     "cls":cls.squeeze,
                     "names":self.data["names"],
                     "save_dir":self.save_dir,
                     "on_plot":self.on_plot}
        '''


