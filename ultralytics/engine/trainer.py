import subprocess
import warnings

import torch
import torch.nn as nn
import torch.cuda
import os
import math
import time
import numpy as np
from ultralytics.utils import (DEFAULT_CFG, RANK, yaml_save, emojis, LOGGER, colorstr, __version__)
from ultralytics.utils.auto_anchors import check_anchors
from ultralytics.cfg import get_cfg, get_save_dir
from pathlib import Path
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.checks import check_file, check_model_file_from_stem, check_amp, check_imgsz
from ultralytics.nn.tasks import attempt_load_weights,attempt_load_one_weight
from ultralytics.utils.torch_utils import select_device, init_seeds, ModelEMA,one_cycle,EarlyStopping,de_parallel, strip_optimizer
from ultralytics.data.utils import check_cls_dataset,check_det_dataset
from ultralytics.utils.downloads import clean_url
from ultralytics.utils.dist import generate_ddp_command, ddp_cleanup
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.nn.modules.head import V5Detect, V5Segment
from torch import distributed as dist
from torch import optim
from datetime import datetime, timedelta
from copy import deepcopy

class BaseTrainer:
    """
    一个用于创建训练对象的基础类
    Attributes:
        args(SimpleNamespace): 训练参数
        validator(BaseValidator): 验证实例
        model(nn.Module): 模型实例
        callbacks(defaultdict): 回调字典
        save_dir(Path): 保存训练结果的目录
        wdir(Path): 保存权重的目录
        last(Path): 保存最后训练结果的路径
        best(Path): 保存最好训练结果的路径
        save_period(int): 保存每X个epoch保存一次训练结果
        batch_size(int): 训练批大下
        epochs(int): 训练周期迭代次数
        start_epoch(int): 开始训练时的周期数
        device(torch.device): 训练使用的驱动CPU/GPU
        amp(bool): 是否使用AMP自动混合精度
        scaler(amp.GradScaler): amp的梯度缩放器
        data(str):数据集路径
        trainset(torch.utils.data.Dataset): 训练集
        testset(torch.utils.data.Dataset): 测试集
        ema(nn.Module): 模型的EMA（Exponential Moving Average）
        resume(bool): 从某一个训练结果中恢复训练
        lf(nn.Module): 损失函数
        scheduler(torch.optim.lr_scheduler._LRScheduler): 学习率曲线
        best_fitness(float): 最好的训练拟合值
        fitness(float): 当前的训练拟合值
        loss(float): 当前损失值
        tloss(float): 全过程损失值
        loss_names(List): 损失名称列表
        csv(Path): CSV结果文件路径
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        初始化
        :param cfg（str|dict|SimpleNamespace, optional）: 配置文件的路径。默认DEFAULT_CFG
        :param overrides: 覆盖cfg的配置参数字典，默认None
        :param _callbacks:
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)  #驱动 cuda:0/cpu
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 +RANK, deterministic=self.args.deterministic)

        #Dirs 路径
        self.save_dir = get_save_dir(self.args)  #Path Project/name
        self.args.name = self.save_dir.name
        self.wdir = self.save_dir / "weights"   #权重路径
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)   #make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0

        if self.device.type in ("cpu", "mps"):
            self.args.workers = 0  #更快的训练速度取决于推理而不是数据加载

        #Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)   #yolov8n -> yolov8n.pt
        try:
            if self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in ("yaml", "yml") or self.args.task in ("detect", "segment", "pose"):
                self.data = check_det_dataset(self.args.data)
                if "yaml_file" in self.data:
                    self.args.data = self.data["yaml_file"]
        except Exception as e:
            raise  RuntimeError(emojis(f"数据集‘{clean_url(self.args.data)}’ error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        #学习率参数
        self.lf = None
        self.scheduler = None

        #训练指标
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]



    def train(self):
        if isinstance(self.args.device, str) and len(self.args.device):   #i.e.  device='0'  or  device = '0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):   #i.e. device=[0,1,2,3]
            world_size = len(self.args.device)
        elif torch.cuda.is_available():    #i.e.  device=None or device='' or device=number
            world_size = 1
        else: #i.e. device='cpu' or 'mps'
            world_size = 0

        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ rect不兼容于多GPU训练，设置其为False")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning("WARNING ⚠️ 'batch=-1'不兼容于多GPU训练，设置其为8")
                self.args.batch = 8
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)



    @staticmethod
    def get_dataset(data):
        """获取训练集和验证集的路径"""
        return data["train"], data.get("val") or data.get("test")

    def check_resume(self, overrides):
        """检测是否需要恢复的训练节点，并且更新对应参数"""
        resume = self.args.resume
        if resume:
            try:
                exist = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exist else get_latest_run())    #搜索resume文件或者下载，返回恢复训练节点文件路径

                #检测恢复数据YAML是否存在，不存在则强制重新加载数据集
                ckpt_args = attempt_load_weights(last).args    #cfg参数
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = str(last)   # 恢复最后一次训练模型
                for k in "imgsz", "batch":
                    if k in overrides:
                        setattr(self.args, k, overrides[k])
            except Exception as e:
                raise FileNotFoundError(
                    "未找到恢复节点，请使用一个有效的节点去恢复"
                ) from e
        self.resume = resume

    def _setup_ddp(self, world_size):
        """初始化并设置分布式并行参数"""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["NCCL_BLOCKING_WAIT"] = "1"  #设置强制超时
        dist.init_process_group(
            "nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800), #3 hours
            rank=RANK,    #进程排名
            world_size=world_size,  #进程数
        )  #初始化分布式训练环境

    def _setup_train(self, world_size):
        """在正确的线程上建立dataloaders和optimizer"""
        #Model
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        #Freeze layers
        freeze_list = (
            self.args.freeze if isinstance(self.args.freeze, list) else
            range(self.args.freeze) if isinstance(self.args.freeze, int) else
            []
        )
        always_freeze_names = [".dfl"]   #一直冻结这些层 不更新权重
        freeze_layer_names = [f"model.{x}" for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"冻结'{k}'")
                v.requires_grad = False
            elif not v. requires_grad:
                LOGGER.warning(f"WARNING ⚠️ 对被冻结的层'{k}'解冻：设置‘requires_grad=True’")
                v.requires_grad = True
        #Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  #True/False
        if self.amp and RANK in (-1, 0):  #单一GPU
            self.amp = torch.tensor(check_amp(self.model), device=self.device)   #True/False
        if RANK > -1 and world_size > 1: #DDP
            dist.broadcast(self.amp, src=0)   #将amp广播到所有进程
        self.amp = bool(self.amp)  #
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)   #放大训练时的loss误差，消除amp的精度损失
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])  #并行运算

        #Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)   #grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        #Batch size
        if self.batch_size == -1 and RANK == -1:
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        #Dataloaders
        batch_size = self.batch_size // max(world_size, 1)  #多GPU训练 均分batch size
        self.train_loader, dataset = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val")[0]
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()
        
        #v5自适应预选框
        if not self.args.resume and isinstance(self.model.model[-1], (V5Segment,  V5Detect)): #V5检测模型
            if not self.args.noautoanchor:
                check_anchors(dataset, model=self.model, thr=self.args.anchor_t, img_sz=self.args.imgsz)  # run AutoAnchor

        #Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)   #在优化之前累加损失
        weight_decay = self.args.weight_decay *self.batch_size * self.accumulate / self.args.nbs  #scale weight decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs))  * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        #Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1




    def setup_model(self):
        """下载/创建/加载模型"""
        if isinstance(self.model, torch.nn.Module):    #模型已经加载
            return
        model, weights = self.model, None
        ckpt = None
        if str(model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt["model"].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK==-1)
        return ckpt

    def _setup_scheduler(self):
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cos  1->hyp['lrf']
        else:
            self.lf = lambda  x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf    #linear 1 -> hyp['lrf']
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取模型"""
        raise NotImplementedError("该任务训练器不支持加载cfg文件")

    def set_model_attributes(self):
        """在训练之前设置或者更新模型参数"""
        self.model.names = self.data["names"]

    def label_loss_items(self, loss_items=None, prefix="train"):
        """返回一个带标签的损失字典
        Note:
            classification不用这个函数，segmentation 和 detection 需要用到这个函数"""
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)  #加载模型、数据集，验证参数，初始化优化器、学习率参数

        nb = len(self.train_loader)   #number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1   #预热训练batch数
        last_opt_step = -1
        self.epoch_time = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        LOGGER.info(
            f"Using{self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb     #不关闭数据增强训练的batch数量
            self.plot_idx.extend([base_idx, base_idx+1, base_idx + 2])

        if RANK in (-1, 0):
            LOGGER.startTrain([self.progress_string(), self.start_epoch, self.epochs])

        epoch =  self.epochs# predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.model.train()
            if RANK != -1:
                self.train_loader.sample.set_epoch(epoch)

            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            self.tloss = None
            self.optimizer.zero_grad()
            total_instance = 0  #实例数量
            for i, batch in enumerate(self.train_loader):
                if LOGGER.stop:
                    torch.cuda.empty_cache()
                    raise ProcessLookupError(f"中断：训练中断成功,已训练{epoch}epoch")
                #预热训练
                ni = i + nb * epoch  #第几batch
                if ni < nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))   #0-nw -> 1-n  # >1
                    for j, x in enumerate(self.optimizer.param_groups):
                        #bias的学习率从0.1下降到lr0， 其他学习率从0.0上升到lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j==0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )  #0-nw -> warm_bias_lr/0-lr0
                        if "momentum" in x: #冲量
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                #Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss *i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items  #各项损失均值


                #Backward
                self.scaler.scale(self.loss).backward()   #放大损失

                #Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    #timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  #DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)   #将self.stop 广播到所有进程
                            self.stop = broadcast_list[0]
                        if self.stop:  #训练时间到
                            break
                #Log
                mem = f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.3g}G"
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1     #不同的检测有不同数量的损失 分类损失数量为1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    total_instance += batch["cls"].shape[0]
                    instances = batch["cls"].shape[0] if i < len(self.train_loader)-1 else total_instance
                    loss_mes = ("%11s" *3 + "%11.4g" * (2 + loss_len)) \
                            % (f"{epoch + 1}/{self.epochs}", f"{i+1}/{len(self.train_loader)}", mem, *losses, instances,batch["img"].shape[-1])
                    LOGGER.batchFinish(loss_mes)
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)


            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model,
                                     include=["yaml", "nc", "args", "names", "stride", "class_weights"])  # 更新ema属性

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness)  # 早停
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)  # 超时

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                LOGGER.epochFinish([loss_mes, epoch+1])
            # Schediler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 必须在optimizer.step() 前 lr_scheduler.step()
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()  # 根据新epochs重新设置学习率
                    self.stop |= epoch >= self.epochs  # 早停
                self.scheduler.step()
            torch.cuda.empty_cache()  # 每一个epoch清空GPU缓存，减少CUDA内存溢出

            # Early Stopping
            if RANK != -1:  # DDP
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # 将self.stop广播到所有RANK
                self.stop = broadcast_list[0]
            if self.stop:
                break
        if RANK in (-1, 0):
            LOGGER.info(
                f"\n 训练了{epoch - self.start_epoch + 1}epochs,"
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours"
            )
            self.final_eval()  #去除优化器，验证验证集
            if self.args.plots:
                self.plot_metrics()
        LOGGER.trainFinish("Train Finish!!")
        torch.cuda.empty_cache()

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""



    def build_optimizer(self, model,name="auto",lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        构建优化器
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname: #bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn): #weight (no decay)
                    g[1].append(param)
                else: # weight(with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"优化器'{name}不在已知列表内，仅支持"
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]"
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})   #g[0]参数使用权重衰减decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})     #g[1]参数使用权重衰减0 -> 不变
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups"
                     f"bn:{len(g[1])} weight(decay=0.0), bias:{len(g[0])} weight(decay={decay}), weight:{len(g[2])} bias(decay=0.0)")
        return optimizer

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        if self.resume:
            assert start_epoch > 0,(f"{self.args.model}训练已完成{self.epochs}epochs，请开始一个新的训练")
            LOGGER.info(f"{self.args.model}从{start_epoch + 1}epoch恢复训练，训练到{self.epochs}epochs")
        if self.epochs < start_epoch:
            LOGGER.info(f"{self.model}已经完成训练了{ckpt['epoch']}epochs, 将再增加训练{self.epochs}epochs")
            self.epochs += ckpt["epoch"]
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):   #最后close_mosaic个epochs关闭数据增强
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("关闭数据增强")
            self.train_loader.datase.close_mosaic(hyp=self.args)

    def preprocess_batch(self, batch):
        return batch

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)   #将放大的还原
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  #clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def validate(self):
        """验证"""
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", self.loss.detach().cpu().numpy())  #如果metrics里没有fitness，则返回loss
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  #number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  #header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def save_model(self):
        """保存模型训练节点"""
        import pandas as pd  #更快开启扫描

        metrics = {**self.metrics, ** {"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema":deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  #as dict
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }
        #save last and best
        torch.save(ckpt, self.last)    #最后一个模型
        if self.best_fitness >= self.fitness:
            torch.save(ckpt, self.best)    #最好的模型
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")   #周期保存

    def final_eval(self):
        """最终验证"""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  #去除优化器
                if f is self.best:
                    LOGGER.info(f"\nValidating{f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass