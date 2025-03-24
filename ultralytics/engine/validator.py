import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.head import V5Segment
from ultralytics.utils import LOGGER, colorstr, emojis, PROGRESS_BAR
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device,smart_inference_mode

class BaseValidator:
    """
    基准验证器
    Attributes:
        args(SimpleNamespace)：验证器参数
        dataloader(Dataloader):验证集
        model(nn.Module):模型
        data(dict):数据集参数字典
        device(torch.device):驱动
        batch_i(int):当前批次索引
        training(bool):模型是否处于训练模式
        names(dict):种类名称
        seen:在验证期间到目前为止的图像数量
        stats:在验证期间统计信息的占位符
        confusion_matrix:混淆矩阵的占位符
        nc:种类数量
        iouv(torch.Tensor):iou阈值，0.5以内和0.5~0.95
        jdict(dict):验证结果（dict）存储为json
        speed(dict):记录每个batch的‘preprocess’,'inference','loss','postprocess'的运行时间
        save_dir(Path):保存结果的字典
        plots(dict):存储plots

    """
    def __init__(self,dataloader=None, save_dir=None, args=None):
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess":0.0, "infrence":0.0, "loss":0.0, "postprocess":0.0}

        self.save_dir = get_save_dir(self.args) or Path(save_dir)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)   #推理过程是否进行数据增强
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            #self.args.half = self.device.type != "cpu"
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            #self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            model = AutoBackend(
                model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half
            )  #加载模型
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1
                LOGGER.info(f"对于不是PyTorch的model，强制使'batch=1'，输入图像大小(1,3,{imgsz},{imgsz})")

            if str(self.args.data).split(".")[-1] in ("yaml", "yml"): #YAML文件路径
                self.data = check_det_dataset(self.args.data)   #检测目标检测数据集
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)  #检测分类数据集
            else:
                raise FileNotFoundError(emojis(f"'task={self.args.task}' 的数据集'{self.args.data}'未发现❌"))

            if self.device.type in ("cpu", "mps"):
                self.args.workers = 0
            if not pt:
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)  #获取数据集

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))   #模型预热
        self.model = model
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        self.init_metrics(de_parallel(model))   #初始化混淆矩阵和预测结果等参数
        self.hdict = []
        LOGGER.startVal(self.get_desc())
        if not self.training:
            PROGRESS_BAR.show("验证中","开始验证")
            PROGRESS_BAR.start(0, len(self.dataloader), True)

        for batch_i, batch in enumerate(self.dataloader):
            self.batch_i = batch_i
            #Preprocess
            with dt[0]:
                batch = self.preprocess(batch)   #对一批次图像进行处理 转换device，归一化等

            #Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)
            #Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            #Postprocess
            with dt[3]:
                preds = self.postprocess(preds)    #NMS
            
            self.update_metrics(preds, batch)    #计算正例矩阵， 更新混淆矩阵 如果save_json 则保存结果到jdict 用于coco
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            if not self.training:
                PROGRESS_BAR.setValue(batch_i+1, str(len(batch["img"])))
                if PROGRESS_BAR.isStop():
                    raise ProcessLookupError("中断：验证中断成功")

        stats = self.get_stats()  #mr, mp, map50, map, fitness 计算指标
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()   #将混淆矩阵和运行速度更新进self.metrics
        self.print_results()
        LOGGER.valFinish("")
        PROGRESS_BAR.close()
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  #小数点后5位
        else:
            LOGGER.info("Speed:%.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image" % tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving{f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)    #重新评估cocomAP
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"结果已保存至{colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        使用IoU匹配预测的和真实的目标框
        Args:
            pred_classes(torch.Tensor): shape(N,) 预测目标种类索引
            true_classes(torch.Tensor): shape(M,) 真实目标种类索引
            iou (torch.Tensor): shape(N,M) 包含用于预测和真实目标的成对IoU值
            use_scipy(bool):是否使用scipy用于匹配
        Returns:
            (torch.Tensor): shape(N,10)
        """
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)  #N*10,   10:0.5，0.55，0.6，0.65，0.7，0.75，0.8，0.85，0.9，0.95
        correct_class = true_classes[:,None] == pred_classes     #M*N   种类预测正确
        iou = iou * correct_class    #种类预测正确的iou   *M*N
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                import scipy
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[[detections_idx], i] = True
            else:
                matches = np.nonzero(iou >= threshold)   #iou>threshold  并且跟种类对应shape(n, 2)   2: rowM(true), cloumnN(pred)
                matches = np.array(matches).T   #n,2
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]   #以iou从大到小进行排序后的matches
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]    #一个预测框对应一个真实框
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]   #一个真实框对应一个预测框
                    correct[matches[:, 1].astype(int), i] = True   #预测正确的为True， 分10个iou 0.5-0.95
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)





    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def get_desc(self):
        """Get results key"""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass