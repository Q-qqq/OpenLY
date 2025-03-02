
import os
from pathlib import Path
import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import box_iou,DetMetrics,ConfusionMatrix
from ultralytics.utils.plotting import plot_images, output_to_target
from ultralytics.nn.modules.head import V5Detect
from ultralytics.nn.tasks import DetectionModel


class DetectionValidator(BaseValidator):
    """
    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
    """

    def __init__(self, dataloader=None, save_dir=None,args=None):
        super().__init__(dataloader, save_dir, args)
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  #iou阈值向量0.5-0.95
        self.niou = self.iouv.numel()  #10
        self.lb = []

    def preprocess(self, batch):
        """预处理"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        if self.args.save_hybrid:   #混合标签，原始的和预测的
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)   #去归一化
            self.lb = (
                [torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1) for i in range(nb)]
            )   #[cls boxxes]  //原始标签
        return batch

    def init_metrics(self, model):
        val = self.data.get(self.args.split, "")  #验证路径
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt") #is coco
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training   #如果训练coco，旨在最后验证的时候运行
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

    def get_desc(self):
        """获取yolo模型指标头部标题"""
        return ("%22s" + "%11s" * 6) % ("Class", "Image", "Instances", "Box(p", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """使用非最大值抑制处理预测结果"""
        m = self.model.model if isinstance(self.model, DetectionModel) else self.model.model.model
        if isinstance(m[-1], V5Detect):
            return ops.v5_non_max_suppression(preds,
                                              self.args.conf,
                                              self.args.iou,
                                              labels=self.lb,
                                              multi_label=True,
                                              agnostic=self.args.single_cls,
                                              max_det=self.args.max_det)
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """准备第si批次的图像标签，并将标签转换至原生空间坐标系"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["radio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1,0,1,0]]   #去归一化
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)   #原生空间标签
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        """将预测的box转换至原生空间坐标系"""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  #将预测的box转换至原生空间
        return predn

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """ 返回正确的预测矩阵"""
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:,5], gt_cls, iou)   #获取预测正确的预测框shape（N， 10） N：预测的数量， 10：0.5-0.95

    def update_metrics(self, preds, batch):
        """根据预测和真实框更新混淆矩阵，并保存预测结果"""
        for si, pred in enumerate(preds):  #List(Tensor(n, 6+nm))*bs
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0,device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            )
            pbatch = self._prepare_batch(si, batch)   #获取一个批次内一张图像的标签
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:   #预测为0
                if nl:     #真实存在
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    #TODO:obb不支持混淆矩阵
                    if self.args.plots and self.args.task != "obb":
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls, im_file=batch["im_file"][si])   #更新真实目标
                continue

            #Predictions
            if self.args.single_cls:
                pred[:, 5] = 0  #单一种类
            predn = self._prepare_pred(pred, pbatch)   #将预测的box转换至原生空间
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            #Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)  #shape(npr, 10) 预测正确的为True， 10：分别在0.5-0.95的iou阈值下
                #TODO:obb不支持混淆矩阵
                if self.args.plots and self.args.task != "obb":
                    self.confusion_matrix.process_batch(predn, bbox, cls, im_file=batch["im_file"][si])  #更新混淆矩阵
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            #Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt"
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def pred_to_json(self, predn, filename):
        """序列化YOLO预测输出至COCOjson格式"""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  #xywh
        box[:, :2] -= box[:, 2:]  / 2  #中心点 -> 左上角点
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5)
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  #whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh((torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())   #归一化
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)   #标签
            with open(file, "a") as f:
                f.write(("%g"*len(line)).rstrip() % line + "\n")   #添加保存

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """返回指标状态和结果字典"""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  #to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)   #计算指标
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )   #每个种类的数量
        return self.metrics.results_dict   #mp,mr,map50,map fitness

    def print_results(self):
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ 在{self.args.task}集内未发现标签，无法计算指标")

        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_results(i)))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.names.values(),
                    normalize=normalize,
                    on_plot=self.on_plot
                )

    def build_dataset(self, img_path, mode="val", batch=None):
        """build YOLO dataset"""

        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def plot_val_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname = self.save_dir / f"val_batch{ni}_labels.jpg",
            names = self.names,
            on_plot=self.on_plot
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname = self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot
        )

    def eval_json(self, stats):
        """使用pycocotools对coco数据集进行再评估mAP"""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"
            pred_json = self.save_dir / "predictions.json"
            LOGGER.info(f"\n使用{pred_json}和{anno_json}评估pycocotools mAP")
            try:
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                for x in anno_json, pred_json:
                    assert x.is_file(), f"未找到文件{x}"
                anno = COCO(str(anno_json))   #init annotations api
                pred = anno.loadRes(str(pred_json))
                eval = COCOeval(anno, pred, "bbox")
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  #image to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[1]], stats[self.metrics.key[-2]] = eval.stats[:2]   #更新mAP50-95 mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools运行{e}失败")
        return stats
