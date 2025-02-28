from multiprocessing.pool import ThreadPool
from pathlib import  Path
import numpy as np
import torch.nn.functional as F
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.head import V5Segment
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import box_iou,SegmentMetrics,mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images

class SegmentationValidator(DetectionValidator):
    def __init__(self,dataloader=None, save_dir=None, args=None):
        super().__init__(dataloader,save_dir,args)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """预处理，转float和device"""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """初始化指标参数"""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
            self.process = ops.process_mask_upsample  #更高的精度
        else:
            self.process = ops.process_mask  #更快的速度
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[])

    def get_desc(self):
        return  ("%22s"+"%11s" * 10)%(
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)"
        )

    def postprocess(self, preds):
        m = self.model.model if isinstance(self.model, SegmentationModel) else self.model.model.model
        if isinstance(m[-1], V5Segment):
            p = ops.v5_non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
                nc=self.nc,)
        else:
            p = ops.non_max_suppression(
                preds[0],  #box classes mc
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
                nc=self.nc,
            )   #bs, n, 6+nm
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
        return p, proto

    def _prepare_batch(self, si, batch):
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto=None):
        predn = super()._prepare_pred(pred,pbatch)
        pred_masks = self.process(proto, pred[:,6:], pred[:,:4], shape=pbatch["imgsz"])
        return predn, pred_masks

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        返回正确的预测矩阵（N, 10）
        Args:
            detections(torch.Tensor):shape(N, 6), x1,y1,x2,y2,conf,cls   预测
            gt_bboxes(torch.Tensor):shape(M, 4), x1,y1,x2,y2 真实框
            gt_cls(torch.Tensor):shape(M,) 真实种类
            pred_masks(torch.Tensor):shape(N, orig_img_h,orig_img_w) 预测masks
            gt_masks(torch.Tensor): shape(M,orig_img_h, orig_img_w) 真实masks
            overlap(bool): 掩膜是否重叠类型
            masks(bool): 是否计算maskIou作为指标或者计算boxIou作为指标
        """
        if masks:
            if overlap:
                nl = len(gt_cls) #真实种类数量
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1   # 1,2,3,4,..
                gt_masks = gt_masks.repeat(nl, 1, 1) # shape(1, h , w) - > (nl, h, w)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)  #将各种类分开
            if gt_masks.shape[1:] != pred_masks.shape[1:]:   #长宽不等
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)  #向预测masks大小适应
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  #boxes
            iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:,5], gt_cls, iou)  #获取预测正确的预测框矩阵（N,10）  通过Iou阈值0.5-0.95

    def update_metrics(self, preds, batch):
        """更新混淆矩阵指标"""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):  #按图像遍历
            self.seen += 1
            npr = len(pred)  #nms后的目标数量
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr,self.niou,dtype=torch.bool, device=self.device)
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)   #真实目标数量
            stat["target_cls"] = cls
            if npr == 0:
                if nl:   #漏检
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls, im_file=batch["im_file"][si])
                continue

            #Masks
            gt_masks = pbatch.pop("masks")
            #Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            #Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)  #目标框的预测矩阵（N, 10）
                stat["tp_m"] = self._process_batch(predn,bbox,cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True) #掩膜的预测矩阵(N, 10)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls, im_file=batch["im_file"][si])
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  #绘制预测图像时，前の张图像跳过前15个目标

            #save
            if self.args.save_json:
                pred_masks = ops.scale_image(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),   #HWC
                    pbatch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.pred_to_json(predn, batch["im_file"][si], pred_masks)

    def pred_to_json(self, predn, filename, pred_masks=None):
        """保存结果到Json格式
        Examples:
             result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode

        def single_encode(x):
            rle = encode(np.asarray(x[:,:,None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2  #中心点xy转为左上角点
        pred_masks =np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i]
                }
            )

    def eval_json(self, stats):
        """返回COCO风格的目标检测评估指标"""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"
            pred_json = self.save_dir / "predictions.json"
            LOGGER.info(f"\n使用{pred_json}和{anno_json}评估pycocotools mAP")
            try:
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import  COCO
                from pycocotools.cocoeval import COCOeval

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x}文件不存在"
                anno = COCO(str(anno_json))
                pred = anno.loadRes(str(pred_json))
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i*4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[:2]  #更新map50 和map50-95
            except Exception as e:
                LOGGER.warning(f"pycocotools运行失败：{e}")
        return stats

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def plot_val_samples(self, batch, ni):
        plot_images(batch["img"],
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    masks=batch["masks"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        plot_images(batch["img"],
                    *output_to_target(preds[0], max_det=15),
                    torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred.jpg",
                    names=self.names,
                    on_plot=self.on_plot)
        self.plot_masks.clear()