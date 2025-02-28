from pathlib import Path
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import batch_probiou,OBBMetrics
from ultralytics.utils.plotting import plot_images,output_to_rotated_target

class OBBValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None):
        super().__init__(dataloader, save_dir,args)
        self.args.task = "obb"
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")
        self.is_dota = isinstance(val, str) and "DOTA" in val   #is COCO

    def postprocess(self, preds):
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       nc=self.nc,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       rotated=True)

    def _prepare_batch(self, si, batch):
        """准备一个batch中第si个图像参数，并将box缩放至适应原始图像"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  #缩放至适应网络输入图像
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)   #缩放至适应原始图像
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        """准备预测结果，将预测框缩放至适应原始图像"""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  #将预测框缩放至适应原始图像
        return predn

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """获取正例矩阵"""
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:,-1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r,b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score":round(predn[i, 4].item(), 5),
                    "rbox":[round(x, 3) for x in r],
                    "poly":[round(x, 3) for x in b]
                }
            )

    def eval_json(self, stats):
        """保存预测结果为jjson"""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"
            pred_txt = self.save_dir / "predictions_txt"
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))

            LOGGER.info(f"保存DOTA格式的预测结果到{pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"]].replace(" ", "-")
                p = d["poly"]

                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

            pred_merged_txt = self.save_dir / "predictions_merged_txt"  #predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"保存DOTA和格式的混合预测结果到{pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+__\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"]
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  #classes
                scores = bbox[:, 5]
                b = bbox[:, :5].clone()
                b[:, :2] += c
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]
                    score = round(x[-2], 3)

                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
        return stats

    def save_one_txt(self, predn, save_conf, shape, file):
        gn = torch.tensor(shape)[[1, 0]]   #whwh
        for *xywh, conf, cls, angle in predn.tolist():
            xywhr = torch.tensor([*xywh, angle]).view(1, 5)
            xyxyxyxy = (ops.xywhr2xyxyxyxy(xywhr) / gn).view(-1).tolist()  #转xyxyxyxy 并归一化
            line = (cls, *xyxyxyxy, conf) if save_conf else (cls, *xyxyxyxy)
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def plot_predictions(self, batch, preds, ni):
        plot_images(batch["img"],
                    *output_to_rotated_target(preds, max_det=self.args.max_det),
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred.jpg",
                    names=self.names,
                    on_plot=self.on_plot)