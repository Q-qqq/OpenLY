from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.nn.modules.head import V5Segment


class SegmentationPredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """im:模型输入， orig_imgs:原始图像输入"""
        m = self.model.model if isinstance(self.model, SegmentationModel) else self.model.model.model
        if isinstance(m[-1], V5Segment):
            p = ops.v5_non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                classes=self.args.classes)
        else:
            p = ops.non_max_suppression(preds[0],
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        nc=len(self.model.names),
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  #torch.Tensor
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):
                masks = None
            elif self.args.retina_masks:  #高分辨率mask
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)  #原始输入图像大小box，已经过填充补正，未变形
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2]) #计算出masks，并将其适应box剪切，masks经填充补正
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  #计算masks，并将其适应box剪切，box未经填充补正
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)  #box填充补正
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results