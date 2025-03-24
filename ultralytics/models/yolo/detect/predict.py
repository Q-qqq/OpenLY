from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import V5Detect

class DetectionPredictor(BasePredictor):
    """目标检测预测"""
    def postprocess(self, preds, img, orig_imgs):
        m = self.model.model if isinstance(self.model, DetectionModel) else self.model.model.model
        if isinstance(m[-1], V5Detect):
            preds = ops.v5_non_max_suppression(
                preds, 
                self.args.conf, 
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes)
        else:
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
            )
            
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]   #batch: source、images、videocapture、s
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
