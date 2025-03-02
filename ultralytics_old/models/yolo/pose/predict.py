from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops

class PosePredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning("WARNING ⚠️使用Apple MPS验证存在bug，建议使用'device=cpu")

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs. list): #Tensor
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]  #n, nkpt, ndim
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)   #将预测数据转换至原始图像坐标系上
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts))
        return results