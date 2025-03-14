

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops



class V5DetectionValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        self.args.task = "v5detect"
        

    def postprocess(self, preds):
        """使用非最大值抑制处理预测结果"""
        return ops.v5_non_max_suppression(preds,
                                            self.args.conf,
                                            self.args.iou,
                                            labels=self.lb,
                                            multi_label=True,
                                            agnostic=self.args.single_cls,
                                            max_det=self.args.max_det)