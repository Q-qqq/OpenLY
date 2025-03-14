
from ultralytics.utils import ops
from ultralytics_old.models.yolo.segment.val import SegmentationValidator

class V5SegmentationValidator(SegmentationValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml")
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "v5segment"


    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p = ops.v5_non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
                nc=self.nc,)
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto