import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked, threaded, LOGGER, TryExcept, PROGRESS_BAR
from ultralytics.data.dataset import YOLODataset

from APP.Utils.base import QInstances


class Yolo(YOLO):
    @threaded
    @TryExcept(msg="Train error", verbose=True)
    def lyTrain(self, trainer=None, **kwargs):
        self.train(trainer, **kwargs)


    @threaded
    @TryExcept(msg="Val error", verbose=True)
    def lyVal(self, validator=None,  **kwargs):
        LOGGER.info("开始验证")
        self.val(validator, **kwargs)

    @TryExcept(msg="Predict Error", verbose=True)
    def lyPredict(self, source=None, stream=False, predictor=None, **kwargs):
        try:
            PROGRESS_BAR.show("推理", "开始推理...")
            results = self.predict(source, stream, predictor, **kwargs)
            labels = {}
            for i, result in enumerate(results):
                if i==0:
                    PROGRESS_BAR.show("推理", "开始推理...")
                    PROGRESS_BAR.start(0, len(source), True)
                boxes = result.boxes.xywhn if result.boxes is not None else None   #取归一化参数
                segments = result.masks.xyn if result.masks is not None else None
                keypoints = result.keypoints.xyn if result.keypoints is not None else None
                obbs = result.obb.xyxyxyxyn if result.obb is not None else None
                prob = result.probs.top1 if result.probs is not None else None
                if boxes is not None:
                    conf = result.boxes.conf
                    cls = result.boxes.cls
                elif obbs is not None:
                    conf = result.obb.conf
                    cls = result.obb.cls
                elif prob is not None:
                    conf = result.probs.top1conf
                    cls = prob
                else:
                    raise SyntaxError("置信度缺失")

                nkpt = 0
                ndim = 0
                if keypoints:
                    nkpt, ndim = keypoints.shape[1:]
                    visiable = result.keypoints.conf  # 可见性
                    if result.keypoints.has_visible:
                        keypoints = torch.cat([keypoints, visiable], -1) if isinstance(keypoints,
                                                                                       torch.Tensor) else np.concatenate(
                            [keypoints, visiable], -1)

                instances = QInstances(boxes, segments or obbs, keypoints, "xywh", normalized=True)
                label = {"names": result.names,
                         "cls": cls,
                         "conf": conf,
                         "nkpt": nkpt,
                         "ndim": ndim,
                         "instances": instances}
                labels[str(Path(result.path))] = label
                PROGRESS_BAR.setValue(i+1, f"{result.path}: {result.speed['inference']:.1f}ms")
                if PROGRESS_BAR.isStop():
                    PROGRESS_BAR.close()
                    return labels
            PROGRESS_BAR.close()
        except Exception as e:
            LOGGER.error("Predict error:" + str(e))
            PROGRESS_BAR.stop()
            PROGRESS_BAR.close()
        return labels

    @threaded
    @TryExcept(msg="Export error", verbose=True)
    def lyExport(self):
        self.export()
