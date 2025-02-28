import cv2
import torch
from PIL import Image
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops

class ClassificationPredictor(BasePredictor):
    def __init__(self, cfg, overrides):
        super().__init__(cfg, overrides)
        self.args.task = "classify"
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

    def preprocess(self, im):
        """裁剪缩放归一化"""
        if not isinstance(im, torch.Tensor):
            is_legacy_transform = any(self._legacy_transform_name in str(transform) for transform in self.transforms.transforms)  #ToTensor
            if is_legacy_transform:
                im = torch.stack([self.transforms(img) for img in im], dim=0)
            else:
                im = torch.stack([self.transforms(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in im], dim=0)
        im = (im if isinstance(im, torch.Tensor) else torch.from_numpy(im)).to(self.model.device)
        return im.half() if self.model.fp16 else im.float()

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, probs=pred))
        return results