
import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import V5DetectionModel
from ultralytics.utils import RANK



class V5DetectionTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import V5DetectionTrainer

        args = dict(model="yolo1v5-anchhors.pt", data="coco8.yaml", epochs=3)
        trainer = V5DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = V5DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "obj_loss"
        return V5DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

