from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


class PoseTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning("WARNING ⚠️ 使用Apple MPS训练存在bug，建议使用'device=cpu'")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PoseModel(cfg, ch=3, nc = self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        self.loss_names = "box_loss", "poss_loss", "kobj_loss", "cls_loss", " dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args)
        )

    def plot_training_samples(self, batch, ni):
        plot_images(batch["img"],
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    kpts=batch["keypoints"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"train_batch{ni}.jpg",
                    on_plot=self.on_plot)

    def plot_metrics(self):
        plot_results(files=self.csv, pose=True, on_plot=self.on_plot)