import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images

class ClassificationValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None):
        super().__init__(dataloader, save_dir, args)
        self.targets = None
        self.pred =None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        return ("%22s" + "%11s" * 2) %("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []
        self.im_files = []

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """获取每张图像前5的种类"""
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5])  #将每个种类的预测值按衰减排列，并取其对应索引为种类
        self.targets.append(batch["cls"])
        self.im_files.append(batch["im_file"])

    def finalize_metrics(self, *args, **kwargs):
        self.confusion_matrix.process_cls_preds(self.pred, self.targets, self.im_files)  #更新混淆矩阵
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot)

        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        plot_images(images=batch["img"],
                    batch_idx=torch.arange(len(batch["img"])),
                    cls=batch["cls"].view(-1),
                    fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        plot_images(batch["img"],
                    batch_idx=torch.arange(len(batch["img"])),
                    cls=torch.argmax(preds, dim=-1),
                    fname=self.save_dir / f"val_batch{ni}_pred.jpg",
                    names=self.names,
                    on_plot=self.on_plot)