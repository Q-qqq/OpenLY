from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel

class YOLO(Model):

    @property
    def task_map(self):
        """检测任务映射模型类、训练器、验证器、推理器"""
        return {
            "classify":
                {
                    "model": ClassificationModel,
                    "trainer": yolo.classify.ClassificationTrainer,
                    "validator": yolo.classify.ClassificationValidator,
                    "predictor": yolo.classify.ClassificationPredictor,
                },
            "detect":
                {
                    "model": DetectionModel,
                    "trainer": yolo.detect.DetectionTrainer,
                    "validator": yolo.detect.DetectionValidator,
                    "predictor": yolo.detect.DetectionPredictor,
                },
            "segment":
                {
                    "model": SegmentationModel,
                    "trainer": yolo.segment.SegmentationTrainer,
                    "validator": yolo.segment.SegmentationValidator,
                    "predictor": yolo.segment.SegmentationPredictor,
                },
            "pose":
                {
                    "model": PoseModel,
                    "trainer": yolo.pose.PoseTrainer,
                    "validator": yolo.pose.PoseValidator,
                    "predictor": yolo.pose.PosePredictor,
                },
            "obb":
                {
                    "model": OBBModel,
                    "trainer": yolo.obb.OBBTrainer,
                    "validator": yolo.obb.OBBValidator,
                    "predictor": yolo.obb.OBBPredictor,
                },

        }