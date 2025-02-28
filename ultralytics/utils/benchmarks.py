import numpy as np
import torch.cuda
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import  export_formats
from ultralytics.utils import ASSETS, LINUX, LOGGER, MACOS, WEIGHTS_DIR
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device


def benchmark(
        model=WEIGHTS_DIR / "yolov8n.pt",
        data=None,
        imgsz=160,
        half=False,
        int8=False,
        device="cpu",
        verbose=False
):
    """
    对不同格式的YOLO模型进行基准测试，确保速度和准确性

    Args:
        model(str | PAth | optional): 模型文件路径。默认Path(SETTINGS['weights_dir']) / 'yolov8n.pt'
        data(str, optional): 评估用数据集，如果没有，则从TASK2DATA继承， 默认None
        imgsz(int, optional): 图像大小，默认160
        half(bool, optional): 模型使用用半浮点精度，默认False
        int8(bool, optional): 模型使用Int8精度，默认False
        device(str, optional): 运行驱动，'cpu'/'cuda'，默认'cpu'
        verbose(bool | float| optional): 如果是True或者float，则确保beckmarks通过给定的指标，默认False
    Returns:
        df(pandas.DataFrame): 一个pandas数据帧附带每一种格式的benchmark结果，包含文件大小、评估指标、推理时间
    """
