from pathlib import  Path
import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.data.utils import IMG_FORMATS

from APP import PROJ_SETTINGS


def getDefaultDataset():
    project_p = PROJ_SETTINGS["name"]
    with open(Path(project_p) / "data" / "train.txt", "w") as f:
        pass
    with open(Path(project_p) / "data" / "val.txt", "w") as f:
        pass
    return {"names": [],
            "path": str(Path(project_p) / "data"),
            "train": "train.txt",
            "val": "val.txt"}


def getNoLabelPath():
    project_p = PROJ_SETTINGS["name"]
    return str(Path(project_p) / "data" / "no_label")


def format_im_files(im_files):
    im_files = im_files if isinstance(im_files, list) else [im_files]
    im_files = [str(Path(f)) for f in im_files]
    return im_files



def readLabelFile(label_file):
    """读取标签文件为list(nd.array(1,m)) 格式"""
    with open(label_file) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        lb = [np.array(b, dtype=np.float32) for b in lb]
    return lb

def writeLabelFile(label_file, lb):
    """将list(np.array(1,m))格式的数据保存为标签文件txt"""
    texts = []
    for line in lb:
        texts.append(("%g " * len(line)).rstrip() % line)
    Path(label_file).parent.mkdir(parents=True, exist_ok=True)
    with open(label_file, "a") as f:
        f.writelines(text + "\n" for text in texts)


def check_cls_train_dataset(data):
    train_set = Path(data) / "train"
    if not train_set.exists():
        train_set.mkdir(parents=True, exist_ok=True)
    names =[x.name for x in (train_set).iterdir() if x.is_dir()]  #种类名称 训练集目录下文件夹名称
    names = dict(enumerate(sorted(names)))
    if len(names) == 0:
        LOGGER.warning("训练数据集为空")
    else:
        files = [path for path in train_set.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]  # 数据集路径和其子文件夹下的图像文件
        nf = len(files)  # 找到的文件数量
        if nf == 0:
            LOGGER.warning(f"训练数据集{train_set}存在种类{len(names)},其内无图像数据")
    return train_set, names

def check_cls_val_dataset(data):
    data = Path(data)
    val_set = (
        data / "val" if (data / "val").exists() else
        data / "validation" if (data / "validation").exists() else
        None
    )  # 验证集路径
    test_set = data / "test" if (data / "test").exists() else None  # 测试集路径
    val_set= val_set or test_set
    if not val_set:
        (data / "val").mkdir(parents=True, exist_ok=True)
    names = [x.name for x in (val_set).iterdir() if x.is_dir()]  # 种类名称 训练集目录下文件夹名称
    names = dict(enumerate(sorted(names)))
    if len(names) == 0:
        LOGGER.warning("验证数据集为空")
    else:
        files = [path for path in val_set.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]  # 数据集路径和其子文件夹下的图像文件
        nf = len(files)  # 找到的文件数量
        if nf == 0:
            LOGGER.warning(f"验证数据集{val_set}存在种类{len(names)},其内无图像数据")
    return val_set, names
