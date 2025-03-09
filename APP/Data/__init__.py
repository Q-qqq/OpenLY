from pathlib import  Path
import numpy as np
import torch

from ultralytics.utils import LOGGER, yaml_load
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.data.utils import img2label_paths
from APP  import PROJ_SETTINGS


def getDefaultDataset():
    """获取默认数据集"""
    project_p = PROJ_SETTINGS["name"]
    with open(Path(project_p) / "data" / "train.txt", "w") as f:
        pass  #创建空白训练集txt文件
    with open(Path(project_p) / "data" / "val.txt", "w") as f:
        pass #创建空白验证集txt文件
    return {"names": [],
            "path": "./",
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
        line = (line.view(-1) if isinstance(line, torch.Tensor) else line.reshape(-1)).tolist()
        line[0] = int[line[0]]
        texts.append(("%g " * len(line)).rstrip() % tuple(line))
    if not Path(label_file).parent.exists():
        Path(label_file).parent.mkdir(parents=True, exist_ok=True)
    with open(label_file, "W") as f:
        f.writelines(text + "\n" for text in texts)


def check_cls_train_dataset(data):
    train_set = Path(data) / "train"
    if not train_set.exists():
        train_set.mkdir(parents=True, exist_ok=True)
    names =[x.name for x in (train_set).iterdir() if x.is_dir()]  #种类名称 训练集目录下文件夹名称
    names = dict(enumerate(sorted(names)))
    if len(names) == 0:
        LOGGER.info("训练数据集为空， 请添加种类进行创建（编辑->种类->添加）")
    else:
        files = [path for path in train_set.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]  # 数据集路径和其子文件夹下的图像文件
        nf = len(files)  # 找到的文件数量
        if nf == 0:
            LOGGER.info(f"训练数据集{train_set}存在种类{len(names)},其内无图像数据")
    return train_set, names

def check_cls_val_dataset(data, train_names):
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
    for _, val_name in names.items():
        if val_name not in train_names.values():
            LOGGER.warning(f"验证集种类{val_name}不在训练集种类中,请添加训练集种类{val_name}或者删除验证集种类{val_name}")
    if len(names) == 0:
        LOGGER.info("验证数据集为空")
    else:
        files = [path for path in val_set.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]  # 数据集路径和其子文件夹下的图像文件
        nf = len(files)  # 找到的文件数量
        if nf == 0:
            LOGGER.info(f"验证数据集{val_set}存在种类{len(names)},其内无图像数据")
    return val_set, names

def guess_dataset_task(dataset):
    """检查数据集任务类型"""
    dataset = Path(dataset)
    if dataset.is_dir():
        return ["classify"]
    elif dataset.suffix == ".yaml":
        data = check_det_dataset(dataset)
        train_path = data["train"]
        val_path = data["val"]
        with open(train_path) as f:
            train_img = f.read().strip().splitlines()
        with open(val_path) as f:
            val_img = f.read().strip().splitlines()
        if len(train_img) == 0 and len(val_img) == 0:
            return "null"
        if len(train_img) != 0:
            for img in train_img:
                label_path = img2label_paths([img])[0]
                label = readLabelFile(label_path)
                if len(label) == 0:
                    continue
                else:
                    if len(label[0]) == 5:
                        return ["detect"]
                    elif len(label[0]) == 6:
                        return ["obb"]
                    else:
                        return ["segment", "keypoint"]
        else:
            for img in val_img:
                label_path = img2label_paths([img])[0]
                label = readLabelFile(label_path)
                if len(label) == 0:
                    continue
                else:
                    if len(label[0]) == 5:
                        return "detect"
                    elif len(label[0]) == 6:
                        return ["obb"]
                    else:
                        return ["segment", "keypoint"]
        LOGGER.warning(f"数据集{dataset}标签全为空无法识别任务类型")
        return ["error"]
                
