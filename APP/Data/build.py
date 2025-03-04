import copy
from pathlib import Path

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import IterableSimpleNamespace,  yaml_save, LOGGER, yaml_load


from APP.Data import check_cls_val_dataset, check_cls_train_dataset, getDefaultDataset
from APP.Data.datasets import ClassifyDataset, DetectDataset
from APP  import PROJ_SETTINGS

def check_detect_dataset(root, args):
    """构建检测数据集"""
    if not root or not Path(root).exists():
        LOGGER.warning(f"数据集{root}不存在，默认使用{PROJ_SETTINGS['name']}//data数据集")
        root = Path(PROJ_SETTINGS["name"]) / "data"/ "dataset.yaml"

    if not Path(root).exists():
        yaml_save(root, getDefaultDataset())
    data = yaml_load(root)
    if not Path(data["path"]).is_absolute():
        path = (Path(root).parent / data["path"]).resolve()
        if path.exists():
            data["path"] = str(path)
        else:
            raise FileNotFoundError(f"数据集路径{data['path']}和{path}不存在,请检查数据集参数：dataset.yaml")
    elif not Path(data["path"]).exists():
        if Path(root.parent / "train.txt").exists and Path(root.parent / "val.txt").exists():
            data["path"] = str(root.parent)
        else:
            raise FileNotFoundError(f"数据集路径{data['path']}不存在,请检查数据集参数：dataset.yaml")
    if len(data["names"]):
        data = check_det_dataset(root)
    else:
        data["train"] = (Path(data["path"]) / data["train"]).resolve()
        data["val"] = (Path(data["path"])/ data["val"]).resolve()

    data["yaml_file"] = root
    if not Path(data["train"]).exists():
        with open(data["train"], "w") as f:  # 创建训练集空白txt文件
            pass
    if not Path(data["val"]).exists():  # 创建验证集空白txt文件
        with open(data["val"], "w") as f:
            pass

    train_dataset = buildDetectDataset(data["train"], "detect", args, data)
    val_dataset = buildDetectDataset(data["val"] or data["test"], "detect", args, data)
    train_path = str(Path(train_dataset.im_files[0]).parent) if train_dataset.im_files else str(
        Path(train_dataset.img_path).parent / "images" / "train")  # 训练集图像文件路径
    val_path = str(Path(val_dataset.im_files[0]).parent) if val_dataset.im_files else str(
        Path(val_dataset.img_path).parent / "images" / "val")  # 验证集图像文件路径
    return root, train_dataset, val_dataset, train_path, val_path

def check_cls_dataset(root, args):
    """构建分类数据集"""
    if not root or not Path(root).exists():
        LOGGER.warning(f"数据集{root}不存在，默认使用{PROJ_SETTINGS['name']}//data数据集")
        root = Path(PROJ_SETTINGS["name"]) / "data"

    if isinstance(args, dict):
        args = IterableSimpleNamespace(**args)
    train_path, train_names = check_cls_train_dataset(root)
    train_dataset = ClassifyDataset(root=train_path, names=train_names, args=args)
    val_path, _ = check_cls_val_dataset(root)
    val_dataset = ClassifyDataset(root=val_path, names=train_names, args=args) #使用训练集种类
    return root, train_dataset, val_dataset, train_path, val_path



def buildDetectDataset(img_path,task, args, data):
    return DetectDataset(
        img_path=img_path,
        imgsz=100,
        batch_size=1,
        augment=False,
        hyp=args,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        task=task,
        classes=None,
        data=data,
        fraction=1.0,
    )
