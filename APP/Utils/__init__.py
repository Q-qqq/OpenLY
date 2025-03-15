import glob
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import numpy as np
import torch
from pathlib import Path

from APP.Data import readLabelFile
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import ROOT

def get_widget(parent:QFrame, name):
    """获取父控件中指定objectname的子控件"""
    for widget in parent.children():
        if widget:
            if widget.objectName() == name:
                return widget
            else:
                w = get_widget(widget, name)
                if w != None:
                    return w
    return None

def getcat(value):
    """获取拼接函数"""
    return torch.cat if isinstance(value, torch.Tensor) else np.concatenate


def append_formatted_text(text_edit, text, font_size, color=None, bold=False):
    # 获取当前光标并移动到文档末尾
    cursor = text_edit.textCursor()
    cursor.movePosition(QTextCursor.End)
    
    # 创建格式并设置属性
    fmt = QTextCharFormat()
    fmt.setFont(QFont("Consolas", font_size))  # 字体大小
    if color:
        fmt.setForeground(color)
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)       # 粗体
    
    # 插入带格式的文本并换行
    cursor.insertText(text + "\n", fmt)
    
    # 更新光标位置
    text_edit.setTextCursor(cursor)
    text_edit.ensureCursorVisible()

def guess_dataset_task(dataset):
    """
    检查数据集可用任务类型
    Args:
        dataset(str | Path): 若dataset为文件夹路径，则为分类数据集；若为yaml文件，则为detect,segment,pose 或 obb数据集
    Returns:
        (list): ["classify","detect", "v5detect", "segment", "v5segment", "obb", "pose"]
    """
    if dataset == "" or not Path(dataset).exists():
        return ["classify","detect", "v5detect", "segment", "v5segment", "obb", "pose"]
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
        #数据集为空，可自由决定任务类型
        if len(train_img) == 0 and len(val_img) == 0:
            return [ "detect", "v5detect", "segment", "v5segment", "obb", "pose"]
        if len(train_img) != 0:  #从训练集判断
            for img in train_img:
                img = str(Path(data["path"]) / img if img.startswith(".") else img)
                label_path = img2label_paths([img])[0]
                label = readLabelFile(label_path)
                if len(label) == 0:  #ok样本
                    continue
                else:
                    if len(label[0]) == 5:  #目标检测数据集
                        return ["detect", "v5detect"]
                    elif len(label[0]) == 6:  #obb目标检测数据集
                        return ["obb"]
                    else:  #segment 或者 pose
                        return ["detect", "v5detect","segment","v5segment", "pose"]
        else:    #从验证集判断
            for img in val_img:
                label_path = img2label_paths([img])[0]
                label = readLabelFile(label_path)
                if len(label) == 0:
                    continue
                else:
                    if len(label[0]) == 5:
                        return ["detect", "v5detect"]
                    elif len(label[0]) == 6:
                        return ["obb"]
                    else:
                        return ["detect", "v5detect","segment","v5segment", "pose"]
    else:
        QMessageBox.critical(None, "提示", f"数据集{dataset}格式错误, 应为分类文件夹或者yaml数据集文件")
        return []
    return [ "detect", "v5detect", "segment", "v5segment", "obb", "pose"]
    
def get_models(task):
    """根据任务类型获取可用的神经网络模型"""
    models = glob.glob(str(ROOT /"cfg" / "models" / task / "**"), recursive=False)
    return [Path(m).name for m in models]

def judge_pt_task(pt_model, task):
    """判断pt模型是否兼容task"""
    ckpt = torch.load(pt_model, map_location="cpu")
    args = ckpt["train_args"]
    if args["task"] != task:
        return False
    return True