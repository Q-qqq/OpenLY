from functools import wraps
import glob
import math
from random import sample
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import numpy as np
import torch
from pathlib import Path

from APP import APP_ROOT, APP_SETTINGS, EXPERIMENT_SETTINGS, PROJ_SETTINGS, __version__
from APP.Data import readLabelFile
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_version
from ultralytics.nn.tasks import guess_model_task

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
        label_len = []
        if len(train_img) != 0:  #从训练集判断
            for img in sample(train_img, 10 if len(train_img)>10 else len(train_img)):
                img = str(Path(data["path"]) / img if img.startswith(".") else img)
                label_path = img2label_paths([img])[0]
                label = readLabelFile(label_path)
                if len(label) == 0:  #ok样本
                    continue
                else:
                    label_len.append([len(l) for l in label])
        else:    #从验证集判断
            for img in sample(val_img, 10 if len(val_img)>10 else len(val_img)):
                label_path = img2label_paths([img])[0]
                label = readLabelFile(label_path)
                if len(label) == 0:
                    continue
                else:
                    label_len.append([len(l) for l in label])
        if len(label_len) == 0:
            return [ "detect", "v5detect", "segment", "v5segment", "obb", "pose"]
        elif label_len == [5]*len(label_len):
            return ["detect", "v5detect"]
        elif label_len == [9]*len(label_len):
            return ["obb"]
        elif min(label_len) == max(label_len):
            return ["pose"]
        else:
            return ["detect", "v5detect","segment","v5segment"]
    else:
        QMessageBox.critical(None, "提示", f"数据集{dataset}格式错误, 应为分类文件夹或者yaml数据集文件")
        return []
    
def get_models(task):
    """根据任务类型获取可用的神经网络模型"""
    models = glob.glob(str(ROOT /"cfg" / "models" / task / "**"), recursive=False)
    return [Path(m).name for m in models]



def check_pt_task(model, task):
    """判断pt模型是否与任务匹配
    Args:
        model(str): pt模型路径
        task(str): 学习任务：detect、segment、v5detect、v5segment、obb、pose、classify
    """
    def judge_pt_task(pt_model, task):
        ckpt = torch.load(pt_model, map_location="cpu")
        args = ckpt["train_args"]
        if args["task"] != task:
            QMessageBox.information(None, "提示", f"模型{model}执行的任务{args["task"]}与当前任务{task}不符")
            return False
        return True
    if Path(model).exists() and Path(model).suffix == ".pt":
        if judge_pt_task(model, task):
            return  True
    else:
        QMessageBox.information(None, "提示", f"模型{model}不可用，请选择一个可用的pt模型")
    items = glob.glob(str(Path(getExperimentPath()) / "**" / "*.pt"), recursive=True)
    if len(items):
        model, ok = QInputDialog.getItem(None, "pt模型", "model:", items, 0, True)
        if ok and model != "":
            if Path(model).exists() and Path(model).suffix == ".pt" and judge_pt_task(model, task):
                return True
            else:
                QMessageBox.information(None, "提示", f"模型错误{model}， 请确定模型路径存在或是否pt模型或模型任务是否正确")
    return False

def check_yaml_task(model, task):
    """检查yaml模型是否与学习任务匹配
    Args:
        model(str): yaml模型路径
        task(str): 学习任务：detect、segment、v5detect、v5segment、obb、pose、classify
    """
    if Path(model).exists() and Path(model).suffix in (".yaml", "yml"):
        net = yaml_load(model)
        net_task = guess_model_task(net) 
        if net_task == task:
            return True
        else:
            QMessageBox.information(None, "提示", f"模型任务{net_task}与目标任务{task}不符")
    else:
        QMessageBox.information(None, "提示", "请确定模型路径是否存在或模型是否未yaml网络模型")
    return False


def getExperimentPath(name = ""):
    """获取实验路径""" 
    project = PROJ_SETTINGS["name"]
    experiment = name if name != "" else EXPERIMENT_SETTINGS["name"]
    return str(Path(project) / "experiments" / experiment)

def getExistDirectory(*args, **kwargs):
    """获取系统存在的一个文件夹并保存历史路径
    Args:
        parent(QWidget|None): 父窗口
        caption(str): 标题
        dir(str): 初始打开文件路径，默认为APP_SETTINGS["default_dir"]"""
    if len(args) < 3 and not kwargs.get("dir"):
        kwargs["dir"] = APP_SETTINGS["default_dir"]
    dir = QFileDialog.getExistingDirectory(*args, **kwargs)
    APP_SETTINGS.update({"default_dir": dir }) if dir != "" else None
    return dir

def getOpenFileName(*args, **kwargs):
    """
    获取单个文件
    Args：
        parent(QWidget|None): 父窗口
        caption(str): 标题
        dir(str): 初始打开文件路径，默认为APP_SETTINGS["default_file"]
        filter(str): 文件过滤 'file (*.jpg *.png *.gif)'"""
    if len(args) < 3 and  not kwargs.get("dir"):
        kwargs["dir"] = APP_SETTINGS["default_file"]
    file, file_type = QFileDialog.getOpenFileName(*args, **kwargs)
    APP_SETTINGS.update({"default_file": str(Path(file).parent)}) if file != "" else None
    return file, file_type

def getOpenFileNames(*args, **kwargs):
    """
    获取多个文件
    Args：
        parent(QWidget|None): 父窗口
        caption(str): 标题
        dir(str): 初始打开文件路径,默认APP_SETTINGS["default_file"]
        filter(str): 文件过滤 'file (*.jpg *.png *.gif)'"""
    if len(args) < 3 and  not kwargs.get("dir"):
        kwargs["dir"] = APP_SETTINGS["default_file"]
    files, files_type = QFileDialog.getOpenFileNames(*args, **kwargs)
    APP_SETTINGS.update({"default_file": str(Path(files[0]).parent)}) if len(files) else None
    return files, files_type


def checkProject(project_path):
    """核查项目路径是否一个可用的项目"""
    project_path = Path(project_path)
    pro_set_p = project_path / "SETTINGS.yaml"

    #检测存在
    if not project_path.exists() or not pro_set_p.exists():
        QMessageBox.show(None, "提示", "项目不存在")
        return False

    #检测是否项目路径
    proj_set = yaml_load(pro_set_p)
    if not proj_set.get("current_experiment", None): #不存在关键参数
        QMessageBox.show(None, "提示", "项目参数缺失，该项目可能不是OpenLY项目")
        return False
    
    #检测版本
    version = proj_set.get("version", None)
    if not version or version.split(":")[0] != __version__.split(":")[0]:
        QMessageBox.show(None, "提示", "软件版本错误")
        return False

    return True
    


def loadQssStyleSheet(app,train_ui):
    """加载QSS样式表"""
    #train_ui.setAttribute(Qt.WA_TranslucentBackground) 
    if APP_SETTINGS["style"] == "cute":
        styleSheetPath = APP_ROOT / "APP" / "resources" / "styles" / "cute.qss"
        train_ui.centralWidget().setGraphicsEffect(None) 
    elif APP_SETTINGS["style"] == "technology":
        styleSheetPath = APP_ROOT / "APP" / "resources" / "styles" / "technology.qss"
        color = QColor(255,  255, 255, 200)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25) 
        shadow.setColor(color)
        shadow.setOffset(3,  3)
        train_ui.centralWidget().setGraphicsEffect(shadow) 
        init_progress_effect(train_ui.Train_progressbar)
    elif APP_SETTINGS["style"] == "light":
        styleSheetPath = APP_ROOT / "APP" / "resources" / "styles" / "light.qss"
    
    
    with open(styleSheetPath, "r", encoding="utf-8") as f:
        qss = f.read()
    app.setStyleSheet(qss)

# 量子隧穿动画引擎 
def init_progress_effect(progress_bar):
    # 创建多维度光晕 
    holo_glow  = QGraphicsDropShadowEffect()
    holo_glow.setBlurRadius(30) 
    holo_glow.setColor(QColor(61,122,254,80)) 
    
    # 配置时空涟漪动画 
    ripple_anim  = QVariantAnimation()
    ripple_anim.setDuration(1500) 
    ripple_anim.setStartValue(0) 
    ripple_anim.setEndValue(360) 
    ripple_anim.valueChanged.connect( 
        lambda val: holo_glow.setOffset( 
            val/36, 
            math.sin(math.radians(val))*5  
        )
    )
   
    
    # 启动平行宇宙渲染 
    progress_bar.setGraphicsEffect(holo_glow)
    ripple_anim.setLoopCount(-1)
    ripple_anim.start() 



def debounce(delay_ms: int):
    """防抖装饰器"""
    def decorator(func):
        timer = QTimer()
        timer.setSingleShot(True)

        @wraps(func)
        def wrapper(*args, **kwargs):
            timer.start(delay_ms)
            # 保存最新参数
            wrapper._args = args
            wrapper._kwargs = kwargs

        # 连接定时器到实际函数
        timer.timeout.connect(lambda: func(*wrapper._args, **wrapper._kwargs))
        return wrapper
    return decorator