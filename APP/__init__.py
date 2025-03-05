import torch
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


from pathlib import Path
from ultralytics.utils import yaml_load, yaml_save, LOGGER
from ultralytics.utils.checks import check_yaml


import time
import copy
import math
import numpy as np
import glob

__version__ = "洛业LY:2.0.0:"

FILE = Path(__file__).resolve()
APP_ROOT = FILE.parents[1]  # APP根目录


FILL_RULE = Qt.WindingFill

class Setting(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 创建字典
        self.file = None

    def save(self):
        """保存当前配置到YAML文件"""
        if self.file:
            yaml_save(self.file, dict(self))

    def load(self, path):
        """加载新的项目或者切换项目"""
        from ultralytics.utils.checks import check_version
        self.file = Path(path)

        if not self.file.exists():
            self.save()
        self.update(yaml_load(self.file))
        correct_keys = self.keys() == self.defaults.keys()
        correct_type = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
        if self.get("version"):
            correct_verssion = check_version(self["version"], self.defaults["version"])
        if not (correct_keys, correct_type, correct_verssion):
            LOGGER.warning(f"项目配置与软件不匹配，配置将恢复默认状态，更新至{self.file}")
            self.reset()

    def checkVersion(self, version):
        """检查版本是否与软件版本匹配"""
        pass

    def update(self, *args, **kwargs):
        """更新一个配置值"""
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        """重置"""
        self.clear()
        self.update(self.defaults)
        self.save()

class APPSetting(Setting):
    """
        管理存储在yaml文件中的APP设置
        """

    def __init__(self):
        self.defaults = {
            "version": __version__,
            "projects": [],
            "current_project":"",
            "default_dir": "C://",
            "default_file": "C://",
            "style": "cute",
        }
        super().__init__(copy.deepcopy(self.defaults))  # 创建字典
        self.load(APP_ROOT / "SETTINGS.yaml")
        self.checkProjects()

    def updateProject(self, project):
        project = str(Path(project))
        """更新项目列表"""
        self["projects"] = [str(Path(p)) for p in self["projects"]]
        if project in self["projects"]:
            self["projects"].remove(project)
        self["projects"].append(project)
        self["current_project"] = project
        self.save()
    
    def checkProjects(self):
        """检查项目是否存在且不重复"""
        self["projects"] = [str(Path(p)) for p in self["projects"]]
        for project in self["projects"]:
            if self["projects"].count(project) > 1:
                self["projects"].remove(project)
            if not Path(project).exists() or project == "":
                self["projects"].remove(project)
        self.save()

class ProjectSetting(Setting):
    def __init__(self):
        self.defaults = {
            "name": "",
            "version": __version__,
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "save_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "experiments":[],
            "current_experiment":"",
            "task": "detect",
        }
        super().__init__(copy.deepcopy(self.defaults))

    def save(self):
        """保存当前配置到YAML文件"""
        if self.file:
            self["save_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            yaml_save(self.file, dict(self))

    def load(self,project_path):
        """加载新的项目或者切换项目"""
        super().load(Path(project_path) / "SETTINGS.yaml")
        self["name"] = str(Path(project_path))

    def updateExperiment(self, experiment):
        experiment = experiment
        """更新项目列表"""
        if experiment in self["experiments"]:
            self["experiments"].remove(experiment)
        self["experiments"].append(experiment)
        self["current_experiment"] = experiment
        self.save()

    def updateMode(self, mode):
        self["mode"] = mode


class ExperimentSetting(Setting):
    def __init__(self):
        self.defaults = {
            "name": "",
            "version": __version__,
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "save_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        }
        super().__init__(copy.deepcopy(self.defaults))

    def save(self):
        """保存当前配置到YAML文件"""
        if self.file:
            self["save_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            yaml_save(self.file, dict(self))

    def load(self, experiment):
        """加载新的实验或者切换实验"""
        super().load(Path(ProjectSetting["name"]) / "experiments" /  experiment / "SETTINGS.yaml")
        self["name"] = str(Path(experiment))

    def updateMode(self):
        self.save()

APP_SETTINGS = APPSetting()
PROJ_SETTINGS = ProjectSetting()
EXPERIMENT_SETTINGS = ExperimentSetting()


def getExperimentPath():
    """获取实验路径""" 
    project = PROJ_SETTINGS["name"]
    experiment = EXPERIMENT_SETTINGS["name"]
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
    if not project_path.exists():
        return False
    elif not pro_set_p.exists():
        return False
    return True

class ComboBox(QComboBox):
    def wheelEvent(self, e):
        pass

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













