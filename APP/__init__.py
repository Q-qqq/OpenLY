from functools import wraps
import torch
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


from pathlib import Path
from ultralytics.utils import yaml_load, yaml_save, LOGGER
from ultralytics.utils.checks import check_version, check_yaml


import time
import copy
import math
import numpy as np
import glob

__version__ = "洛业LY:2.2.0"

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
        self.file = Path(path)

        if not self.file.exists():
            self.save()
        self.update(yaml_load(self.file))
        correct_keys = self.keys() == self.defaults.keys()
        correct_type = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
        if self.get("version"):
            correct_version = self["version"].split(":")[0] == self.defaults["version"].split(":")[0]
        if not (correct_keys and correct_type and correct_version):
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
        projects = copy.copy(self["projects"])
        for project in projects:
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
            "current_experiment": "",
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
        super().load(Path(PROJ_SETTINGS["name"]) / "experiments" /  experiment / "SETTINGS.yaml")
        self["name"] = experiment
        self.save()
        PROJ_SETTINGS["current_experiment"] = experiment
        PROJ_SETTINGS.save()


APP_SETTINGS = APPSetting()
PROJ_SETTINGS = ProjectSetting()
EXPERIMENT_SETTINGS = ExperimentSetting()














