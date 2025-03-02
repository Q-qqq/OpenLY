import contextlib
import os
import re
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Union
from types import SimpleNamespace
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ultralytics import __version__
import yaml
import platform
import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *



class Logger(QObject):
    """信息显示"""
    Show_Mes_Signal = Signal(str)
    Start_Train_Signal = Signal(list)
    Batch_Finish_Signal = Signal(str)
    Epoch_Finish_Signal = Signal(list)
    Train_Finish_Signal = Signal(str)
    Train_Interrupt_Signal = Signal()
    Start_Val_Signal = Signal(str)
    Val_Finish_Signal = Signal(str)
    Error_Signal = Signal(str)
    def __init__(self, parent=None):
        super(Logger, self).__init__(parent)
        self.errorFormat = '<font color="red" size="5">{}</font>'
        self.warningFormat = '<font color="orange" size="5">{}</font>'
        self.validFormat = '<font color="green" size="5">{}</font>'
        self.stop = False  #停止训练


    def error(self,msg):
        """错误信号"""
        errorMsg = self.errorFormat.format(msg)
        self.Error_Signal.emit(msg)
        self.Show_Mes_Signal.emit(errorMsg)

    def warning(self,msg):
        """警告信号"""
        warningMsg = self.warningFormat.format(msg)
        self.Show_Mes_Signal.emit(warningMsg)

    def info(self,msg):
        """正常信号"""
        validMsg = self.validFormat.format(msg)
        self.Show_Mes_Signal.emit(validMsg)

    def startTrain(self, msg_epochs):
        """开始训练信号"""
        msg_epochs[0] = self.validFormat.format(msg_epochs[0])
        self.Start_Train_Signal.emit(msg_epochs)

    def batchFinish(self, msg):
        """完成一个batch信号"""
        self.Batch_Finish_Signal.emit(msg)

    def epochFinish(self, msg_epoch):
        """完成一个epoch信号"""
        msg_epoch[0] = self.validFormat.format(msg_epoch[0])
        self.Epoch_Finish_Signal.emit(msg_epoch)


    def trainFinish(self, msg):
        """训练结束信号"""
        self.Train_Finish_Signal.emit(self.validFormat.format(msg))

    def trainInterrupt(self):
        """训练停止信号"""
        self.Train_Interrupt_Signal.emit()

    def startVal(self, msg):
        """开始验证信号"""
        self.Start_Val_Signal.emit(self.validFormat.format(msg))

    def valFinish(self, msg):
        """验证结束信号"""
        self.Val_Finish_Signal.emit(msg)






LOGGER = Logger()



class ThreadingLocked:
    """
    确保线程安全的装饰器，能够使被装饰的函数在同一时间只被一个线程使用
    """
    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, f):
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            with self.lock:
                return f(*args, **kwargs)
        return decorated

class TryExcept(contextlib.ContextDecorator):
    """
    使用@TryExcept() 修饰或者 'with TryExcept()' 进行上下文管理
    Examples:
        As a decorator:
        @TryExcept(msg="Error occurred in func", verbose=True)
        def func():
            # Function logic here
             pass

        As a context manager:
         with TryExcept(msg="Error occurred in block", verbose=True):
             # Code block here
             pass
    """
    def __init__(self, msg="", verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """进入TryExcenot上下文时运行"""
        pass

    def __exit__(self, exc_type, value, traceback):
        """当退出‘with’代码块时运行"""
        if self.verbose and value:
            if str(value).startswith("中断"):
                LOGGER.error(f"{value}")
            else:
                LOGGER.error(f"{self.msg}{':' if self.msg else ''}{value}\n"
                            f"error file:{traceback.tb_frame}\n"
                            f"error line:{traceback.tb_lineno}")
        return True


#常量

# 多GPU DDP 常量
RANK = int(os.getenv("RANK", -1)) #进程号
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  #GPU编号
#其他常量
NUM_THREADS = min(8,max(os.cpu_count() -1,1))  #线程数
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
ASSETS = ROOT / "assets"  #default images


def is_dir_writeable(dir_path:Union[str,Path]) -> bool:
    #判断输入路径是否可写入
    return os.access(str(dir_path),os.W_OK)

def emojis(string =""):
    '''
    返回平台（Linus、Windows）相关的字符串
    '''
    return string.encode().decode("ascii","ignore") if WINDOWS else string


def yaml_save(file="data.yaml", data=None, header=""):
    """
    保存data到yaml文件
    :param file（str, optional）: 文件路径
    :param data（dict）: 需要保存的数据
    :param header(str, optional): 文件开头说明
    :return: None
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)  #创建目录

    #将无效的数据类型转换为string
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    '''
    加载yaml文件
    :param file:  yaml后缀的参数文件
    :param append_filename: 是否将yaml文件名称添加至读取结果字典
    :return(dic):  结果字典
    '''
    assert Path(file).suffix in (".yaml", ".yml"), f"试图使用yaml_load加载非yaml文件"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  #string
        #移除指定字符串（非ASCII码）
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+","", s)
        data = yaml.safe_load(s) or {}  #加载yaml
        if append_filename:
            data["yaml_file"] = str(file)
        return data

class IterableSimpleNamespace(SimpleNamespace):
    '''
    SimpleNamespace:输入字典**dict，使其可以像访问属性一样获取字典的值，class.key = value
    SimpleNamespace的扩展，增加了iterable函数使其dict()可以用于for迭代
    '''
    def __iter__(self):
        '''返回一个key-value pairs的iterator，在for迭代循环开始时被调用'''
        return iter(vars(self).items())

    def __str__(self):
        '''返回人类可读的str -- key-value'''
        return "\n".join(f"{k}={v}" for k,v in vars(self).items())

    def __getattr__(self, attr):
        '''访问不存在属性时执行'''
        name = self.__class__.__name__
        raise AttributeError(
            f"""
                    '{name}' 对象不存在 '{attr}'.这个情况可能是由一个修改操作或者想要获取一个不在参数文件内的参数导致 。
                    """)

    def get(self, key, default=None):
        '''返回key指定的value， 如果key不存在，则返回default'''
        return getattr(self,key, default)

#Default configuration
DEFAULT_CFG_PATH =  ROOT / "cfg/default.yaml"
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def deprecation_warn(arg, new_arg, version=None):
    '''当一个废弃的参数被使用时发出一个废弃的警告，并建议使用新参数代替'''
    if not version:
        version = float(__version__[:3]) + 0.2   # 在2个版本后启用
    LOGGER.warning(
        f"WARNING ⚠️ '{arg}' 已被废弃，并且将在YOLO版本{version}被移除"
        f"请用 '{new_arg}' 代替"
    )


def colorstr(*input):
    '''
    给字体赋予颜色格式
         -colorstr('color', 'style', 'your string')
         -colorstr('your string')    默认蓝色，黑体
    Args:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'
    Returns:
        (str):带颜色和格式的字符串
    '''
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    #return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
    return string
def clean_url(url):
    """清除URL中的auth， i.e.https://url.com/file.txt?auth -> https://url.com/file.txt"""
    url = Path(url).as_posix().replace(":/","://")
    return urllib.parse.unquote(url).split("?")[0] # '%2F'-> '/'  split https://url.com/file.txt?auth

def url2file(url):
    """转换URL为文件名， i.e. https://url.com/file.txt?auth -> file.txt"""
    return Path(clean_url(url)).name

def is_online() -> bool:
    """
    检查网络连接状态，并尝试连接已知网络
    :return: 连接是否成功
    """
    import socket

    for host in "1.1.1.1", "8.8.8.8", "223.5.5.5":   #Clounflare, Google, AliDNS
        try:
            test_connection =  socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            test_connection.close()
            return True
    return False

ONLINE = is_online()   #网络是否连接

def get_user_config_dir(sub_dir="Ultralytics"):
    """
    基于环境操作系统返回对应的配置目录
    :param sub_dir(str):将创建的子目录名称
    :return(Path):配置目录的路径
    """

    if WINDOWS:
        path = Path.home() /"AppData"/ "Roaming" / sub_dir
    elif MACOS:
        path = Path.home() /"Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"不支持的操作系统: {platform.system()}")

    if not is_dir_writeable(path.parent):
        LOGGER.warning(f"WARNING ⚠️ 用户配置目录‘{path}’不可写入，默认‘/tmp’ 或 CWD"
                       f"或者你可以为路径定义一个YOLO_CONFIG_DIR环境变量")
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() /sub_dir     #tmp 临时文件   #cwd 终端路径
    #create
    path.mkdir(parents=True, exist_ok=True)
    return path

USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())   #C盘用户目录->Ultralytics 配置目录
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"


def get_git_dir():
    """
    确定当前文件是否git存储库的一部分，如果是，返回存储库的根目录，如果不是，返回None
    :return（Path |None）: Git root directory
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d

class SettingsManager(dict):
    """
    管理存储在yaml文件中的Ultralytics设置
    """
    def __init__(self, file=SETTINGS_YAML, version="0.0.4"):
        """
        初始化
        :param file: Ultralytics配置文件路径，默认 USER_CONFIG_DIR / "settings.yaml"
        :param version: 配置版本
        """
        import copy
        import hashlib

        from ultralytics.utils.checks import check_version
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
        from ultralytics.utils.files import increment_path

        abs_root = Path().resolve()
        root = increment_path(abs_root / "project",mkdir=False)
        self.file = Path(file)

        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),   #唯一标识码
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptrue": True,
            "raytune": True,
            "tensorboard": True,
            "wandb": True,
        }
        super().__init__(copy.deepcopy(self.defaults))   #创建字典

        with torch_distributed_zero_first(RANK):
            if not self.file.exists():
                self.save()
            self.load()
            if not self.get("projects"):
                self.update({"projects":[]})
            correct_keys = self.keys() == self.defaults.keys()  #检查关键字
            correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self. defaults.values()))    #检测值类型
            correct_version = check_version(self["settings_version"], self.version)       #检测版本号
            if not (correct_keys and correct_types and correct_version):
                LOGGER.warning("WARNING ⚠️ Ultralytics 配置将重置至默认状态，可能由你的配置或一个最近的Ultralytics包更新造成\n"
                               f"查看你的‘yolo settings’ 或 {self.file}"
                               "\n 更新配置‘yolo settings key=value’")
                self.reset()


    def save(self):
        """保存当前配置到YAML文件"""
        yaml_save(self.file, dict(self))

    def load(self):
        super().update(yaml_load(self.file))

    def update(self, *args, **kwargs):
        """更新一个配置值"""
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        """重置"""
        self.clear()
        self.update(self.defaults)
        self.save()



def is_pytest_runing():
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(sys.argv[0]).stem)



PREFIX = colorstr("Ultralytics: ")

SETTINGS = SettingsManager()
DATASETS_DIR = Path(SETTINGS["datasets_dir"])   #数据集路径
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])     #模型权重路径
RUNS_DIR = Path(SETTINGS["runs_dir"])           #运行结果路径






def is_pytest_running():
    """确定pytest是否现在正在运行"""
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(sys.argv[0]).stem)

def is_github_action_running() -> bool:
    """确定现在的运行环境时否GitHub"""
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ
TESTS_RUNNING = is_pytest_running() #or is_github_action_running()


class SimpleClass:
    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

def plt_settings(rcparams=None, backend="Agg"):
    """装饰器"""
    if rcparams is None:
        rcparams = {"font.size": 11}
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_backend =plt.get_backend()
            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(backend)
            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)
            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)
            return result
        return wrapper
    return  decorator


def threaded(func):
    """多线程化一个目标函数并返回线程或函数结果的装饰器"""

    def wrapper(*args, **kwargs):
        if kwargs.pop("threaded", True):  #在子线程中运行
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return  func(*args, **kwargs)
    return wrapper

def is_colab():
    """检测当前脚本运行环境是或否Google Colab"""
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ

def is_kaggle():
    """检测当前简本是否在kaggle渠道"""
    return os.environ.get("PMD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"

def cv2_readimg(img_path, color=cv2.IMREAD_COLOR):
    """使用cv2读取图像
    Args:
        img_path(str): 图像路径
        color(bool): 是否彩色图像RGB"""
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8), color)
    return img



