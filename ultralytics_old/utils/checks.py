import glob
import os
import platform
import re
import shutil
from importlib import metadata
from ultralytics.utils import (
    TryExcept,
    LOGGER,
    emojis,
    url2file,
    clean_url,
    ROOT,
    colorstr,
    yaml_load,
    ThreadingLocked,
    USER_CONFIG_DIR,
    ASSETS,
    SimpleNamespace)
from ultralytics.utils import downloads
from matplotlib import font_manager

from pathlib import Path
import torch
import math

def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  #'2.0.1+cpu' -> (2,0,1)
    except Exception as ex:
        LOGGER.warning(f"WARNING ⚠️ 解析版本号{version}失败，返回0,0,0:{ex}")
        return 0,0,0

def check_version(
        current: str="0.0.0",
        required: str="0.0.0",
        name: str="version",
        hard: bool=False,
        verbose: bool=False,
        msg: str=""
) -> bool:
    '''检查现在的版本号（current）是否在要求的版本号（required）范围内
    Args:
        current(str): 现有的版本号或包名称
        requied(str)：要求的版本号或者范围
        name(str,optional)：警告信息的名称
        hard(bool)：若为True，当没有要求的版本时，报错
        cerbose(bool)：若为True，当没有要求的版本时，警告
        msg(str)：显示的警告信息
    Returns:
        (bool)：版本符合要求时，返回True，否则返回False
    Example:

        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
    '''
    if not current:
        LOGGER.warning(f"WARNING⚠️ 无效的调用：check_version({current},{required})")
        return True
    elif not current[0].isdigit():  #current不是版本号而是包名
        try:
            name = current
            current = metadata.version(current)  #获取版本号
        except metadata.PackageNotFoundError as e: #未发现包
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING⚠️{current}包未下载")) from e
            else:
                return False

    if not required:
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  #'>=22.04'  ->('>=', '22.04')
        v= parse_version(version)
        if op == "==" and c!= v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in (">=", "") and not(c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ 要求{name}{op}{version}，但是{name}={current}不满足需求 {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))
        if verbose:
            LOGGER.warning(warning)
    return result

def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """检测文件后最"""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list,tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  #文件后缀
            if len(s):
                assert s in suffix, f"{msg}{f}可接受的后缀是{suffix}，不是{s}"

def check_yolo5u_filename(file: str, verbose: bool = True):
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")   #i.e. yolo5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  #i.e. tolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                LOGGER.info(f"💡 将'model={original_file}' 替换为 'model={file}'.\n"
                             f"与标准的由https://github.com/ultralytics/yolov5训练的yolov5模型相比，"
                             f"由https://github.com/ultralytics/ultralytics进行训练的YOLOv5 'u'models性能明显提高\n")
    return file

def check_file(file, suffix="", download=True, hard=True):
    """hard:找不到报警"""
    #搜索或者下载文件，并且返回路径
    check_suffix(file, suffix)
    file = str(file).strip() #转为string 并去空格
    file  = check_yolo5u_filename(file)   #yolov5n -> yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())   #python版本低于3.10时需要检测'://"
        or file.lower().startswith("grpc://")
    ):  #文件存在或者gRPC Trition 映像
        return file
    elif download and file.lower().startswith(("https://", "https://", "rtsp://","rtmp://","tcp://")): #下载
        url = file
        file = url2file(file)  #获取网址的文件名
        if Path(file).exists():
            LOGGER.info(f"发现{clean_url(url)} 已经存在，位于{file}")
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  #搜寻
        files = glob.glob(str(ROOT / "cfg" / "**"/ file), recursive=True)  #查找cfg文件夹内file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' 不存在")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"超过一个文件被找到‘{file}’, {files}")
        return files[0] if len(files) else []

def check_python(minimum: str = "3.8.0") -> bool:
    """
    检测当前的pyhon版本是否复合最低版本需求
    :param minimum（str）:要求的python最低版本
    :return（bool）:
    """
    return check_version(platform.python_version(), minimum, name="Python", hard=True)

def check_torchvision():
    """
    检查已下载的PyTorch和Torchvision版本的兼容性
    """
    import torchvision
    compatibility_table = {"2.0":["0.15"], "1.13": ["0.14"], "1.12":["0.13"], "1.9":["0.10"]}

    v_torch = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    v_torchvision = ".".join(torchvision.__version__.split("+")[0].split(".")[:2])

    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        if all(v_torchvision != v for v in compatible_versions):
            LOGGER.warning(f"WARNING ⚠️ torchvision=={v_torchvision}跟torch=={v_torch}不兼容\n"
                           f"运行‘pip install torchvision=={compatible_versions[0]}’ 修复torchvision版本"
                           f"或者'pip install -U torch torchvision' 更新两者\n")

def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    '''
    解析一个requirements.txt文件-里面存储需求的模块包名称
    Args:
        file_path(Path): requirements.txt文件的路径Path
        package(str, optional): python模块包，用来替换requirements.txt文件 例如：package = 'ultralytics'
    Returns:
        (List[Dict[str,str]]): 一列表的字典name:soecifier
    Example:
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package='ultralytics')
        ```
    '''
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]    #获取包的元数据
    else:
        requires = Path(file_path).read_text().splitlines()
    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):  #去除注释行
            line = line.split("#")[0].strip()   #忽略注释
            match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line)
            if match:
                requirements.append(SimpleNamespace(name=match[1], spacifier=match[2].strip() if match[2] else ""))
    return requirements




@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude =(), install=True, cmds=""):
    """
    检测已安装的依赖是否满足yolov8需求，如果需要尝试自动更新依赖
    Example:
        ```python
        from ultralytics.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements('path/to/requirements.txt')

        # Check a single package
        check_requirements('ultralytics>=8.0.0')

        # Check multiple packages
        check_requirements(['numpy', 'ultralytics>=8.0.0'])
    :param requirements(Path, str, List[str]): 依赖文件路径，一个单一依赖字符串；多依赖字符串列表
    :param exclude(tuple[str]): 从检查中排除的依赖包名称
    :param install(bool): 是否自动更新缺失的依赖包
    :param cmds(str): 自动更新时添加给pip install命令的其他命令
    """
    prefix = colorstr("red", "bold", "requirements:")
    check_python()  #检查python版本
    check_torchvision() # 检查torch- torchvision 兼容性
    if isinstance(requirements, Path):
        file = requirements.resolve()  #获取绝对路径
        assert file.exists(), f"{prefix} 未找到文件：{file}, 检测失败"
        requirements = [f"{x.name} {x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "") # replace git+https://org/repo.git -> 'repo'
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        try:
            assert check_version(metadata.version(name), required)
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)
    if len(pkgs):
        return False
    return True


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """查找/下载YAML文件并且返回路径、检测后缀"""
    return check_file(file, suffix, hard=hard)

def check_model_file_from_stem(model="yolov8n"):
    """将一个有效的模型stem转换为带后缀.pt的文件名称"""
    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix(".pt")  # yolov8n -> yolov8n.pt
    else:
        return model

def check_class_names(names):
    """
    检查种类名称
    Convert lists to dicts
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-类数据集要求种类索引0-{n-1}，但是存在无效的种类索引\n"
                f"{min(names.keys())}-{max(names.keys())}这是你的定义"
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):   #imagenet class codes, i.e 'n01440764'
            names_map = yaml_load(ROOT/"cfg/datasets/ImageNet.yaml")["map"]  #映射
            names = {k:names_map[v] for k, v in names.items()}
    return names

@ThreadingLocked()
def check_font(font="Arial.ttf"):
    """
    Args:
        font(str): Path 或者字体名称
    Returns:
        file(Path): 字体文件绝对路径
    """
    name = Path(font).name

    #检查USER_CONFIG_DIR
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    #检查系统字体
    matches = [s for s in font_manager.findSystemFonts() if font.lower() in s.lower()]
    if any(matches):
        return matches[0]

    #下载
    url = f"https//ultralytics.com/assets/{name}"
    if downloads.is_url(url):
        downloads.safe_download(url=url, file=file)
        return file

def is_ascii(s) -> bool:
    """检查字符串是否仅有ASCII组成"""
    s = str(s)
    return all(ord(c) < 128 for c in s)

def check_amp(model):
    """yolov8检查Pytorch自动混合精度（AMP）方法，若如检查失败，说明系统AMP存在异常，可能
    导致NaN的损失或者0map，所以将取消使用amp"""
    device = next(model.parameters()).device
    if device.type in ("cpu", "mps"):
        return False  #AMP只兼容CUDA

    def amp_allclose(m, im):
        a = m(im, device=device, verbose=False)[0].boxes.data  #FP32 推理结果
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.data  #AMP 推理解雇哦
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5) #形状是否相同，值是否在容忍度0.5内相等

    im = ASSETS / "bus.jpg"  #检查用图像
    prefix = colorstr("AMP: ")
    LOGGER.info(f"{prefix}运行YOLOv8n检测自动混合精度（AMP）")
    warning_msg = "已设置'amp=True'。如果出现NaN losses还活着0mAP的情况，请设置'amp=False'"
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO("yolov8n.pt"), im)
        LOGGER.info(f"{prefix}通过 ✅")
    except ConnectionError:
        LOGGER.warning(f"{prefix} 跳过检查 ⚠️，离线导致下载YOLOv8n失败，{warning_msg}")
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(f"{prefix}跳过检查 ⚠️，由于ultralytics包错误，无法加载YOLOv8n，{warning_msg}")
    except AssertionError:
        LOGGER.warning(f"{prefix}检查失败❌，你的系统的AMP存在异常，可能导致NaN losses或者0mAP，所以以下训练将不使用AMP")
        return False
    return True

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
        验证图像尺寸是否stride的倍数，如果不是，则改变它为比stride大的最接近stride倍数的尺寸
    Args:
        imgsz(int | cList[int]): Image size.
        stride(int): stride value.
        min_dim(int): Minumum number of dimensions.
        max_dim(int): Maximum number of dimensions.
        floor(int): Minimum allowed value for image size.
    Returns:
        (List[int]): Updated image size
    """
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}'是一个无效的类型{type(imgsz).__name__}\n"
            f"有效的imgsz类型如'imgsz=640' | 'imgsz=[640,640]'"
        )

    #Apply max_dim
    if len(imgsz) > max_dim:   #2 > 1
        msg = (
            "'train'和'val'的imgsz必须是整数int，但'predict'和'export'的imgsz则可能时一个[h.w]list或者一个整数int"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} 是一个无效的图像尺寸，{msg}")
        LOGGER.warning(f"WARNING ⚠️ 更新'imgsz={max(imgsz)}'。{msg}")
        imgsz = [max(imgsz)]

    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]  #update

    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz}必须是最大stride{stride}的倍数，imgsz更新为{sz}")

    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz
    return sz


