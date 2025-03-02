import math
import os
from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
from ultralytics.utils import LOGGER,DEFAULT_CFG_DICT,DEFAULT_CFG_KEYS
from ultralytics.utils.checks import check_version
import thop
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import thop

TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")
TORCHVISION_0_10 = check_version(torchvision.__version__, "0.10.0")
TORCHVISION_0_11 = check_version(torchvision.__version__, "0.11.0")
TORCHVISION_0_13 = check_version(torchvision.__version__, "0.13.0")

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """多卡训练，使分布式训练的所有进程等待每个local_master执行某些操作"""
    initialized = dist.is_available() and dist.is_initialized()
    if initialized and local_rank not in (-1, 0):   #非主线程
        dist.barrier(device_ids=[local_rank])
    yield          #迭代生成器 中断此函数继续执行上下文后再继续执行此函数下文
    if initialized and local_rank ==0:
        dist.barrier(device_ids=[0])

def time_sync():
     if torch.cuda.is_available():
         torch.cuda.synchronize()   #等待Gpu计算完成
     return time.time()

def fuse_conv_and_bn(conv, bn):
    """混合Conv2d和BatchNorm2d"""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).requires_grad_(False).to(conv.weight.device)
    )
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(deconv.weight.device)
    )

    # Prepare filters
    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(deconv.weight.size(1), device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

def is_parallel(model):
    """模型是否DP/DDP"""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    """反并行化模型，如果模型是DP或者DDP模型，返回单GPU模型"""
    return model.module if is_parallel(model) else model

def get_num_params(model):
    """获取模型全部参数量"""
    return sum(x.numel() for x in model.parameters())

def get_num_gradients(model):
    """获取模型带梯度回传的参数量"""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def get_flops(model ,imgsz=640):
    """获取模型的FLOPs"""
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  #max stride
            im = torch.empty(1, p.shape[1], stride, stride, device=p.device)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  #stride  GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride   #imgsz GFLOPs
        except Exception:
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
    except Exception:
        return 0.0


def model_info(model, detailed=False, verbose=True, imgsz=640):
    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    n_l = len(list(model.modules())) #number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info("%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype))

    flops = get_flops(model, imgsz)
    fused = "(fuse)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f",{flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo","YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}")
    return n_l, n_p, n_g, flops

def intersect_dicts(da, db, exclude=()):
    """返回da和db相交的字典，其中键值不包含exclude"""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def make_divisible(x, divisor):
    """取最接近x的divisor倍数的值"""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max()) #to int
    return math.ceil(x / divisor) * divisor

def initialize_weights(model):
    """初始化模型权重为随机值"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """在给定的比例radio和网格大小gs的基础下缩放填充图像的尺寸"""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  #resize
    if not same_shape:   #不填充/剪切
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))  #gs倍数
    return F.pad(img, [0, w-s[1], 0, h - s[0]], value=0.447)

def get_cpu_info():
    import cpuinfo
    k = "brand_raw", "hardware_raw", "arch_string_raw"
    info = cpuinfo.get_cpu_info()
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
    return string.replace("(R)", "").replace("CPU","").replace("@", "")

def select_device(device="", batch=0, newline=False, verbose=True):
    """
    根据提供的参数选择合适的Pytorch device
    Args:
        device(str | torch.device, optional):'None'.'cpu','cuda','0','0,1,2,3'
        batch(int, optional): 批次大小，默认0，
        newline(bool, optional): 如果为真，在日志的末尾加回车换行，默认false,
        verbose(bool, optional): 如果为真，显示device信息日志,默认true
    Returns:
        (torch.device): selected device
    Examples:
        select_device('cuda:0')
        device(type='cuda', index=0)

        select_device('cpu')
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device):
        return device

    s = ""
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")
    cpu = device == "cpu"
    mps = device in ("mps", "mps:0")  #苹果系统驱动
    #确保device可用
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #强制设置 torch.cuda.is_available() -> False
    elif device:
        if device == "cuda":
            device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", ""))):
            raise ValueError(
                f"无效的'CUDA'device={device}被请求,"
                f"使用'device=cpu'或者使用有效的CUDA device(s),"
                f"例如‘device=0’ 或 ‘device=0,1,2,3’\n"
                f"torch.cuda.is_available(): {torch.cuda.is_available()}\n"
                f"torch.cuda.device_count(): {torch.cuda.device_count()}\n"
                f"os.environ['CUDA_VISIBLE_DEVICES']: {visible}"
            )
    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        n = len(devices)  #device count
        if n > 0 and batch > 0 and batch % n != 0:   #batch_size要整除于device count
            raise ValueError(f"'batch={batch}'必须是GPU数量{n}的整数")
        #GPU信息
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i==0 else space}CUDA:{d} ({p.name},{p.total_memory/(1<<20):.0f}MiB)\n"
        arg = "cuda:0"
    #elif mps and TORCH_2_0 and torch.backends.mps.is_available():
    #    s += f"MPS ({get_cpu_info()})\n"
    #    args = "mps"
    else:
        s += f"CPU({get_cpu_info()})"
        arg = "cpu"

    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)

def init_seeds(seed=0, deterministic=False):
    """初始化随机数字生成器种子，保证每次随机都一样"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #多GPU

    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

def profile(input, ops, n=10, device=None):
    """
    ultralytics速度、内存和计算量分析器
    Example:
        ```python
        from ultralytics.utils.torch_utils import profile

        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
        ```
    """
    results = []
    if not isinstance(device, torch.device):
        device=select_device(device)
    LOGGER.info(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else[input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [input]:
            m = m.to(device) if hasattr(m,"to") else m
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]
            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2   #GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()  #运行模型时间
                    try:
                        (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()  #反馈时间
                    except Exception:
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n  #每一个forward的时间 单位ms
                    tb += (t[2] - t[1]) * 1000 / n  #每一个backward的时间 单位ms
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0 #GB
                s_in, s_out = (tuple(x.shape) if isinstance(x. torch.Tensor) else "list" for x in (x, y))  #shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0 #参数量
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.warning(str(e))
                results.append(None)
            torch.cuda.empty_cache()
    return results

def copy_attr(a, b, include=(), exclude=()):
    """将b的属性复制到a中"""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """返回一个lambda函数，其以sin的趋势从y1到y2"""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

class EarlyStopping:
    """早停类：当训练达到指定epoch且没有再提升时停止训练"""

    def __init__(self, patience=50):
        self.best_fitness = 0.0 #i.e. mAP
        self.best_epoch= 0
        self.patience = patience or float("inf")   #没有提升的epochs达到即停止
        self.possible_stop = False  #下一个epoch可能会停止

    def __call__(self, epoch, fitness):
        """检测是否停止"""
        if fitness is None:
            return False

        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        if stop:
            LOGGER.info(
                f"在最后的{self.patience}epochs里模型训练未提升，触发早停"
                f"最好的训练结果在{self.best_epoch}epoch"
            )
        return stop

def strip_optimizer(f="best.pt", s=""):
    """将优化器从‘f’中去除，并将新的pt文件保存到‘s’，如果s为'',那么将覆盖原先的f文件"""
    x = torch.load(f, map_location=torch.device("cpu"))
    if "model" not in x:
        LOGGER.info(f"跳过，{f}不是一个有效的ultralytics模型")
        return
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  #dict
    args = {**DEFAULT_CFG_DICT, **x["train_args"]} if "train_args" in x else None
    if x.get("ema"):
        x["model"] = x["ema"]   #将model替换成ema
    for k in "optimizer","best_fitness", "ema", "updates":
        x[k] = None
    x["epoch"] = -1
    x["model"].half() #FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    x["train_args"] = {k:v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  #去除非默认键值
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  #dile size
    LOGGER.info(f"优化器已经从{f}中去除，{f'并将新的pt保存到{s}' if s else ''} {mb:.1f}MB")

def smart_inference_mode():
    def decorate(fn):
        if torch.is_inference_mode_enabled() and TORCH_1_9:
            return fn
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)
    return decorate
