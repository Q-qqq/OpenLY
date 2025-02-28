import contextlib

import torch.nn as nn
import torch
import torch.nn.functional as F
from ultralytics.utils import emojis, LOGGER, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS,yaml_load, auto_anchors
from ultralytics.utils import checks
from ultralytics.utils.loss import v8DetectionLoss,v8SegmentationLoss, v8PoseLoss, v8OBBLoss, v8ClassificationLoss
from ultralytics.utils.checks import check_requirements, check_suffix,check_yaml
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    time_sync,
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    model_info,
    intersect_dicts,
    make_divisible,
    initialize_weights,
    scale_img,
)
from pathlib import Path
import thop
from copy import deepcopy
from ultralytics.nn.modules.head import V5Detect, V5Segment
from ultralytics.utils.loss import V5DetectLoss, V5SegmentLoss
from ultralytics.nn.modules import(
    #AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Pose,
    RepC3,
    RepConv,
    ResNetLayer,
    #RTDETRDecoder,
    Segment,
)


class BaseModel(nn.Module):

    def forward(self, x, *args, **kwargs):
        """模型在单一尺度上的前向传递，是_forward_once的包装器"""
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        通过网络执行前向传递
        Args:
            x(torch.Tensor): 模型的输入tensor
            profile(bool): 如果是True，输出每一层的计算时间，默认False
            visualize(bool): 如果是真，保存模型特征图，默认False
            augment(bool): 如果是真，在预测过程中增强图像，默认False
            embed(bool): 一个返回的特征向量列表
        Returns:
            (torch.Tensor): 模型的最后一个输出
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_augment(self, x):
        LOGGER.warning(f"WARNING ⚠️ {self.__class__.__name__} 不支持推理增强")
        return self._predict_once(x)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """前向传递一次"""
        y, dt, embeddings = [], [], [] #outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j==-1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  #flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


    def _profile_one_layer(self, m, x, dt):
        """统计模型的计算速度FLOPS和计算量FLOPs"""
        c = m==self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) *100)
        if m == self.model[0]:
            LOGGER.info(f"{'time(ms)':>10s} {'GFLOPs':>10s} {'params':>10s} module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f} {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s} Total")

    def fuse(self, verbose=True):
        """为了提高计算效率，将Conv2d()和BatchNorm2d（）混合为一层"""
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  #fuse to conv
                    delattr(m, "bn")    # remove batchnorm
                    m.forward = m.forward_fuse  #update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)
        return self


    def is_fused(self, thresh=10):
        """检查模型是否至少包含thresh个bn层"""
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        """输出模型信息"""
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """将函数应用于模型中不是参数或注册缓冲区的所有张量"""
        self = super()._apply(fn)
        m = self.model[-1] #Head()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        加载模型权重
        Args:
            weights (dict | torch.nn.Module): 预训练权重
            verbose(bool, optional): 是否记录加载过程，默认True
        """
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())  #获取csd 和 self.state_dict() 的字典交集，值使用csd
        self.load_state_dict(csd, strict=False)  #load
        if verbose:
            LOGGER.info(f"从预训练权重中转移了{len(csd)}/{len(self.model.state_dict())}项")

    def loss(self, batch, preds=None):
        """
        计算损失值
        Args:
            batch(dict): 去计算损失的原始批数据
            preds(torch.Tensor | List[torch.Tensor]): 预测输出
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError("compute_loss()需要被实施")


class DetectionModel(BaseModel):
    """Yolov8 detection model"""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  #cfg dict

        #model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"用种类数量{nc}覆盖{cfg}的种类数量{self.yaml['nc']}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  #default names dict
        self.inplace = self.yaml.get("inplace", True)



        

        #build strides
        m = self.model[-1]  #Detect()
        forward = lambda x: self.forward(x)[0] if isinstance(m, (V5Segment, Segment, Pose, OBB)) else self.forward(x) #fyorward
        if isinstance(m, ( Detect, Segment, Pose, OBB)):
            s = 256  #最小stride不超过256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        elif isinstance(m, (V5Detect, V5Segment)):
            s=256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            auto_anchors.check_anchors_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)  # 将预选框缩放到grid_size大小
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        #Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """对输入图像x进行增强，并输出增强的推理和训练输出"""
        img_size = x.shape[-2:]  # h w
        s = [1, 0.83, 0.67]  #缩放
        f = [None, 3, None]  #翻转 2上下   3左右
        y = []            #outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs = int(self.stride.max()))
            yi = super().predict(xi)[0]  #forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None    #增强推理输出，训练输出


    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """对推理前的增强进行反操作"""
        p[:, :4] /= scale  #de-scale
        x, y, wh, cls = p.split((1,1,2,p.shape[dim]-4), dim)
        if flips == 2:
            y = img_size[0] - y  #反翻转 上下
        elif flips == 3:
            x = img_size[1] - x  #反翻转 左右
        return torch.cat((x,y,wh,cls), dim)

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))   #1, 4, 16, 64
        e = 1
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))
        y[0]= y[0][..., :-i]   #laget
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]  #small
        return y

    def init_criterion(self):
        if isinstance(self.model[-1], V5Detect):
            return V5DetectLoss(self)
        return v8DetectionLoss(self)

class OBBModel(DetectionModel):
    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8OBBLoss(self)

class SegmentationModel(DetectionModel):
    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
    def init_criterion(self):
        if isinstance(self.model[-1], V5Segment):
            return V5SegmentLoss(self, overlap=self.args.overlap_mask)
        return v8SegmentationLoss(self)

class PoseModel(DetectionModel):
    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None,None), verbose=True):
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  #加载模型参数
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.warning(f"WARNING ⚠️ 使用kpt_shape={data_kpt_shape}覆盖 model.yaml的kpt_shape={cfg['kpt_shape']}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8PoseLoss(self)

class ClassificationModel(BaseModel):
    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch) #input channel
        if nc and nc != self.yaml["nc"]:
            LOGGER.warning(f"WARNING ⚠️ 使用nc={nc}覆盖 model.yaml的nc={self.yaml['nc']}")
            self.yaml["nc"] = nc
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("没有定义模型的种类数量")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)   #load model
        self.stride = torch.Tensor([1])
        self.name = {i: f"{i}" for i in range(self.yaml["nc"])}  #default names
        self.info()  #输出模型信息

    @staticmethod
    def reshape_outputs(model, nc):
        """修改检测头输出通道数"""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]   #最后一个模块 检测头
        if isinstance(m, Classify):  #YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.Linear.in_features, nc)
        elif isinstance(m, nn.Linear): #ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
                elif nn.Conv2d in types:
                    i = types.index(nn.Conv2d)
                    if m[i].out_channels != nc:
                        m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        return v8ClassificationLoss()


class Ensemble(nn.ModuleList):
    """模型集合"""
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """生成YOLO网络的最后一层"""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)   # nms 集合，y.shape(B,HW,C)
        return y, None  #推挤结果， 训练输出

@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    上下文管理去暂时添加或修改python模块缓存(sys.modules)中的模块

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.
    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path

    :param modules（dict, optional）:将旧模块路径映射到新模块路径的字典

    """
    if not modules:
        modules = {}
    import  importlib
    import  sys
    try:
        # 在sys.modules中以旧模块名称设置modules
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)
        yield
    finally:
        #移除暂时的模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    使用torch.load()尝试去加载一个PyTorch模型
    :param weight（str）:PyTorch模型的路径
    :return（dict）: 加载的字典，模型在其中
    """
    from ultralytics.utils.downloads import attempt_download_asset

    checks.check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)   #如果本地没有weight，则线上搜索weight
    try:
        with temporary_modules(
                {
                    "ultralytics.yolo.utils": "ultralytics.utils",
                    "ultralytics.yolo.v8": "ultralytics.models.yolo",
                    "ultralytics.yolo.data": "ultralytics.data",
                }
        ):   #对于旧版8，0的分类和位姿模型
            ckpt = torch.load(file, map_location="cpu")
    except ModuleNotFoundError as e:  #e.name是未发现的模块名称
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight}似乎是最初用'https://github.com/ultralytics/yolov5'训练的模型\n"
                    f"在‘https://github.com/ultralytics/ultralytics’中这个模型是不与yolov8兼容的\n"
                    f"建议使用最新的‘ultralytics’包训练一个新的模型或者用一个有效的yolov8模型去运行"
                )
            ) from e
        LOGGER.warning(
            f"ERROR ❌️ {weight}似乎是最初用'https://github.com/ultralytics/yolov5'训练的模型\n"
            f"在‘https://github.com/ultralytics/ultralytics’中这个模型是不与yolov8兼容的\n"
            f"建议使用最新的‘ultralytics’包训练一个新的模型或者用一个有效的yolov8模型去运行"
        )
        #check_requirements(e.name)   #检测缺少的module
        ckpt = torch.load(file, map_location="cpu")
    if not isinstance(ckpt, dict):
        LOGGER.warning(f"WARNING ⚠️ 文件‘{weight}’没完全保存或者格式错误")
        ckpt = {"model": ckpt.model}
    return ckpt, file

def guess_model_task(model):
    """
    从model的结构体系和配置猜测其任务
    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """
    def cfg2task(cfg):
        """从yaml字典猜测"""
        m = cfg["head"][-1][-2].lower() #输出模块的名称
        if m in ("classify", "classifier", "cls", "fc"):
            return "classify"
        if m == "detect":
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"
    #cfg
    if isinstance(model,dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    #Python model
    if isinstance(model, nn.Module):
        for x in "model.args", "model.model.args","model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return  cfg2task(eval(x))
        for m in model.midules():
            if isinstance(m, Detect):
                return "detect"
            elif isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m,Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
        #filename
        if isinstance(model, (str, Path)):
            model = Path(model)
            if "-seg" in model.stem or "segment" in model.parts:
                return"segment"
            elif "-cls" in model.stem or "classify" in model.parts:
                return "classify"
            elif "-pose" in model.stem or "pose" in model.parts:
                return "pose"
            elif "-obb" in model.stem or "obb" in model.parts:
                return "obb"
            elif "detect" in model.parts:
                return "detect"

        #无法猜测
        LOGGER.warning("WARNING ⚠️ 无法猜测除模型任务，默认‘task=detect’")
        return "detect"

def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """加载一个单一模型的所有权重"""

    ckpt, f = torch_safe_load(weight) #load ckpt
    args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None   #合并参数
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()

    model.args = {k:v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = f
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()

    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment, Pose, OBB):
            m.inplace= inplace
        #elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
        #    m.recompute_scale_factor = None       #兼容 torch 1.11.0

    return model, ckpt





def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """加载整个模型的权重{a,b,c}或者一个单一模型权重{a}或者权重a"""
    ensemble = Ensemble()   #modulelist
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None #合并参数
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  #FP32 model

        #update
        model.args = args
        model.pt_path = w
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        #Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())

    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment,Pose, OBB):
            m.inplace = inplace
        #elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
        #    m.recompute_scale_factor = None #兼容 torch 1.11.0


    #返回 model
    if len(ensemble) ==1:
        return ensemble[-1]

    #返回ensemble
    LOGGER.info(f"Ensemble 用{weights} 创建")
    for k in "names","nc","yaml":
        setattr(ensemble, k, getattr(ensemble[0],k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride  #拥有最大stride的model的stride
    assert  all(ensemble[0].nc == m.nc for m in ensemble), f"模型种类数量冲突{[m.nc for m in ensemble]}"
    return ensemble

def guess_model_scale(model_path):
    """从模型yaml文件路径中猜测模型尺度nsmlx"""
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)
    return ""

def yaml_model_load(path):
    """加载模型yaml文件参数字典"""
    import re

    path = Path(path)
    if path.stem in (f"yolo{d}{x}6" for x in "nsmlx" for d in (5,8)):
        new_stem = re.sub(r"(\d+)([nslmx])5(.+)?$", r"\1\2-p6-3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 model现在使用-p6后缀， 重新命名{path.stem}为{new_stem}")
        path = path.with_name(new_stem + path.suffix)

    #unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file) # model dict
    d["scale"] = guess_model_scale(path)  #nslmx
    d["yaml_file"] = str(path)
    return d


def parse_model(d, ch, verbose=True): #model_dict, input_channels
    """解析yaml模型文件"""
    import ast

    #Args
    max_channels = float("inf")
    nc, act, scales, anchors = (d.get(x) for x in ("nc", "activation", "scales","anchors"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ 没有模型尺度nsmlx存在，假设其为{scale}")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  #重新定义默认激活函数
        if verbose:
            LOGGER.info(f"activation={act}")
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1] #layers, savelist, ch out
    for i, (f,n,m,args) in enumerate(d["backbone"] + d["head"]): #from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m] #get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n #depth gain
        if m in (
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
        ):
            c1, c2 = ch[f], args[0]    #input_channels output_channels
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n) #复制数量
                n = 1
        #elif m is AIFI:
        #    args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (V5Segment, V5Detect, Detect, Segment, Pose, OBB):
            args.append([ch[x] for x in f])   #？个检测头对应输入的channels
            if isinstance(args[1], int) and m in (V5Segment, V5Detect):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m in (V5Segment, Segment):
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        #elif m is RTDETRDecoder:
        #    args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  #module
        t = str(m)[8:-2].replace("__main__", "")   #module type
        m.np = sum(x.numel() for x in m_.parameters()) #number of params
        m_.i, m_.f, m_.type = i, f, t   #顺序索引， 输入索引， type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  #append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

