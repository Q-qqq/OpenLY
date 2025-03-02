from ultralytics.utils import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    yaml_load,
    IterableSimpleNamespace,
    colorstr,
    LOGGER,
    ROOT,
    RUNS_DIR,
    RANK
)
from pathlib import  Path
from types import SimpleNamespace
from typing import Dict, List, Union
import yaml
from ultralytics.utils import deprecation_warn

MODELS = "train", "val", "predict", "export", "track", "benchmark"
TASKS = "detect", "segment", "classify", "pose", "obb"
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml"
}
TASK2MODEL = {
    "detect": "yolov8n.pt",
    "segment": "yolov8n-seg.pt",
    "classify": "yolov8n-cls.pt",
    "pose": "yolov8n-pose.pt",
    "obb": "yolov8n-obb.pt"
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)"
}

#分类（type）定义arg
CFG_FLOAT_KEYS = (
     "warmup_epochs",  #3.0, warmup epochs
     "box",            #7.5, box loss gain
     "cls",            #0.5, cls loss gain
     "dfl",            #1.5, dfl loss gain
     "pose",           #12.0, pose loss gain
     "degrees",        #0.0, image rotation (+/-deg) for augment
     "shear",          #0.0, image shear (+/-deg) for augment
     "time"           #number of hours to train for override epochs if supplied
)   #浮点数参数0 ~ ∞

CFG_FRACTION_KEYS = (
     "dropout",   #0.0, use dropout regularization (正则化，classify train only)随机失活
     "lr0",       #0.01, initial learning rate (i.e SGD=1E-2,  Adam=1E-3)
     "lrf",       #0.01, final learning rate (真实最终学习率：lr0*lrf)
     "momentum",  #0.937, SGD momentum/Adam beta1
     "weight_decay",  #0.0005, optimizer weight decay 5E-4
     "warmup_momentum",   #0.8, warmup initial momentum
     "warmuo_bias_lr",    #0.1, warmup initial bias lr
     "label_smoothing",   #0.0, label smoothing
     "hsv_h",     #0.015, image HSV-Hue augmentation
     "hsv_s",     #0.7, image HSV-Saturation augmentation
     "hsv_v",     #0.4, image HSV-Value augmentation
     "translate",   #0.1, image translate(+/-) for augment
     "scale",       #0.5, image scale(+/-) for augment
     "perspective", #0.0, iamge perspective(+/-), range 0 ~ 0.001
     "flipud",      #0.0, image flip up-down probability
     "fliplr",      #0.5, image flip left-right probability
     "mosaic",      #1.0, image mosaic(图像拼接) probability
     "mixup",       #0.0, image mixup(图像重叠) probability
     "copy_paste",  #0.0, segment copy-paste
     "erasing",     #0.4, probability of random erasing during classification training
     "crop_fraction",  #1.0, image crop fraction for classification evaluation/inference
     "conf",        #0.5, confidence thshold for detection (default 0.25 predict, 0.001 val)
     "iou",         #0.7, intersection over union (IOU): thresold for NMS
     "fraction",    #1.0, dataset fraction to train (default all image to train)
)   #fraction float 0.0 ~ 1.0

CFG_INT_KEYS = (
     "epochs",   #100, number of epochs to train
     "patience",  #50, epochs to wait for no observable improvement for early stopping of training (在该epochs内不早停)
     "batch",     #16， number of images per batch (-1 is AutoBatch)
     "workers",   #8, number of workers threads for data loading (per RANK if DDP)
     "seed",      #0, random seed for reproducibility
     "close_mosaic", #1, enable mosaic augmentation for final epoch (0 to disable)
     "mask_ratio",   #4, mask downsample ratio (segment train only)
     "max_det",      #300, maximum number of detections per image
     "vid_stride",   #1, vidio frame rate stride
     "line_width",   #None, line width of the bounding boxes. Scale to image size if None
     "workspace",    #4, TensorRT: workspace size(GB)
     "nbs",          #64, nominal batch size
     "save_period",  #-1, Save checkpoint every x epochs (disabled if < 1)
)

CFG_BOOL_KEYS = (
     "save",      #True, Save checkpoints and predict result
     "exist_ok",  #False, Whether to overwrite exsiting experiment
     "verbose",   #True, Whether to print verbose output
     "deterministic",  #True, whether to enable deteministic mode
     "single_cls",   #False, train multi-class data as single-class
     "rect",      #False, rectangular training if mode = "train" or rectangular validate if mode = "val"
     "cos_lr",    #False, use cosine learning rate scheduler
     "amp",       #True, whether Automatic Mixed Precision(AMP) training
     "profile",   #False, profile ONNX and TensorRT speeds during training for loggers
     "overlap_mask",  #True, masks should overlap during training (segment train only)
     "val",         #True, validate/test during training
     "save_json",   #False, save results to Json file
     "save_hybrid", #False, save hybrid vision of labels (labels + addication predictions)
     "half",        #False, use half precision(FP16)
     "dnn",         #False, use OpenCV DNN for DNN inference
     "plots",       #True, save plots and image during train/val
     "show",        #False, show predicted iamges and videos if environment allows
     "save_txt",    #False, save results as .txt file
     "save_conf",   #False, save results with confidence scores
     "save_crop",   #False, save cropped images with results
     "save_frames",  #False, save predicted individual video frames
     "show_labels",  #True, show prediction labels
     "show_conf",    #True, show prediction confidence
     "show_boxes",   #True, show prediction boxes
     "stream_buffer", #False, buffer all streaming frames(True) or return the most recent frame(False)
     "visualize",    #False, visualize model features
     "augment",      #False, apply iamge augmentation to prediction sources
     "agnostic_nms", #False, class agnostic NMS
     "retina_masks", #False, use high-resolution segmentaion masks
     "keras",        #False, use keras = s
     "optimize",     #False, TorchScript: optimize for mobile
     "int8",         #False, CoreML/TF INT8 quantization
     "dynamic",      #False, ONNX/TF/TensorRT: dynamic axes
     "simplify",     #False, ONNX: simplify model
     "nms",          #False, CoreML:add NMS
     "multi_scale",  #False, whether to use multi-scale during training
)
CFG_OTHER_KEYS = (
     "task",  #(str)detect, YOLO task, i.e. detect, segmetn, classify, pose
     "mode",  #(str)train, YOLO mode, i.e. train, val, predict, export, track,benchmask
     "model",  #(str)modeln.pt, path to model file i.e. yolov8n.pt, yalov8n.yaml
     "data",   #(str)data.yaml, path to data file, i.e. coco128.yaml
     "cache",   #(bool| str)False, Use cache for data loading ,i.e. True(ram, disk)/False
     "device",   #(int|list[int] | str)0, device to run, i.e. cufa device=0 pr device=0,1,2,3, cpu device=cpu
     "project",  #(str)proName, project name
     "name",     #(str)expName, experiname, results saved to "project//name" directory
     "pretrained",   #(bool | str)True, whether to use a pretrained model(bool) or amodel to load weights from(str)
     "optimizer",   #(str)auto,  #optimizer to use, choices=[SGD, ADam, Adamx, AdamW,NAdam, RAdam, RMSProp, auto]
     "freeze",      #(int | list[int])None, freeze first n layers, or freeze list of layer indices during training(int | list)
     "split",       #(str)val, dataset split to use for validation, i.e. val, test or train
     "source",    #(str)source directory for images or video for predict
     "classes",   #(int | list[int])[0,2,3], filter results by class, i.e. classes=0 or classes=[0,2,3]
     "embed",     #(list[int])return feature vectors/embeddings from given layers
     "format",    #(str)torchscript,  format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
     "opset",     #(int,optional) ONNX:opset version
     "auto_augment",     #(str)randaugment, auto augmentation policy for classification (randaugment, autoaugment,augmix)
     "cfg",       #(str,optional) for overriding defaults.yaml
     "tracker",   #(str)bootsort.yaml, tracker type, choices=[botsrt.yaml, bytetrack.yaml]
     "resume",    # (bool|str)False, resume training from last checkpoint
     "imgsz",     #(int | list)640, image size  width,height
)





def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    '''
    将一个文件或字典里的配置参数数据进行加载并合并
        :param cfg（str|Path|Dict|SimpleNamespace）:  配置参数文件名称、Path，或已加载的字典Dict，SimpleNamespace
        :param overrides(str|Dict|optionnal): 将来自一个文件或字典的数据合并进cfg，以overrides为主
        :return(SimpleNamespace): 训练参数命名空间
    '''
    cfg = cfg2dict(cfg)

    # 合并
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # 忽略特殊的keys
        check_dict_alignment(cfg, overrides)  # 校对
        cfg = {**cfg, **overrides}  # 合并两个字典，值以overrides为主
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):  # 项目名称，结果保存文件夹为数字
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # 名称不能等于model
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ name=model, 自动更新 name={cfg['name']}.")

    # 检查type和value
    for k, v in cfg.items():
        if v is not None:  # None值可能是可选参数
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(
                    f"'{k}={v}'是无效的type-{type(v).__name__}."
                    f"有效的'{k}'应该是int 或者float"
                )
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"'{k}={v}'是无效的type-{type(v).__name__}."
                        f"有效的'{k}'必须是int/float"
                    )
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}'是一个无效的值"
                                     f"'{k}'必须在0.0-1.0之间")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(
                    f"'{k}={v}'是无效的type-{type(v).__name__}."
                    f"有效的'{k}'必须是bool -- True/False"
                )
    return IterableSimpleNamespace(**cfg)


def cfg2dict(cfg):
    '''
    将一个配置对象转换为字典，无论它是一个文件路径Path，一个字符串路径，一个字典或者一个SimpleNamespace对象
        :param cfg(str | Path | dict | SimpleNamespace): 需要去转换的配置对象
        :return(dict):  字典格式的配置
    '''
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load 字典
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # 转换为字典
    return cfg

def check_dict_alignment(base: Dict, custom: Dict, e=None):
    '''
    检查两个字典的keys是否一一对应
        :param base:基字典
        :param custom:需要核对的字典
        :param e:
    '''
    custom = _handle_deprecation(custom)
    base_key, custom_keys = (set(x.keys()) for x in (base, custom))  # 将字典的keys对象转化为set集合
    mismatched = [k for k in custom_keys if k not in base_key]  # 只存在于custom不存在于base的key   并集 - 交集
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_key)  # 根据x在base_key中的相似度进行排列
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f" 相似的参数： i.e. {matches}" if matches else ""
            string += f"'{x}' 不是一个有效的yolo参数。 \n{match_str}"
        LOGGER.error(string)

def _handle_deprecation(custom):
    '''处理废弃的参数'''
    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")
    return custom


def get_save_dir(args, name=None):
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path
        project = args.project or RUNS_DIR / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)
    return Path(save_dir)



