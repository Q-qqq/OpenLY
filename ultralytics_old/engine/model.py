import inspect
import sys
from pathlib import Path
from typing import Union
from torch import nn

from ultralytics.cfg import get_cfg, get_save_dir, TASK2DATA
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, SETTINGS, checks, emojis, yaml_load

class Model(nn.Module):
    """
   Attributes:
       model(Union[str,Path], optional): 模型加载的路径或者新创建模型的名称，默认‘yolov8n.pt’
       task(Any,optional):任务类型由YOLO模型决定，能够指定模型的应用领域，例如目标检测，分割等，默认None
       verbose(bool, optional): 是否显示运行信息
       predictor(BasePredictor):预测对象
       trainer(BaseTrainer): 训练对象
       ckpt(dict): 模型训练节点，从pt文件种加载
       ckpt_path(str): 模型节点文件路径
       overrides(dict):与模型参数重合的字典，将覆盖上模型参数
       metrics(dict): 最后训练或者验证评估指标
       model_name(str): 模型名称
   """
    def __init__(self, model: Union[str,Path]="yolov8n.pt", task=None, verbose=False):
        super().__init__()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt = None
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None
        self.task = task
        self.model_name = model = str(model).strip()

        # load or create
        model = checks.check_model_file_from_stem(model)  # 添加后缀.pt
        if Path(model).suffix in (".yaml", ".yml"):
            self._new(model, task=task, verbose=verbose)   #新建模型
        else:
            self._load(model, task=task)        #加载模型

        self.model_name = model

    @property
    def task_map(self):
        """根据任务映射到各模型的trainer, validator, predictor类
        返回的是一个映射字典"""
        raise NotImplementedError("请为你的模型提供任务映射!")

    def _smart_load(self, key):
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  #获取函数名
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}'模型不支持'{self.task}'任务的'{mode}'模式")
            ) from e

    def _new(self, cfg:str, task=None, model=None,verbose=False):
        """
        创建一个新的模型，并且推断出模型任务
        Args:
            cfg(str): 模型配置文件
            task(str | None): 模型任务
            model(BaseModel): 定制的模型对象
            verbose(bool): 显示"""
        cfg_dict = yaml_model_load(cfg)  #加载模型文件字典
        self.cfg = cfg   #path
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK==-1)  #创建模型
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides} #合并默认的配置参数和模型参数
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """加载一个已有模型.pt文件
        Args:
            weights(str): 模型节点文件
            task(str|None): 模型任务"""
        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task




    @staticmethod
    def _reset_ckpt_args(args):
        """当加载一个PyTorch模型时，重新设置参数"""
        include = {"imgsz", "data", "task", "single_cls"}  #只记得这些参数
        return {k:v for k, v in args.items() if k in include}

    def reset_weights(self):
        """重新将模型参数设置为随机初始值，有效地丢弃所有训练信息"""
        self._check_is_pythorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grab = True
        return self

    def load(self, weights="yolov8n.pt"):
        """加载指定权重到模型中
        该方法支持从一个文件或者文件夹中加载权重，通过名称和shape匹配参数进行加载"""
        self._check_is_pythorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename="model.pt"):
        """保存现在地模型状态到指定文件中"""
        self._check_is_pythorch_model()
        import torch
        torch.save(self.ckpt, filename)

    def info(self, detailed=False, verbose=True):
        """记录或者返回模型参数
        Args:
            detailed(bool)： 是否显示模型地细节信息
            verbose(bool): 是否输出模型信息，如果为否，只返回模型信息
        Returns:
            (list): 多种类型的模型信息列表"""
        self._check_is_pythorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """将模型的Conv2d层和BatchNorm2d层进行混合，提高推理速度"""
        self._check_is_pythorch_model()
        self.model.fuse()


    def embed(self, source=None, stream=False, **kwargs):
        """给予给定源生成图像特征向量
        Args:
            source (str | int | PIL.Image | np.ndarray): The source of the image for generating embeddings.
                The source can be a file path, URL, PIL image, numpy array, etc. Defaults to None.
            stream (bool): If True, predictions are streamed. Defaults to False.
            **kwargs (dict): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings."""
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  #默认倒数第二层
        return self.predict(source, stream, **kwargs)

    def __call__(self, source=None, stream=False, **kwargs):
        """模型预测
        Args:
            source(str | int | PIL.Image | np.ndarray, optional): 预测图像源，可以时文件路径，URLs网址，PIL图像和numpy arrays。默认None
            stream(bool, optional): 是否将输入源当作一个实时流来对待
        Returns:
            (List[ultralytics.engine.results.Results]):预测结果列表"""
        return self.predict(source, stream, **kwargs)

    def train(self, trainer=None, **kwargs):
        """使用指定数据集和配置参数进行训练
            Args:
                trainer(BaseTrainer,optional): 训练模型的一个实例对象，如果None, 使用默认的trainer
            Return:
                (dict | None): 训练评估指标
            """
        self._check_is_pythorch_model()
        overrides = yaml_load(checks.check_yaml(kwargs["cfg"]) if kwargs.get("cfg") else self.overrides)
        custom = {"data": DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task]}   #*.pt
        args = {**overrides, **custom,  **kwargs, "mode": "train"}   #最高优先级的参数
        if args.get("resume"):
            args["resume"] = self.ckpt_path  #恢复训练

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args)
        if not args.get("resume"):   #如果不恢复训练自动设置模型
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def val(self, validator=None, **kwargs):
        """
        使用一个指定的数据集和参数验证模型
        Args:
            validator(BaseValidator, optional):  一个用于验证模型的自定义验证类的实例，如果None，则使用默认的验证器，默认None
            **kwargs(dict): 验证参数的字典
        Returns:
            (dict): 从验证过程中获得的验证指标
        """
        custom = {"rect": True}
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  #最高优先级的参数

        validator = (validator or self._smart_load("validator"))(save_dir= kwargs.get("save_dir"), args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """
        使用YOLO模型在所给的图像源上执行推理
        Args:
            source(str | int | PIL.Image | np.adrray, optional): 图像源：文件路径、URLs，PIL图像，numpy图像
            stream(bool, optional): 图像源作为一个实时流去推理，默认False
            predictor（BasePredictor, optional）: 自定义推理类实例
            **kwargs(dict): 增多的参数
        Returns:
            （List[ultralytics.engine.results.Results]）: 推理结果列表
            """
        if source is None:
            if source is None:
                source = ASSETS  #默认图像
                LOGGER.warning(f"WARNING ⚠️ 推理源丢失，使用'source={source}'")
        is_cli = (sys.argv[0].endswith("yolo") or sys.argv[0].endswith("ultralytics")) and any(
            x in sys.argv for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf":0.25, "save":is_cli, "mode":"predict"}
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else: #只更新参数
            self.predictor.args = get_cfg(self.predictor.args, args)  #高优先级参数覆盖
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, presist=False, **kwargs):
        """
        使用已注册的跟踪器在指定的输入源上执行对象跟踪。
        Args:
            source(str, optional): 目标跟踪的输入源
            stream(bool, optional): 将输入源作为一个持续视频流对待，默认False
            persist(bool, optional): 在不同的调用此方法之间保留跟踪器，默认False
        Returns:
            (List[ultralytic.engine.results.Results])：跟踪结果列表，封装在Results类中
        """
        pass


    def export(self, **kwargs):
        """
        将模型导出为适合部署的不同格式
        """
        pass





    def _check_is_pythorch_model(self):
        """如果self.model不是Pytorch模型，报类型错误"""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix ==".pt"   #pt路径
        pt_model = isinstance(self.model, nn.Module) #pt模型
        if not(pt_model or pt_str):
            raise TypeError(
                f"'model={self.model}'不是一个*.pt的PyTorch模型"
            )

    @property
    def decive(self):
        """返回模型驱动"""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

