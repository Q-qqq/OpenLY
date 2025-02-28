import torch.nn as nn
import torch
import ast
import contextlib
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import  Path
import cv2
import numpy as np
from PIL import  Image

from ultralytics.utils import LINUX, LOGGER, ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml
from ultralytics.utils.downloads import attempt_download_asset, is_url


def check_class_names(names):
    """检查种类名称是否格式正确1:class1,2:class2..."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}种类数据集要求种类索引 0-{n-1}，但现在存在无效的种类索引,"
                f"在数据集YAML参数文件中定义了种类索引{min(names.keys())}-{max((names.keys()))}"
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):
            names_map = yaml_load(ROOT/"cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """如果data.yaml文件内有names，则加载，否则返回默认种类名称"""
    if data:
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data))["names"]
    return {i: f"class{i}" for i in range(999)}


class AutoBackend(nn.Module):
    """为运行YOLO模型推理动态选择backend
    支持的模型格式有：
            | Format                | File Suffix      |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | TorchScript           | *.torchscript    |
            | ONNX Runtime          | *.onnx           |
            | ONNX OpenCV DNN       | *.onnx (dnn=True)|
            | OpenVINO              | *openvino_model/ |
            | CoreML                | *.mlpackage      |
            | TensorRT              | *.engine         |
            | TensorFlow SavedModel | *_saved_model    |
            | TensorFlow GraphDef   | *.pb             |
            | TensorFlow Lite       | *.tflite         |
            | TensorFlow Edge TPU   | *_edgetpu.tflite |
            | PaddlePaddle          | *_paddle_model   |
            | ncnn                  | *_ncnn_model     |
    """

    @torch.no_grad()
    def __init__(self,
                 weights="yolov8n.pt",
                 device=torch.device("cpu"),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True,
                 verbose=True):
        """
        Args:
            weights(str): 模型权重路径， 默认‘yolov8n.pt’
            device(torch.device): 运行模型的驱动，默认CPU
            dnn(bool):使用OpenCV DNN模型推理ONNX，默认False
            data(str | Path |optional): data.yaml文件的路径，其内包含种类名等参数
            fp16(bool): 使能半精度推理，默认False
            fuse(bool): 混合Conv2D+BatchNorm， 默认True
            verbose(bool):使能信息显示，默认True
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            ncnn,
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton #FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  #BHWC formats
        stride = 32
        model, metadata = None, None

        #Set device
        cuda = torch.cuda.is_available() and device.type != "cpu" #use CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx]):
            device = torch.device("cpu")
            cuda = False

        #Download if not loacal
        if not (pt or triton or nn_module):
            w = attempt_download_asset(w)   #本地未发现则从网上下载

        #load model
        if nn_module:  #in-memory PyTorch model
            model = weights.to(device)
            model = model.fuse(verbose=verbose) if fuse else model
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape
            stride = max(int(model.stride.max()), 32)  #model stride
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            self.model = model
            pt = True
        elif pt:    #PyTorch
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape
            stride = max(int(model.stride.max()), 32)
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            self.model = model
        elif jit: #TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  #模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  #加载元数据字典
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))
        elif dnn:  #ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:
            LOGGER.info(f"Loading {w} for ONNX Euntime inference..")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = {"CUDAExecutionProvider", "CPUExecutionProvider"} if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map  #模型输入输出元数据

        elif xml:  #OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")
            from openvino.runtime import Core,Layout, get_batch #noqa

            core = Core()
            w = Path(w)
            if not w.is_file():
                w = next(w.glob("*.xml"))  #获取w路径下的xml文件
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")
            metadata = w.parent/"metadata.yaml"
            '''
        elif engine:   #TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt
            check_version(trt.__version__, "7.0.0", hard=True)
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.logger(trt.Logger.INFO)
            #Read file
            with open(w, "rb") as f, trt.Runrime(logger) as runtime:
                meta_len = int.from_bytes(f.read(4), byteorder="little") #读取元数据
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_excution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
                elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"
        elif coreml:      #CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct
            model = ct.models.MLModel(w)
            metadata = dict(model.user_defined_metadata)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            from ultralytics.engine.exporter import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wrap frozen graphs for deployment."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # Load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
        elif tfjs:  # TF.js
            raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi  # noqa

            w = Path(w)
            if not w.is_file():  # if not *.pdmodel
                w = next(w.rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w.parents[1] / "metadata.yaml"
        elif ncnn:  # ncnn
            LOGGER.info(f"Loading {w} for ncnn inference...")
            check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # requires ncnn
            import ncnn as pyncnn

            net = pyncnn.Net()
            net.opt.use_vulkan_compute = cuda
            w = Path(w)
            if not w.is_file():  # if not *.param
                w = next(w.glob("*.param"))  # get *.param file from *_ncnn_model dir
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"
        elif triton:  # NVIDIA Triton Inference Server
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
        
        '''
        else:
            raise TypeError(
                f"model='{w}' is not a supported model format. "
            )

        #加载额外的元数据YAML
        if isinstance(metadata,(str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata:
            for k, v in metadata.items():
                if k in ("stride", "batch"):
                    metadata[k] = int(v)
                elif k in ("imgsz", "names", "kpt_shape") and isinstance(v ,str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING ⚠️ 没有找到'model={weights}的元数据'")

        #Check names
        if "names" not in locals():   #本地变量
            names = default_class_names(data)
        names = check_class_names(names)

        #取消梯度回传
        if pt:
            for p in model.parameters():
                p.requires_grad = False
        self.__dict__.update(locals())

    def forward(self, im, augment=False, visualize=False, embed=None):
        b, ch, h, w = im.shape
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  #BCHW to BHWC (1,320,192,3)
        if self.pt or self.nn_module:  #PyTorch
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)
        elif self.jit:  #TorchScript
            y = self.model(im)
        elif self.dnn: #ONNX OpenCV DNN
            im = im.cpu().numpy()
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  #ONNX Runtime
            im = im.cpu().numpy()
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  #OpenVINO
            im = im.cpu().numpy()
            y = list(self.ov_compiled_model(im).values())
            """
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized
            if "confidence" in y:
                raise TypeError(
                    "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                    f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                )
                # TODO: CoreML NMS inference handling
                # from ultralytics.utils.ops import xywh2xyxy
                # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            elif len(y) == 1:  # classification model
                y = list(y.values())
            elif len(y) == 2:  # segmentation model
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.ncnn:  # ncnn
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            ex = self.net.create_extractor()
            input_names, output_names = self.net.input_names(), self.net.output_names()
            ex.input(input_names[0], mat_in)
            y = []
            for output_name in output_names:
                mat_out = self.pyncnn.Mat()
                ex.extract(output_name, mat_out)
                y.append(np.array(mat_out)[None])
        elif self.triton:  # NVIDIA Triton Inference Server
            im = im.cpu().numpy()  # torch to numpy
            y = self.model(im)
            
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
                if len(y) == 2 and len(self.names) == 999:  # segments and names not defined
                    ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # index of protos, boxes
                    nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                    self.names = {i: f"class{i}" for i in range(nc)}
            else:  # Lite or Edge TPU
                details = self.input_details[0]
                integer = details["dtype"] in (np.int8, np.int16)  # is TFLite quantized int8 or int16 model
                if integer:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])  # de-scale
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if integer:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    if x.ndim > 2:  # if task is not classification
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                        x[:, [0, 2]] *= w
                        x[:, [1, 3]] *= h
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            """

        if isinstance(y , (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)


    def from_numpy(self, x):
        """numpy to torch"""
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """预热运行"""
        warmip_types = self.pt, self.jit,self.onnx,self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmip_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device) #input
            for _ in range(2 if self.jit else 1):
                self.forward(im)    #warmup



    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """判断模型类型"""
        from ultralytics.engine.exporter import export_formats

        sf = list(export_formats().Suffix)  #export suffixes
        if not is_url(p, check=False) and not isinstance(p, str):
            check_suffix(p, sf)  #check
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")  #保留对旧苹果系统CoreML *mlmodel的支持
        types[8] &= not types[9]   #8和9后缀有重合
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = url.netloc and url.path and url.scheme in {"http", "grpc"}
        return types + [triton]