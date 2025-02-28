import platform
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
from multiprocessing.pool import ThreadPool

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, colorstr, ops, PROGRESS_BAR, NUM_THREADS
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs"""

class BasePredictor:
    """
    Attributes:
        args(SimpleNamespace)： 配置参数
        save_dir(Path): 保存结果目录
        done_warmup(bool): 预热是否已经完成
        model(nn.Midule): 预测模型
        data(dict): 数据集配置参数
        device(torch.device): 预测驱动
        dataset(Dataset):预测数据集
        vid_path(str): 视频文件路径
        vid_writer(cv2.VideoWriter): 保存视频输出的video Writer
        data_path(str): 数据路径
        """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25
        self.done_warmup = False

        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.txt_path = None
        self._lock = threading.Lock()    #自动线程安全推理

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """在图像或者流上进行推理"""
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)  #返回一个迭代器
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  #将结果混合为一个列表

    def preprocess(self, im):
        """准备推理用输入图像"""
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))   #使图像大小适应模型输入
            im = im[..., ::-1].transpose((0,3,1,2))  #BGR->RGB. BHWC->BCHW (b, 1/3, h, w)
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)
        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im /= 255
        return im

    def postprocess(self, preds, img, orig_imgs):
        """处理推理结果，获取准确目标"""
        return preds

    def pre_transform(self, im):
        """适应图像大小，以缩放填充的方式"""
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def inference(self, im, *args, **kwargs):
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True) if self.args.visualize and (not self. source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """返回的是一个迭代器"""
        if self.args.verbose:
            LOGGER.info("")

        #启动模型
        if not self.model:
            self.setup_model(model)

        with self._lock: #线程安全推理
            try:
                self.setup_source(source if source is not None else self.args.source)
                if self.args.save or self.args.save_txt:
                    (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

                #Warmup model
                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))  #color or gray
                    self.done_warmup = True
                self.seen, self.windows, self.batch = 0, [], None
                profilers = (
                    ops.Profile(device=self.device),
                    ops.Profile(device=self.device),
                    ops.Profile(device=self.device)
                )
                PROGRESS_BAR.show("预测中")
                PROGRESS_BAR.start(0, len(self.dataset), True)
                for batch in self.dataset: #sources, images. videocaptrue, s
                    self.batch = batch
                    path, im0s, vid_cap, s = batch

                    with profilers[0]:
                        im = self.preprocess(im0s)  #缩放归一化转Tensor

                    #推理
                    with profilers[1]:
                        preds = self.inference(im, *args, **kwargs)
                        if self.args.embed:
                            yield from [preds] if isinstance(preds, torch.Tensor) else preds  #返回每个推理直接结果
                            continue

                    #后处理
                    with profilers[2]:
                        self.results = self.postprocess(preds, im, im0s)

                    #可视化，保存
                    n = len(im0s)
                    for i in range(n):
                        self.seen += 1
                        self.results[i] = self.results[i].cpu()
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 /n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }    #批次内平均速度
                        p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                        p = Path(p)

                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s += self.write_results(i, self.results, (p,im,im0))
                        if self.args.save or self.args.save_txt:
                            self.results[i].save_dir = self.save_dir.__str__()
                        if self.args.show and self.plotted_img is not None:
                            self.show(p)
                        if self.args.save and self.plotted_img is not None:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))
                    torch.cuda.empty_cache()
                    yield from self.results
                    PROGRESS_BAR.setValue(self.seen, f"{s} {profilers[1].dt *1e3:.1f}ms")
                    if PROGRESS_BAR.isStop():
                        PROGRESS_BAR.close()
                        raise ProcessLookupError("中断：预测中断成功")
            except Exception as ex:
                PROGRESS_BAR.stop()
                PROGRESS_BAR.close()
                raise ProcessLookupError(f"预测失败：{ex}")
                    
        #Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()

        #Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers) #所有图像的平均速度
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape"
                f"{(1,3,*im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  #标签文件数量
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"预测结果保存至{colorstr('bold', self.save_dir)}{s}")


    def setup_model(self, model, verbose=True):
        self.model = AutoBackend(
            model or self.args.model,
            device=select_device(self.args.device,verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=verbose
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()

    def setup_source(self, source):
        """设置推理源"""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        self.transforms = (
            getattr(self.model.model, "transforms",
                    classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction))
        ) if self.args.task == "classify"  else None
        #加载推理源
        self.dataset = load_inference_source(source=source, vid_stride=self.args.vid_stride, buffer=self.args.stream_buffer)
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and(
                self.dataset.mode == "stream"  #视频流 - 实时
                or len(self.dataset)>1000     #源数量/图像数量
                or any(getattr(self.dataset, "video_flag", [False]))  #视频
             ):  #视频
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] *self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs


    def write_results(self, idx, results, batch):
        p, im, _ = batch
        log_string = ""
        if len(im.shape) == 3:   #(c,h,w)
            im = im[None]   #(1,c,h,w)
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # bs >=1
            log_string += f"{idx}: "
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, "frame", 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / "labels" / p.stem) + ("" if self.dataset.mode == "image" else f"_{frame}")
        log_string += "h*w:%gx%g " % im.shape[2:]
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:   #Add bbox to image
            plot_args = {
                "line_width": self.args.line_width,
                "boxes": self.args.show_boxes,
                "conf": self.args.show_conf,
                "labels": self.args.show_labels,
            }
            if not self.args.retina_masks:
                plot_args["im_gpu"] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        #Write
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(
                save_dir = self.save_dir / "crops",
                file_name = self.data_path.stem + ("" if self.dataset.mode == "image" else f"_{frame}"),
            )
        return log_string


    def show(self, p):
        pass

    def save_preds(self, vid_cap, idx, save_path):
        """将预测的视频/流保存到mp4文件中"""
        im0 = self.plotted_img
        #Save imgs
        if self.dataset.mode == "image":
            cv2.imwrite(save_path, im0)  #图像直接保存
        else: #视频/流
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if self.vid_path[idx] != save_path:  #新视频
                self.vid_path[idx] = save_path
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                    self.vid_frame[idx] = 0
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  #释放当前视频写入器
                if vid_cap:  #cv2.Videocapture
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # 要求整数，mp4不允许浮点数的帧率
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else: #stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi","MJPG")
                self.vid_writer[idx] = cv2.VideoWriter(
                    str(Path(save_path).with_suffix(suffix)), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            #write_video
            self.vid_writer[idx].write(im0)

            #write frame  #保存每一帧
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{self.vid_frame[idx]}.jpg", im0)
                self.vid_frame[idx] += 1

    def predict_cli(self, source=None, model=None):
        """空跑"""
        gen = self.stream_inference(source, model)
        for _ in gen:
            pass