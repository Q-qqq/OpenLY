import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse
import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, ops, is_colab, is_kaggle, cv2_readimg, PROGRESS_BAR, NUM_THREADS
from ultralytics.utils.checks import check_requirements

@dataclass
class SourceTypes:
    """用于预测的输入源的种类"""
    webcam: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


def get_best_youtube_url(url, use_pafy=True):
    """获取youtube视频最高质量的下载url"""
    if use_pafy:
        check_requirements(("pafy", "youtube_dl>=2020.12.2"))
        import pafy
        return  pafy.new(url).getbestvideo(preftype="mp4").url
    else:
        check_requirements("yt_dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get("formats", [])):
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")



class LoadStreams:
    """
    Attributes:
        sources(str): 源输入路径或者URLs
        vid_stride(int): 视频流帧率步长，默认1
        buffer(bool): 是否缓存输入流，默认False
        running(bool): 流线程是否正在运行
        mode(str):设置为’stream‘表示实时捕获
        imgs(list): 每一个流的图像帧列表
        fps(list): 每一个流的FPS
        frames(List):每一个流的全部帧
        threds(list):每一个流的线程
        shape(List):每一个流的shape
        caps(List):每一个流的cv2.CidioCapture对象
        bs(int): 处理的批大小
    """

    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        torch.backends.cudnn.benchmark = True   #使固定大小输入的推理更快
        self.buffer = buffer
        self.running = True
        self.mode = "stream"
        self.vid_stride = vid_stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n
        self.imgs = [[] for _ in range(n)]
        self.shape = [[] for _ in range(n)]
        self.sources = [ops.clean_str(x) for x in sources]
        for i, s in enumerate(sources):
            st = f"{i + 1}/{n}: {s}..."
            if urlparse(s).hostname in ("www.youtube.com", "youtube.com", "youtu.be"):  #youtube视频
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s
            if s == 0 and (is_colab() or is_kaggle()):
                raise NotImplementedError(
                    "'source=0'不支持在Colab和Kaggle notebooks运行，请在本地环境尝试运行"
                )
            self.caps[i] = cv2.VideoCapture(s)  #store Video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}打开失败 {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGH))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")  #帧数
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  #30FPS
            success, im = self.caps[i].read()  #第一帧
            if not success or im is None:
                raise ConnectionError(f"{st}从{s}读取图像失败")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")

        self.bs = self.__len__()


    def update(self, i, cap, stream):
        """读取流的第i帧"""
        n, f = 0, self.frames[i]  #frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:   #保持图像缓存数量少于30
                n += 1
                cap.grab()  #开始取流
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()   #获取一帧
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("WARNING ⚠️ 视频流没反应，请检测相机连接")
                        cap.open(stream)  #重新打开
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)   #等待清空缓存

    def close(self):
        self.running = False #停止
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  #超时即销毁
        for cap in self.caps:
            try:
                cap.release()
            except Exception as e:
                LOGGER.warning(f"释放VideoCapture失败{e}")
        cv2.destroyAllWindows()

    def __iter__(self):
        """重启"""
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):   #循环获取各个流的帧
            while not x:  #无限循环等待子线程获取帧
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  #q退出
                    self.close()
                    raise StopIteration
                time.sleep(1/ min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"WARNING ⚠️ 流-{i}：等待中")
            if self.buffer:
                images.append(x.pop(0)) #缓存图像丢出首帧，每次如此
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()
        return self.sources, images, None, ""

    def __len__(self):
        return  len(self.sources)

class LoadScreenshots:
    """屏幕截图
    Attributes:
        source(str): 指示截取的屏幕
        screen(int): 屏幕数量
        left(int):屏幕截图的左上角点x
        right(int): 屏幕截图左上角点y
        width(int): 屏幕截图宽度
        height(int): 屏幕截图高度
        mode(str): 设置'stream'，表示实时捕获
        frame(int): 捕获帧数
        sct(mss.mss): 来自mss库的屏幕截图对象
        bs(int): 批大小，默认1
        monitor：监视配置细节"""

    def __init__(self, source):
        """source = [source, screen_number, left, top, width, height](单位：pixels)"""
        check_requirements("mss")
        import  mss
        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None #默认全屏
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "Stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        #解析监视器shape
        monitor = self.sct.monitor[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}



    def __iter__(self):
        return self

    def __next__(self):
        im0 = np.asarray(self.sct.grab(self.monitor))[:,:,:3]  #BGRA -> BGR
        s = f"screen {self.screen} (LTWG): {self.left}, {self.top}, {self.width}, {self.height}:"
        self.frame += 1
        return [str(self.screen)], [im0], None, s

class LoadImages:
    """
    加载图像或者视频
    Attributes:
        files(List): 图像或者视频文件的路径列表
        nf(int): 文件数量
        video_flag(list): 文件是否一个视频文件
        bs(int): 批大小
        cap(cv2.Videocapture): OpenCV的视频捕获对象
        frame(int): 视频帧计数器
        frames(int): 视频全部帧数
        count(int):迭代器计数器
        """
    def __init__(self, path, vid_stride=1):
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":   #路径文本文件
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  #图像/视频文件路径

        files = []
        for p in sorted(path) if isinstance(path, (list,tuple)) else [path]:
            a = str(Path(p).resolve() if parent is None else parent / Path(p).resolve())
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))
            elif os.path.isfile(a):
                files.append(a)
            else:
                raise FileNotFoundError(f"文件{p}不存在")
        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  #文件数量
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.vid_stride = vid_stride  #视频帧率步长
        self.bs = 1
        if any(videos):
            self._new_video(videos[0])  #第一个视频
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(
                f"在{p}内未发现图像或者视频"
            )
    def _new_video(self, path):
        """创建一个新的video capture对象"""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            #read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()  #取第frame+vid_stride帧
            success, im0 = self.cap.retrieve()
            while not success:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                success, im0 = self.cap.read()
            self.frame += 1
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}:"

        else:
            self.count += 1
            im0 = cv2_readimg(path)  #BGR
            if im0 is None:
                raise FileNotFoundError(f"未找到图像{path}")
            s = f"image {self.count} / {self.nf} {path}"
        return [path], [im0], self.cap, s

    def __len__(self):
        return self.nf


class LoadPilAndNumpy:
    """
        为了批处理从PIL和Numpy arrays加载图像
    Attributes:
        paths(list): 图像路径的列表，可自动生成
        im0(list): 存储图像的列表，Numpy arrays
        mode(str): 被处理的数据类型，默认"image"
        bs(int): 批大小，等于im0的长度
        count(int): 迭代器的计数器，当count为1时，退出迭代，只迭代一次，返回一个batch
    """

    def __init__(self, im0):
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]
        self.im0 = self.checkImgs(im0)
        self.mode = "image"
        self.bs = len(self.im0)

    def checkImgs(self, ims):
        imgs = []
        PROGRESS_BAR.show("预测图像集加载", "开始加载")
        PROGRESS_BAR.start(0, len(ims), True)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=self._single_check,
                                iterable=ims)
            for i, im in enumerate(results):
                imgs.append(im)
                PROGRESS_BAR.setValue(i+1, self.paths[i])
                if PROGRESS_BAR.isStop():
                    PROGRESS_BAR.close()
                    raise ProcessLookupError("中断：预测图像集加载中断成功")
        PROGRESS_BAR.close()
        return imgs



    @staticmethod
    def _single_check(im):
        assert isinstance(im, (Image.Image, np.ndarray)), f"只接收PIL/np.ndarray图像类型，但type(im)={type(im)}"
        if isinstance(im, Image.Image):
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)
        return im

    def __len__(self):
        return len(self.im0)

    def __next__(self):
        if self.count == len(self.im0):
            raise StopIteration  #只迭代一次，返回所有数据
        self.count += 1
        return [self.paths[self.count-1]], [self.im0[self.count-1]], None, ""

    def __iter__(self):
        self.count = 0
        return self

class LoadTensor:
    """
    从torch.Tensor加载图像-预处理 检测图像shape，归一化
    Attributes：
        im0(torch.Tensor): 输出Tensor，包含图像
        bs(int): 批大小
        mode(str): 运行模式。默认'image'
        paths(list): 图像路径或者文件名称列表
        count(int)：迭代器的计数器
        """
    def __init__(self, im0):
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]


    @staticmethod
    def _single_check(im, stride=32):
        s = (
            f"WARNING ⚠️ 输入torch.Tensor的shape应该是BCHW，例如shape(1,3,640,640)"
            f"且应该被{stride}整除，但现在的输入shap{tuple(im.shape)}并不满足要求"
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:
            LOGGER.warning(f"WARNING ⚠️ 输入torch.Tensor应该归一化至0.0-1.0，但最大值却为{im.max()},将为其归一化：/255")
            im = im.float() / 255.0
        return im

    def __iter__(self):
        self.count = 0

    def __next__(self):
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ""

    def __len__(self):
        return self.bs

LOADERS = LoadStreams, LoadPilAndNumpy, LoadImages, LoadScreenshots

def autocast_list(source):
    """将不同的源混合进一个列表内"""
    files = []
    for im in source:
        if isinstance(im, (str, Path)): #file or url
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im))
        elif isinstance(im, (Image.Image, np.ndarray)):
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} 是一个不支持的预测源类型"
            )
    return files