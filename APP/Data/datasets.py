import glob
from itertools import repeat
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
import shutil
import time
from PIL import Image
import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from ultralytics.data.utils import get_hash, img2label_paths,IMG_FORMATS, load_dataset_cache_file, save_dataset_cache_file, verify_image, verify_image_label
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset, ClassificationDataset
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, PROGRESS_BAR, TQDM, yaml_save, yaml_load,NUM_THREADS, LOGGER
from ultralytics.data.augment import classify_transforms



from APP.Label.base import QInstances
from APP.Data import format_im_files,readLabelFile,writeLabelFile


def get_im_files(img_pathes):
    """获取所有图像文件路径"""
    f = []
    with open(img_pathes) as t:
        t = t.read().strip().splitlines()
        parent = str(Path(img_pathes).parent) + os.sep  # 上级路径
        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
    im_files = [str(Path(file)) for file in f]
    return im_files

def write_im_files(img_paths, im_files):
    """将图像文件路径写入txt文件"""
    parent = str(Path(img_paths).parent)
    im_files = ["." + f.replace(parent, "").replace("\\", "/") for f in im_files]
    with open(img_paths, "w") as f:
        f.writelines(im_file + "\n" for im_file in im_files)




class DetectDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_files = img2label_paths(self.im_files)

    def get_img_files(self, img_path):
        """从所有图像路径文件中获取所有图像路径"""
        try:
            f = [] #image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.suffix.split(".")[-1].lower() in IMG_FORMATS:
                    f += p
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep  #上级路径
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]   #local to global path
                else:
                    raise FileNotFoundError(f"{self.prefix}'{p}'路径不存在")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            if len(im_files) == 0:
                LOGGER.warning(f"{self.prefix}在{img_path}内未发现图像")
        except Exception as e:
            LOGGER.error(f"{self.prefix}从{img_path}错误加载数据: {str(e)}")
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        im_files = [str(Path(f)) for f in im_files]
        return im_files
    
    def cache_labels(self,path, im_files=[], progress=True):
        if not len(im_files):
            im_files = self.im_files
        label_files = img2label_paths(im_files)
        path =Path(path)
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        if progress:
            PROGRESS_BAR.start("DataLoader", "Start...", [0, total], False)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            for i, (im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg) in enumerate(results):
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                    if progress:
                        PROGRESS_BAR.setValue(i+1, f"Dataset loading...{im_file if im_file else msg}")
                        
            if progress:
                PROGRESS_BAR.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(label_files + im_files)
        x["results"] = nf, nm, ne, nc, len(im_files)
        x["msgs"] = msgs  # warnings
        if len(im_file) != len(self.im_files):
            x["version"] = DATASET_CACHE_VERSION  # add cache version
        else:
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x
    

    def getLabel(self, im_file):
        """获取图像对应的标签"""
        ind = self.im_files.index(str(Path(im_file)))
        label = self.get_image_and_label(ind)
        label["names"] = self.data["names"]
        label["cls"] = label["cls"].reshape(-1).tolist()
        return label

    def getImShapes(self):
        return dict((im_file, list(reversed(label["shape"])))  for im_file, label in zip(self.im_files, self.labels))


    def update_labels_info(self, label):
        "label : 单个图像标签"
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", None)
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        if len(segments) == 0 and bboxes is not None:  #不存在分割数据集但存在目标检测数据集，取消segments
            segments = None
        label["instances"] = QInstances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label



    def addData(self, im_files, data_path):
        """添加样本
        Args:
            im_files(str|list): 图像文件
            data_path(str): 数据图像文件的存储路径"""
        if not Path(data_path).exists():
            Path(data_path).mkdir(parents=True, exist_ok=True)
        im_files = format_im_files(im_files)
        self.im_files = [str(Path(f)) for f in self.im_files]
        label_files = img2label_paths(im_files)
        new_im_files = []
        exist_im_files = get_im_files(self.img_path)
        for im_file, label_file in zip(im_files, label_files):
            #image
            new_im_file = str(Path(data_path) / Path(im_file).name)
            shutil.move(im_file, data_path)
            self.im_files.append(new_im_file)
            #label
            new_label_file = img2label_paths(new_im_file)[0]
            if Path(label_file).exists():
                if not Path(new_label_file).parent.exists():
                    Path(new_label_file).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(label_file, Path(new_label_file).parent)  #已存在，移动标签至数据集路径
            else:
                with open(new_label_file, "w") as f:  #新建空白txt
                    pass
            self.label_files.append(str(label_file))
            new_im_files.append(new_im_file)
            if new_im_file not in exist_im_files:  #更新总图像路径
                exist_im_files.append(new_im_file)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")  # 缓存文件
        labels = self.cache_labels(cache_path, new_im_files, False)
        [labels.pop(k) for k in ("hash", "version", "msgs")]  # 去除没用的项
        self.labels += labels["labels"]
        #self.shapes += labels["shapes"]
        write_im_files(self.img_path, exist_im_files)  #重新写入总图像文件路径
        #重置参数
        self.ni = len(self.labels)
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        return new_im_files

    def removeData(self,im_files, no_label_path=""):
        """删除指定图像文件的数据集信息
        Args:
            im_files(str| list)： 图像文件路径
            no_label_path(str|Path): 未标注图像的存储路径.如果为空，将彻底删除数据集"""
        if no_label_path and not Path(no_label_path).exists():
            Path(no_label_path).mkdir(parents=True, exist_ok=True)
        im_files = format_im_files(im_files)
        self.im_files = [str(Path(f)) for f in self.im_files]
        label_files = img2label_paths(im_files)
        exist_im_files = get_im_files(self.img_path)
        for im_file, label_file in zip(im_files, label_files):
            if im_file in self.im_files:
                ind = self.im_files.index(im_file)
                self.im_files.pop(ind)
                label_file = self.label_files.pop(ind)
                self.labels.pop(ind)
                #self.shapes.pop(ind)
                if Path(im_file).exists():
                    if no_label_path and no_label_path != "":
                        shutil.move(im_file, no_label_path)  # 图像移动至未标注
                    else:
                        os.remove(im_file)  # 彻底删除
                if Path(label_file).exists():
                    os.remove(label_file)  # 标签删除
                if im_file in exist_im_files:
                    exist_im_files.remove(str(Path(im_file)))  #将图像文件路径从总文件路径删除
        write_im_files(self.img_path, exist_im_files)
        # 重置参数
        self.ni = len(self.labels)
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

    def changeData(self,im_files):
        """改变样本的标签，该标签已经在txt文件上被改变"""
        im_files = format_im_files(im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")  # 缓存文件
        labels = self.cache_labels(cache_path, im_files, False)
        [labels.pop(k) for k in ("hash", "version", "msgs")]  # 去除没用的项
        for i in range(len(im_files)):
            ind = self.im_files.index(str(Path(im_files[i])))
            self.labels[ind] = {**self.labels[ind], **labels["labels"][i]}


    def deleteClass(self, cls, no_label_path=None):
        """删除某个种类
        Args:
            cls(int):将删除的种类的索引
            no_label_path(str):被删除的种类图像将移动至未标注路径内"""
        label_files = img2label_paths(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=readLabelFile,
                iterable=label_files
            )
            for label_file, lb in zip(label_files, results):
                new_lb = []
                for b in lb:
                    if b[0] != cls:
                        if b[0] > cls:
                            b[0] -= 1
                        new_lb.append(b)
                writeLabelFile(label_file, new_lb)
        #数据集更新标签
        self.labels = self.get_labels() if self.im_files else []
        self.ni = len(self.labels)  # number of images
        if cls in self.data["names"].keys():
            self.data["names"].pop(cls)
            self.data["names"] = dict(enumerate(self.data["names"].values()))

    def addClass(self, cls_name):
        """添加种类
        Args:
            cls_name(str):新添加的种类名称"""
        self.data["names"].update({len(self.data["names"]): cls_name})
        self.data["nc"] = len(self.data["names"])
        data = yaml_load(self.data["yaml_file"])
        data["names"] = list(self.data["names"].values())
        data["nc"] = self.data["nc"]
        yaml_save(self.data["yaml_file"], data)

    def renameClass(self, cls, cls_name):
        """重命名种类名称，将重命名数据集对应种类文件夹名称
        Args:
            cls(int):种类索引
            cls_name(str):重命名的种类名称"""
        self.data["names"][cls] = cls_name
        data = yaml_load(self.data["yaml_file"])
        data["names"] = list(self.data["names"].values())
        yaml_save(self.data["yaml_file"], data)

    def build_transforms(self, hyp=None):
        """数据增强"""
        return None





class ClassifyDataset(ClassificationDataset):
    def __init__(self, root,names, args=DEFAULT_CFG, augment=False, cache=False, prefix=""):
        self.prefix = f"{prefix}: " if prefix else ""
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.args = args
        if names:
            super().__init__(root, args)  # 通过所给分类数据集路径，获取图像，图像路径，标签，分类类别名称等数据
        else:
            self.root = root
            self.samples = []
            self.torch_transforms = classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        self.labels = {}
        self.im_files = [str(Path(x[0])) for x in self.samples]
        self.names = names
    
    def verify_images(self, im_cls, progress=True):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            if progress:
                PROGRESS_BAR.start("Classify dataset Load", "Start", [0,len(im_cls)], False)
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(enumerate(results), desc=desc, total=len(im_cls))
                for i, sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                    if progress:
                        PROGRESS_BAR.setValue(i+1, f"数据集加载中...{sample[0]}, {sample[1]}")
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            if progress:
                PROGRESS_BAR.close()
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples

    def addData(self, im_files, classes):
        """
        添加样本
        Args:
            im_files(list|str): 旧图像文件路径
            classes(list|str):图像对应种类
        """
        im_files = format_im_files(im_files)
        classes = classes if isinstance(classes, list) else [classes]
        im_cls = []
        new_im_files = []
        for im_file, cls in zip(im_files, classes):
            new_im_file = Path(self.root) / self.names[cls] / Path(im_file).name
            if not new_im_file.parent.exists():
                new_im_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(im_file, new_im_file.parent)   #移动图像文件到数据集对应种类文件内
            im_cls.append((str(new_im_file), cls))
            new_im_files.append(str(new_im_file))
        #验证图像
        samples, shapes = self.verify_images(im_cls,False)
        for sample in samples:
            self.samples.append([sample[0], sample[1], Path(sample[0]).with_suffix(".npy"), None])
        self.im_files = [str(Path(x[0])) for x in self.samples]
        self.shapes += shapes
        return new_im_files

    def changeData(self, im_files, classes):
        """将图像从原种类转换至种类classes"""
        im_files = format_im_files(im_files)
        classes = classes if isinstance(classes, list) else [classes]
        new_im_files = []
        for im_file, cls in zip(im_files, classes):
            new_im_file = Path(self.root) / self.names[cls] / Path(im_file).name
            new_im_files.append(new_im_file)
            if Path(im_file).parent.name != self.names[cls]:
                ind = self.im_files.index(im_file)
                if not new_im_file.parent.exists():
                    new_im_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(im_file, new_im_file.parent)    #移动图像至新种类文件夹
                self.samples[ind][1] = cls   #改变样本种类  file, cls, cache, img
        return new_im_files

    def removeData(self, im_files, no_label_path=""):
        """删除指定图像文件样本信息
        Args:
            im_files(str| list)： 图像文件路径
            no_label_path(str|Path): 未标注图像的存储文件夹，如果为空，将彻底删除样本"""
        if no_label_path and not Path(no_label_path).exists():
            Path(no_label_path).mkdir(parents=True, exist_ok=True)
        im_files = format_im_files(im_files)
        for im_file in im_files:
            if im_file in self.im_files:
                if Path(im_file).exists():
                    if no_label_path and no_label_path != "":
                        shutil.move(im_file, no_label_path)  # 移动至未标注
                    else:
                        os.remove(im_file)
                ind = self.im_files.index(im_file)
                self.im_files.pop(ind)
                self.samples.pop(ind)
                self.shapes.pop(ind)

    def getLabel(self, im_file):
        """获取图像文件对应的标签信息
        Args:
            im_file(str|Path):图像文件路径 """
        im_file = str(Path(im_file))
        ind = self.im_files.index(im_file)
        item = self.__getitem__(ind)
        label = {}
        label["img"] = item["img"]
        label["im_file"] = item["im_file"]
        label["cls"] = item["cls"]
        label["names"] = self.names
        return label

    def deleteClass(self, cls, no_label_path):
        """删除某个种类
        Args:
            cls(int):将删除的种类的索引
            no_label_path(str):被删除的种类图像将移动至未标注路径内"""
        # 移动删除种类的图像文件到未标注文件夹
        if no_label_path and not Path(no_label_path).exists():
            Path(no_label_path).mkdir(parents=True, exist_ok=True)
        cls_name = self.names[cls]
        train_files = glob.glob(str(Path(self.root) / cls_name / "**"), recursive=True)
        for f in train_files:
            shutil.move(f, no_label_path)
        os.rmdir(Path(self.root) / cls_name)
        self.names.pop(cls)
        names = dict(enumerate(sorted(self.names.values())))
        self.__init__(self.root, names, self.args)

    def addClass(self, cls_name):
        """添加种类
        Args:
            cls_name(str):新添加的种类名称"""
        (Path(self.root) / cls_name).mkdir(parents=True, exist_ok=True)
        self.names[len(self.names)] = cls_name

    def renameClass(self, cls, cls_name):
        """重命名种类名称，将重命名数据集对应种类文件夹名称
        Args:
            cls(int):种类索引
            cls_name(str):重命名的种类名称"""
        os.rename(Path(self.root) / self.names[cls], Path(self.root) / cls_name)
        self.names[cls] = cls_name