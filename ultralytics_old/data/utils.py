import os
import hashlib
import subprocess
import zipfile
from tarfile import is_tarfile
from pathlib import Path
from PIL import Image, ImageOps
import threading
from ultralytics.utils import LOGGER,DATASETS_DIR,ROOT,colorstr,emojis, yaml_load,clean_url,SETTINGS_YAML
from ultralytics.utils.ops import *
from ultralytics.utils.downloads import safe_download,download
from ultralytics.utils.checks import check_file, check_class_names, check_font, is_ascii

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  #global pin_memory for dataloaders

FILE_LOCK = threading.Lock()


def img2label_paths(img_paths):
    """将图像路径转为标签路径"""
    if not isinstance(img_paths, list):
        img_paths = [img_paths]
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(str(Path(x)).rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def get_hash(paths):
    """返回一个属于路径列表的hash值，检验文件一致性"""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()

def exif_size(img: Image.Image):
    s = img.size  #wh
    if img.format == "JPEG":
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274,None)
                if rotation in [6,8]:
                    s = s[1],s[0]
    return s

def verify_image(args):
    """验证单个图像"""
    (im_file, cls), prefix = args
    nf, nc, msg, shape = 0, 0, "", (0,0)
    try:
        im = Image.open(im_file)
        im.verify()  #PIL验证
        shape = exif_size(im)  #image size
        shape  = (shape[1], shape[0])  #h, w
        assert (shape[0]> 9) &(shape[1] > 9), f"图像大小{shape}小于10个像素"
        assert im.format.lower() in IMG_FORMATS, f"无效的图像格式{im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2,2)
                if f.read() != b"\xff\xd9":   #损坏的JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}:将损坏的JPEG重新存储"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ 忽略损坏的图像文件{im_file}:{e}"
    return [im_file, cls], nf, nc, msg, shape




def verify_image_label(args):
    "验证并读取图像和标签"
    with FILE_LOCK:
        im_file, lb_file, keypoint, num_cls, nkpt, ndim = args
        nm,nf,ne,nc,npc,msg,segments,keypoints = 0,0,0,0,np.array([0] * num_cls),"",[],None
        try:
            #验证图像
            im = Image.open(im_file)
            im.verify()    #检验图像文件完整性
            shape = exif_size(im)
            shape = (shape[1],shape[0])  #hw
            assert (shape[0] > 9) &(shape[1] > 9), f"图像大小{shape} 需要大于 10 像素"
            assert im.format.lower() in IMG_FORMATS,f"无效的图像格式 {im.format}"
            if im.format.lower() in ("jpg","jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2,2)       #读取指针倒数2
                    if f.read() != b"\xff\xd9":   #corrupt JPEG   已损坏
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file,"JPEG",subsampling=0, quality=100)  #修复并保存图像文件
                        msg += f"警告⚠️ {im_file}: 损坏的JPEG文件已修复保存"
            #验证标签
            if os.path.isfile(lb_file):
                nf = 1
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb) and (not keypoint):  #分割
                        #分割标签转目标检测框
                        classes = np.array([x[0] for x in lb],dtype=np.float32)
                        segments = [np.array(x[1:],dtype=np.float32).reshape(-1,2) for x in lb]  # 分割标签格式（cls, x1,y1,x2,y2...）
                        lb = np.concatenate((classes.reshape(-1,1),segments2boxes(segments)),1)  #(cls x y x y)
                    lb = np.array(lb,dtype=np.float32)
                nl = len(lb)
                if nl:
                    if keypoint:
                        assert lb.shape[1] == (5 + nkpt *ndim), f"labels require {(5+nkpt*ndim)} columns each"
                        points = lb[:,5:].reshape(-1,ndim)[:,:2]
                    else:
                        assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                        points = lb[:,1:]
                    assert points.max() <= 1, f"non-normalized or out of bound coordinates {points[points > 1]}"
                    assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                    #All label
                    #验证种类数量
                    max_cls = lb[:,0].max() #max label cls count
                    assert max_cls < num_cls, (f"Label class {int(max_cls)} exceeds dataset class count {num_cls}."
                                                f"Possible class labels are 0 - {num_cls - 1}")
                    #检测重复行
                    _,i = np.unique(lb,axis=0, return_index=True)
                    if len(i) < nl:
                        lb = lb[i]  #去除重复
                        if segments:
                            segments = [segments[x] for x in i]
                        msg += f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
                else:
                    ne = 1 #empty label
                    lb = np.zeros((0,(5+nkpt*ndim) if keypoint else 5), dtype=np.float32)
            else:
                nm = 1 #missing label
                lb = np.zeros((0,(5+nkpt*ndim) if keypoint else 5), dtype=np.float32)
            if keypoint:
                keypoints = lb[:,5:].reshape(-1,nkpt,ndim)   #nl, nkpt, ndim
                if ndim == 2:
                    kpt_mask = np.where((keypoints[...,0] < 0) | (keypoints[..., 1] < 0),0.0, 1.0).astype(np.float32)   #大于0的kptpoints
                    keypoints = np.concatenate([keypoints,kpt_mask[..., None]], axis=-1)  #(nl,nkpt,3) 3-> ndim+1
            lb = lb[:,:5]
            #种类特征数量
            for c in lb[:,0]:
                npc[int(c)] += 1
            return im_file, lb, shape, segments, keypoints, nm,nf,ne,nc,npc,msg
        except Exception as e:
            nc = 1
            msg += f"WARNING⚠️ {im_file}: ignoring corrupt image/label: {e}"
            return [None,None,None,None,None,nm,nf,ne,nc,npc,msg]

def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    '''
    将一个多边形坐标列表转换为一个指定图像大小的二值掩膜
    :param imgsz(tuple): 新图像大小
    :param polygons（List[np.ndarray]）: 多边形列表
    :param color(int): 填充多边形的颜色
    :param downsample_ratio(int):  下采样的比值
    :return: （np.ndarray）一张imgsz大小的掩膜图像
    '''
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    return cv2.resize(mask, (nw, nh))

def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    '''
    将n个多边形segments标签转换为对应的n个mask
    :param imgsz(tuple):图像大小
    :param polygons(List[np.ndarray)): segments 分割数据集标签
    :param color(int): 多边形填充颜色-像素值
    :param downsample_ratio（int）: 下采样比值
    :return(np.ndarray): n个mask的集合masks
    '''
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])

def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    '''
    将segments的多边形所转换的掩膜进行重叠
    :param imgsz: 图像大小
    :param segments: 分割数据集
    :param downsample_ratio: 下采样比值
    :return: （masks，index），masks：重叠起来的掩膜，index：按segments各多边形面积排序的索引
    '''
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8     #masks的数据格式由segments的数量决定
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)  #按面积排序  从大到小
    ms = np.array(ms)[index]
    for i in range (len(segments)):
        mask = ms[i] * (i+1)  #对掩膜的多边形填充赋值i+1
        masks = masks + mask  #将所有掩膜重叠起来+
        masks = np.clip(masks, a_min=0, a_max=i + 1)      #最后加显示在最顶上
    return masks, index

class Format:
    def __init__(self,
                 bbox_format="xywh",
                 normalize=True,
                 return_mask=False,
                 return_keypoint=False,
                 return_obb=False,
                 mask_ratio=4,
                 mask_overlap=True,
                 batch_idx=True):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask   #只有训练目标检测检测时才设置False
        self.return_keyppoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx   #保持批索引
    def __call__(self, labels):
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)  #目标检测框bbox数量

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio)
            labels["masks"] = masks
        if self.normalize:
            instances.normalize(w, h)
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keyppoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
        if self.return_obb:
            labels["bboxes"] = (xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0,5)))

        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)   #bbox的数量
        return labels



    def _format_segments(self, instances, cls, w, h):
        segments = instances.segments
        if self.mask_overlap:     #重叠起来的mask
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]   #(640, 640) -> (1,640,640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)
        return masks, instances, cls

    def _format_img(self, img):
        '''转换图像格式从Numpy array 到 PyTorch tensor 去适应YOLO'''
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)     #在最后面增加一个维度（w,h）-> (h,w,1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])  #(d,h,w)
        img = torch.from_numpy(img.copy())
        return img

def check_cls_dataset(dataset, split=""):
    """
    检测分类数据集种类数量和图片数量
    接收一个数据集路径名称，如果本地没有，则尝试从网络下载到本地
    Args:
        dataset (str | Path): 数据集的路径名称
        split (str, optioinl): 数据集名称的分割符号，可以是'val', 'test' 或者 'train'， 默认'val'
    Returns:
        (dict):
            'train'(Path): 训练集路径
            'val'(Path): 验证集路径
            'test'(Path): 测试集路径
            'nc'(int): 种类数量
            'names'(dict): 种类名称
    """

    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        LOGGER.warning(f"\nWARNING ⚠️ 未找到数据，丢失路径{data_dir}, 尝试下载...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f"https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        LOGGER.info(f"数据集下载成功 ✅ ({time.time() - t:.1f}s), 保存到 {colorstr('bold', data_dir)}\n")
    train_set = data_dir/"train"  #训练集路径
    val_set = (
        data_dir / "val" if (data_dir / "val").exists() else
        data_dir / "validation" if (data_dir / "validation").exists() else
        None
    )  #验证集路径
    test_set = data_dir / "test" if (data_dir / "test").exists() else None   #测试集路径
    if split == "val" and not val_set:
        LOGGER.warning("WARNING ⚠️ 数据集'split=val'未找到，使用'split=test'代替")
    elif split == "test" and not test_set:
        LOGGER.warning("WARNING ⚠️ 数据集'split=test'未找到，使用'split=val'代替")

    nc = len( [x for x in (data_dir / "train").glob("*") if x.is_dir()])  #种类数量 训练集目录下文件夹数量
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  #种类名称 训练集目录下文件夹名称
    names = dict(enumerate(sorted(names)))

    for k, v in{"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f'{colorstr(f"{k}:")} {v}...'
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS] #数据集路径和其子文件夹下的图像文件
            nf = len(files)  #找到的文件数量
            nd = len({file.parent for file in files})  #数据集下的子文件夹数量
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' 未找到训练图像❌"))
                else:
                    LOGGER.warning(f"{prefix} WARNING ⚠️ 未找到图像")
            elif nd != nc:
                LOGGER.warning(f"{prefix} 在{nd}个种类内存在{nf}个图像,\nERROR ❌️ 种类数量错误，要求的是{nc}个种类，而不是{nd}")
            else:
                LOGGER.info(f"{prefix} 在{nd}个种类内存在{nf}个图像")
    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}

def find_dataset_yaml(path: Path) -> Path:
    """找到并返回关联一个目标检测数据集、分割数据集或位姿数据集的YAML文件
    Args:
        path(Path): 寻找YAML文件的目录
    Returns:
        (Path): YAML文件的路径
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  #在path下或其子文件夹下的yaml文件
    assert files, f"在{path.resolve()}内未发现YAML文件"
    if len(files) > 1:
        files = [ f for f in files if f.stem == path.stem]  #yaml的文件名称应该与目录名称相同
    assert len(files) == 1, f"只要求一个YAML文件在'{path.resolve()}',但找到了{len(files)}个。\n{files}"
    return files[0]

def check_det_dataset(dataset, autodownload=True):
    """
    检查数据集
    如果没在本地找到，则尝试下载
    Args:
        dataset(str): 数据集路径或者数据集描述者(比如yaml文件)
        autodownload(bool): 当没在本地找到数据集时，是否尝试自动下载数据集，默认True
    Returns:
        (dict): 解析出来的数据集信息和路径
    """
    file = check_file(dataset)   #文件路径
    #下载
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):  #压缩文件
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    data = yaml_load(file, append_filename=True)

    #checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:  #训练集或者验证集不在data内
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' 键丢失❌\n 'train' 和'val'不可或缺")
                )
            LOGGER.WARNING("WARNING ⚠️ 重命名YAML文件内'validation'为'val'，使其适应yolo格式")
            data["val"] = data.pop("validation")
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} 键丢失❌\n 确保'names'或'nc'键的存在"))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise  SyntaxError(emojis(f"{dataset} 'len(names)={len(data['names'])}'的数量与'nc={data['nc']}'不等"))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]  #命名为1，2，3，4，5...
    else:
        data["nc"] = len(data["names"])
    data["names"] = check_class_names(data["names"])

    #绝对路径
    path = Path (extract_dir or data.get("path") or Path(data.get("yaml_file","")).parent) #数据集根目录
    if not path.is_absolute():
        path = (DATASETS_DIR/path).resolve()

    #获取训练集、验证集、测试集的绝对路径
    data["path"] = path
    for k in "train", "val", "test":
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path/x).resolve()) for x in data[k]]

    #parse yaml
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]   #验证集路径
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  #clean "auth"
            m = f"\n⚠️数据集中未找到图像：'{name}'，丢失路径'{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote 数据集下载目录'{DATASETS_DIR}'，可以在'{SETTINGS_YAML}'对其进行更新"
                raise FileNotFoundError(m)
            t = time.time()
            r = None #success
            if s.startswith("http") and s.endswith(".zip"): #URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "): #bash script
                LOGGER.info(f"Running{s} ...")
                r = os.system(s)
            else: #python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"成功下载 ✅ 用时{dt}， 保存至{colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"下载失败 {dt}❌"
            LOGGER.info(f"数据集 {s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  #下载字体
    return data #dict


