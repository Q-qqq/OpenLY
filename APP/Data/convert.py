import glob
import shutil
from multiprocessing.pool import ThreadPool
from pathlib import Path
import os
import copy
import xml.etree.ElementTree as ET
import json
from itertools import repeat

import cv2
import numpy as np
import torch

from ultralytics.utils import LOGGER,NUM_THREADS, yaml_save, PROGRESS_BAR, cv2_readimg
from ultralytics.data.utils import IMG_FORMATS, verify_image,xyxy2xywh


def img2label_path(img_path, img="JPEGImages", label="Annotations", suffix=".xml"):
    """将图像路径转为标签路径"""
    sa, sb = f"{os.sep}{img}{os.sep}", f"{os.sep}{label}{os.sep}"
    return sb.join(str(Path(img_path)).rsplit(sa, 1)).rsplit(".", 1)[0] + suffix


def check_im_files(im_files):
    cls = [None for f in im_files]
    pf = [None for f in im_files]
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(
            func=verify_image,
            iterable=zip(zip(im_files, cls), pf)
        )
        for (im_file, c), nf, nc, msg,shape in results:
            if msg:
                LOGGER.warning(msg)
                im_files.remove(im_file)
                continue
    return im_files

def mask2segments(mask):
    segments = []
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        if len(c) > 3:
            segments.append(c.reshape(-1).astype("float32"))
    return segments

class VOC2YOLO:

    def __call__(self,voc_path, yolo_path, img_dn="JPEGImages", label_dn="Annotations", suffix=".xml", sets_name="ImageSets//Main"):
        self.voc_path = voc_path
        self.yolo_path = yolo_path
        self.im_files = self.getImFiles(img_dn)
        self.names = self.getClassesName()
        self.label_files = self.getLabelFiles(img_dn, label_dn, suffix)
        self.train_files, self.val_files, self.test_files = self.getSets(sets_name)
        self.createPathTxt(yolo_path)
        self.createYolo()

    def createPathTxt(self, yolo_path):
        with open(Path(yolo_path) / "train.txt", "w") as f:
            pass
        with open(Path(yolo_path) / "val.txt", "w") as f:
            pass


    def getImFiles(self, img_dir_name):
        """获取数据集图像文件夹中的所有图像"""
        files = glob.glob(f"{self.voc_path}//{img_dir_name}//**//*.*", recursive=True)
        im_files = [f for f in files if f.split(".")[-1].lower() in IMG_FORMATS]
        return check_im_files(im_files)

    def getClassesName(self):
        """获取种类id和名称"""
        Images_path = Path(self.im_files[0]).parent.parent
        fs = glob.glob(str(Images_path / "**"))
        names = [Path(f).name for f in fs]
        names = dict((v.lower(), k) for k, v in enumerate(sorted(names)))
        return names


    def getLabelFiles(self, img_dn, label_dn, suffix):
        """获取所有图像对应的标签文件，若不存在标签文件的图像将被删除"""
        label_files = []
        fs = copy.copy(self.im_files)
        for im_file in fs:
            label_file = img2label_path(im_file, img_dn, label_dn, suffix)
            if Path(label_file).exists():
                label_files.append(label_file)
            else:
                self.im_files.remove(im_file)
        return label_files

    def getSets(self, sets_name):
        """获取训练集、验证集、测试集对应的图像集"""
        f = Path(self.voc_path) / sets_name
        sets = glob.glob(str(f / "**"), recursive=True)
        train_files = copy.copy(self.im_files)
        val_files = []
        test_files = []
        for set in sets:
            set = Path(set)
            if set.suffix == ".txt":
                if set.stem == "trainval":
                    continue
                with open(set, "r") as f:
                    f = f.read().strip().splitlines()
                    parent = str(set.parent) + os.sep  # 上级路径
                    im_files = [x.replace("./", parent) if x.startswith("./") else x for x in f]
                    im_files = [f for f in im_files if f in self.im_files]
                    if set.stem == "train":
                        train_files.clear()
                        train_files += im_files
                    if set.stem == "val":
                        val_files += im_files
                    if set.stem == "text":
                        test_files += im_files
        return train_files, val_files, test_files

    def xml2txt(self, xml_p, label_p):
        """将xml标签文件转换程yolo1的label标签文件"""
        annotation = ET.parse(xml_p)
        root = annotation.getroot()
        file_name = root.find("filename").text
        if file_name.split(".")[0] != Path(label_p).stem:
            LOGGER.warning(f"{xml_p}中文件名称{file_name}与标签文件名称{label_p}不对应")
            return
        is_segmented = bool(int(root.find("segmented").text))
        if is_segmented:
            LOGGER.warning(f"{xml_p}为分割数据，暂不支持转yolo")
            return
        objects = root.findall("object")
        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)
        for obj in objects:
            cls_name = obj.find("name").text
            assert cls_name in self.names.keys(), f"现有种类不存在{cls_name}"
            cls = self.names[cls_name.lower()]
            box = obj.find("bndbox")
            x1 = float(box.find("xmin").text)
            y1 = float(box.find("ymin").text)
            x2 = float(box.find("xmax").text)
            y2 = float(box.find("ymax").text)
            box = torch.Tensor([x1, y1, x2, y2])
            if max(box) > 1:
                box[0] /= img_w
                box[1] /= img_h
                box[2] /= img_w
                box[3] /= img_h
            xywh = xyxy2xywh(box)
            with open(label_p, "a") as f:
                f.write(("%g " * 5).rstrip() % (cls, *xywh.view(-1)) + "\n")

    def fileSaveToTxt(self, im_file, train_val_test):
        """将图像文件路径保存到txt总路径文件中"""
        im_file = str(Path(im_file))
        self.yolo_path = str(Path(self.yolo_path))
        al_p = im_file.replace(self.yolo_path, "")
        al_p = "." + al_p
        al_p = al_p.replace("\\" , "/")
        txt_p = Path(self.yolo_path) / f"{train_val_test}.txt"
        with open(txt_p, "a") as f:
            f.write(al_p + "\n")

    def write(self, files):
        im_file, label_file, yolo_train_path, yolo_val_path, yolo_test_path, sa, sp = files
        if im_file in self.train_files:
            yolo_im_path = yolo_train_path
            train_val_test = "train"
        elif im_file in self.val_files:
            yolo_im_path = yolo_val_path
            train_val_test = "val"
        else:
            yolo_im_path = yolo_test_path
            train_val_test = "test"
        yolo_label_path = Path(str(yolo_train_path).replace(sa, sp)) / (Path(label_file).stem + ".txt")
        if not Path(yolo_im_path).exists():
            Path(yolo_im_path).mkdir(parents=True, exist_ok=True)
        if not Path(yolo_label_path).parent.exists():
            Path(yolo_label_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(im_file, yolo_im_path)  # image
        self.xml2txt(label_file, yolo_label_path)  # label
        new_im_file = str(Path(yolo_im_path) / Path(im_file).name)
        self.fileSaveToTxt(new_im_file, train_val_test)
        return im_file

    def createYolo(self):
        """创建yolo数据集"""
        yolo_train_path = Path(self.yolo_path) / "images" / "train"
        yolo_val_path = Path(self.yolo_path) / "images" / "val"
        yolo_test_path = Path(self.yolo_path) / "images" / "test"
        sa, sp = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        PROGRESS_BAR.show("VOC2YOLO", "开始转换")
        PROGRESS_BAR.start(0, len(self.im_files), True)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.write,
                                iterable=zip(self.im_files,
                                             self.label_files,
                                             repeat(yolo_train_path),
                                             repeat(yolo_val_path),
                                             repeat(yolo_test_path),
                                             repeat(sa),
                                             repeat(sp)))
            for i, (im_file) in enumerate(results):
                PROGRESS_BAR.setValue(i+1, f"转换中...{im_file}")
                if PROGRESS_BAR.isStop():
                    raise ProcessLookupError("中断：VOC转YOLO中断成功")
        data = {"names":list(self.names.keys()), "train": "train.txt", "val":"val.txt", "path":self.yolo_path}
        if self.test_files:
            data.pop("val")
            data.update({"test": "test.txt"})
        yaml_save(self.yolo_path + "\\dataset.yaml", data)
        PROGRESS_BAR.setValue(i + 1, f"转换完成！")



class COCO2YOLO:
    def __call__(self, coco_path, yolo_path,train_img_dn="train2017", val_img_dn="val2017", label_dn="annotations", label_type="instances", task="detect", suffix=".json"):
        train_files = self.getImFiles(train_img_dn)     #存在的训练集图像文件
        train_annotations = self.getAnnotations(label_dn, label_type, train_img_dn, suffix)  #训练集标签文件解析
        train_id_files, train_id_sizes = self.getImages(train_annotations["images"], train_img_dn, train_files)  #训练集图像信息 id 对应文件名称和图像大小
        names = self.getClassesName(train_annotations["categories"])  #获取种类名称及id
        train_labels = self.getLabels(train_annotations, task, train_id_files, train_id_sizes)   #获取id对应标签 id: im_file, (w, h), classes, instances
        self.createYolo(Path(yolo_path) / "images" / "train", train_labels, yolo_path  +"//train.txt")  #创建yolo数据集

        val_files = self.getImFiles(val_img_dn)
        if val_files:
            val_annotations = self.getAnnotations(label_dn, label_type, val_img_dn)
            val_id_files, val_id_sizes = self.getImages(val_annotations["images"], val_img_dn, val_files)
            val_labels = self.getLabels(val_annotations, task, val_id_files)
            self.createYolo(Path(yolo_path)/"images"/"val", val_labels, yolo_path +"//val.txt")

        self.buildData(names, yolo_path)   #创建yolo数据集解析文件


    def getImFiles(self,dir_name):
        """获取数据集图像文件夹中的所有图像"""
        files = glob.glob(f"{self.coco_path}//{dir_name}//*.*", recursive=True)
        im_files = [str(Path(f)) for f in files if f.split(".")[-1].lower() in IMG_FORMATS]
        return check_im_files(im_files)

    def getAnnotations(self, label_dir_name, label_type, train_val, suffix):
        """读取标签文件"""
        label_p = Path(self.coco_path) / label_dir_name / f"{label_type}_{train_val}{suffix}"
        with open(label_p, "r") as f:
            annotations = json.load(f)
        return annotations

    def getImages(self, images, img_dn, exist_files):
        """获取图像文件对应id 和 图像大小"""
        id_files = {}
        id_sizes = {}
        for image in images:
            p = str(Path(self.coco_path) / img_dn / image["file_name"])
            if p not in exist_files:
                LOGGER.warning(f"图像{p}不存在")
                continue
            id = image["id"]
            width = image["width"]
            height = image["height"]
            id_files.update({id: p})
            id_sizes.update({id: (width, height)})
        return id_files, id_sizes

    def getClassesName(self, categories):
        """获取种类id和中种类名称的字典"""
        return dict((category["id"], category["name"]) for category in categories)

    def getLabels(self, annotations, task, id_files, id_sizes):
        """获取标签 box/segmen, cls, im_dile
        Args:
            annotations(dict): 标签字典，包含segmentation(list(lsit)分割数据), image_id（图像id）, category_id（种类id）,
                area（分割面积）, is_crowd（分割数据是否RLE格式）, bbox（目标框数据）,id（标签id）等参数;
            task(str): segment/detect, 指定数据集的格式
            id_files(dict): 图像id对应的图像名称
            id_sizes(dict)：图像id对应的图像大小（w, h）"""
        labels = {}  # id: im_file, (w, h),  classes， instances
        for annotation in annotations:
            assert not annotation["iscrowd"], "不支持RLE格式的分割数据"
            if task == "segment":
                assert len(annotation["segmentation"])>0, f"标签{annotation['id']}不存在分割数据集"
            if annotation["image_id"] not in id_files.keys():
                LOGGER.warning(f"图像{annotation['image_id']}不存在images标签中")
                continue
            img_id = annotation["image_id"]
            box = annotation["bbox"]
            segment = annotation["segmentation"][0]
            cls = [annotation["category_id"]]
            im_file = id_files[img_id]
            (w, h) = id_sizes[img_id]
            if task == "detect":
                instance = [box]
            else:
                instance = [segment]

            if labels.get(img_id, None):
                labels[img_id] = [im_file, (w, h), cls, instance]
            else:
                labels[img_id][2].append(cls)
                labels[img_id][3].append(instance[0])
        return labels


    def write(self, label_mes):
        """将标签保存到yolo标注文件txt中"""
        (id, label), yolo_img_path, yolo_label_path, path_txt = label_mes
        im_file, (w, h), classes, instances = label
        instances = np.array(instances, dtype=np.float32)
        if len(instances) and instances.max() > 1:
            n = len(instances)
            arr_inst = instances.reshape(n, -1, 2)
            arr_inst[..., 0] /= w
            arr_inst[..., 1] /= h
            instances = arr_inst.reshape(n, -1)
        lines = []
        for cls, instance in zip(classes, instances):
            line = ("%g " * (len(instance) + 1)).rstrip() % (cls, *(instance.reshape(-1)))
            lines.append(line)
        label_path = yolo_label_path / (Path(im_file).stem + ".txt")
        with open(label_path, "a") as f:
            f.writelines(line + "\n" for line in lines)
        shutil.copy(im_file, yolo_img_path)  # 复制图像
        new_im_file = str(Path(yolo_img_path) / Path(im_file).name)
        self.fileSaveToTxt(new_im_file, path_txt)  # 保存图像路径
        return new_im_file

    def fileSaveToTxt(self, im_file,  path_txt):
        im_file = str(Path(im_file))
        yolo_path = str(Path(path_txt).parent)
        al_p = im_file.replace(yolo_path, "")   #相对路径
        al_p = "." + al_p
        al_p = al_p.replace("\\", "/")
        with open(path_txt, "a") as f:
            f.write(al_p + "\n")

    def createYolo(self, yolo_img_path, labels, path_txt):
        """创建yolo数据集
        Args:
            yolo_img_path(Path): yolo图像文件夹
            instances(list)：标签数据 box / segment
            classes(list): 标签对应种类
            im_files(list): 标签对应图像
            path_txt(str): 图像文件路径存储路径"""
        if not Path(yolo_img_path).exists():
            Path(yolo_img_path).mkdir(parents=True, exist_ok=True)
        sa, sp = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        yolo_label_path = Path(str(yolo_img_path).replace(sa, sp))
        if not yolo_label_path.exists():
            yolo_label_path.mkdir(parents=True, exist_ok=True)
        else:
            fs = glob.glob(str(yolo_label_path / "*.txt"))
            if len(fs):
                shutil.rmtree(yolo_label_path, True)
        PROGRESS_BAR.show("COCO2YOLO", "开始转换")
        PROGRESS_BAR.start(0, len(labels), True)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.write,
                      iterable=zip(labels.items(),
                                   repeat(yolo_img_path),
                                   repeat(yolo_label_path),
                                   repeat(path_txt)))
            for i, im_file in enumerate(results):
                PROGRESS_BAR.setValue(i+1, "转换中..." + im_file)
                if PROGRESS_BAR.isStop():
                    raise ProcessLookupError("中断：COCO转YOLO中断成功")


    def buildData(self,names,  yolo_path):
        """建立数据集文件"""
        data = {"names": list(names.values()), "path":yolo_path, "train":"train.txt", "val": "val.txt"}
        yaml_save(Path(yolo_path) / "data.yaml", data)




class PNG2YOLO:
    def __call__(self, seg_train_p, seg_val_p, seg_suffix, ori_train_p, ori_val_p, ori_suffix, yolo_p):
        dataset_p = Path(f"{yolo_p}//dataset.yaml")
        train_p = Path(f"{yolo_p}//train.txt")
        val_p = Path(f"{yolo_p}//val.txt")
        for p in [train_p, val_p]:
            with open(p, "w") as f:
                f.write("")
        data = {"names":[], "train": "train.txt", "val":"val.txt", "path":yolo_p}
        assert seg_train_p and Path(seg_train_p).exists() and ori_train_p and Path(ori_train_p).exists(), f"路径出错，请检查训练集路径是否存在"
        train_cls_dirs = glob.glob(f"{seg_train_p}//**")
        names = []
        for i,train_dir in enumerate(train_cls_dirs):
            cls_name = Path(train_dir).name

            seg_train_imfiles = glob.glob(f"{train_dir}//**//*.{seg_suffix}",recursive=True)
            seg_train_imfiles = check_im_files(seg_train_imfiles)
            ori_train_imfiles = glob.glob(f"{train_dir}//**//*.{ori_suffix}", recursive=True)
            ori_train_imfiles = check_im_files(ori_train_imfiles)
            seg_train_imfiles, ori_train_imfiles = self.checkSegAndOri(seg_train_imfiles, ori_train_imfiles)
            segments = self.getSegments(seg_train_imfiles)
            PROGRESS_BAR.show(f"训练集-种类{cls_name}")
            self.toYolo("train", i, ori_train_imfiles, segments, yolo_p)
            names.append(cls_name)
        data.update({"names":names})
        yaml_save(f"{yolo_p}//dataset.yaml", data)




        if seg_val_p and Path(seg_val_p).exists() and ori_val_p and Path(ori_val_p).exists():
            val_cls_dirs = glob.glob(f"{seg_val_p}//**")
            for i, val_dir in enumerate(val_cls_dirs):
                cls_name = Path(val_dir).name
                assert cls_name in names, f"验证集存在训练集不存在的种类{cls_name}"
                seg_val_imfiles = glob.glob(f"{val_dir}//*.{seg_suffix}", recursive=True)
                seg_val_imfiles = check_im_files(seg_val_imfiles)
                ori_val_imfiles = glob.glob(f"{val_dir}//*.{ori_suffix}", recursive=True)
                ori_val_imfiles = check_im_files(ori_val_imfiles)
                seg_val_imfiles, ori_val_imfiles = self.checkSegAndOri(seg_val_imfiles, ori_val_imfiles)
                segments = self.getSegments(seg_val_imfiles)
                PROGRESS_BAR.show(f"验证集-种类{cls_name}")
                self.toYolo("val", i, ori_val_imfiles, segments, yolo_p)
        PROGRESS_BAR.close()


    def checkSegAndOri(self, seg_imfiles, ori_imfiles):
        seg_names = [Path(f).stem for f in seg_imfiles]
        new_seg_imfiles = []
        for i, ori_imfile in enumerate(ori_imfiles):
            ori_name = Path(ori_imfile).stem
            if ori_name not in seg_names:
                ori_imfiles.remove(ori_imfile)
                continue
            new_seg_imfiles.append(seg_imfiles[i])
        return new_seg_imfiles, ori_imfiles

    def getSegments(self, seg_imfiles):
        segments = []
        for im_file in seg_imfiles:
            mask = cv2_readimg(im_file, cv2.IMREAD_GRAYSCALE)
            segment = mask2segments(mask)
            if len(segment) and segment[0].max() > 1:
                h, w = mask.shape[:2]
                for i, seg in enumerate(segment):
                    seg = seg.reshape(-1,2)
                    seg[...,0] /= w
                    seg[..., 1] /=h
                    segment[i] = seg.reshape(-1)
            segments.append(segment)
        return segments

    def toYolo(self, train_val, cls, ori_imfiles, segments, yolo_p):
        PROGRESS_BAR.start(0, len(ori_imfiles))
        yolo_img_p = f"{yolo_p}//images//{train_val}"
        yolo_label_p = f"{yolo_p}//labels//{train_val}"
        paths = f"{yolo_p}//{train_val}.txt"
        for p in (yolo_img_p, yolo_label_p):
            if not Path(p).exists():
                Path(p).mkdir(parents=True, exist_ok=True)
        for i, (ori_f,segment) in enumerate(zip(ori_imfiles,segments)):
            lines = []
            for seg in segment:
                line = (cls, *seg)
                lines.append(("%g " * len(line)).rstrip() % line)
            shutil.copy(ori_f, yolo_img_p)
            with open(f"{yolo_label_p}//{Path(ori_f).stem}.txt", "w") as f:
                f.writelines(line + "\n" for line in lines)
            with open(paths, "a") as pf:
                pf.write(f".//images//{train_val}//{Path(ori_f).name}\n")
            PROGRESS_BAR.setValue(i+1, f"种类{cls}: {ori_f}")
















