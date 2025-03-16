# ultralytics 适应性修改

## 修改一：验证时混淆矩阵输出验证结果正反例对应的文件路径  

### ultralytics.utils.metrics.ConfusionMatrix  

+ 初始化添加文件存储参数self.im_files

```python
def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
    ...
    ...
    self.im_files = [[[] for i in range(nc)] for j in range(nc)] if self.task != "classify" \
                else [[[] for i in range(nc+1)] for j in range(nc+1)]   # pred,true:im_files
```

+ 增加函数addImFile(...）向self.im_files添加路径

```python
def addImFile(self, pred_cls, gt_cls, im_file):
    """添加对应种类的图像文件到self.im_files中
    Args:
        pred_cls: im_file或其中中某个目标预测的种类
        gt_cls: im_file或其中某个目标真实的种类
        im_file: 目标文件
    """
    if isinstance(gt_cls, torch.Tensor):
        gt_cls = gt_cls.item()
    if im_file not in self.im_files[pred_cls][gt_cls]:
        self.im_files[pred_cls][gt_cls].append(im_file)
```

+ 对于分类任务，在process_cls_preds函数中增加输入参数im_files，并使用addImFile将对应路径添加到self.im_files中

```python
def process_cls_preds(self, preds, targets, im_files):
    ...
    ...
    for p, t, im_file in zip(preds.cpu().numpy(), targets.cpu().numpy(), im_files):
        self.matrix[p][t] += 1
        self.addImFile(p, t, im_file)  #add
```

+ 对于目标检测、分割、定向框和姿态任务，在process_batch中使用addImFile添加路径（5个位置需要添加）

```python
def process_batch(self, detections, gt_bboxes, gt_cls, im_file):
    ...
    if gt_cls.shape[0] == 0:  # Check if labels is empty
        if detections is not None:
            detections = detections[detections[:, 4] > self.conf]
            detection_classes = detections[:, 5].int()
            for dc in detection_classes:
                self.matrix[dc, self.nc] += 1  # false positives
                self.addImFile(dc, self.nc, im_file)  #add
        return
    if detections is None:
        gt_classes = gt_cls.int()
        for gc in gt_classes:
            self.matrix[self.nc, gc] += 1  # background FN
            self.addImFile(self.nc, gc, im_file)   #add
        return
    ...
    ...
    for i, gc in enumerate(gt_classes):
        j = m0 == i
        if n and sum(j) == 1:
            self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            self.addImFile(detection_classes[m1[j]], gc, im_file)  #add
        else:
            self.matrix[self.nc, gc] += 1  # true background
            self.addImFile(self.nc, gc, im_file)  #add

    for i, dc in enumerate(detection_classes):
        if not any(m1 == i):
            self.matrix[dc, self.nc] += 1  # predicted background
            self.addImFile(dc, self.nc, im_file)  #add
```

+ 最后在绘制混淆矩阵的时候进行保存，并修改显示只显示种类序号，修改xy轴字体大小, 并添加保存结果的函数

```python
@TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
@plt_settings()
def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
    ...
    ...
    ticklabels = [str(i) for i in range(nn + 1)] if labels else "auto" #only class number
    #ticklabels = (list(names) + ["background"]) if labels else "auto"
    ...
    ...
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    ...
    ...
    if normalize:   #save once for im_files
        self.saveImageFiles(save_dir, names)
    if on_plot:
        on_plot(plot_fname)

def saveImageFiles(self, save_dir, names):
        save_dict = {}
        names = list(names)
        if self.task != "classify":
            names.append("null")
        for pred_c in range(len(self.im_files)):
            for gt_c in range(len(self.im_files[pred_c])):
                save_dict[f"pred-{names[pred_c]},true-{names[gt_c]}${pred_c},{gt_c}"] = self.im_files[pred_c][gt_c]
        yaml_save(Path(save_dir) /"Confusion_Matrix_Imfiles.yaml",save_dict)
```

### ultralytics.models.yolo.classify.val

+ 初始化评估指标时，初始化self.im_files参数，其将存储所有验证图像路径

```python
def init_metrics(self, model):
    ...
    ...
    self.im_files = []   #metris files name
```  

+ 每一次更新指标时，将验证的图像按验证顺序添加到im_files中

```python
def update_metrics(self, preds, batch):
    ...
    ...
    self.im_files.append(batch["im_file"])
```

+ 在完成评估指标更新时，将self.im_files传入ConfusionMatrix中

```python
def finalize_metrics(self, *args, **kwargs):
    """Finalizes metrics of the model such as confusion_matrix and speed."""
    self.confusion_matrix.process_cls_preds(self.pred, self.targets, im_files=self.im_files)  #传入
    ...
    ...
```

### ultralytics.models.yolo.detect.val

+ 每一次更新都将对应的路径传入ConfusionMatrix中

```python
def update_metrics(self, preds, batch):
    """Metrics."""
    for si, pred in enumerate(preds):
        ...
        ...
        if npr == 0:
            if nl:
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])
                if self.args.plots:
                    self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls, im_file=batch["im_file"][si])   #更新
            continue
        ...
        ...
        if self.args.plots:
            self.confusion_matrix.process_batch(predn, bbox, cls, im_file=batch["im_file"][si])  #更新
        ...
        ...
```

### ultralytics.models.yolo.segment.val 同 ultralytics.models.yolo.detect.val

### ultralytics.models.yolo.pose.val 同 ultralytics.models.yolo.detect.val

## 修改二：适应中文图像路径输入

### ultralytics.data.utils

+ 自定义读取中文路径的cv2_readimg

```python
def cv2_readimg(img_path, color=cv2.IMREAD_COLOR):
    """使用cv2读取图像
    Args:
        img_path(str): 图像路径
        color(bool): 是否彩色图像RGB"""
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8), color)
    return img
```

### ultralytics.data

+ 在data文件夹使用替换，将cv2.imread全部替换为cv2_readimg, 且在各文件中添加from ultralytics.data.utils import cv2_readimg

## 修改三：进度条

### ultralytics.utils.__init

+ 添加Qt类Progress， 通过对其信号进行槽连接获取进度数据从而进行进度条的显示，实例化为全局变量PROGRESS_BAR

```python
from PySide6.QtCore import *
class Progress(QObject):
    """进度条"""
    Start_Signal = Signal(str, str, list)
    Set_Value_Signal = Signal(int, str)
    Reset_Signal = Signal()
    Close_Signal = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False    #停止进度条对应的加载
        self.permit_stop = False  #允许中断进度
        self.loading = False   #进度条加载中


    def stop(self):
        """停止进度条"""
        if self.permit_stop:
            self.loading = False
            self._stop = True

    def isStop(self):
        """判断是否停止"""
        return self._stop

    def start(self, title="Load", head_txt="start load...", range=[0,100], permit_stop=False):
        """开始进度条"""
        self._stop = False
        self.loading = True
        self.permit_stop = permit_stop
        self.Start_Signal.emit( title, head_txt, range)

    def setValue(self, value, text):
        """设置进度条的值"""
        self.Set_Value_Signal.emit(value, text)
    

    def reset(self):
        """重置进度条"""
        self._stop = False
        self.Reset_Signal.emit()

    def close(self):
        """关闭进度条"""
        self.loading = False
        self.Close_Signal.emit()

PROGRESS_BAR = Progress()
```

+ 修改TryExcept修饰器，使其可以在程序出错时强制中断进度条, 并输出训练中断信号

```python
class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if value:
            value = str(value).replace("：", ":")
            if value.startswith("Interrupt"):
                LOGGER.interruptError(value.split(":")[1].strip())
            elif self.verbose:
                LOGGER.error(f"{self.msg}{': ' if self.msg else ''}{value}")
            if PROGRESS_BAR.loading:
                PROGRESS_BAR._stop = True
                PROGRESS_BAR.close()
        return True
```

### ultralytics.data.dataset

+ YoloDataset类中cache_labels进行标签的加载，对其进行进度条显示

```python
def cache_labels(self, path=Path("./labels.cache")):
    ...
    ...
    PROGRESS_BAR.start("DataLoader", "Start...", [0, total], False)
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(
            ...
            ...
        )
        pbar = TQDM(results, desc=desc, total=total)
        for i, (im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg) in enumerate(pbar):
            ...
            ...
            if msg:
                msgs.append(msg)
            PROGRESS_BAR.setValue(i+1, f"Dataset loading...{im_file if im_file else msg}")
            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            LOGGER.info(pbar.desc)
        pbar.close()
        PROGRESS_BAR.close()
    ...
    ...
    return x
```

+ ClassifyDataset类中verify_images进行图像的检测，对其进行进度条显示

```python
 def verify_images(self):
    ...
    ...
    except (FileNotFoundError, AssertionError, AttributeError):
        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        PROGRESS_BAR.start("Classify dataset Load", "Start", [0,len(self.samples)], False)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(enumerate(results), desc=desc, total=len(self.samples))
            for i, sample, nf_f, nc_f, msg,_ in pbar:
                ...
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                PROGRESS_BAR.setValue(i+1, f"数据集加载中...{sample[0]}, {sample[1]}")
            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        PROGRESS_BAR.close()
        ...
        ...
```

### ultralytics.data.base

+ BaseDataset类中cache_images进行图像的缓存，对其进行进度条显示

```python
def cache_images(self):
    """Cache images to memory or disk."""
    b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
    fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
    PROGRESS_BAR.start("Cache images", "Start...", [0, self.ni], False)
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(fcn, range(self.ni))
        pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
        for i, x in pbar:
            if self.cache == "disk":
                b += self.npy_files[i].stat().st_size
            else:  # 'ram'
                self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                b += self.ims[i].nbytes
            pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            PROGRESS_BAR.setValue(i+1, pbar.desc)
        PROGRESS_BAR.close()
        pbar.close()
```

### ultralytics.engine.validator

+ BaseValidator类中__call__进行数据集的验证。对其在非训练期间进行验证进度条显示

```python
@smart_inference_mode()
def __call__(self, trainer=None, model=None):
    ...
    ...
    if not self.training:
        PROGRESS_BAR.start("Validator", "Start val...", [0, len(self.dataloader)], True)
    for batch_i, batch in enumerate(bar):
        ...
        ...
        self.run_callbacks("on_val_batch_end")
        if not self.training:
            PROGRESS_BAR.setValue(batch_i+1, str(len(batch["img"])))
            if PROGRESS_BAR.isStop():
                PROGRESS_BAR.close()
                raise ProcessLookupError("Interrupt：Val interrupt successful")
    ...
    ...
    self.run_callbacks("on_val_end")
    if not self.training:
        PROGRESS_BAR.close()
    i...
    ...
```

### ultralytics.engine.predictor

+ BasePredictor类中stram_inference进行流推理，对其进行进度条显示, 并且将预测结果self.results转移至cpu，减少显存的占用

```python
@smart_inference_mode()
def stream_inference(self, source=None, model=None, *args, **kwargs):
    ...
    ...
    with self._lock:  # for thread-safe inference
        ...
        ...
        self.run_callbacks("on_predict_start")
        PROGRESS_BAR.start("Predict", "Start predicting...", [0, len(self.dataset)], True)
        for data_i, self.batch in enumerate(self.dataset):
            ...
            for i in range(n):
                self.seen += 1
                self.results[i] = self.results[i].cpu() #Reduce memory
                ...
            ...
            self.run_callbacks("on_predict_batch_end")
            yield from self.results
            PROGRESS_BAR.setValue(data_i+1, f"{s} {profilers[1].dt *1e3:.1f}ms")
            if PROGRESS_BAR.isStop():
                PROGRESS_BAR.close()
                break
        PROGRESS_BAR.close()
```

### ultralytics.data.loaders

+ 在LoadPilAndNumpy类中，需要对每个输入图像进行check，对其checks进行进度条显示

```python
def __init__(self, im0):
    ...
    self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
    self.im0 = self.checkImgs(im0)
    ...

def checkImgs(self, ims):
    imgs = []
    PROGRESS_BAR.start("预测图像集加载", "开始加载", [0, len(ims)], True)
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(func=self._single_check,
                            iterable=ims)
        for i, im in enumerate(results):
            imgs.append(im)
            PROGRESS_BAR.setValue(i+1, self.paths[i])
            if PROGRESS_BAR.isStop():
                PROGRESS_BAR.close()
                raise ProcessLookupError("Interrupt：Load img of predict interrupt success")
    return imgs
```

### ultralytics.enginer.exporter

+ 在Exporter类的__call__函数中，添加导出进度条显示

```python
def __call__(self, model=None) -> str:
        """Returns list of exported files/dirs after running callbacks."""
        PROGRESS_BAR.start("模型导出， ”导出中", 0, 0, False)
        self.run_callbacks("on_export_start")
        ...
        ...
        self.run_callbacks("on_export_end")
        PROGRESS_BAR.close()
        return f  # return list of exported files/dirs
```

## 修改四：加载预测图像时不要一次性加载，一次一张，减少显存需求

### ultralytics.data.loader

+ 在LoadPilAndNumpy类中，修改__next__函数，获取每一次迭代的值为一张图像

```python
def __next__(self):
        """Returns the next batch of images, paths, and metadata for processing."""
        if self.count == len(self.im0):
            raise StopIteration  #迭代结束信号
        self.count += 1
        return [self.paths[self.count-1]], [self.im0[self.count-1]],[""]
```

+ 在LoadTensor类中，同LoadPilAndNumpy一样修改

## 修改五：验证图像和标签时确保加载文件的线程安全

### ultralytics.data.utils

+ 在verify_iamge_label函数中添加线程锁

```python
_LOCK = threading.Lock()
def verify_image_label(args):
    """Verify one image-label pair."""
    with _LOCK:
        im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
        ...
        ...
```

## 修改六：验证结果绘制自定义

### ultralytics.engine.validator

+ 在BaseValidator类的__call__函数中，注释掉训练期间self.args.plot的赋值，使其自定义是否绘制结果且每次验证重绘一次

```python
@smart_inference_mode()
def __call__(self, trainer=None, model=None):
    """Executes validation process, running inference on dataloader and computing performance metrics."""
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
        ...
        ...
        #self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        model.eval()
    ...
    ...
```

## 修改七：添加Keys

### ultralytics.cfg.__init__.py

+ 往全部变量中添加CFG_OTHERS_KEYS, 使KEYS全局变量包含全部参数, 将batch从float类型移动到int类型

```python
CFG_INT_KEYS = frozenset(
    {  # integer-only arguments
        ...
        "batch",
    }
)
CFG_BOOL_KEYS = frozenset(
    {  # boolean-only arguments
        ...
        ...
        "amp",
    }
)

CFG_OTHER_KEYS = frozenset(
    {
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
    "copy_paste_mode", #(str) "flip"  the method to do copy_paste augmentation (flip, mixup)
    "auto_augment",     #(str)randaugment, auto augmentation policy for classification (randaugment, autoaugment,augmix)
    "cfg",       #(str,optional) for overriding defaults.yaml
    "tracker",   #(str)bootsort.yaml, tracker type, choices=[botsrt.yaml, bytetrack.yaml]
    "resume",    # (bool|str)False, resume training from last checkpoint
    "imgsz",     #(int | list)640, image size  width,height
    }
)
```

## 修改八：LOGGER

### ultralytics.utils.__init__

+ 添加Logging类，并实例化为LOGGER， 替换原先的LOGGER. 其中定义了多个信号对训练、验证信息进行界面显示

```python
# Set logger
_LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
class Logger(QObject):
    """信息显示"""
    Show_Mes_Signal = Signal(str, str)
    Start_Train_Signal = Signal(list)
    Batch_Finish_Signal = Signal(str)
    Epoch_Finish_Signal = Signal(list)
    Train_Finish_Signal = Signal(str)
    interrupt_error_Signal = Signal(str)
    Start_Val_Signal = Signal(str)
    Val_Finish_Signal = Signal(str)
    Error_Signal = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stop = False  #停止训练


    def error(self,msg):
        """错误信号"""
        _LOGGER.error(msg)
        self.Error_Signal.emit(msg)
        self.Show_Mes_Signal.emit("error", msg)

    def warning(self,msg):
        """警告信号"""
        _LOGGER.warning(msg)
        self.Show_Mes_Signal.emit("warning", msg)

    def info(self,msg):
        """正常信号"""
        _LOGGER.info(msg)
        self.Show_Mes_Signal.emit("info", msg)

    def startTrain(self, msg_epochs):
        """开始训练信号"""
        _LOGGER.info(msg_epochs[0])
        self.Start_Train_Signal.emit(msg_epochs)

    def batchFinish(self, msg):
        """完成一个batch信号"""
        self.Batch_Finish_Signal.emit(msg)

    def epochFinish(self, msg_epoch):
        """完成一个epoch信号"""
        _LOGGER.info(msg_epoch[0])
        self.Epoch_Finish_Signal.emit(msg_epoch)


    def trainFinish(self, msg):
        """训练结束信号"""
        _LOGGER.info(msg)
        self.Train_Finish_Signal.emit(msg)

    def interruptError(self, msg):
        """中断信号"""
        self.interrupt_error_Signal.emit(msg)

    def startVal(self, msg):
        """开始验证信号"""
        _LOGGER.info(msg)
        self.Start_Val_Signal.emit(msg)

    def valFinish(self, msg):
        """验证结束信号"""
        _LOGGER.info(msg)
        self.Val_Finish_Signal.emit(msg)

LOGGER  = Logger()
```

### ultralytics.engine.trainer

+ 在BaseTrainer类的_do_train函数中添加LOGGER对训练进度信息进行显示, 并添加total_instances变量计算每一次训练的总实例数量，在最后一个batch输出

```python
def _do_train(self, world_size=1):
    ...
    ...
    LOGGER.info(
        f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
        f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if RANK in (-1, 0):
        LOGGER.startTrain([self.progress_string(), self.start_epoch, self.epochs]) #start train sinal
        LOGGER.startVal(self.validator.get_desc())
    ....
    ....
    while True:
        ...
        ...
        if RANK in {-1, 0}:
            #LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        ...
        total_instance = 0 # all instances
        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            ...
            ...
            # Log
            if RANK in {-1, 0}:
                total_instance += batch["cls"].shape[0]
                instances = batch["cls"].shape[0] if i < len(self.train_loader)-1 else total_instance
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                loss_mes = ("%11s" * 2 + "%11.4g" * (2 + loss_length))% (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                        instances,  # batch size, i.e. 8
                        batch["img"].shape[-1],  # imgsz, i.e 640
                    )
                pbar.set_description(loss_mes)
                LOGGER.batchFinish(loss_mes)
                self.run_callbacks("on_batch_end")
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            self.run_callbacks("on_train_batch_end")
            if LOGGER.stop:
                LOGGER.trainInterrupt()
                raise ProcessLookupError(f"Interrupt：训练中断成功,已训练{epoch}epoch")

        ...
        if RANK in {-1, 0}:
            ...
            ...
            LOGGER.epochFinish([loss_mes, epoch+1])

        ....
        ...
    if RANK in {-1, 0}:
        ...
        ...
        self.run_callbacks("on_train_end")
        LOGGER.trainFinish("Train Finish!!")
    self._clear_memory()
    self.run_callbacks("teardown")
```

### ultralytics.engine.validator

+ 在BaseValidator类的__call__函数中添加LOGGER发送验证完成信号

```python
@smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        ...
        ...
        LOGGER.valFinish("")
        self.run_callbacks("on_val_end")
        ...
        ...
```

## 修改九：验证图像增加输出shape

### ultralytics.data.utils

+ 在verify_image函数中添加图像shape的输出

```python
def verify_image(args):
    ...
    ...
    return (im_file, cls), nf, nc, msg, shape
```

+ 在dataset文件中的classifyDataset类中同步修改verify_image的输出获取

```python
# Run scan if *.cache retrieval failed
nf, nc, msgs, samples, x = 0, 0, [], [], {}
PROGRESS_BAR.start("Classify dataset Load", "Start", [0,len(self.samples)], False)
with ThreadPool(NUM_THREADS) as pool:
    results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
    pbar = TQDM(enumerate(results), desc=desc, total=len(self.samples))
    for i, sample, nf_f, nc_f, msg,_ in pbar:  #<---
```
## 修改十： 检测det数据集默认父路径修改

### ultralytics.data.utils

+ 在check_det_dataset函数中，修改数据集默认父路径DATASETS_DIR为Path(file).parent

```python
def check_det_dataset(dataset, autodownload=True):
    file = check_file(dataset)
    ...
    ...
    # Resolve paths
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        path = (Path(file).parent / path).resolve()   #DATASETS_DIR->Path(file).parent
    ...
    ...
```

## 修改十一： 增加/修改训练进行时输出参数

### ultralytics.enginer.trainer

+ 在BaseTrainer类的_do_train函数中，增加训练时输出batch信息，修改每一个epoch最后的输出instances为该epoch的总instances

```python
def _do_train(self, world_size=1):
    ...
    ...
    self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
    while True:
       ...
       ...
            ...
            # Log
            if RANK in {-1, 0}:
                total_instance += batch["cls"].shape[0]
                instances = batch["cls"].shape[0] if i < len(self.train_loader)-1 else total_instance
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                loss_mes = ("%11s" * 3 + "%11.4g" * (2 + loss_length))% (
                        f"{epoch + 1}/{self.epochs}",
                        f"{i+1}/{len(self.train_loader)}",   #batch
                        f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                        instances,  # batch size, i.e. 8
                        batch["img"].shape[-1],  # imgsz, i.e 640
                    )
                pbar.set_description(loss_mes)
                LOGGER.batchFinish(loss_mes)
                self.run_callbacks("on_batch_end")
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            self.run_callbacks("on_train_batch_end")
        ...
        ...
```

### ultralytics.models.yolo.detect.train

+ 在DetectionTrainer类的progress_string函数中添加batch输出,适应训练时添加batch数据输出

```python
def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (5 + len(self.loss_names))) % (
            "Epoch",
            "batch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
```

## 修改十二： 增加旧版yoloV5预选框训练方式

### 解析yolov5模型

1. ultralytics.cfg.models.v5

+ 添加yoloV5神经网络文件夹yolov5-anchors.yaml和yolov5-seg-anchors

2. ultralytics.nn.head

+ 添加检测头v5Detect

```python
class v5Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # 输入输出图像倍数 每个检测头的stride [8, 16, 32]
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor  cls + (xywh+conf)
        self.nl = len(anchors)  # number of detection layers 检测头个数
        self.na = len(anchors[0]) // 2  # number of anchors  每个检测头的anchor数
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl): #每个检测头
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, v5Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4) # x, y, w, h, conf, mask
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4) #xy wh conf mask (bs,3,ny,nx,(2+2+nc+1+nm))  3->一个检测头3个预选框 no = nc+5+nm
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x) #train x; val z,x; export z   x原始数据 z目标信息 

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    def bias_init(self, cf = None):
        #初始化检测头的偏置
        m = self  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
```

+ 添加检测头v5Segment

```python
class v5Segment(v5Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=()):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = V5Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1]) #x[0]（bs,3*nx*ny*nl, xywh+cls+conf+nm）
```

+ 将v5Detect和v5Segment添加到__init__

```python
__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "v5Detect", "v5Sefment"
```

3。 ultralytics.nn.modules.__init__

+ 将v5Detect和v5Segment添加到modules.__init__的引用

```python
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect, v5Detect, v5Segment
```

4. ultralytics.nn.tasks

+ 添加引用 v5Detect和v5Segment

```python
from ultralytics.nn.modules import (
    ...
    ...
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
    v5Detect,   #<-
    v5Segment,  #<-
    A2C2f,
)
```

+ 修改解析模型函数parse_model，使其能解析yolov5神经网路

```python
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    ...
    ...
    nc, act, scales, anchors = (d.get(x) for x in ("nc", "activation", "scales","anchors"))   #v5 add anchors
    no=0
    if anchors:
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)   # number of outputs = anchors * (classes + 5)
    ...
    ...
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        ...
        ...
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc and c2 != no:  # if c2 not equal to number of classes (i.e. for Classify() output) and c2 no equal to no for yolov5 output to detect head
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            ...
            ...
        ...
        ...
        elif m in frozenset({Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect, v5Detect, v5Segment}):
            args.append([ch[x] for x in f])
            if isinstance(args[1], int) and m in (v5Segment, v5Detect):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m in [Segment,v5Segment]:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8) #number of masks
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        ...
        ...
```

+ 修改guess_model_task函数，使其可以猜测yolov5任务

```python
def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # returns n, s, m, l, or x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"
        if m == "v5detect":
            return "v5detect"
        if m == "v5segment":
            return "v5segment"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"
            elif isinstance(m, v5Detect):
                return "v5detect"
            elif isinstance(m, v5Segment):
                return "v5segment"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem and "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"
        elif "v5detect" in model.parts:
            return "v5detect"
        elif "v5segment" in model.parts:
            return "v5segment"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
```

+ 修改DetectionModel类的初始化函数，使其对yolov5的检测头进行初始化

```python
def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
    ...
    ...
    def _forward(x):
            """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
            if self.end2end:
                return self.forward(x)["one2many"]
            return self.forward(x)[0] if isinstance(m, (v5Segment, Segment, Pose, OBB)) else self.forward(x)
    # Build strides
    m = self.model[-1]  # Detect()
    if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
        s = 256  # 2x min stride
        m.inplace = self.inplace
        m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
        self.stride = m.stride
        m.bias_init()  # only run once
    elif isinstance(m, (v5Detect, v5Segment)):
        s=256
        m.inplace = self.inplace
        m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
        autoanchor.check_anchors_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)  # 将预选框缩放到grid_size大小
        self.stride = m.stride
        m.bias_init()
    else:
        self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
    ...
    ...
```

+ 添加v5 detection模型

``` python
class V5DetectionModel(DetectionModel):
    def __init__(self, cfg="yolov5-anchors.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return V5DetectLoss(self)
```

+ 添加v5 segmentation模型

```python
class V5SegmentationModel(DetectionModel):
    def __init__(self, cfg="yolov5-seg-anchors.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return V5SegmentLoss(self, overlap=self.args.overlap_mask)
```

### 计算yolov5损失值

1. ultralytics.utils.loss

+ 添加v5目标检测损失函数

```python
class V5DetectLoss:
    def __init__(self,model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.cls_pw], device=device),reduction="mean")
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.obj_pw], device=device),reduction="mean")

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = self.smooth_BCE(eps= h.label_smoothing)  # positive, negative BCE targets

        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = model.model[-1]  # Detect() module
        self.model = model
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, h.gr, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def smooth_BCE(self, eps=0.5):
        # 返回类别正负样本的值 0< eps < 2
        return 1. - 0.5 * eps, 0.5 * eps

    def __call__(self, pred_out,targets):  # predictions, targets
        pred_out = pred_out[1] if isinstance(pred_out, tuple) else pred_out
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        img = targets["batch_idx"].unsqueeze(1)
        cls = targets["cls"] if targets["cls"].ndim == 2 else targets["cls"].unsqueeze(1)
        box = targets["bboxes"]
        new_targets = torch.cat((img, cls, box),-1)

        tcls, tbox, indices, anchors = self.build_targets(pred_out, new_targets.to(self.device), self.model)
        nt = 0
        for i, p in enumerate(pred_out):
            im, a, gj, gi = indices[i]  # image anchors grid_y grid_x
            tobj = torch.zeros_like(p[..., 4], dtype=p.dtype, device=self.device)

            nb = im.shape[0]  # target数量
            if nb > 0:
                nt += nb  # 总target数量
                p_sub = p[im, a, gj, gi]  # 从总预测值中抽出真实框对应位置的预测值

                # Giou
                pxy = p_sub[:, :2].sigmoid() * 2. - 0.5  # 使中点不落在方格边界上
                pwh = (p_sub[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # wh = an*(wh*2)^2
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], xywh=True,CIoU=True).squeeze()  # 计算ciou
                #iou_index = iou > self.hyp["iou_t"]
                #iou = iou*iou_index
                lbox += (1.0 - iou).mean()  # box的giou损失

                # obj
                tobj[im, a, gj, gi] = (1 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # 用ciou代替真实置信度


                # cls
                if self.nc > 1:  # cls loss (only if multiple classes)
                    tc = torch.full_like(p_sub[:, 5:], self.cn)  # 种类负样本
                    tc[range(nb), tcls[i]] = self.cp  # 种类正样本 one-hot形式
                    lcls += self.BCEcls(p_sub[:, 5:], tc)
            lobj += self.BCEobj(p[..., 4], tobj)
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / lobj.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp.v5_box
        lcls *= self.hyp.v5_cls
        lobj *= self.hyp.obj

        bs = tobj.shape[0]
        loss = lbox + lcls + lobj
        return loss * bs, torch.cat((lbox, lcls, lobj)).detach()


    def build_targets(self, p, targets, model):
        # p：（nl,batch_size,anchors,gs,gs,cls+5]
        # targets:(image,cls,x,y,w,h)
        det = model.module.model[-1] if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) \
            else model.model[-1]  # Detect() module
        na, nt = det.na, targets.shape[0]  # 单个检测头的预选框数量，物体框数量
        tcls, tbox, indices, anch = [], [], [], []  # 定义输出
        gain = torch.ones(7, device=targets.device)
        at = torch.arange(na).view(na, 1).repeat(1, nt).to(targets.device)  # [na,nt] y方向递增网格grid
        targets = torch.cat((targets.repeat(na, 1, 1), at[..., None]),2)  # append anchor indices[anchors_num,targets_num,7(img,cls,x,y,w,h,anchor)]
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=targets.device).float() * g  # offsets

        for i in range(det.nl):  # Detect头数量
            anchors = det.anchors[i]  # 该层检测头的预选框，已经在Model初始化时将其缩放至gridsize
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # [1,1,grid_w,grid_h,grid_w,grid_h]

            # 将真实框与预选框进行匹配
            t = targets * gain  # 将targets的x,y,w,h转为检测头grid_size的大小
            a = []  # 预选框索引
            offsets = 0
            if nt:  # 有物体
                radio = t[..., 4:6] / anchors[:, None]  # 将每一个真实框的wh分别除以检测头内的每一个anchors的wh，得到wh的比值[na,nt,2]
                index_t_head_anch = torch.max(radio, 1. / radio).max(2)[0] < self.hyp.anchor_t  # 获取合适比值的索引即属于该检测头的真实框各对应预选框的索引[na,nt]：torch.max(r,1./r)过滤小的比值，将小比值变大，相当于将比值限定在1/anchors_t  ~~  anchors_t
                a = at[index_t_head_anch]  # 真实框对应的预选框[0,0,1,2,2]  0表示第一个预选框，1表示第二个，以此类推
                t = t[index_t_head_anch]  # 根据索引获取属于这一层检测头的真实框[[box0],[box0],[box1],[box2],[box2]]

                gxy = t[:, 2:4]  # 真实框在grid中的中点
                gxi = gain[[2, 3]] - gxy  # inverse
                z = torch.zeros_like(gxy)

                # 将框中心点扩增到上下左右四个方格
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # 真实框中点x，y分别位于各小方格左边，上边，且不属于左边和上边的第一排方格
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # 真实框中点x，y分别位于各小方格右边，下边，且不属于右边和下边的第一排方格
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]    # 将属于上下左右的框中心整合到一起
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 分别对左上右下补偿xyxy->gg-g-g
            else:
                t = targets[0]
                offsets = 0

            # Define
            imc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (im, c) = a.long().view(-1), imc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()  # 原中心框不变，靠近左边的t[lf]向左边移动0.5，落入左边方格内，靠近上边的t[tp]向上移动0.5，落入上边方格内，以此类推
            gi, gj = gij.T                  #grid indices    框中心在grid中的坐标

            # Append
            indices.append((im, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box真实框 [中心点相对方格右上点的坐标，真实框长宽]
            anch.append(anchors[a])  # anchors 根据索引获取各预测框对应预选框
            tcls.append(c)  # class
        return tcls, tbox, indices, anch
```

+ 添加v5分割损失函数

```python
class V5SegmentLoss:
    """Computes the YOLOv5 model's loss components including classification, objectness, box, and mask losses."""

    def __init__(self, model, autobalance=False, overlap=False):
        """Initializes the compute loss function for YOLOv5 models with options for autobalancing and overlap
        handling.
        """
        self.sort_obj_iou = False
        self.overlap = overlap
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.obj_pw], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = self.smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets

        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, h.gr, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.nm = m.nm  # number of masks
        self.anchors = m.anchors
        self.device = device

    def __call__(self, preds, batch):  # predictions, batch, model
        """Evaluates YOLOv5 model's loss for given predictions, batch, returns total loss components."""
        p, proto = preds if len(preds) == 2 else (preds[2], preds[1])
        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lseg = torch.zeros(1, device=self.device)

        img = batch["batch_idx"].unsqueeze(1)
        cls = batch["cls"] if batch["cls"].ndim == 2 else batch["cls"].unsqueeze(1)
        box = batch["bboxes"]
        masks = batch["masks"].to(self.device).float()
        new_targets = torch.cat((img, cls, box),-1)
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, new_targets.to(self.device))


        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1, self.nc, nm), 1)  # subset of predictions

                # Box regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                marea = xywhn[i][:, 2:].prod(1)  # mask width, height normalized
                mxyxy = xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device))
                for bi in b.unique():
                    j = b == bi  # matching index
                    if self.overlap:
                        mask_gti = torch.where(masks[bi][None] == tidxs[i][j].view(-1, 1, 1), 1.0, 0.0)
                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji #* self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp.v5_box
        lobj *= self.hyp.obj
        lcls *= self.hyp.v5_cls
        lseg *= self.hyp.v5_box / bs

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()
    
    def smooth_BCE(self, eps=0.5):
        # 返回类别正负样本的值 0< eps < 2
        return 1. - 0.5 * eps, 0.5 * eps

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Calculates and normalizes single mask loss for YOLOv5 between predicted and ground truth masks."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n,32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

    def build_targets(self, p, targets):
        """Prepares YOLOv5 targets for loss computation; inputs targets (image, class, x, y, w, h), output target
        classes/boxes.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
        gain = torch.ones(8, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        if self.overlap:
            batch = p[0].shape[0]
            ti = []
            for i in range(batch):
                num = (targets[:, 0] == i).sum()  # find number of targets of each image
                ti.append(torch.arange(num, device=self.device).float().view(1, num).repeat(na, 1) + 1)  # (na, num)
            ti = torch.cat(ti, 1)  # (na, nt)
        else:
            ti = torch.arange(nt, device=self.device).float().view(1, nt).repeat(na, 1)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], ti[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, at = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            (a, tidx), (b, c) = at.long().T, bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tidxs.append(tidx)
            xywhn.append(torch.cat((gxy, gwh), 1) / gain[2:6])  # xywh normalized

        return tcls, tbox, indices, anch, tidxs, xywhn
```

### 自适应预选框

1. ultralytics.utils目录下创建autoanchor.py，该文件用于yolov5自适应瞄框

```python
import random
import numpy as np
import torch
import yaml
from ultralytics.utils import TryExcept, LOGGER

#region 检查预选框顺序是否与检测头大小对应,输入检测层
def check_anchors_order(m):
    area_anchors_detect = m.anchors.prod(-1).mean(-1).view(-1)   #每个检测头内所有anchors的平均面积
    da = area_anchors_detect[-1] - area_anchors_detect[0]   #平均面积顺序排列，最后一个减第一个，要么小于0要么大于0
    ds = m.stride[-1] - m.stride[0]    #输入图像/检测头输出的比值顺序排列（比值大，平均面积大），最后一个减第一个，要么小于0要么大于0
    if da and (da.sign() != ds.sign()): #da！=0 且顺序反向
        m.anchors[:] = m.anchors.flip(0)   #反向排序
#endregion
#region 检查预选框是否符合数据集聚类
@TryExcept("AutoAnchor: ERROR")
def check_anchors(dataset, model, thr=4.0, img_sz=640):
    m = model.model[-1]    #检测头
    shapes = np.array(dataset.shapes)
    shapes = img_sz * shapes / shapes.max(1,keepdims=True)    #将原图像尺寸shape最长边改为img_sz，短边适应
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 随机0.9-1.1的图像尺寸比例

    wh = torch.tensor(np.concatenate([l[:, 2:4] * s for s, l in zip(shapes * scale, dataset.bboxes)])).float()  # 将labels的box宽高缩放到img_sz且随机0.9-1.1缩放

    def metric(k):
        r = wh[:, None] / k[None]          #每一个wh分别除以n个anchors
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors  输入图像的anchors乘以各自检测头大小与输入图像的比值，获得输入图像的anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    LOGGER.info(f'{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). \n')
    if bpr > 0.98:
        LOGGER.info('Current anchors are a good fit to dataset ✅\n')
    else:
        LOGGER.warning('Anchors are a poor fit to dataset ⚠️, attempting to improve...\n')
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(dataset, n=na, img_size=img_sz, thr=thr, gen=1000, verbose=False)

        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchors_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            LOGGER.info('Done ✅ (optional: update model *.yaml to use these anchors in the future)\n')
        else:
            LOGGER.warning('Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)')

def kmean_anchors(dataset="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    Creates kmeans-evolved anchors from training dataset.

    Arguments:
        dataset: path to data.yaml, or a loaded dataset
        n: number of anchors
        img_size: image size used for training
        thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
        gen: generations to evolve anchors using genetic algorithm
        verbose: print all results

    Return:
        k: kmeans evolved anchors

    Usage:
        from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr
    PREFIX = 'AutoAnchor: '
    def metric(k, wh):  # compute metrics
        """Computes ratio metric, anchors above threshold, and best possible recall for YOLOv5 anchor evaluation."""
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        """Evaluates fitness of YOLOv5 anchors by computing recall and ratio metrics for an anchor evolution process."""
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        """Sorts and logs kmeans-evolved anchor metrics and best possible recall values for YOLOv5 anchor evaluation."""
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = (
            f"{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n"
            f"{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, "
            f"past_thr={x[x > thr].mean():.3f}-mean: "
        )
        for x in k:
            s += "%i,%i, " % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k


    # Get label wh
    shapes = np.array(dataset.shapes)
    shapes = img_size * shapes / shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 2:4] * s for s, l in zip(shapes, dataset.bboxes)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f"{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size")
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f"{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...")
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init")
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in range(gen):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            LOGGER.info(f"{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}")
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
```

2. ultralytics.models.yolo.detect.train

+ 在DetectionTrainer类中修改get_dataloder方法，使其输出loader和dataset

```python
def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        loder = build_dataloader(dataset, batch_size, workers, shuffle, rank)  #  dataloader
        return loder, dataset
```

3. ultralytics.engine.trainer

+ 在BaseTrainer类的_setup_train函数中修改get_dataloader函数的调用输出，使其适应lader和dataset两输出， 并添加自适应预选框代码

```python
def _setup_train(self, world_size):
    ...
    ...
    # Dataloaders
    batch_size = self.batch_size // max(world_size, 1)
    self.train_loader, dataset = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
    if RANK in {-1, 0}:
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
        )[0]
        ...
        ...

    #v5自适应预选框
    if not self.args.resume and self.args.task in ["v5segment",  "v5detect"]: #V5检测任务
        if not self.args.noautoanchor:
            check_anchors(dataset, model=self.model, thr=self.args.anchor_t, img_sz=self.args.imgsz)  # run AutoAnchor
```

4. ultralytics.data.dataset

+ 在YOloDataset类的get_labels函数中添加self.shapes和self.bboxes用于自适应预选框的计算

```python
def get_labels(self):
    ...
    ...
    self.im_files = [lb["im_file"] for lb in labels]  # update im_files
    self.shapes = [lb["shape"] for lb in labels]   #get shapes
    self.bboxes = [lb["bboxes"] for lb in labels]  #get bboxes
    ...
    ...
    return labels
```

### yolov5最大值抑制

1. ultralytics.ultis.ops

+ 添加 yolov5的最大值抑制方法

```python
def v5_non_max_suppression(
        prediction,
        conf_thres = 0.35,
        iou_thres = 0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0):
    #prediction (bs, h*w*nl, 4+nc+nm)
    assert 0<=conf_thres<=1, "无效的置信度阈值"
    assert 0<=iou_thres<=1, "无效的IoU阈值"
    if isinstance(prediction, (list, tuple)):  #YOLOv8模型在验证时的输出为（inference_out, loss_out）
        prediction = prediction[0]  # 只选推理输出
    bs = prediction.shape[0]  #batch size
    nc = nc or prediction.shape[2] - 5  #种类数量
    nm = prediction.shape[2] - nc - 5
    xc = prediction[...,4] > conf_thres   #置信度大于阈值的索引

    max_wh = 7680       #最大的图像长宽
    max_nms = 30000   #计算nms时一张图像内最大检测目标数目
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6+nm), device=prediction.device)] * bs
    for img_i,x in enumerate(prediction):   #image index,  pred in a image
        x = x[xc[img_i]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[img_i]):
            lb = labels[img_i]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)  # append labels
        
        if not x.shape[0]:   #图像内无检测到框，下一张图像
            continue

        x[:,5:] *= x[:,4:5]  #类别概率乘以置信度
        

        box = xywh2xyxy(x[:,0:4])        #xywh  to xyxy
        mask = x[:, mi:]   #分割掩膜

        #[box conf cls]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5+j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf,j = x[:, 5:mi].max(1,keepdim = True)    #最大的置信度   类别索引
            x = torch.cat((box, conf, j.float(), mask),1)[conf.view(-1) > conf_thres]      #置信度大于阈值的[box conf cls]  box - xyxy

        #Filter by class
        if classes is not None:
            x = x[(x[:,5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:        #无目标，下一张图像
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        #NMS
        c = x[:,5:6] * (0 if agnostic else max_wh)  #类别 * 4096 放大类别差
        boxes, scores = x[:,:4] + c, x[:,4]     #将不同类别的框加上不同的偏差，进行区分，scores为各个框的置信度分数
        i = torchvision.ops.boxes.nms(boxes,scores,iou_thres)       #去除相同类别相近（iou > iou_thres)的框，并按置信度排序输出
        i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[img_i] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output # (bs, 6) xywh conf cls
```

### yolov5训练、验证、预测模块添加

1. utralytics.models.yolo添加v5detect文件夹, 内含train、val、predict、\_\_init\_\_

```
yolo
————v5detect
    ————__init__.py
    ————train.py
    ————val.py
    ————predict.py
```

+ \_\_init\_\_.py

```python
from .predict import V5DetectionPredictor
from .train import V5DetectionTrainer
from .val import V5DetectionValidator

__all__ = "V5DetectionPredictor", "V5DetectionTrainer", "V5DetectionValidator"
```

+ train.py

```python
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.v5detect.val import V5DetectionValidator
from ultralytics.nn.tasks import V5DetectionModel
from ultralytics.utils import RANK

class V5DetectionTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a detection model.

    Example:
        python
        from ultralytics.models.yolo.detect import V5DetectionTrainer

        args = dict(model="yolo1v5-anchhors.pt", data="coco8.yaml", epochs=3)
        trainer = V5DetectionTrainer(overrides=args)
        trainer.train()
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "v5detect"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = V5DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "obj_loss"
        return V5DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
```

+ val.py

```python
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

class V5DetectionValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a detection model.

    Example:
        python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        self.args.task = "v5detect"
        

    def postprocess(self, preds):
        """使用非最大值抑制处理预测结果"""
        return ops.v5_non_max_suppression(preds,
                                            self.args.conf,
                                            self.args.iou,
                                            labels=self.lb,
                                            multi_label=True,
                                            agnostic=self.args.single_cls,
                                            max_det=self.args.max_det)
```

+ predict.py

```python
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops

class DetectionPredictor(DetectionPredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.v5_non_max_suppression(
                preds, 
                self.args.conf, 
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
```

2. utralytics.models.yolo添加v5segment文件夹, 内含train、val、predict、\_\_init\_\_

```
yolo
————v5segment
    ————__init__.py
    ————train.py
    ————val.py
    ————predict.py
```

+ __init__.py

```python
from .predict import V5SegmentationPredictor
from .train import V5SegmentationTrainer
from .val import V5SegmentationValidator

__all__ = "V5SegmentationPredictor", "V5SegmentationTrainer", "V5SegmentationValidator"
```

+ train.py

```python
from copy import copy

from ultralytics.models import yolo
from ultralytics.models.yolo.v5segment.val import V5SegmentationValidator
from ultralytics.nn.tasks import  V5SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK

class V5SegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml", epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "v5segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = V5SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "obj_loss"
        return V5SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
```

+ val.py

```python

from ultralytics.utils import ops
from ultralytics_old.models.yolo.segment.val import SegmentationValidator

class V5SegmentationValidator(SegmentationValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml")
        validator = SegmentationValidator(args=args)
        validator()
        
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "v5segment"


    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p = ops.v5_non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
                nc=self.nc,)
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto
```

+ predict.py

```python
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class V5SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolov8n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "v5segment"

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.v5_non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
```

3. ultralytics,models.yolo.\_\_init\_\_

+ 引用yolov5库至__all__

```python
from ultralytics.models.yolo import classify, detect, obb, pose, segment, world, v5detect, v5segment

from .model import YOLO, YOLOWorld

__all__ = "classify", "segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld", "v5detect", "v5segment"
```

4. ultralytics.models.yolo.model

+ 在YOLO类的task_map函数中添加yolov5对应模型

```python
@property
def task_map(self):
    """Map head to model, trainer, validator, and predictor classes."""
    return {
        ...
        ...
        "v5detect":{
            "model": V5DetectionModel,
            "trainer": yolo.v5detect.V5DetectionTrainer,
            "validator": yolo.v5detect.V5DetectionValidator,
            "predictor": yolo.v5detect.V5DetectionPredictor,
        },
        "v5segment": {
            "model": V5SegmentationModel,
            "trainer": yolo.v5segment.V5SegmentationTrainer,
            "validator": yolo.v5segment.V5SegmentationValidator,
            "predictor": yolo.v5segment.V5SegmentationPredictor,
        },
    }
```

### 数据集适应yolov5

1. ultralytic.data.dataset

+ 修改初始化函数，使其兼容yolov5任务

```python
class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task in ["segment", "v5segment"] 
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)
```

