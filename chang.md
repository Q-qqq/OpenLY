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
        if self.task == "detect":
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

+ 修改TryExcept修饰器，使其可以在程序出错时强制中断进度条

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
            if self.verbose:
                LOGGER.error(f"{self.msg}{': ' if self.msg else ''}{value}")
            if PROGRESS_BAR.loadiing:
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
        try:
            for data_i, self.batch in enumerate(self.dataset):
                ...
                ...
                self.run_callbacks("on_predict_batch_end")
                yield from self.results
                PROGRESS_BAR.setValue(data_i+1, f"{s} {profilers[1].dt *1e3:.1f}ms")
                if PROGRESS_BAR.isStop():
                    PROGRESS_BAR.close()
                    break
            PROGRESS_BAR.close()
        except Exception as ex:
            PROGRESS_BAR.stop()
            PROGRESS_BAR.close()
            raise ProcessLookupError(f"预测失败：{ex}")
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

+ 往全部变量中添加CFG_OTHERS_KEYS, 使KEYS全局变量包含全部参数

```python
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
    "copy_paste_mode", #(str) "flip"  the method to do copy_paste augmentation (flip, mixup)
    "auto_augment",     #(str)randaugment, auto augmentation policy for classification (randaugment, autoaugment,augmix)
    "cfg",       #(str,optional) for overriding defaults.yaml
    "tracker",   #(str)bootsort.yaml, tracker type, choices=[botsrt.yaml, bytetrack.yaml]
    "resume",    # (bool|str)False, resume training from last checkpoint
    "imgsz",     #(int | list)640, image size  width,height
)
```

## 修改七：LOGGER

### ultralytics.utils.__init__

+ 添加Logging类，并实例化为LOGGER， 替换原先的LOGGER. 其中定义了多个信号对训练、验证信息进行界面显示

```python
# Set logger
_LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
class Logger:
    """信息显示"""
    Show_Mes_Signal = Signal(str)
    Start_Train_Signal = Signal(list)
    Batch_Finish_Signal = Signal(str)
    Epoch_Finish_Signal = Signal(list)
    Train_Finish_Signal = Signal(str)
    Train_Interrupt_Signal = Signal()
    Start_Val_Signal = Signal(str)
    Val_Finish_Signal = Signal(str)
    Error_Signal = Signal(str)
    def __init__(self, parent=None):
        super(Logger, self).__init__(parent)
        self.errorFormat = '<font color="red" size="5">{}</font>'
        self.warningFormat = '<font color="orange" size="5">{}</font>'
        self.stop = False  #停止训练


    def error(self,msg):
        """错误信号"""
        _LOGGER.error(msg)
        errorMsg = self.errorFormat.format(msg)
        self.Error_Signal.emit(msg)
        self.Show_Mes_Signal.emit(errorMsg)

    def warning(self,msg):
        """警告信号"""
        _LOGGER.warning(msg)
        warningMsg = self.warningFormat.format(msg)
        self.Show_Mes_Signal.emit(warningMsg)

    def info(self,msg):
        """正常信号"""
        _LOGGER.info(msg)
        self.Show_Mes_Signal.emit(msg)

    def startTrain(self, msg_epochs):
        """开始训练信号"""
        _LOGGER.info(msg_epochs[0])
        self.Start_Train_Signal.emit(msg_epochs)

    def batchFinish(self, msg):
        """完成一个batch信号"""
        _LOGGER.info(msg)
        self.Batch_Finish_Signal.emit(msg)

    def epochFinish(self, msg_epoch):
        """完成一个epoch信号"""
        _LOGGER.info(msg_epoch[0])
        self.Epoch_Finish_Signal.emit(msg_epoch)


    def trainFinish(self, msg):
        """训练结束信号"""
        _LOGGER.info(msg)
        self.Train_Finish_Signal.emit(msg)

    def trainInterrupt(self):
        """训练停止信号"""
        _LOGGER.info(msg)
        self.Train_Interrupt_Signal.emit()

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
            if LOGGER.stop:
                LOGGER.trainInterrupt()
                raise ProcessLookupError(f"Interrupt：训练中断成功,已训练{epoch}epoch")
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

## 修改八：验证图像增加输出shape

### ultralytics.data.utils

+ 在verify_image函数中添加图像shape的输出

```python
def verify_image(args):
    ...
    ...
    return (im_file, cls), nf, nc, msg, shape
```

## 修改九： 检测det数据集默认父路径修改

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

