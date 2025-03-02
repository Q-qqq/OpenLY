import numpy as np
import torch
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import profile

def check_train_batch_size(model, imgsz=640, amp=True):
    """自动选择最优batch"""
    with torch.cuda.amp.autocast(amp):
        return

def autobatch(model, imgsz=640, fraction=0.60, batch_size=DEFAULT_CFG.batch):
    """
    根据可用的CDA内存选择子最优的batch
    Args:
        model (torch.nn.module): YOLO model to compute batch size for.
        imgsz (int, optional): The image size used as input for the YOLO model. Defaults to 640.
        fraction (float, optional): The fraction of available CUDA memory to use. Defaults to 0.60.
        batch_size (int, optional): The default batch size to use if an error is detected. Defaults to 16.

    Returns:
        (int): The optimal batch size.
    """
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}为imgsz={imgsz}计算最优batch size")
    device = next(model.parameters()).device
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA 未使用，CPU使用默认batch-size{batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:  #最优卷积算法
        LOGGER.info(f"{prefix} ⚠️ 要求torch.backends.cudnn.benchmark=False，使用默认的batch-size{batch_size}")
        return batch_size

    #Inspect CUDA memory
    gb = 1 << 30  #byte to GB
    d = str(device).upper()
    properties = torch.cuda.get_device_properties(device)  #驱动信息
    t = properties.total_memory / gb  #total
    r = torch.cuda.memory_reserved(device) / gb  #缓存
    a = torch.cuda.memory_allocated(device) / gb  #已分配
    f = t - (r + a)  #free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    #Profile batch sizes
    batch_sizes = [1,2,4,8,16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device= device)

        #Fit a solution
        y = [x[2] for x in results if x]   #缓存
        p = np.polyfit(batch_sizes[: len(y)], y, deg=1)   #2项式拟合，返回系数
        b = int((f * fraction - p[1]) / p[0])   #优化的batch size
        if None in results:
            i = results.index(None)
            if b >= batch_sizes[i]:
                b = batch_sizes[max(i-1, 0)]   #选择安全的节点
        if b < 1 or b >1024:
            b = batch_size
            LOGGER.info(f"{prefix}WARNING ⚠️ CUDA检测异常，使用默认的batch-size{batch_size}")

        fraction = (np.polyval(p, b) + r + a) / t
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}WARNING ⚠️ 检测错误：{e},使用默认的batch-size{batch_size}")
        return batch_size


