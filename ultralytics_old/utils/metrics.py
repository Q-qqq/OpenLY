import warnings

import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from ultralytics.utils import SimpleClass,plt_settings,TryExcept,LOGGER, yaml_save
from pathlib import Path
OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])/10.0
)

def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """计算box1和box2的交集面积与box2的面积的比值"""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    #交集面积
    inter_area = (np.minimum(b1_x2[:,None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    #box2面积
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = area2 + area1[:,None] - inter_area   #并集
    return inter_area / (area2 + eps)  #inter/box2   inter/union

def box_iou(box1, box2, eps=1e-7):
    """
    计算两个boxes之间的Iou
    :param box1: shape(N,4)
    :param box2: shape(M,4)
    :param eps:
    :return: shape(N,M)
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    #IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """计算box1(1,4)和box2(n,4)的iou"""
    if xywh:
        (x1, y1, w1, h1),(x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_  #box1 （xyxy）
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_  #box2  (xyxy)
    else: #xyxy
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    #交集
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    #并集
    union = w1 * h1 + w2 * h2 - inter + eps

    #Iou
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)   #box1和box2外接矩形的宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)   #box1和box2外接矩形的高度
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps    #外接矩形对角线的平方
            rho2 = (((b2_x2 +b2_x1) - (b1_x2 + b1_x1))**2 + ((b2_y2 + b2_y1) -(b1_y2 + b1_y1))**2)/4   #两box中心点距离
            if CIoU:
                v = ( 4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha) #CIoU
            return iou - rho2 / c2  #DIoU
        c_area = cw * ch + eps # 外接矩形面积
        return iou - (c_area - union)/c_area    #GIoU
    return iou

def _get_covariance_matrix(boxes):
    """
    生成obbs的协方差矩阵
    Args:
        boxes(torch.Tensor): shape(N,5), xywhr"""
    gbbs = torch.cat((torch.pow(boxes[:, 2:4], 2)/ 12, boxes[:, 4:]), dim=-1)   #长宽的平方除以12
    a, b, c = gbbs.split(1, dim=-1)
    return (a*torch.cos(c) ** 2 + b * torch.sin(c) ** 2,
            a*torch.sin(c) ** 2 + b * torch.cos(c) ** 2,
            a*torch.cos(c)*torch.sin(c) - b*torch.sin(c)*torch.cos(c))



def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    计算两个目标框之间的旋转iou
    Args:
        obb1(torch.Tensor): shape(N, 5)  xywhr
        obb2(torch.Tensor): shape(N, 5)  xywhr
    Returns:
        (torch.Tensor): shape(N,) 指示两个目标框相似度的iou值
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = ( ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2)* (torch.pow(x1 - x2 , 2)))
            / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) *0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (torch.log(
        ((a1+ a2) *(b1 +b2) - (torch.pow(c1+ c2, 2)))
        /(4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps) + eps) *0.5
          )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:  #只考虑wh
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * (torch.atan(w2/h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v *alpha #CIoU
    return iou

def batch_probiou(obb1, obb2, eps=1e-7):
    """计算定向边界框之间的prob Iou
    Args:
        obb1:(N,5)  真实框
        obb2:(M,5)  预测框
    Returns:
        (torch.Tensor):(N,M)"""
    obb1 =torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb1, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)   #(N,1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))   #(1,N)

    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
                 ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
                 / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
         ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
            torch.log(
                ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
                / (4 * torch.sqrt(
                    (a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
                + eps
            )
            * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd

def mask_iou(mask1, mask2, eps=1e-7):
    """
    计算分割掩膜IoU
    Args:
        mask1(torch.Tensor): shape(N, h*w), 真实目标
        mask2(torch.Tensor): shape(M, h*w), 预测目标
    Returns：
        (torch.Tensor): (N,M) 两两相对的IoU
        """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)  #矩阵乘法 （N，M） mask交集
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) -intersection  #并集  (area1+area2) - inter
    return intersection / (union + eps)

def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    计算目标关键点的相似度（OKS）
    Args:
        kpt1(torch.Tensor): shape(N, 17, 3)真实目标
        kpt2(torch.Tensor): shape(M, 17, 3)预测目标
        area(torch.Tensor): shape(N,) 真实目标面积
        sigma(list): 一个包含17个值的列表，分别对应17个点的权重
    Retruns:
        (torch.Tensor): shape(N,M)
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]) ** 2 + (kpt1[:, None, :, 1] - kpt2[..., 1]) ** 2  #(N, M, 17) #目标之间17点对应的距离
    sigma = torch.Tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  #(17,)
    kpt_mask = kpt1[..., 2] != 0  #(N, 17)  #可见的点
    e = d / (2*sigma)**2 / (area[:, None, None] + eps) / 2
    return (torch.exp(-e) * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)




def smooth(y, f=0.05):
    nf = round(len(y) *f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p* y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf)/ nf, mode="valid")

@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None):
    fig, ax = plt.subplots(1, 1, figsize=(9,6), tight_layout=True)
    py = np.stack(py, axis=1)
    if 0 < len(names) < 21:
        for i,y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  #plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  #plot{recall,precision)
    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlable="Confidence", ylabel="Metric", on_plot=None):
    fig,ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")   #plot(confidece, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all class {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlable)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidece Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)




def compute_ap(recall, precision):
    """
    根据给定的recall和precision计算average precision(AP)
    Args:
        recall(list, np.ndarray): 某一个置信度阈值下的回召率曲线(1,2,3,4,5,5...)/nl对应置信度越来越小
        precision(list, np.ndarray): 某一个置信度与之下的精确率曲线（1,2,3,4,5,5...）/(fp+tp) 对应置信度越来越小
    Returns:
        (float): Average precision
        (np.ndarray):
        (np.ndarray):修改了回召率曲线，在开始和结束的地方添加了定值
        """

    #在开始和结束的位置添加定值
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  #[0,0.3,0.1,0.5,0.4] -> [0, 0.3, 0.3,0.5,0.5]  确保mpre是一个递减的矢量

    method = "interp"  #"continuous" "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)   #101 point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  #计算0-1范围内的积分-面积
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]  #去除最后一个索引 0 - n-1
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])   #曲线下方面积
    return ap, mpre, mrec





def ap_per_class(
        tp,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        on_plot=None,
        save_dir=Path(),
        names=(),
        eps=1e-16,
        prefix=""
):
    """
    计算目标检测每个种类的平均精确率（average precision）
    Args:
        tp(np.ndarray): 一个二维矩阵，指示预测的目标是否检测准确（True）或者不准确（False） (n, 10) 0.5-0.95阈值下
        conf(np.ndarray): 置信度分数的矩阵
        pred_cls(np.ndarray): 预测种类矩阵
        target_cls(np.ndarray): 真实种类矩阵
        plot(bool, optional): 是否绘制PR曲线
        on_plot(func, optional)：一个用于回调的函数，当绘制曲线时，传递其路径和数据
        save_dir(Path, optional): 字典保存PR曲线路径
        names(tuple, optional): 种类名称
        prefix(str,optional):保存plot file时的开头字符串
    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
        ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
        unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
        p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves. Shape: (1000,).
        prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """
    #按置信度排序
    i = np.argsort(-conf)   #从大到小排序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    #找种类
    unique_classes, nt = np.unique(target_cls, return_counts=True)   #种类， 对应目标数量
    nc = unique_classes.shape[0]   #种类数量

    #创建precision-recall曲线
    x, prec_values = np.linspace(0, 1, 1000), []

    #Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  #对应种类真实目标数量
        n_p = i.sum()  #对应种类预测目标数量
        if n_p == 0 or n_l == 0:
            continue

        #累加拘束 FPs  TPs
        fpc = (1 - tp[i]).cumsum(0)    #每一列的累加和（每一列表示一个iou_thres)[1,1,1]   [1,2,3]
        tpc = tp[i].cumsum(0)

        #Recall
        recall = tpc / (n_l + eps)   #单一种类的回召率  预测的正例数量/真实目标数量
        #对于不同的置顶度范围0-1内，单一种类的回召率
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  #iou 0.5   #负号x,xp是因为conf是递减的，interp输入要的是递增的

        #Precision
        precision = tpc / (tpc + fpc)   #单一种类的精确率曲线   预测的正例数量/预测的总数量
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        #AP from recall_precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                prec_values.append(np.interp(x, mrec, mpre))   #mrec 0-1   mpre 1-0

    prec_values = np.array(prec_values)   #(nc,1000)

    #Compute F1
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]
    names = dict(enumerate(names))
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)
    i = smooth(f1_curve.mean(0), 0.1).argmax()  #max F1 index
    p , r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()  #对应种类的回召率*种类数量 获得该种类的正例数
    fp = (tp / (p+eps) - tp).round()  #反例数
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values




class Metric(SimpleClass):
    """计算Yolov8模型的评估指标"""
    def __init__(self):
        self.p = []  #(nc,)
        self.r = []  #(nc,)
        self.f1 = [] #(nc,)
        self.all_ap = []  #(nc, 10)
        self.ap_class_index = [] #(nc,)
        self.nc = 0

    @property
    def ap50(self):
        """返回所有种类在iou阈值0.5时的平均精确率（AP）"""
        return self.all_ap[:, 0] if len(self.all_ap) else []   #(nc ,0)

    @property
    def ap(self):
        """返回所有种类在iou阈值分别在0.5-0.95时的平均精确率（AP）"""
        return self.all_ap.mean(1) if len(self.all_ap) else []   #(nc, 0)

    @property
    def mp(self):
        """返回所有种类的平均精确率"""
        return self.p.mean() if len(self.p) else 0.0   #float

    @property
    def mr(self):
        """返回所有种类的平均召回率"""
        return self.r.mean() if len(self.r) else 0.0   #float

    @property
    def map50(self):
        """返回iou阈值未0.5时的平均精确率的平均(mAP)"""
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """返回iou阈值在0.75时的平均精确率的平均（mAP）"""
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0   #float

    @property
    def map(self):
        """返回iou阈值分别在0.5-0.95时的map的平均值"""
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """返回指标结果的均值：mp,mr,map50,map"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """返回指定种类的指标结果"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """每个种类的ap"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps         #np.array

    def fitness(self):
        """模型拟合能的数字化"""
        w = [0.0, 0.0, 0.1, 0.9]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """用一个新的结果集更新模型评估指标"""
        (self.p,
         self.r,
         self.f1,
         self.all_ap,
         self.ap_class_index,
         self.p_curve,
         self.r_curve,
         self.f1_curve,
         self.px,
         self.prec_values) = results

    @property
    def curves(self):
        """返回用于访问指定指标曲线的曲线列表"""
        return []

    @property
    def curves_results(self):
        """返回用于访问指定指标曲线的曲线列表"""
        return[
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"]
        ]




class DetMetrics(SimpleClass):
    """
    用于计算目标检测指标的通用类，例如precision, recall, map
    Args:
        save_dir(Path): 保存output plots的路径
        plot(bool):是否画出每一个种类的precision-recall曲线
        on_plot(func):
        names(tuple of str):种类名称
    """
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()):
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.metrics = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"

    def process(self, tp, conf, pred_cls, target_cls):
        """计算目标检测预测结果的评估指标并更新"""
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot
        )[2:]
        self.metrics.nc = len(self.names)
        self.metrics.update(results)

    @property
    def keys(self):
        """返回一个访问指定指标的关键字列表"""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-90(B)"]

    def mean_results(self):
        """返回平均指标 mprecision, mrecall, mAP50, mAP50-95"""
        return self.metrics.mean_results()

    def class_results(self, i):
        """返回指定种类的评估指标"""
        return self.metrics.class_result(i)

    @property
    def maps(self):
        """返回每个种类的mAP"""
        return self.metrics.maps

    @property
    def fitness(self):
        """返回模型拟合性能指标"""
        return self.metrics.fitness()

    @property
    def ap_class_index(self):
        """返回当前存在的种类索引"""
        return self.metrics.ap_class_index

    @property
    def results_dict(self):
        """返回结果字典"""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """返回访问指定指标曲线的曲线名称列表"""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """返回曲线数据结果 recall-precision, confidence-F1, confidence-precision, confidence-recall """
        return self.metrics.curves_results

class ConfusionMatrix:
    """
    对目标检测和分类任务计算和更新混淆矩阵
    Attributes：
        task(str): 'detect'/'classify'
        matricx(np.array): 混淆矩阵的维度取决于task
        nc(int): 种类数量
        conf(float): 目标置信度阈值
        iou_thres(float): IoU阈值
        im_files(list(list(list(pred_nc)*gt_nc)))): 存储预测种类和真实种类对应的图像文件im_files[pred_c][gt_c] = list()"""

    def __init__(self,nc, conf=0.25, iou_thres=0.45, task="detect"):
        self.task = task
        self.matrix = np.zeros((nc + 1, nc +1)) if self.task == "detect" else np.zeros((nc , nc))
        self.im_files = [[[] for i in range(nc+1)] for j in range(nc+1)] if self.task == "detect" else [[[] for i in range(nc)] for j in range(nc)]
        self.nc = nc
        self.conf = 0.25 if conf in(None, 0.001) else conf
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets, im_files):
        """更新分类任务的混淆矩阵
        Args:
            preds(Array[N, min(nc, 5)]): 预测的种类标签
            targets(Array[M, 1]): 真实的种类标签
            im_files(List(List(str))): 图像文件路径，一个子list表示一个batch"""
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        im_files = [f for fs in im_files for f in fs]   #等同view(-1)
        for p, t, im_file in zip(preds.cpu().numpy(), targets.cpu().numpy(), im_files):
            self.matrix[p][t] += 1
            self.addImFile(p, t, im_file)

    def addImFile(self, pred_cls, gt_cls, im_file):
        """添加对应种类的图像文件到self.im_files中
        Args:
            pred_cls: im_file或其中中某个目标预测的种类
            gt_cls: im_file或其中某个目标真实的种类
            im_file: 目标文件
        """
        if im_file not in self.im_files[pred_cls][gt_cls]:
            self.im_files[pred_cls][gt_cls].append(im_file)


    def process_batch(self, detections, gt_bboxes, gt_cls, im_file):
        """更新目标检测任务的混淆矩阵
        Args:
            detections(Array[N,6]): 预测的目标检测框boxes和其相关信息conf、cls （x1,y1,x2,y2,conf,cls）
            gt_bboxes(Array[M,4])：真实目标框xyxy
            gt_cls(Array[M]):真实目标框对应的种类
            im_file(str): 图像文件路径
        """
        if gt_cls.size(0) == 0:
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detections_classes = detections[:,5].int()
                for dc in detections_classes:
                    self.matrix[dc, self.nc] += 1  #过杀  检测出来了，但没有对应真实目标
                    self.addImFile(dc, self.nc, im_file)
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1 #漏检   有目标，没检测出来
                self.addImFile(self.nc, gc, im_file)
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detections_classes = detections[:, 5].int()
        iou = box_iou(gt_bboxes, detections[:, :4])    #(M，N)
        x = torch.where(iou > self.iou_thres)   # 获取满足iou条件的框的位置（row(gt) coloum(detections)）
        if x[0].shape[0]:     #存在匹配项
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  #shape(n. 3)  gt detections iou
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]   #排序 iou从大到小
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]   #一个预测框对应一个真实框, 取分数最高的
                matches = matches[matches[:,2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]   #一个真实框对应一个预测框
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)   #按列分割 索引
        for i, gc in enumerate(gt_classes):
            j = m0 == i   #对应第i个真实框索引
            if n and sum(j) == 1:
                self.matrix[detections_classes[m1[j]], gc] += 1 #correct  #m1[j]第i个真实框对应的预测框索引
                self.addImFile(detections_classes[m1[j]], gc, im_file)
            else:
                self.matrix[self.nc, gc] += 1   #漏检   有目标，没检测出来
                self.addImFile(self.nc, gc, im_file)
        if n:
            for i, dc in enumerate(detections_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  #过杀  检测出来了，但没有对应真实目标
                    self.addImFile(dc, self.nc, im_file)

    def matrix(self):
        """返回混淆矩阵"""
        return self.matrix

    def tp_fp(self):
        """返回正例和反例"""
        tp = self.matrix.diagonal()   #对角线处为正例
        fp = self.matrix.sum(1) - tp  #反例
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)   #对目标检测移除背景类

    @TryExcept("WARNING ⚠️ 绘制混淆矩阵失败")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """绘制混淆矩阵并保存"""
        import seaborn as sn
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)   #按列归一化
        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
        nc ,nn = self.nc, len(names)   #所有种类数量， 现有种类数量
        sn.set(font_scale=0.6 if nc < 50 else 0.3)
        labels = (0 < nn < 99) and (nn == nc)   #应用种类名称到ticklabels
        ticklabels = [str(i) for i in range(len(names))] if labels else "auto"
        ticklabels = ticklabels + [str(len(names))] if labels and self.task=="detect" else ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  #suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 20,
                annot_kws={"size":8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1,1,1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True",fontsize=8)
        ax.set_ylabel("Predicted",fontsize=8)
        ax.set_title(title)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ','_')}.png"
        fig.savefig(plot_fname, dpi=150)
        plt.close(fig)
        if normalize:   #保存一次
            self.saveImageFiles(save_dir, names)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """输出混淆矩阵文本"""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def saveImageFiles(self, save_dir, names):
        save_dict = {}
        names = list(names)
        if self.task == "detect":
            names.append("null")
        for pred_c in range(len(self.im_files)):
            for gt_c in range(len(self.im_files[pred_c])):
                save_dict[f"pred-{names[pred_c]},true-{names[gt_c]}${pred_c},{gt_c}"] = self.im_files[pred_c][gt_c]
        yaml_save(Path(save_dir) /"Confusion_Matrix_Imfiles.yaml",save_dict)



class SegmentMetrics(SimpleClass):
    """计算分割目标评估指标
    Args:
        save_dir(Path): 保存路径
        plot(bool):是否绘制, 默认False
        names(list):种类名称
    Attributes:
          box(Metric): 计算目标框评估指标的类
          seg(Metric): 计算分割掩膜的评估指标的类
          speed(dict):存储推理时间的字典
    """
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()):
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {"preprocess":0.0, "inference":0.0, "loss":0.0, "postprocess":0.0}
        self.task = "Segment"

    def process(self, tp, tp_m, conf, pred_cls, target_cls):
        """计算目标框和分割掩膜的评估指标
        Args:
            tp(list):正例框
            tp_m(lsit): 正例掩膜
            conf(list): 正例对应置信度
            pred_cls(list): 正例对应预测种类
            target_cls(list): 真实目标种类
        """

        results_masks = ap_per_class(tp_m,
                                     conf,
                                     pred_cls,
                                     target_cls,
                                     plot=self.plot,
                                     on_plot=self.on_plot,
                                     save_dir=self.save_dir,
                                     names=self.names,
                                     prefix="Mask",
                                     )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_masks)
        results_box = ap_per_class(tp,
                                   conf,
                                   pred_cls,
                                   target_cls,
                                   plot=self.plot,
                                   on_plot=self.on_plot,
                                   save_dir=self.save_dir,
                                   names=self.names,
                                   prefix="Box",
                                   )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """返回访问评估指标的关键字列表B-box，M-mask"""
        return ["metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(M)",
                "metrics/recall(M)",
                "metrics/mAP50(M)",
                "metrics/mAP50-90(M)"]

    def mean_results(self):
        """返回box和seg的平均指标，box(mp、mr、map50，map) seg(mp、mr、map50、map)"""
        return self.box.mean_results() + self.seg.mean_results()

    def class_results(self,i):
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        return self.box.ap_class_index

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        return ["Precision-Recall(B)",
                "F1-Confidence(B)",
                "Precision-Confidence(B)",
                "Recall-Confidence(B)",
                "Precision-Recall(M)",
                "F1-Confidence(M)",
                "Precision-Confidence(M)",
                "Recall-Confidece(M)"]

    @property
    def curves_results(self):
        return self.box.curves_results + self.seg.curves_results

class PoseMetric(SegmentMetrics):
    """
    Attributes:
        save_dir(Path): 保存输出plots的路径
        plot(bool):是否绘制目标，默认False
        names(list): 种类名称
        box(Metric): 计算目标框指标的实例
        pose(Metric): 计算关键点指标的实例
        speed(dict): 存储推理速度的字典
    """
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()):
        super().__init__(save_dir, plot, names)
        self.save_dir = save_dir
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.pose = Metric()
        self.speed = {"preprocess":0.0, "inference":0.0, "loss":0.0, "postprocess":0.0}

    def process(self, tp, tp_p, conf, pred_cls, target_cls):
        """基于所给数据计算指标
        Args:
            tp(np.ndarray): 正例boxes  （N, 10）, N个目标，10个置信度阈值0.5-0.95, 满足阈值为1.不满足为0
            tp_p(np.ndarray)： 正例keypoints  (N, 10)
            conf(np.ndarray): (N, )目标对应的置信度
            pred_cls(np.ndarray): （N， ）预测种类
            target_cls(np.ndarray): (N, )真实种类
        """
        results_pose = ap_per_class(tp_p,
                                    conf,
                                    pred_cls,
                                    target_cls,
                                    plot=self.plot,
                                    on_plot=self.on_plot,
                                    save_dir=self.save_dir,
                                    names=self.names,
                                    prefix="Pose")[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)

        results_box = ap_per_class(tp,
                                   conf,
                                   pred_cls,
                                   target_cls,
                                   plot=self.plot,
                                   on_plot=self.on_plot,
                                   save_dir=self.save_dir,
                                   names=self.names,
                                   prefix="Box")[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        return ["metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(P)",
                "metrics/recall(P)",
                "metrics/mAP(P)",
                "metrics/mAP50-95(P)"]

    def mean_results(self):
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self,i):
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        return self.box.fitness() + self.pose.fitness()

    #property
    def curves(self):
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self):
        return self.box.curves_results + self.pose.curves_results


class OBBMetrics(SimpleClass):
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()):
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference":0.0, "loss":0.0, "postprocess":0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        results = ap_per_class(tp,
                               conf,
                               pred_cls,
                               target_cls,
                               plot=self.plot,
                               save_dir=self.save_dir,
                               names=self.names,
                               on_plot=self.on_plot)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        return ["metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)"]

    def mean_results(self):
        return self.box.mean_results()

    def class_result(self, i):
        return self.box.class_result(i)

    @property
    def maps(self):
        return self.box.maps

    @property
    def fitness(self):
        return self.box.fitness()

    @property
    def ap_class_index(self):
        return self.box.ap_class_index

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return []


class ClassifyMetrics(SimpleClass):
    """计算分类矩阵包含前1-5的准确率"""
    def __init__(self):
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess":0.0, "inference":0.0, "loss":0.0, "postprocess":0.0}
        self.task="classify"

    def process(self, targets, pred):
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:,0], correct.max(1).values), dim=1)
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        return ["metrics/accuracy_top1",
                "metrics/accuracy_top5"]

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return []