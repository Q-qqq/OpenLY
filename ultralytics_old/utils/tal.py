import torch
import torch.nn as nn
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_iou,probiou
from ultralytics.utils.ops import xywhr2xyxyxyxy

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """从feats中生成anchors"""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape  #检测头 宽高
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))   #网格坐标点(w*h , 2)
        stride_tensor.append(torch.full((h*w, 1), stride, dtype=dtype, device=device))  #h*w个stride
    return torch.cat(anchor_points),torch.cat(stride_tensor)   #(head_num * w*h, 2)   (head_num*w*h, 1)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Args:
        distance: (batch,w*h*head_num,4)
        anchor_points: (head_num*w*h,2)
    Returns:
        (batch, head_num* w*h, 4)
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)  #模型输出图像大小的box

def bbox2dist(anchor_points, bbox, reg_max):
    """
    将目标框转换为相对预选点的ltrb
    Args:
        anchor_points(Tensor)：（n_anchors, 2）
        bbox(Tensor)：(b,n_anchors, 4) xyxy  每个预选点对应的真实框
        reg_max（int）：
    Returns:
        (Tensor):(b,n_anchors, 4)  ltrb
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  #dist(lt, rb)   regmax为最大的边的一半，

def dist2rbox(pred_dist, pred_angle, anchor_points,dim=-1):
    """
    根据anchor points 和 distribution 预测预测的目标框坐标
    Args:
        pred_dist (torch.Tensor): Predicted distance, (bs, n_anchors, 4).  ltrb
        pred_angle (torch.Tensor): Predicted angle, (bs, n_achors, 1).
        anchor_points (torch.Tensor): Anchor points, (n_anchors, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, n_anchors, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)   #left top    right bottom
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)

    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf *sin + yf * cos   #中心点到预选点的xy方向距离
    xy = torch.cat([x, y], dim=dim) + anchor_points  #中心点坐标
    return torch.cat([xy, lt+rb], dim=dim)   #xywh

class TaskAlignedAssigner(nn.Module):
    """
    获取真实框对应预选点的索引Tensor
    获取真实框bboxes和one-hot格式的种类分数
    根据预测框和真实框的Iou指标（定位）和预测种类分数（分类）计算新的真实框种类分数
    Attributes:
        topk(int): 考虑的最佳候选人数量
        num_classes(int): 目标种类数量
        alpha(float):任务对齐指标的分类参数
        beta(float): 任务对齐指标的定位参数
        eps(float): 接近0的数值，避免除0
    """
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        计算任务对齐分配
        n_anchors = h*w*head_num
        Args:
            pd_scores(Tensor): shape(bs, n_anchors, num_classes)  分类分数
            pd_bboxes(Tensor): shape(bs, n_anchors, 4)   预测定位框  （xyxy）
            anc_points(Tensor): shape(n_anchors, 2)    预选框中心点 -> 预选点
            gt_labels(Tensor): shape(bs, n_max_boxes, 1)       真实种类
            gt_bboxes(Tensor): shape(bs, n_max_boxes, 4)       真实框
            mask_gt(Tensor): shape(bs, n_mboxes, 1)        掩膜
        Returns:
            target_labels(Tensor): shape(bs, n_anchors)
            target_bboxes(Tensor): shape(bs, n_anchors, 4)
            target_scores(Tensor): shape(bs, n_anchors, num_classes)
            fg_mask(Tensor): shape(bs, n_anchors)      预选框是否有对应的真实框
            target_gt_idx(Tensor): shape(bs, n_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device)
            )
        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        #获取位于真实框内且IoU指标前topk个的预选点mask   每个预选点都会对应一个预测框，根据预测款计算Iou
        #(b, n_mboxes, n_anchors) (b, n_mboxes, n_anchors)
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)  #获取在真实框内的前topk个预选点 - 1个真实框对应topk个预选点
        #(b, n_anchors) (b, n_anchors) (b, n_mboxes, n_anchors)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)   #获取预选点和真实框1对1的mask
        #(b, n_anchors) (b, n_anchors, 4) (b, n_anchors, num_classes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)   #获取各个预选框对应的真实框种类、位置、one_hot

        #归一化
        align_metric *= mask_pos   #(b, n_mboxes, n_anchors)
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  #(b, n_mboxes)  真实框对应的最大的总指标
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True) #(b, n_mboxes) 真实框对应的最大的Iou指标
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)  #(b, n_mboxes, 1) 归一化
        target_scores = target_scores * norm_align_metric #修改真实框分数，结合了预测结果的种类分数0-1

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx




    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """获取每个真实框其内的预选点，且根据种类分数和iou指标只取前topk个预选点"""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  #获取在真实框内的预选点mask  (bs, n_mboxes, n_anchors)   一个真实框对应n_anchors个预选点
        #(b, n_mboxes, n_anchor) = (b,n_mboxes, n_anchors)*(b, n_mboxes,1)
        mask  = mask_in_gts * mask_gt       #乘以真实框的掩膜获取各真实框对应的预选点 -- 一对n
        #(b, n_mboxes, n_anchors)，(b, n_mboxes, n_anchors)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask)   #获取预测框和真实框的Iou指标   每一个真实框对应n个预选框中心点
        #(b, n_mboxes, n_anchors)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())   #获取每个真实框取前topk个最大指标预选点的mask  一对topk
        #Merge all mask to a final mask (b, n_mboxes, n_anchors)
        mask_pos = mask_topk * mask
        return mask_pos, align_metric, overlaps



    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        获取位于真实框内的预选点的mask，因为一个预选点可能同时在多个真实框内，所以每个真实框跟n个预选点计算是否在框内，结果（b, n_mboxes, n_anchors）
        Args：
            xy_centers(Tensor): shape(n_anchors, 2)预选点
            gt_bboxes(Tensor): shape(b, n_mboxes, 4)
        Returns:
            (Tensor): shape(b, n_mboxes, n_anchors)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  #left_top, right_bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)    #shape(n_mboxes*bs, n_anchors, 4)  在真实框内的预选点 delta大于0
        bbox_deltas = bbox_deltas.view(bs, n_boxes, n_anchors, -1)   #b, n_mboxes, n_anchors, 4
        return bbox_deltas.amin(3).gt_(eps)   #获取deltas的最小值并与eps比较，小于eps的为0，排除不在真实框内的预选点（b, n_mboxes, n_acnhors）

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """计算预测框和真实框的对齐指标
        Returns:
            align_metric(Tensor): shape(b, n_mboxes, n_anchors)  结合种类预测分数和Iou的总指标
            overlaps(Tensor): shape(b, n_mboxes, n_anchors)  预测框和真实框的Iou指标"""
        na = pd_bboxes.shape[-2]  #n_anchors
        mask_gt = mask_gt.bool()  #b, n_mboxes, n_anchors
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype = pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype,device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  #2, b, n_mboxes
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)   #b, n_mboxes    [[0,0,0..],[1,1,1..],...]
        ind[1] = gt_labels.squeeze(-1)   #b, n_mboxes
        #将预测的种类分数赋值bbox_scores           ind[0]批次对应所有anchors的种类ind[1]的预测分数
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]   #(b,n_mboxes, n_anchors)[mask_gt] = ((b,n_mboxes), n_anchors,(b, n_mboxes))[mask_gt]

        #计算真实框内每个预测框与真实框的IoU
        pd_bboxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]    #(b, n_mboxes, n_anchors, 4)[mask_gt]  -> (m, 4)
        gt_bboxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]                  #(b, n_mboxes, n_anchors, 4)[mask_gt]  -> (m, 4)
        overlaps[mask_gt] = self.iou_calculation(gt_bboxes, pd_bboxes)
        #计算种类预测分数的alpha次方乘以iou的beta次方，为总指标
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return bbox_iou(gt_bboxes, pd_bboxes, xywh = False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        根据指标选出每个真实框对应的前topk个最大/最小指标的预选点
        Args:
            metrics(Tensor):shape(b, n_mboxes, n_anchors)
            largest(bool):若为真，选择最大的数值，否则选择最小的数值
            topk_mask(Tensor):shape(b, n_mboxes, topk)
        Return:
            mask(Tensor):shape(b, n_mboxes, n_anchors)每个真实框最多对应topk个预选点，即n_anchors个数据中最多只有topk个不为0
        """
        #(b, n_mboxes, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)   #获取指定维度顺序排列后的前topk个数值及其序号索引，顺序由largest指定
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        #(b, n_mboxes, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)  #topk_idx中 对应于topk_mask为1的地方用0填充  去除真实框外的索引

        #（b, n_mboxes, topk, n_anchors) -> (b, n_mboxes, n_anchors)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)  #（b, n_mboxes, n_anchors）
        ones = torch.ones_like(topk_idxs[:,:, :1], dtype=torch.int8, device=topk_idxs.device)  #(b, n_mboxes,1)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, : , k:k+1], ones)   #每个真实框对应指定索引的预选点数量+1+1+1
        count_tensor.masked_fill_(count_tensor > 1, 0)   #去除真实框外的(0)和在候选外的预选框(1)均替换为0，其余的都是根据topk指标选上的预选点,每个真实框对应topk个预选点
        return count_tensor.to(metrics.dtype)   #(b, n_mboxes, n_anchors) 每个真实框最多对应topk个预选点，即n_anchors个数据中最多只有topk个不为0

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        如果一个预选点对应多个真实框，有着最高IOU的真实框将被选中
        Args：
            mask_pos(Tensor): shape(b, n_mboxes,n_anchors)
            overlaps(Tensor): shape(b, n_mboxes, n_anchors)
        Return:
            target_gt_idx(Tensor): shape(b, n_anchors)  预选点对应真实框的索引
            fg_mask(Tensor): shape(b, n_anchors)      每个预选点对应的真实框数量
            mask_pos(Tensor): shape(b, n_mboxes, n_anchors)  一个预选点对应一个真实框的mask
        """
        fg_mask = mask_pos.sum(-2)  #（b,n_anchors） 预选点对应真实框数量
        if fg_mask.max() > 1:  #一个预选点匹配了多个真实框
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)   #（b, n_mboxes， n_anchors）  匹配多个真实框的预选点为1
            max_overlaps_idx = overlaps.argmax(1)   #(b, n_anchors)     获取每个预测框与所有真实框的最大Iou的索引

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  #（b, n_mboxes, n_anchors)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)   #（b, n_mboxes, n_anchors）  将最大iou的预选点置为1 ，其余为0

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()    #(b, n_max_boxes, n_anchors) 匹配多个真实框的预选点取is_max_overlaps对应位置的值，其余的取mask_pos原值
            fg_mask = mask_pos.sum(-2)  #(b, n_anchors)   预选点有框的为1，无框的为0
        target_gt_idx = mask_pos.argmax(-2)  #(b, n_anchors)  每个预选点对应的真实框索引
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """

        Args:
            gt_labels(Tensor): 真实目标框种类（b,n_mboxes,1)
            gt_bboxes(Tensor): 真实目标框位置（b,n_mboxes,4)
            target_gt_idx(Tensor): 预选点对应真实框的索引（b,n_anchors)
            fg_mask（Tensor）：一个布尔值张量，显示有预选框的位置（b, n_anchors）
        Returns:
            target_labels(Tensor): 对应正预选框的真实框种类 （b, n_anchors）
            target_bboxes(Tensor): 对应正预选框的真实框位置 （b, n_anchors, 4)
            target_scores(Tensor): one-hot格式的真实框种类分数，值为1 （b, n_anchors, num_classes)
        """
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device )[..., None]   #[[0],[1],[2]...]
        target_gt_idx = target_gt_idx + batch_ind*self.n_max_boxes    #(b, n_anchors)  index + bs*n_mboxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]     #获取预选点对应真实框种类标签 (b, n_anchors)

        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]  #获取预选点对应真实框位置标签（b, n_anchors, 4)

        target_labels.clamp_(0)
        #one_hot
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device
        )  #(b, n_anchors, number of classes)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)  #d对应种类位置赋值1（one-hot）

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  #(b, n_anchors, num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)   #预选点没有对应真实框的全归为0

        return target_labels, target_bboxes, target_scores

class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return  probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        获取位于真实框内的预选点的mask，因为一个预选点可能同时在多个真实框内，所以每个真实框跟n个预选点计算是否在框内，结果（b, n_mboxes, n_anchors）
        Args：
            xy_centers(Tensor): shape(n_anchors, 2)
            gt_bboxes(Tensor): shape(b, n_mboxes, 4)
        Returns:
            (Tensor): shape(b, n_mboxes, n_anchors)
        """
        #(b, n_boxes,5)   -->   (b, n_boxes, 4, 2)
        #xywhr -> xy xy xy xy
        corners = xywhr2xyxyxyxy(gt_bboxes)
        #(b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        #(b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap *ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  #点在旋转框



