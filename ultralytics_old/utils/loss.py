import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import TaskAlignedAssigner,RotatedTaskAlignedAssigner, bbox2dist,make_anchors,dist2bbox, dist2rbox
from ultralytics.utils.metrics import bbox_iou, probiou,OKS_SIGMA
from ultralytics.utils.ops import smooth_BCE, xywh2xyxy,xyxy2xywh,crop_mask
from ultralytics.utils.torch_utils import de_parallel


class BboxLoss(nn.Module):
    """计算目标检测框的损失"""
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Iou Loss"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)   #(b, n_anchors, 1)  种类分数
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)   #计算预测框和目标框的IoU
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum   #iou以种类分数为权重后归一化

        #DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) *weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        tl = target.long() #target left  5.6 -> 5
        tr = tl + 1 #target right        5 > 6
        wl = tr - target #weight left    6-5.6 -> 0.4
        wr = 1- wl  #weight right        1-0.4 -> 0.6
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)         #交叉熵 softmax - log

class RotatedBboxLoss(BboxLoss):
    """计算旋转框损失"""
    def __init__(self, reg_max, use_dfl=False):
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        #DFL
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max+1), target_ltrb[fg_mask]) * weight   #剔除角度，只计算ltrb
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

class KeypointLoss(nn.Module):
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2  #两点之间的距离
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)   #(obj_num, 1)  nkpt/visible_num
        e = d / (2 * self.sigmas) **2 / (area + 1e-9) / 2
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

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
        lobj *= self.hyp.dfl

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
                index_t_head_anch = torch.max(radio, 1. / radio).max(2)[0] < self.hyp.iou_t  # 获取合适比值的索引即属于该检测头的真实框各对应预选框的索引[na,nt]：torch.max(r,1./r)过滤小的比值，将小比值变大，相当于将比值限定在1/anchors_t  ~~  anchors_t
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
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets

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

class v8DetectionLoss:
    """计算v8目标检测训练损失的标准类"""
    def __init__(self, model): #模型必须去并行化
        device = next(model.parameters()).device  #驱动
        h = model.args  #超参数

        m = model.model[-1] #Detect()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride   #各检测头的步长
        self.nc = m.nc     #number of classess
        self.no = m.no     #输出通道数
        self.reg_max = m.reg_max   #输出box的层数  输出的每一层对结果边长一半所占倍数的分布
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner =TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        预处理(排序，取消归一化)真实目标计数并匹配批处理大小输出一个新的targets（batch, n_max_box, n_anchors）
        Args:
            targets(Tensor):shape(n,6) 6:batch_idx, cls, bboxes
            batch_size(int): 批大小
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]   #image index
            _, counts = i.unique(return_counts=True)   #按顺序排列的图像索引各有几个（每个图像有几个标签）
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j   #与索引j相同的图像
                n = matches.sum()  #存在数量
                if n:
                    out[j, :n] = targets[matches, 1:]      #[cls1 box4]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor)) #取消归一化->xyxy  shape(batch_size, counts.max, 5）
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """从anchor_points 和 distribution对预测的目标框进行解码"""
        if self.use_dfl: # reg_max>1
            b, a, c = pred_dist.shape  #batch, w*h*head_num, channels(reg_max*4)
            pred_dist = pred_dist.view(b, a, 4, c//4).softmax(3).matmul(self.proj.type(pred_dist.dtype))   #(b,a,4)  DFLmodel
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def __call__(self, preds, batch):
        """
            累加batch大小的box、cls和dfl的损失
            preds:(head_num, batch, no, h, w)
        """
        loss = torch.zeros(3, device=self.device)  # box cls dfl
        feats = preds[1] if isinstance(preds, tuple) else preds   #模型输出
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )         #  box,  classes   pred_distri(batch, self.reg_max*4, w*h*head_num) pred_scores(batch, self.nc, w*h*head_num)

        pred_scores = pred_scores.permute(0,2,1).contiguous()  #(batch, w*h*head_num, self.nc)
        pred_distri = pred_distri.permute(0,2,1).contiguous()  #(batch, w*h*head_num, self.reg_max*4)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  #模型输出大小*stride = 输入图像大小 image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)   #获取网络坐标点，offset0.5->坐标点偏移至四点中间


        #Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1,1), batch["bboxes"]), 1)  #batch,cls ,box
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1,0,1,0]])   #取消targets的归一化并按图像分类targets(batch,max_box_count,5)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  #cls(batch,max_box_count,1), xyxy(batch,max_box_count,4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  #大于0的为1，小于0的为0 有框的为1.无框的为0  (batch, max_box_count)

        #Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  #(batch, head_num*w*h, 4)  xyxy

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,  #将检测头的描点映射到真实图像上，stride越大越稀疏
            gt_labels,
            gt_bboxes,
            mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)   #取真实目标种类分数总和1的最大值

        #cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  #BCE

        #Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  #box gain
        loss[1] *= self.hyp.cls  #cls gain
        loss[2] *= self.hyp.dfl  #dfl gain

        return loss.sum() * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device) #box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape   #barch, num_mask, mask_height, mask_width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  #预测的box_dist(batch,reg_max*4, h*w*head_num), 预测的cls_scores(batch, nc, h*w*head_num)

        #8,grids,..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   #batch, n_anchors, nc
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()   #batch, n_anchors, reg_max*4
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()     #batch, n_anchors, num_mask(32)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        #Target
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1,1), batch["bboxes"]), 1)  #(num_targets, 6)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1,0,1,0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  #cls, box(xyxy)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) #大于0的为1，小于0的为0 有框的为1.无框的为0  (batch, max_box_count)
        except RuntimeError as e:
            raise TypeError(
                "检查数据集是否分割数据集格式或者网络结构是否分割网络结构"
            ) from e

        #Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)   #dist -> xyxy
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )       #筛选预选框对应的真实框，去掉无对应真实框的预选框

        target_scores_sum = max(target_scores.sum(), 1)  #分数总和，小于1则取1

        #cls loss
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  #BCE

        if fg_mask.sum():
            #Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            #Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):        #downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]   #将mask缩放至proto输出的大小
            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap)
        else:
            loss[1] += (proto *0).sum() + (pred_masks * 0).sum()  #无穷小的相加可能导致损失值nan

        loss[0] *= self.hyp.box  #box gain
        loss[1] *= self.hyp.box  #seg gain
        loss[2] *= self.hyp.cls  #cls gain
        loss[3] *= self.hyp.dfl  #dfl gain
        return loss.sum() * batch_size, loss.detach()

    def single_mask_loss(self,gt_mask, pred, proto, xyxy, area):
        """
        计算单个图像的分割损失
        Args:
            gt_mask(torch.Tensor): shape(n, H, W) 真实mask, n是目标数量
            pred(torch.Tensor): shape(n, 32) 预测的mask系数
            proto(torch.Tensor): shape(32, H, W) 原型masks
            xyxy(torch.Tensor): shape(n, 4)，xyxy格式的真实框，已归一化"
            area(torch.Tensor): shape(n,)，真实框的面积
        Returns:
            (torch.Tensor): 单个图像的分割损失
            """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  #矩阵乘法 (n, 32)  @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")  #交叉熵   #(n, 80, 80)
        return (crop_mask(loss, xyxy).mean(dim=(1,2))/area).sum()   #去除目标框外的损失值



    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, overlap):
        """
        累计分割损失
        Args:
            fg_mask(torch.Tensor): shape（b, n_anchors） 二值mask，标识正预选点位置
            masks(torch.Tensor): 真实masks，如果不使用overlap shape(b, h, w)，否则shape(b,？,h, w)
            target_gt_idx(torch.Tensor): shape(b, n_anchors)每个预选点对应的真实目标索引
            target_bboxes(torch.Tensor):shape(b, n_anchors,4) 每个预选点对应的真实框
            batch_idx(torch.Tensor): shape(n_labels_in_batch, 1)  标签对应批次索引
            proto(torch.Tensor): shape(b, 32, h, w) hw是网络输出的mask大小，不等于输入图像hw
            pred_masks(torch.Tensor): shape(b, n_anchors, 32) 每个预选点预测出来的mask
            imgsz(torch.Tensor): shape(2) i.e(H,W) 输入图像大小
            overlap(bool): Whether the masks in `masks` tensor overlap.
        Returns:
            (torch.Tensor): 分割损失
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        #归一化
        target_bboxes_normalized = target_bboxes / imgsz[[1,0,1,0]]

        #真实框面积
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        #将真实框缩放至mask大小
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_mask_i, proto_i, mxyxy_i, marea_i, masks_i = single_i     #一张图像内的数据
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]  #真实目标索引
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)   #将重叠的masks分开
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask,pred_mask_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto*0).sum() + (pred_masks *0).sum()  #损失0  没有真实目标
        return loss / fg_mask.sum()   #损失均值


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)
    def preprocess(self, targets, batch_size, scale_tensor):
        '''预处理：将target取消归一化，并分图像存储到各个维度'''
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  #image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i== j   #一张图像
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        将预测的ltrb格式框和预选点解码为xywhr格式框
        Args:
            anchor_points(torch.Tensor): shape(n_anchors, 2)  预选点
            pred_dist(torch.Tensor): shape(b, n_anchors, reg_max *4)  预测ltrb
            pred_angle(torch.Tensor): shape(b, n_anchors, 1) 预测角度
        Returns:
            (torch.Tensor):
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c//4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)  #xywhr


    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  #box cls dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  #b, 1, n_anchors
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #b,n_anchors, num_class
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #b, n_anchors, reg_max*4
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()  #b, n_anchors, 1

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  #image size (h, w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  #n_anchors, 2

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1,5)), 1)  #n, 7
            rw, rh = targets[:, 4] *imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  #去除面积小于4个像素的目标
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1,0,1,0]])  #b, n_mboxes, 6
            gt_labels, gt_bboxes = targets.split((1, 5), 2)   #cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB数据集格式错误") from e

        #Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)   #xywhr. (b, n_anchors, 5)

        bboxes_for_assigner = pred_bboxes.clone().detach()

        bboxes_for_assigner[..., 4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        #cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  #BCE

        #Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor  #归一化
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box #box gain
        loss[1] *= self.hyp.cls #cls gain
        loss[2] *= self.hyp.dfl #dfl gain

        return loss.sum() * batch_size, loss.detach()  #loss(box, cls, dfl)

class v8PoseLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # 关键点数量
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts):
        """计算关键点累计损失和关键点目标损失，关键点目标损失是一种二分类损失，指示关键点是否存在
        Args:
            fg_mask(torch.Tensor): shape(b, n_anchors) 二值mask，指示预选点上是否存在目标
            target_gt_idx(torch.Tensor): shape(b, n_anchors) 每个预选点对应的真实目标索引
            keypoints(torch.Tensor): shape(n_object_in_batch, nkpt(per_object), ndim)  真实的关键点
            batch_idx(torch.Tensor): shape(n_object_in_batch, 1)           每个真实关键点对应的图像索引
            stride_tensor(torch.Tensor): shape(n_anchors, 1) 预选点对应的将输出图缩放至原图大小的比例尺
            target_bboxes(torch.Tensor): shape(b, n_anchors, 4) 预选点对应的真实框 xyxy
            pred_kpts（torch.Tensor）: shape(b, n_anchors, nkpt, ndim) 预测的关键点"""

        batch_idx = batch_idx.flatten()
        batch_size = len(fg_mask)
        nkpt, ndim = keypoints.shape[1:]
        #单张图像内最多存在的关键点目标数量
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        #将对应图像的关键点移动到对应的batch_size
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, nkpt, ndim), device=keypoints.device
        )
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        selected_keypoints = batched_keypoints.gather(1, target_gt_idx_expanded.expand(-1, -1, nkpt, ndim))   #(b,n_anchors, nkpt, ndim) 获取每个预选点对应的关键点目标
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)  #归一化

        #loss
        kpts_loss = 0
        kpts_obj_loss = 0

        if fg_mask.any():
            gt_kpt = selected_keypoints[fg_mask]  #obj_num, nkpt, ndim
            area = xyxy2xywh(target_bboxes[fg_mask])[:, 2:].prod(1,keepdim=True)  #w*h
            pred_kpt = pred_kpts[fg_mask]         #obj_num, nkpt, ndim
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)  #可见性mask  （obj_num, nkpt, 1）
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  #关键点损失，关于面积

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())   #关键点目标可见性损失
        return kpts_loss, kpts_obj_loss


    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)  #box,  kpt_location, kpt_visibility, cls, dfl
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[1].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max*4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   #b, n_anchors, num_class
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()   #b, n_anchors, reg_max*4
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()       #b, n_anchors, nk


        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]   #image size (h, w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1,0,1,0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  #cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  #xyxy (b, n_anchors, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        #cls loss
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        #Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)   #box, dfl
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]  # x  w
            keypoints[..., 1] *= imgsz[0]  # y  h

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts)

        loss[0] *= self.hyp.bpx #box gain
        loss[1] *= self.hyp.pose #pose gain
        loss[2] *= self.hyp.kobj  #kobj gain
        loss[3] *= self.hyp.cls #cls gain
        loss[4] *= self.hyp.dfl  #dfl gain

        return loss.sum() * batch_size, loss.detach()

class v8ClassificationLoss:
    def __call__(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


