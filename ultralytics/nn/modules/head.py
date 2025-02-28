import math
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL, Proto
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import make_anchors, bbox2dist, dist2bbox, dist2rbox

__all__ = (
    "Detect",
    "Segment", 
    "Pose",
    "Classify",
    "OBB",
    #"RTDETRDecoder",
)
class V5Detect(nn.Module):
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

                if isinstance(self, V5Segment):  # (boxes + masks)
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

class V5Segment(V5Detect):
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

class Detect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc   # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16 # DFL channels
        self.no = nc + self.reg_max * 4   #number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides 在建立的过程中计算出来
        c2, c3 = max((16,ch[0]//4, self.reg_max * 4)), max(ch[0], min(self.nc, 100)) #channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.reg_max*4, 1)) for x in ch
        )   #bbox
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)  #class
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """x(list(Tensor)): nl个检测头的输入"""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)   #b, h, w, reg_max*4+class_num
        if self.training:
            return x

        #Inference
        shape = x[0].shape  #batch, channel, height, weight
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  #no = channel  List(batch, h*w*nl, nc + reg_max*4)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max *4]
            cls = x_cat[:, self.reg_max*4:]
        else:
            box, cls = x_cat.split((self.reg_max*4, self.nc), 1)
        dbox = self.decode_bboxes(box)   # dist to box xywh

        if self.export and self.format in ("tflite", "edgetpu"):
            #计算归一化提升稳定性
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1,4,1)
            norm = self.strides / (self.strides[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)
        y = torch.cat((dbox, cls.sigmoid()), 1) #xywh nc*cls
        return y if self.export else (y, x)      #y(b, h*w*nl, xywh+nc)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

    def bias_init(self):
        m = self #self.model[-1] Detect()
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0 #box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  #cls

class Segment(Detect):
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm #number of masks
        self.npr = npr #number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])  #x[0]:shape(b, ch[0],h, w) - > (b, nm ,2h, 2w)
        bs = p.shape[0]   #batch size
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2) # mask 系数  #(batch, nm, h*w*nl)
        x = self.detect(self, x)  #box classses
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc ], 1), (x[1], mc, p))  #shape (b h*w*nl, xywh+nc+nm,)

class OBB(Detect):
    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne #number if extra parameters
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0] #batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)   #OBB theta logits  b , ne, h*w*num_head
        #note: 将‘angle’当作一个属性，‘decode_bboxes’就能使用它
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides

class Pose(Detect):
    def __init__(self, nc=80, kpt_shape=(17,3), ch=()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  #number of keypoints, number of dims(2 for x, y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  #number of keyppoints total
        self.detect = Detect.forward
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x ,c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]  #batch
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  #(bs, 17*3(nk), h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs,kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt]))


    def kpts_decode(self, bs, kpts):
        """解码 keypoints"""
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2:3] = y[:, 2::3].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y

class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        #c2种类数
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to (b, c_, 1, 1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)   #to x(b, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


