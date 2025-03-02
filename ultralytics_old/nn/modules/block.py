import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, LightConv,RepConv, DWConv, GhostConv
from ultralytics.nn.modules.transformer import TransformerBlock


__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
)
class DFL(nn.Module):
    """
        Distribution Focal Loss（DFL）的集成模块

    """
    def __init__(self, c1=16):
        """使用c1初始化Conv"""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1,bias=False).requires_grad_(False)   #输入通道c1，输出通道1，卷积块大小1
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1,c1,1,1))    # 卷积核参数设置为0，1，2，3，4，5....(c1-1)
        self.c1 = c1

    def forward(self, x):
        b, c ,a = x.shape   #batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b,4,a)   #4*c1=c


class Proto(nn.Module):
    """Yolov8用于分割models的mask原始module  stride=2"""
    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2,2,0,bias=True)   #反卷积 上采样HW 2倍
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self,x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class HGStem(nn.Module):
    """
        使用了5个卷积和1个最大池化层的stemBlock(of PPHGNetV2)
    """
    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm//2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm//2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm*2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm,c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])  #手动填充
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class HGBlock(nn.Module):
    """
        使用了两个标准卷积和LightConv的HG_Block(of PPHGNetV2)
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i==0 else cm, cm, k=k,act=act) for i in range(n))
        self.sc = Conv(c1 + n *cm, c2 // 2, 1, 1, act=act)    #降维
        self.ec = Conv(c2//2, c2, 1, 1, act=act)          #升维
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class SPP(nn.Module):
    """金字塔池化层"""
    def __init__(self, c1, c2, k=(5,9,13)):
        super().__init__()
        c_ = c1//2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k)+1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    """金字塔池化层-Fast"""
    def __init__(self, c1, c2, k=5):
        #This module is equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1//2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x,y1,y2,self.m(y2)),1))

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1,c2, shortcut=True, g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g = g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x+self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C1(nn.Module):
    #使用1个标准卷积的CSP Bottleneck
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        y = self.cv1(x)
        return self.m(y) + y

class C2(nn.Module):
    #使用2个标注卷积的CSP Bottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1,2 *self.c, 1, 1)
        self.cv2 = Conv(2*self.c, c2, 1)   #可选act = FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0)for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2,1)  #切割
        return self.cv2(torch.cat((self.m(a), b), 1))

class C2f(nn.Module):
    """更快速的C2"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2+n) *self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k= ((3,3),(3,3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))

    def forward_split(self, x):
        #使用split替代chunk
        y = list(self.cv1(x).split((self.c, self.c),1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y,1))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2*c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_,c_, shortcut, g, k=((1,1),(3,3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)),self.cv2(x)), 1))

class C3x(C3):
    """交叉卷积的C3"""
    def __init__(self,c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int (c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut,g, k=((1,3),(3,1)), e=1) for _ in range(n)))

class RepC3(nn.Module):
    """Rep C3"""
    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

class C3TR(C3):
    """带注意力机制(TransformerBlock())的C3"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2//2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  #pw
            DWConv(c_, c_, k, s, act=False) if s==2 else nn.Identity(), #dw
            GhostConv(c_, c2, 1, 1, act=False) #pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s==2 else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)



class BottleneckCSP(nn.Module):
    """CSP Bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2*c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2*c_)
        self.act=nn.SiLU()
        self.m = nn.Sequential( * (Bottleneck(c_,c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class ResNetBlock(nn.Module):
    """残差模块 with standard convolution Layers"""
    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        c3 =e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s!=1 or c1 !=c3 else nn.Identity()

    def forward(self, x):
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))

class ResNetLayer(nn.Module):
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()
        self.is_first = is_first
        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n-1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layer(x)


class C3Ghost(C3):
    """使用GhostBottleneck()的C3module"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

