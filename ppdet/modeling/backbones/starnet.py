import math
import paddle
import paddle.nn as nn
from numbers import Integral
from ..shape_spec import ShapeSpec
from paddle.nn import AdaptiveAvgPool2D, Linear
from paddle import ParamAttr
from paddle.nn.initializer import Constant

from ppdet.core.workspace import register, serializable
from .transformer_utils import DropPath, trunc_normal_, zeros_, ones_

__all__ = ['StarNet']

class ConvBN(paddle.nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        if with_bn:
            self.bn = nn.BatchNorm2D(out_planes,weight_attr=ParamAttr(initializer=Constant(1.)),bias_attr=ParamAttr(initializer=Constant(0.)))
    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return x

class Block(nn.Layer):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

@register
@serializable
class StarNet(nn.Layer):
    def __init__(self, 
                 base_dim=32,
                 depths=[3, 3, 12, 5], 
                 mlp_ratio=4, 
                 drop_path_rate=0.0, 
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3], 
                 num_stages=4,
                 freeze_stem_only=False,
                 **kwargs):
        super().__init__()
        self.in_channel = 32
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        self._out_channels = [ x * base_dim for x in [1, 2, 4, 8]]
        self._out_strides = [0, 1, 2, 3]
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        stages = []
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            stages.append(nn.Sequential(down_sampler, *blocks))
        self.stages = nn.Sequential(*stages)

        self.apply(self._init_weights)
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, num_stages)):
                    self._freeze_parameters(self.stages[i])

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,nn.Conv2D)):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm,nn.BatchNorm2D)):
            zeros_(m.bias)
            ones_(m.weight)
            
    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.stop_gradient = True

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


def starnet_s1(pretrained=False, **kwargs):
    return StarNet(24, [2, 2, 8, 3], **kwargs)


def starnet_s2(pretrained=False, **kwargs):
    return StarNet(32, [1, 2, 6, 2], **kwargs)


def starnet_s3(pretrained=False, **kwargs):
    return StarNet(32, [2, 2, 8, 4], **kwargs)


def starnet_s4(pretrained=False, **kwargs):
    return StarNet(32, [3, 3, 12, 5], **kwargs)


# very small networks #
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)