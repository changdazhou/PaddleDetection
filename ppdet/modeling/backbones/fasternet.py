import math
from numbers import Integral

import paddle
import paddle.nn as nn
from ..shape_spec import ShapeSpec
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

__all__ = ['FasterNet']

MODEL_cfg = {
    'fasternet_t0': dict(
        embed_dim=40,
        depths=[1, 2, 8, 2],
        drop_path_rate=0.0,
        act_layer=nn.GELU),
    'fasternet_t1': dict(
        embed_dim=64,
        depths=[1, 2, 8, 2],
        drop_path_rate=0.02,
        act_layer=nn.GELU),
    'fasternet_t2': dict(
        embed_dim=96,
        depths=[1, 2, 8, 2],
        drop_path_rate=0.05,
        act_layer=nn.ReLU),
    'fasternet_s': dict(
        embed_dim=40,
        depths=[1, 2, 8, 2],
        drop_path_rate=0.0,
        act_layer=nn.GELU),
    'fasternet_m': dict(
        embed_dim=40,
        depths=[1, 2, 8, 2],
        drop_path_rate=0.0,
        act_layer=nn.GELU),
    'fasternet_l': dict(
        embed_dim=40,
        depths=[1, 2, 8, 2],
        drop_path_rate=0.0,
        act_layer=nn.GELU),
}


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, training=True, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PConv(nn.Layer):
    def __init__(self, dim, kernel_size=3, n_div=4):
        super(PConv, self).__init__()

        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2D(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias_attr=False)

    def forward(self, x):

        x1, x2 = paddle.split(x, [self.dim_conv, self.dim_untouched], axis=1)
        x1 = self.conv(x1)
        x = paddle.concat([x1, x2], axis=1)

        return x


class BasicBlock(nn.Layer):
    def __init__(self,
                 dim,
                 expand_ratio=2,
                 act_layer=nn.ReLU,
                 drop_path_rate=0.0):
        super(BasicBlock, self).__init__()

        self.pconv = PConv(dim)

        self.conv1 = nn.Conv2D(dim, dim * expand_ratio, 1, bias_attr=False)

        self.bn = nn.BatchNorm2D(dim * expand_ratio)
        self.act_layer = act_layer()

        self.conv2 = nn.Conv2D(dim * expand_ratio, dim, 1, bias_attr=False)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)

        x = residual + self.drop_path(x)
        return x


@register
@serializable
class FasterNet(nn.Layer):
    def __init__(self,
                 return_idx=[0, 1, 2, 3],
                 in_channel=3,
                 embed_dim=40,
                 act_layer=nn.ReLU,
                 depths=[1, 2, 8, 2],
                 drop_path=0.0):
        super().__init__()

        if act_layer == 'GELU':
            act_layer = nn.GELU
        else:
            act_layer = nn.ReLU

        self.stem = nn.Sequential(
            nn.Conv2D(
                in_channel, embed_dim, 4, stride=4, bias_attr=False),
            nn.BatchNorm2D(embed_dim),
            act_layer())
        if isinstance(return_idx, list):
            return_idx = [(x * 2) for x in return_idx]
        self.return_idx = return_idx

        drop_path_list = [
            x.item() for x in paddle.linspace(0, drop_path, sum(depths))
        ]

        self._out_channels = [ x * embed_dim for x in [1, 2, 4, 8]]

        self.features = []
        embed_dim = embed_dim
        for idx, depth in enumerate(depths):
            self.features.append(
                nn.Sequential(* [
                    BasicBlock(
                        embed_dim,
                        act_layer=act_layer,
                        drop_path_rate=drop_path_list[sum(depths[:idx]) + i])
                    for i in range(depth)
                ]))

            if idx < len(depths) - 1:
                self.features.append(
                    nn.Sequential(
                        nn.Conv2D(
                            embed_dim,
                            embed_dim * 2,
                            2,
                            stride=2,
                            bias_attr=False),
                        nn.BatchNorm2D(embed_dim * 2),
                        act_layer()))

                embed_dim = embed_dim * 2
                
        self.features = nn.Sequential(*self.features)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i])
            for i in range(len(self._out_channels))
        ]

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.features):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

def FasterNet_t0():
    return FasterNet(MODEL_cfg['fasternet_t0'])

def FasterNet_t1():
    return FasterNet(MODEL_cfg['fasternet_t1'])

def FasterNet_t2():
    return FasterNet(MODEL_cfg['fasternet_t2'])

def FasterNet_s():
    return FasterNet(MODEL_cfg['fasternet_s'])

def FasterNet_m():
    return FasterNet(MODEL_cfg['fasternet_m'])

def FasterNet_l():
    return FasterNet(MODEL_cfg['fasternet_l'])

if __name__ == '__main__':
    import paddle
    paddle.set_device('gpu')
    model = FasterNet_t0()
    paddle.summary(model, (1, 3, 224, 224))
