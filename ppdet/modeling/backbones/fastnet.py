import math
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec
from paddle import Tensor
from typing import List

__all__ = ['FastNet']

FastNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
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


class Partial_conv3(nn.Layer):
    def __init__(self, dim, n_div, forward):
        super(Partial_conv3, self).__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2D(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        # FIXME: This is a hack to avoid the bug in paddle.split
        x = x.clone(
        )  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(
            x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = paddle.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = paddle.concat((x1, x2), 1)

        return x


class BasicBlock(nn.Layer):
    def __init__(self, dim, n_div, mlp_ratio, drop_path, layer_scale_init_value,
                 act_layer, norm_layer, pconv_fw_type):
        super(BasicBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        self.n_div = n_div
        hidden_dim = int(dim * mlp_ratio)
        layers: List[nn.Layer] = [
            nn.Conv2D()(dim, hidden_dim, 1, bias=False), norm_layer(hidden_dim),
            act_layer(), nn.Conv2D(
                hidden_dim, dim, 1, bias=False)
        ]
        self.layers = nn.Sequential(*layers)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        if layer_scale_init_value > 0:
            # FIXME: This is a hack to avoid the bug
            self.layer_scale = ParamAttr(
                layer_scale_init_value * paddle.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layers(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.layers(x))
        return x
