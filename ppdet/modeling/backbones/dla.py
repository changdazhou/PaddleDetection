# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

DLA_cfg = {34: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512]), }


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvNormLayer(
            ch_in,
            ch_out,
            filter_size=3,
            stride=stride,
            bias_on=False,
            norm_decay=None)
        self.conv2 = ConvNormLayer(
            ch_out,
            ch_out,
            filter_size=3,
            stride=1,
            bias_on=False,
            norm_decay=None)

    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs

        out = self.conv1(inputs)
        out = F.relu(out)

        out = self.conv2(out)

        out = paddle.add(x=out, y=residual)
        out = F.relu(out)

        return out
    
def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.full(shape=[], fill_value=1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ConvBN(paddle.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1,
        padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_sublayer(name='conv', sublayer=paddle.nn.Conv2D(
            in_channels=in_planes, out_channels=out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups))
        if with_bn:
            self.add_sublayer(name='bn', sublayer=paddle.nn.BatchNorm2D(
                num_features=out_planes))
            init_Constant = paddle.nn.initializer.Constant(value=1)
            init_Constant(self.bn.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(self.bn.bias)
    
class StarBasic(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, dilation=1,mlp_ratio=3, drop_path=0, **cargs):
        super(StarBasic, self).__init__()
        self.dwconv = ConvBN(inplanes, inplanes, 7, 1, (7 - 1) // 2, groups=inplanes,
            with_bn=True)
        self.f1 = ConvBN(inplanes, mlp_ratio * inplanes, 1, with_bn=False)
        self.f2 = ConvBN(inplanes, mlp_ratio * inplanes, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * inplanes, inplanes, 1, with_bn=True)
        self.dwconv2 = ConvBN(inplanes, inplanes, 7, 1, (7 - 1) // 2, groups=inplanes,
            with_bn=False)
        self.act = paddle.nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else paddle.nn.Identity()
        self.convbn = ConvBN(inplanes,planes,kernel_size=3,stride=stride,padding=dilation,dilation=dilation,with_bn=True)

    def forward(self, x,residual=None):
        if residual is None:
            residual = x
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        x = self.convbn(x)
        x = x + residual
        return x


class Root(nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2D(
            ch_in,
            ch_out,
            1,
            stride=1,
            bias_attr=False,
            padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2D(ch_out)
        self.residual = residual

    def forward(self, inputs):
        children = inputs
        out = self.conv(paddle.concat(inputs, axis=1))
        out = self.bn(out)
        if self.residual:
            out = paddle.add(x=out, y=children[0])
        out = F.relu(out)

        return out


class Tree(nn.Layer):
    def __init__(self,
                 level,
                 block,
                 ch_in,
                 ch_out,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * ch_out
        if level_root:
            root_dim += ch_in
        if level == 1:
            self.tree1 = block(ch_in, ch_out, stride)
            self.tree2 = block(ch_out, ch_out, 1)
        else:
            self.tree1 = Tree(
                level - 1,
                block,
                ch_in,
                ch_out,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual)
            self.tree2 = Tree(
                level - 1,
                block,
                ch_out,
                ch_out,
                1,
                root_dim=root_dim + ch_out,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual)

        if level == 1:
            self.root = Root(root_dim, ch_out, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.level = level
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)
        if ch_in != ch_out:
            self.project = nn.Sequential(
                    nn.Conv2D(
                        ch_in,
                        ch_out,
                        kernel_size=1,
                        stride=1,
                        bias_attr=False),
                    nn.BatchNorm2D(ch_out))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@register
@serializable
class DLA(nn.Layer):
    """
    DLA, see https://arxiv.org/pdf/1707.06484.pdf

    Args:
        depth (int): DLA depth, only support 34 now.
        residual_root (bool): whether use a reidual layer in the root block
        pre_img (bool): add pre_img, only used in CenterTrack
        pre_hm (bool): add pre_hm, only used in CenterTrack
    """

    def __init__(self,
                 depth=34,
                 block="BasicBlock",
                 residual_root=False,
                 pre_img=False,
                 pre_hm=False):
        super(DLA, self).__init__()
        assert depth == 34, 'Only support DLA with depth of 34 now.'
        if block == "BasicBlock":
            block = BasicBlock
        elif block == "StarBasic":
            block = StarBasic
        levels, channels = DLA_cfg[depth]
        self.channels = channels
        self.num_levels = len(levels)

        self.base_layer = nn.Sequential(
            nn.Conv2D(
                3,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias_attr=False),
            nn.BatchNorm2D(channels[0]),
            nn.ReLU())
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root)

        if pre_img:
            self.pre_img_layer = nn.Sequential(
                ConvNormLayer(
                    3,
                    channels[0],
                    filter_size=7,
                    stride=1,
                    bias_on=False,
                    norm_decay=None),
                nn.ReLU())
        if pre_hm:
            self.pre_hm_layer = nn.Sequential(
                ConvNormLayer(
                    1,
                    channels[0],
                    filter_size=7,
                    stride=1,
                    bias_on=False,
                    norm_decay=None),
                nn.ReLU())
        self.pre_img = pre_img
        self.pre_hm = pre_hm

    def _make_conv_level(self, ch_in, ch_out, conv_num, stride=1):
        modules = []
        for i in range(conv_num):
            modules.extend([
                nn.Conv2D(
                    ch_in,
                    ch_out,
                    kernel_size=3,
                    padding=1,
                    stride=stride if i == 0 else 1,
                    bias_attr=False), nn.BatchNorm2D(ch_out), nn.ReLU(),
            ])
            ch_in = ch_out
        return nn.Sequential(*modules)

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.channels[i]) for i in range(self.num_levels)
        ]

    def forward(self, inputs):
        outs = []
        feats = self.base_layer(inputs['image'])

        if self.pre_img and 'pre_image' in inputs and inputs[
                'pre_image'] is not None:
            feats = feats + self.pre_img_layer(inputs['pre_image'])

        if self.pre_hm and 'pre_hm' in inputs and inputs['pre_hm'] is not None:
            feats = feats + self.pre_hm_layer(inputs['pre_hm'])

        for i in range(self.num_levels):
            feats = getattr(self, 'level{}'.format(i))(feats)
            outs.append(feats)

        return outs
