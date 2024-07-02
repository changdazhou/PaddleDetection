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

class ConvBN(paddle.nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        if with_bn:
            self.bn = nn.BatchNorm2D(out_planes,weight_attr=nn.initializer.Constant(value=1.),bias_attr=nn.initializer.Constant(value=0.))
    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return x
    
    
class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


# class BasicBlock(nn.Layer):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.dwconv2(self.g(x))
#         x = input + self.drop_path(x)
#         return x
    
class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, stride=1, drop_path=0.):
        super(BasicBlock, self).__init__()
        self.dwconv = ConvBN(ch_in, ch_in, 7, stride, (7 - 1) // 2, groups=ch_in, with_bn=True)
        self.f1 = ConvBN(ch_in, ch_out, stride, with_bn=False)
        self.f2 = ConvBN(ch_in, ch_out, stride, with_bn=False)
        self.g = ConvBN(ch_out, ch_in, stride, with_bn=True)
        self.dwconv2 = ConvBN(ch_in, ch_out, 7, stride, (7 - 1) // 2, groups=ch_in, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs
        
        x = self.dwconv(inputs)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        # x = inputs + self.drop_path(x)
        
        x = paddle.add(x=x, y=residual)
        # out = F.relu(out)
        
        return x

# class BasicBlock(nn.Layer):
#     def __init__(self, ch_in, ch_out, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = ConvNormLayer(
#             ch_in,
#             ch_out,
#             filter_size=3,
#             stride=stride,
#             bias_on=False,
#             norm_decay=None)
#         self.conv2 = ConvNormLayer(
#             ch_out,
#             ch_out,
#             filter_size=3,
#             stride=1,
#             bias_on=False,
#             norm_decay=None)

#     def forward(self, inputs, residual=None):
#         if residual is None:
#             residual = inputs

#         out = self.conv1(inputs)
#         out = F.relu(out)

#         out = self.conv2(out)

#         out = paddle.add(x=out, y=residual)
#         out = F.relu(out)

#         return out


class Root(nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = ConvNormLayer(
            ch_in,
            ch_out,
            filter_size=1,
            stride=1,
            bias_on=False,
            norm_decay=None)
        self.residual = residual

    def forward(self, inputs):
        children = inputs
        out = self.conv(paddle.concat(inputs, axis=1))
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
            self.project = ConvNormLayer(
                ch_in,
                ch_out,
                filter_size=1,
                stride=1,
                bias_on=False,
                norm_decay=None)

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
                 residual_root=False,
                 pre_img=False,
                 pre_hm=False):
        super(DLA, self).__init__()
        assert depth == 34, 'Only support DLA with depth of 34 now.'
        if depth == 34:
            block = BasicBlock
        levels, channels = DLA_cfg[depth]
        self.channels = channels
        self.num_levels = len(levels)

        self.base_layer = nn.Sequential(
            ConvNormLayer(
                3,
                channels[0],
                filter_size=7,
                stride=1,
                bias_on=False,
                norm_decay=None),
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
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    filter_size=3,
                    stride=stride if i == 0 else 1,
                    bias_on=False,
                    norm_decay=None), nn.ReLU()
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
