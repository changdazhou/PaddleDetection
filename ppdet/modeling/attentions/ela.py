from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, AdaptiveAvgPool2D

from ppdet.core.workspace import register, serializable
__all__ = ['ELAttentionBlock']

class EfficientLocalizationAttention(nn.Layer):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1D(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias_attr=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # 处理高度维度
        x_h = paddle.mean(x, axis=3, keepdim=True).reshape([b, c, h])
        x_h = self.sigmoid(self.gn(self.conv(x_h)).reshape([b, c, h, 1]))

        # 处理宽度维度
        x_w = paddle.mean(x, axis=2, keepdim=True).reshape([b, c, w])
        x_w = self.sigmoid(self.gn(self.conv(x_w)).reshape([b, c, 1, w]))

        # print(x_h.shape, x_w.shape)
        # 在两个维度上应用注意力
        return x * x_h * x_w




@register
@serializable  
class ELAttentionBlock(nn.Layer):
    def __init__(self,channel_list, kernel_size=7):
        super(ELAttentionBlock, self).__init__()
        if isinstance(channel_list, int):
            channel_list = [channel_list]
        self.attention_num = len(channel_list)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for i in range(self.attention_num)]
        else:
            assert len(kernel_size) == self.attention_num, 'kernel_size length must be attention_num'
        for i in range(self.attention_num):
            setattr(self,'ela_module_{}'.format(i),EfficientLocalizationAttention(channel_list[i],kernel_size[i]))
        
        
    def forward(self, inputs):
        for i in range(self.attention_num):
            inputs[i] = getattr(self,'ela_module_{}'.format(i))(inputs[i])
        return inputs