from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, AdaptiveAvgPool2D

from ppdet.core.workspace import register, serializable
__all__ = ['CBAMAttentionBlock']


# Construction
import paddle
from paddle import nn

class CBAM(nn.Layer):
    def __init__(self, in_channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelAttention = ChannelAttention(in_channel, ratio=ratio)
        self.spatialAttention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelAttention(x)
        x = x * self.spatialAttention(x)
        return x

class ChannelAttention(nn.Layer):
    def __init__(self, in_channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2D(in_channel, in_channel // ratio, 1)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2D(in_channel // ratio, in_channel, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


@register
@serializable  
class CBAMAttentionBlock(nn.Layer):
    def __init__(self,channel_list, ratio=8, kernel_size=7):
        super(CBAMAttentionBlock, self).__init__()
        if isinstance(channel_list, int):
            channel_list = [channel_list]
        self.attention_num = len(channel_list)
        for i in range(self.attention_num):
            setattr(self,'cbam_module_{}'.format(i),CBAM(channel_list[i],ratio,kernel_size))
        
        
    def forward(self, inputs):
        for i in range(self.attention_num):
            inputs[i] = getattr(self,'cbam_module_{}'.format(i))(inputs[i])
        return inputs