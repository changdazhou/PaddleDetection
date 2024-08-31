from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, AdaptiveAvgPool2D

from ppdet.core.workspace import register, serializable
__all__ = ['SEAttentionBlock']


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs)

@register
@serializable  
class SEAttentionBlock(nn.Layer):
    def __init__(self,channel_list, reduction=4):
        super(SEAttentionBlock, self).__init__()
        if isinstance(channel_list, int):
            channel_list = [channel_list]
        self.attention_num = len(channel_list)
        if isinstance(reduction, int):
            reduction = [reduction for i in range(self.attention_num)]
        else:
            assert len(reduction) == self.attention_num, 'reduction length must be attention_num'
        for i in range(self.attention_num):
            setattr(self,'se_module_{}'.format(i),SEModule(channel_list[i],reduction[i]))
        
        
    def forward(self, inputs):
        for i in range(self.attention_num):
            inputs[i] = getattr(self,'se_module_{}'.format(i))(inputs[i])
        return inputs