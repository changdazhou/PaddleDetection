from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, AdaptiveAvgPool2D

from ppdet.core.workspace import register, serializable
__all__ = ['ECAttentionBlock']

class ECANet(nn.Layer):
    def __init__(self, in_channel, b=1, gamma=2):
        super(ECANet, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape) # b,c,h,w
        y = self.avg_pool(x).squeeze(-1).transpose([0,2,1])
        # print(y.shape) # b,1,c
        y = self.conv(y)
        # print(y.shape) # b,1,c
        y = self.sigmoid(y).transpose([0,2,1])
        y = y.reshape([*y.shape, 1])
        # print(y.shape) # b,c,1,1
        return x * y.expand_as(x)




@register
@serializable  
class ECAttentionBlock(nn.Layer):
    def __init__(self,channel_list):
        super(ECAttentionBlock, self).__init__()
        if isinstance(channel_list, int):
            channel_list = [channel_list]
        self.attention_num = len(channel_list)
        for i in range(self.attention_num):
            setattr(self,'eca_module_{}'.format(i),ECANet(channel_list[i]))
        
        
    def forward(self, inputs):
        for i in range(self.attention_num):
            inputs[i] = getattr(self,'eca_module_{}'.format(i))(inputs[i])
        return inputs