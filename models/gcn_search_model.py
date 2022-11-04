import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# https://github.com/narumiruna/efficientnet-pytorch/blob/master/efficientnet/models/efficientnet.py
class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, swish_nonlinearity=False):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if swish_nonlinearity:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                Swish(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )
        
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc[2].weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
# https://gitee.com/yfsong0709/ResGCNv1/blob/master/src/model/attentions.py (https://github.com/yfsong0709/ResGCNv1/blob/master/src/model/attentions.py)
class Channel_Att(nn.Module):
    def __init__(self, channels, reduction=4, swish_nonlinearity=False, **kwargs):
        super(Channel_Att, self).__init__()

        if swish_nonlinearity:
            self.fcn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, kernel_size=1),
                nn.BatchNorm2d(channels//4),
                Swish(inplace=True),
                nn.Conv2d(channels // reduction, channels, kernel_size=1),
                nn.Softmax(dim=1)
            )
        else:
            self.fcn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, kernel_size=1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, kernel_size=1),
                nn.Softmax(dim=1)
            )

        self.bn = nn.BatchNorm2d(channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        
        nn.init.kaiming_normal_(self.fcn[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fcn[4].weight, mode='fan_out', nonlinearity='conv2d')

    def forward(self, x):
        res = x
        x_att = self.fcn(x).squeeze()
        return self.nonlinearity(self.bn(x * x_att[..., None, None]) + res)

# https://gitee.com/yfsong0709/ResGCNv1/blob/master/src/model/attentions.py (https://github.com/yfsong0709/ResGCNv1/blob/master/src/model/attentions.py)
class Frame_Att(nn.Module):
    def __init__(self, channels, kernel_size, swish_nonlinearity=False, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(kernel_size,1), padding=((kernel_size - 1) // 2,0))
        self.bn = nn.BatchNorm2d(channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        res = x
        x_avg = torch.transpose(self.avg_pool(torch.transpose(x, 1, 2)), 1, 2)
        x_max = torch.transpose(self.max_pool(torch.transpose(x, 1, 2)), 1, 2)
        x_att = self.conv(torch.cat([x_avg, x_max], dim=1)).squeeze()
        return self.nonlinearity(self.bn(x * x_att[..., None, :, None]) + res)

# https://gitee.com/yfsong0709/ResGCNv1/blob/master/src/model/attentions.py (https://github.com/yfsong0709/ResGCNv1/blob/master/src/model/attentions.py)
class Joint_Att(nn.Module):
    def __init__(self, channels, num_joints, swish_nonlinearity=False, **kwargs):
        super(Joint_Att, self).__init__()

        if swish_nonlinearity:
            self.fcn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_joints, num_joints//2, kernel_size=1),
                nn.BatchNorm2d(num_joints//2),
                Swish(inplace=True),
                nn.Conv2d(num_joints//2, num_joints, kernel_size=1),
                nn.Softmax(dim=1)
            )
        else:
            self.fcn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_joints, num_joints//2, kernel_size=1),
                nn.BatchNorm2d(num_joints//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_joints//2, num_joints, kernel_size=1),
                nn.Softmax(dim=1)
            )

        self.bn = nn.BatchNorm2d(channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        
        nn.init.kaiming_normal_(self.fcn[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fcn[4].weight, mode='fan_out', nonlinearity='conv2d')

    def forward(self, x):
        res = x
        x_att = self.fcn(torch.transpose(x, 1, 3)).squeeze()
        return self.nonlinearity(self.bn(x * x_att[..., None, None, :]) + res)

# https://github.com/yysijie/st-gcn/blob/master/net/utils/tgcn.py
# The based unit of graph convolutional networks.
class GraphConv(nn.Module):

    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        groups (int, optional): Number of groups in convolution. Default: 1.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 groups=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
            groups=groups)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='conv2d')
        
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        
        x = self.conv(x)
                
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return x.contiguous(), A
    
# https://gitee.com/yfsong0709/ResGCNv1/blob/master/src/model/blocks.py (https://github.com/yfsong0709/ResGCNv1/blob/master/src/model/blocks.py)
class Spatial_Bottleneck_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual=False, reduction=4, basic=False, se_ratio=0, swish_nonlinearity=False, **kwargs):
        super(Spatial_Bottleneck_Block, self).__init__()
        self.basic = basic
        self.se_ratio = se_ratio

        inter_channels = out_channels // reduction if not self.basic else out_channels

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            nn.init.kaiming_normal_(self.residual[0].weight, mode='fan_out', nonlinearity='conv2d')

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        if self.basic:
            self.conv = GraphConv(in_channels, inter_channels, kernel_size)
        else:
            self.conv = GraphConv(inter_channels, inter_channels, kernel_size)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        if self.se_ratio > 0:
            self.se = SELayer(out_channels, reduction=self.se_ratio, swish_nonlinearity=swish_nonlinearity)
        
        nn.init.kaiming_normal_(self.conv_down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_up.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x, A):

        res_block = self.residual(x)

        if not self.basic:
            x = self.conv_down(x)
            x = self.bn_down(x)
            x = self.nonlinearity(x)

        x, A = self.conv(x, A)
        x = self.bn(x)
        
        if not self.basic:
            x = self.nonlinearity(x)

            x = self.conv_up(x)
            x = self.bn_up(x)
            
        if self.se_ratio > 0:
            x = self.se(x)
        
        x = self.nonlinearity(x + res_block)

        return x, A
    
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
class Spatial_MBConv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual=False, expansion=4, se_ratio=0, swish_nonlinearity=False, **kwargs):
        super(Spatial_MBConv_Block, self).__init__()
        self.expansion = expansion
        self.se_ratio = se_ratio

        inter_channels = in_channels * self.expansion if in_channels * self.expansion >= 12 else 12 

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            nn.init.kaiming_normal_(self.residual[0].weight, mode='fan_out', nonlinearity='conv2d')

        self.conv_up = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(inter_channels)
                    
        self.depthwise_conv = GraphConv(inter_channels, inter_channels, kernel_size, groups=inter_channels, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        
        if self.se_ratio > 0:
            self.se = SELayer(inter_channels, reduction=self.se_ratio, swish_nonlinearity=swish_nonlinearity)
        
        self.conv_down = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn_down = nn.BatchNorm2d(out_channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        
        nn.init.kaiming_normal_(self.conv_down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_up.weight, mode='fan_out', nonlinearity='relu')
        
        
    def forward(self, x, A):

        res_block = self.residual(x)

        if not self.expansion == 1:
            x = self.conv_up(x)
            x = self.bn_up(x)
            x = self.nonlinearity(x)

        x, A = self.depthwise_conv(x, A)
        x = self.bn(x)
        x = self.nonlinearity(x)
        
        if self.se_ratio > 0:
            x = self.se(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        
        x = self.nonlinearity(x + res_block)

        return x, A

# https://gitee.com/yfsong0709/ResGCNv1/blob/master/src/model/blocks.py (https://github.com/yfsong0709/ResGCNv1/blob/master/src/model/blocks.py)
class Temporal_Bottleneck_Block(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, residual=False, reduction=4, dropout_factor=0.0, basic=False, inner_se_ratio=0, outer_se_ratio=0, scales=1, swish_nonlinearity=False, **kwargs):
        super(Temporal_Bottleneck_Block, self).__init__()
        self.basic = basic
        self.inner_se_ratio = inner_se_ratio
        self.outer_se_ratio = outer_se_ratio
        
        inter_channels = channels // reduction if not self.basic else channels

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )
            nn.init.kaiming_normal_(self.residual[0].weight, mode='fan_out', nonlinearity='conv2d')

        temporal_branches = []
        for dilation_rate in range(1, scales+1):
            conv_down = nn.Conv2d(channels, inter_channels, 1)
            bn_down = nn.BatchNorm2d(inter_channels)
            
            padding = (((kernel_size - 1) // 2)*dilation_rate, 0)
            conv = nn.Conv2d(inter_channels, inter_channels, (kernel_size,1), (stride,1), padding, dilation=(dilation_rate,1))
            bn = nn.BatchNorm2d(inter_channels)
            
            conv_up = nn.Conv2d(inter_channels, channels, 1)
            bn_up = nn.BatchNorm2d(channels)
            
            dropout = nn.Dropout(dropout_factor, inplace=True)
            if self.inner_se_ratio > 0:
                inner_se = SELayer(channels, reduction=self.inner_se_ratio, swish_nonlinearity=swish_nonlinearity)
            else:
                inner_se = None
            
            nn.init.kaiming_normal_(conv_down.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(conv_up.weight, mode='fan_out', nonlinearity='relu')
            
            temporal_branches.append(nn.ModuleList(([conv_down, bn_down, conv, bn, conv_up, bn_up, dropout, inner_se])))
        self.temporal_branches = nn.ModuleList(temporal_branches)
            
        self.conv = nn.Conv2d(channels*scales, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        if self.outer_se_ratio > 0:
            self.outer_se = SELayer(channels, reduction=self.outer_se_ratio, swish_nonlinearity=swish_nonlinearity)
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='conv2d')

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x_cat = []
        for temporal_branch in self.temporal_branches:
            branch_x = x
            if not self.basic:
                branch_x = temporal_branch[0](branch_x) #conv_down
                branch_x = temporal_branch[1](branch_x) #bn_down
                branch_x = self.nonlinearity(branch_x)

            branch_x = temporal_branch[2](branch_x) #conv
            branch_x = temporal_branch[3](branch_x) #bn

            if not self.basic:
                branch_x = self.nonlinearity(branch_x)

                branch_x = temporal_branch[4](branch_x) #conv_up
                branch_x = temporal_branch[5](branch_x) #bn_up
                branch_x = temporal_branch[6](branch_x) #dropout
                
            if self.inner_se_ratio > 0:
                branch_x = temporal_branch[7](branch_x) #se
            
            x_cat.append(branch_x)
        
        if len(x_cat) > 1:
            x = torch.cat(x_cat, dim=1)
        
            x = self.conv(x)
            x = self.bn(x)
            
            if self.outer_se_ratio > 0:
                x = self.outer_se(x) 
        else:
            x = x_cat[0]
            
        x = self.nonlinearity(x + res_block + res_module)
            
        return x
    
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
class Temporal_MBConv_Block(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, residual=False, expansion=4, dropout_factor=0.0, inner_se_ratio=0, outer_se_ratio=0, scales=1, swish_nonlinearity=False, **kwargs):
        super(Temporal_MBConv_Block, self).__init__()
        self.expansion = expansion
        self.inner_se_ratio = inner_se_ratio
        self.outer_se_ratio = int(outer_se_ratio / self.expansion) if outer_se_ratio / self.expansion >= 1 else 1
        
        inter_channels = channels * self.expansion

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )
            nn.init.kaiming_normal_(self.residual[0].weight, mode='fan_out', nonlinearity='conv2d')

        temporal_branches = []
        for dilation_rate in range(1, scales+1):
            conv_up = nn.Conv2d(channels, inter_channels, 1, bias=False)
            bn_up = nn.BatchNorm2d(inter_channels)
                
            padding = (((kernel_size - 1) // 2)*dilation_rate, 0)
            depthwise_conv = nn.Conv2d(inter_channels, inter_channels, (kernel_size,1), (stride,1), padding, groups=inter_channels, bias=False, dilation=(dilation_rate,1))
            bn = nn.BatchNorm2d(inter_channels)
            
            if self.inner_se_ratio > 0:
                inner_se = SELayer(inter_channels, reduction=self.inner_se_ratio, swish_nonlinearity=swish_nonlinearity)
            else:
                inner_se = None

            conv_down = nn.Conv2d(inter_channels, channels, 1, bias=False)
            bn_down = nn.BatchNorm2d(channels)
            dropout = nn.Dropout(dropout_factor, inplace=True)
        
            nn.init.kaiming_normal_(conv_up.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(depthwise_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(conv_down.weight, mode='fan_out', nonlinearity='relu')
            
            temporal_branches.append(nn.ModuleList([conv_up, bn_up, depthwise_conv, bn, inner_se, conv_down, bn_down, dropout]))
        self.temporal_branches = nn.ModuleList(temporal_branches)
            
        self.conv = nn.Conv2d(channels*scales, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        if swish_nonlinearity:
            self.nonlinearity = Swish(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)
        if self.outer_se_ratio > 0:
            self.outer_se = SELayer(channels, reduction=self.outer_se_ratio, swish_nonlinearity=swish_nonlinearity)
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='conv2d')
        
    def forward(self, x, res_module):
        
        res_block = self.residual(x)

        x_cat = []
        input_x = x
        for temporal_branch in self.temporal_branches:
            branch_x = input_x
                
            if not self.expansion == 1:
                branch_x = temporal_branch[0](branch_x) #conv_up
                branch_x = temporal_branch[1](branch_x) #bn_up
                branch_x = self.nonlinearity(branch_x) 

            branch_x = temporal_branch[2](branch_x) #depthwise_conv
            branch_x = temporal_branch[3](branch_x) #bn
            branch_x = self.nonlinearity(branch_x)

            if self.inner_se_ratio > 0:
                branch_x = temporal_branch[4](branch_x) #se

            branch_x = temporal_branch[5](branch_x) #conv_down
            branch_x = temporal_branch[6](branch_x) #bn_down
            branch_x = temporal_branch[7](branch_x) #dropout
            
            x_cat.append(branch_x)
        
        if len(x_cat) > 1:
            x = torch.cat(x_cat, dim=1)
        
            x = self.conv(x)
            x = self.bn(x)
            
            if self.outer_se_ratio > 0:
                x = self.outer_se(x) 
        else:
            x = x_cat[0]
        
        x = self.nonlinearity(x + res_block + res_module)
        
        return x
    
# https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (float, optional): Dropout rate. Default: 0.0
        reduction (int, optional): Factor of compression in bottleneck convolution. Default: 4
        expansion (int, optional): Factor of expansion in inverted bottleneck convolution. Default: 6
        block_type (string, optional): Type of micro block architecture to apply. Default: 'basic'
        inner_se_ratio (int, optional): Reduction ratio of inner Squeeze-and-Excitation (SE) mechanism. If 0 SE is not applied. Default: 0
        outer_se_ratio (int, optional): Reduction ratio of outer Squeeze-and-Excitation (SE) mechanism. If 0 SE is not applied. Default: 0
        temporal_scales (int, optional): Number of scales in multi-scale temporal convolution. Default: 1
        attention (string, optional): Type of attention to apply to GCN module. Default: 'null'
        num_joints (int, optional): Number of joints in skeleton. Default: 19 (In-Motion skeleton)
        residual (string, optional): Type of residual mechanism to apply. Default: 'dense'
        swish_nonlinearity (bool, optional): If ``True``, utilizes Swish activation function throughout network, else ReLU. Default: False

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.0,
                 reduction=4,
                 expansion=4,
                 block_type='basic',
                 inner_se_ratio=0,
                 outer_se_ratio=0,
                 temporal_scales=1,
                 attention='null',
                 num_joints=19,
                 residual='dense',
                 swish_nonlinearity=False
                 ):
        super().__init__()

        assert len(kernel_size) == 2
        assert int(kernel_size[0]) % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
       
        
        if residual=='null':
            module_res, block_res = False, False
        elif residual=='block':
            module_res, block_res = False, True
        elif residual=='module':
            module_res, block_res = True, False
        elif residual=='dense':
            module_res, block_res = True, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels)
            )
            nn.init.kaiming_normal_(self.residual[0].weight, mode='fan_out', nonlinearity='conv2d')
            
        if block_type=='mbconv':
            self.gcn = Spatial_MBConv_Block(in_channels, out_channels, kernel_size[1], residual=block_res, expansion=expansion, se_ratio=inner_se_ratio, swish_nonlinearity=swish_nonlinearity) 
            self.tcn = Temporal_MBConv_Block(out_channels, kernel_size[0], stride=stride, residual=block_res, expansion=expansion, dropout_factor=dropout, inner_se_ratio=inner_se_ratio, outer_se_ratio=outer_se_ratio, scales=temporal_scales, swish_nonlinearity=swish_nonlinearity) 
        elif block_type=='bottleneck':
            self.gcn = Spatial_Bottleneck_Block(in_channels, out_channels, kernel_size[1], residual=block_res, reduction=reduction, basic=False, se_ratio=inner_se_ratio, swish_nonlinearity=swish_nonlinearity) 
            self.tcn = Temporal_Bottleneck_Block(out_channels, kernel_size[0], stride=stride, residual=block_res, reduction=reduction, dropout_factor=dropout, basic=False, inner_se_ratio=inner_se_ratio, outer_se_ratio=outer_se_ratio, scales=temporal_scales, swish_nonlinearity=swish_nonlinearity)
        elif block_type=='basic':
            self.gcn = Spatial_Bottleneck_Block(in_channels, out_channels, kernel_size[1], residual=block_res, reduction=reduction, basic=True, se_ratio=inner_se_ratio, swish_nonlinearity=swish_nonlinearity) 
            self.tcn = Temporal_Bottleneck_Block(out_channels, kernel_size[0], stride=stride, residual=block_res, reduction=reduction, dropout_factor=dropout, basic=True, inner_se_ratio=inner_se_ratio, outer_se_ratio=outer_se_ratio, scales=temporal_scales, swish_nonlinearity=swish_nonlinearity)    
        self.attention = None
        if attention == 'channel':
            self.attention = Channel_Att(out_channels, swish_nonlinearity=swish_nonlinearity)
        elif attention == 'frame':
            self.attention = Frame_Att(out_channels, kernel_size[0], swish_nonlinearity=swish_nonlinearity)
        elif attention == 'joint':
            self.attention = Joint_Att(out_channels, num_joints, swish_nonlinearity=swish_nonlinearity)
        
    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x, res)
        if self.attention is not None:
            x = self.attention(x)

        return x, A

# https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        num_classes (int): Number of classes for the classification task
        graphs (list): The graphs of spatial convolution
        input_channels (int): Number of channels in the input data
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        dropout (float): Dropout rate. Default: 0.0
        num_input_branches (int, optional): Number of modalities in input tensor. Default: 3
        attention (string, optional): Type of attention to apply to GCN module. Default: 'null'
        spatial_pool (bool, optional): If ``True``, applies spatial global pooling operation before classification. Default: False
        se_outer (bool, optional): If ``True``, applies outer Squeeze-and-Excitation. Default: False
        se_inner (bool, optional): If ``True``, applies inner Squeeze-and-Excitation. Default: False
        initial_residual (string, optional): Type of residual mechanism to apply to initial layer. Default: 'null'
        residual (string, optional): Type of residual mechanism to apply. Default: 'dense'
        initial_block_type (string, optional): Type of micro block architecture to apply to first layer. Default: 'basic'
        block_type (string, optional): Type of micro block architecture to apply througout network. Default: 'basic'
        input_width (int, optional): Number of channels in first convolutional layer. Default: 16
        initial_main_width (int, optional): Number of channels in first convolutional layer of main branch. Default: 32
        temporal_kernel_size (int, optional): Kernel size of temporal convolution. Default: 9
        num_input_modules (int, optional): Number of ST-GCN modules in input branches. Default: 3
        num_main_levels (int, optional): Number of abstration levels of main branch. Default: 2
        num_main_level_modules (int, optional): Number of modules per abstration level of main branch. Default: 2
        input_temporal_scales (list, optional): Number of scales in multi-scale temporal convolution per input block. Default: [1, 1]
        main_temporal_scales (list, optional): Number of scales in multi-scale temporal convolution per main level. Default: [1, 1, 1]
        bottleneck_factor (int, optional): Degree of reduction/expansion in bottleneck conv/MBConv. Default: 4
        se_ratio (int, optional): Degree of downsampling in SE layer. Default: 4
        relative_se (bool, optional): If ``True``, utilizes SE with downsampling relative to layer width. Default: False
        swish_nonlinearity (bool, optional): If ``True``, utilizes Swish activation function throughout network, else ReLU. Default: False
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, input_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_classes)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, num_classes, graphs, input_channels=6, edge_importance_weighting=True, dropout=0.0, num_input_branches=3, 
                 attention='null', spatial_pool=False, se_outer=False, se_inner=False, initial_residual='null', residual='dense', initial_block_type='basic', block_type='basic', input_width=16, initial_main_width=32, temporal_kernel_size=9, num_input_modules=3, num_main_levels=2, num_main_level_modules=2, input_temporal_scales=[1,1,1], main_temporal_scales=[1,1], bottleneck_factor=4, se_ratio=4, relative_se=False, swish_nonlinearity=False, **kwargs):
        super().__init__()
        self.input_channels_per_branch = input_channels // num_input_branches
        self.spatial_pool = spatial_pool
        self.swish_nonlinearity = swish_nonlinearity
        self.num_input_branches = num_input_branches
        self.num_input_modules = num_input_modules

        # graphs and kernel sizes
        self.graphs = graphs
        self.graph_input = self.graphs[0]
        A_input = torch.tensor(self.graph_input.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_input', A_input)
        spatial_kernel_size_input = A_input.size(0)
        kernel_sizes = [(temporal_kernel_size, spatial_kernel_size_input)]
        if len(self.graphs) == 2:
            self.graph_main = self.graphs[1]
            A_main = torch.tensor(self.graph_main.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_main', A_main)
            spatial_kernel_size_main = A_main.size(0)
            kernel_sizes.append((temporal_kernel_size, spatial_kernel_size_main))
        self.num_joints = A_input.size(1)
            
        # build networks
        self.data_bn = nn.BatchNorm1d(input_channels * A_input.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        # input branches
        input_branches = []
        for n in range(self.num_input_branches):
            print("SEARCH:")
            print(input_width, type(input_width))
            print(bottleneck_factor, type(bottleneck_factor))
            input_branch_blocks = [st_gcn(
                self.input_channels_per_branch, 
                int(input_width) if initial_block_type=='mbconv' else int(input_width)*int(bottleneck_factor), 
                kernel_sizes[0], 
                1, 
                dropout=dropout, 
                reduction=bottleneck_factor, 
                block_type=initial_block_type, 
                inner_se_ratio=se_ratio if se_inner else 0, 
                outer_se_ratio=se_ratio if se_outer else 0, 
                temporal_scales=input_temporal_scales[0], 
                expansion=bottleneck_factor, 
                num_joints=self.num_joints, 
                attention=attention, 
                residual=initial_residual, 
                swish_nonlinearity=swish_nonlinearity, 
                **kwargs0
                )]
            for i in range(1, self.num_input_modules):
                if i==1:
                    input_branch_block = st_gcn(input_width if initial_block_type=='mbconv' else input_width*bottleneck_factor, input_width if block_type=='mbconv' else input_width*bottleneck_factor, kernel_sizes[0], 1, dropout=dropout, reduction=bottleneck_factor, block_type=block_type, inner_se_ratio=se_ratio if se_inner else 0, outer_se_ratio=se_ratio if se_outer else 0, temporal_scales=input_temporal_scales[i], expansion=bottleneck_factor, num_joints=self.num_joints, attention=attention, residual=residual, swish_nonlinearity=swish_nonlinearity, **kwargs0)
                else:
                    input_branch_block = st_gcn(input_width if block_type=='mbconv' else input_width*bottleneck_factor, input_width if block_type=='mbconv' else input_width*bottleneck_factor, kernel_sizes[0], 1, dropout=dropout, reduction=bottleneck_factor, block_type=block_type, inner_se_ratio=se_ratio if se_inner else 0, outer_se_ratio=se_ratio if se_outer else 0, temporal_scales=input_temporal_scales[i], expansion=bottleneck_factor, num_joints=self.num_joints, attention=attention, residual=residual, swish_nonlinearity=swish_nonlinearity, **kwargs0)
                input_branch_blocks.append(input_branch_block)
            input_branch = nn.ModuleList(input_branch_blocks)
            input_branches.append(input_branch)
        self.st_gcn_input = nn.ModuleList(input_branches)
        if self.num_input_modules==1:
            self.bottleneck_conv = nn.Conv2d((input_width if initial_block_type=='mbconv' else input_width*bottleneck_factor)*self.num_input_branches, initial_main_width if block_type=='mbconv' else initial_main_width*bottleneck_factor, 1)
        else:
            self.bottleneck_conv = nn.Conv2d((input_width if block_type=='mbconv' else input_width*bottleneck_factor)*self.num_input_branches, initial_main_width if block_type=='mbconv' else initial_main_width*bottleneck_factor, 1)
        self.bottleneck_bn = nn.BatchNorm2d(initial_main_width if block_type=='mbconv' else initial_main_width*bottleneck_factor)
        
        nn.init.kaiming_normal_(self.bottleneck_conv.weight, mode='fan_out', nonlinearity='conv2d')
        
        # main stream
        main_branch = []
        level_input_width = initial_main_width if block_type=='mbconv' else initial_main_width*bottleneck_factor
        level_output_width = initial_main_width if block_type=='mbconv' else initial_main_width*bottleneck_factor
        adaptive_se_ratio = se_ratio
        for i in range(num_main_levels): 
            main_branch.append(st_gcn(level_input_width, level_output_width, kernel_sizes[1] if len(kernel_sizes) == 2 else kernel_sizes[0], 2, dropout=dropout, reduction=bottleneck_factor, block_type=block_type, inner_se_ratio=adaptive_se_ratio if se_inner else 0, outer_se_ratio=adaptive_se_ratio if se_outer else 0, temporal_scales=main_temporal_scales[i], expansion=bottleneck_factor, num_joints=self.num_joints, attention=attention, residual=residual, swish_nonlinearity=swish_nonlinearity, **kwargs0))
            for j in range(1, num_main_level_modules):
                main_branch.append(st_gcn(level_output_width, level_output_width, kernel_sizes[1] if len(kernel_sizes) == 2 else kernel_sizes[0], 1, dropout=dropout, reduction=bottleneck_factor, block_type=block_type, inner_se_ratio=adaptive_se_ratio if se_inner else 0, outer_se_ratio=adaptive_se_ratio if se_outer else 0, temporal_scales=main_temporal_scales[i], expansion=bottleneck_factor, num_joints=self.num_joints, attention=attention, residual=residual, swish_nonlinearity=swish_nonlinearity, **kwargs0))
            level_input_width = level_output_width
            level_output_width = 2*level_input_width
            if relative_se:
                adaptive_se_ratio *= 2
              
        self.st_gcn_main = nn.ModuleList(main_branch)    

        # initialize parameters for edge importance weighting
        num_input_layers = 0
        for module in self.st_gcn_input:
            for layer in module:
                num_input_layers += 1
        num_main_layers = 0
        for gcn in self.st_gcn_main:
            num_main_layers += 1
        if len(self.graphs) == 1:
            num_input_layers += num_main_layers
        if edge_importance_weighting:
            self.edge_importance_input = nn.ParameterDict({str(i).zfill(5): nn.Parameter(torch.ones(self.A_input.size())) for i in range(num_input_layers)})
            if len(self.graphs) == 2:
                self.edge_importance_main = nn.ParameterDict({str(i).zfill(5): nn.Parameter(torch.ones(self.A_main.size())) for i in range(num_main_layers)})
        else:
            self.edge_importance_input = [1] * num_input_layers
            if len(self.graphs) == 2:
                self.edge_importance_main = [1] * num_main_layers
                        
        # fcn for prediction
        if self.spatial_pool:
            self.fcn_pool = nn.Conv2d(level_input_width*self.num_joints, level_input_width, kernel_size=1)
            nn.init.kaiming_normal_(self.fcn_pool.weight, mode='fan_out', nonlinearity='linear')   
        
        self.fcn = nn.Conv2d(level_input_width, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.fcn.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):

        # data normalization
        N, C, T, V = x.size()
        M = 1
        x = x.unsqueeze(4)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        input_x = x
        
        # input branches
        x_cat = []
        for i, branch in enumerate(self.st_gcn_input):
            x = input_x[:,i*self.input_channels_per_branch:i*self.input_channels_per_branch+self.input_channels_per_branch,:,:]
            for gcn, importance in zip(branch, list(self.edge_importance_input.values())[int(i*self.num_input_modules):]):
                x, _ = gcn(x, self.A_input * importance)
            x_cat.append(x)
            
        if len(x_cat) > 1:
            x = torch.cat(x_cat, dim=1) 
            
            x = self.bottleneck_conv(x)
            x = self.bottleneck_bn(x)
            
        else:
            x = x_cat[0]
        
        # main stream
        if len(self.graphs) == 1:
            for gcn, importance in zip(self.st_gcn_main, list(self.edge_importance_input.values())[int(self.num_input_branches*self.num_input_modules):]):
                x, _ = gcn(x, self.A_input * importance)
        else:
            for gcn, importance in zip(self.st_gcn_main, self.edge_importance_main.values()):
                x, _ = gcn(x, self.A_main * importance)
                
        # feature
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # global pooling
        if self.spatial_pool:
            x = F.avg_pool2d(x, (x.size()[2],1)) 
            x = x.view(N, M, -1, 1, self.num_joints).mean(dim=1)  
            N, C, T, V = x.size()
            x = x.view(N, C * V, 1, 1) 
            x = self.fcn_pool(x)
        else:
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x, feature