import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch.autograd import Variable
from graph.graph import Graph
#Graph = import_class(graph)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)




class DilatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(DilatedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)
    
    def forward(self, inputs):
        outputs = super(DilatedConv1d, self).forward(inputs)
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):# 32，512 ，空洞卷积率
        super(ResidualBlock, self).__init__()
        self.filter_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.gate_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, kernel_size=1)
        
    def forward(self,inputs):
        sigmoid_out = F.sigmoid(self.gate_conv(inputs))
        tahn_out = F.tanh(self.filter_conv(inputs))
        output = sigmoid_out * tahn_out
        #
        skip_out = self.skip_conv(output)
        res_out = self.residual_conv(output)
        res_out = res_out + inputs[:, :, -res_out.size(2):]
        # res
        return res_out , skip_out



class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=1):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

   
class Outgcn(nn.Module):
    def __init__(self,A):#输出应该是N*T*1
        super(Outgcn, self).__init__()
        self.gcn1=unit_gcn(512,256,A)
        self.gcn2=unit_gcn(256,128,A)
        self.gcn3=unit_gcn(128,64,A)
        self.gcn4=unit_gcn(64,32,A)
        self.gcn5=unit_gcn(32,16,A)
        self.fc1 = nn.Linear(16*25,100)
        self.fc2 = nn.Linear(100,13)
		#self.softmax=
        
        
    def forward(self,inputs):#V*C*T
        V,C,T=inputs.size()
        inputs=inputs.permute(1,2,0)#C*T*V
        inputs=inputs.view(1,C,T,V)
        inputs=self.gcn1(inputs)
        inputs=self.gcn2(inputs)
        inputs=self.gcn3(inputs)
        inputs=self.gcn4(inputs)
        inputs=self.gcn5(inputs)# 1,C=16,T,V=25
        inputs=inputs.permute(0,2,1,3)
        #print('11',inputs.shape)
        inputs=inputs.contiguous()
        inputs=inputs.view(inputs.size()[1],25*16)
        #print('22',inputs.shape)
        inputs=self.fc1(inputs)
        inputs=torch.relu(inputs)
        outputs=self.fc2(inputs)
        return outputs
class TADNet(nn.Module):
    def __init__(self,dilation_depth=9,in_depth=256, res_channels=3, skip_channels=512):
        super(TADNet, self).__init__()
        

        self.graph = Graph()
        A = self.graph.A
        self.dilations = [2**i for i in range(dilation_depth)]
        self.main = nn.ModuleList([ResidualBlock(res_channels,skip_channels,dilation) for dilation in self.dilations])
        self.post = Outgcn(A)
        
    def forward(self,inputs):#N,T,V,C
        N,T,V,C=inputs.size()
        skip_connections = []
        inputs=inputs.permute(0,2,3,1)
        outputs=inputs.view(N*V,C,T)
        for layer in self.main:
            outputs,skip = layer(outputs)
            skip_connections.append(skip)
        #print(outputs.shape)
        #print(skip.shape)
        outputs = sum([s[:,:,-skip.shape[2]:] for s in skip_connections])
        #print(outputs.shape)
        outputs = self.post(outputs)
        return outputs