
# coding: utf-8
import sys
# In[ ]:

from tadnet import TADNet as SunNet
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
import visdom
import copy


# In[ ]:

class Processor:
    def __init__(self):
        self.dilation_depth=9 #网络有多少层
        self.out_len_redce=2**self.dilation_depth-1  #输出减少了多少
        self.model = SunNet(self.dilation_depth).cuda()
        self.model.load_state_dict(torch.load(r"models/net9_20.pth"))
    def eval(self,input_x):
        input_x=torch.tensor(input_x)
        input_x=input_x.float().unsqueeze(0).to('cuda')
        self.model.eval()
        #print("input_x",input_x.size())
        with torch.no_grad():
            output = self.model(input_x)
        output=output[-1]
        _, predict_label = torch.max(output.data, 0)


        output = F.softmax(output,dim=0)
        #print(predict_label,output)
        return int(predict_label),float(output[predict_label])
        


# In[ ]:



