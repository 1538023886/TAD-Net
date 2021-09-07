
# coding: utf-8

# In[1]:

from tadnet import TADNet as SunNet
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
#import visdom
#import tqdm


# In[2]:

#定义超参数
dilation_depth=9 #网络有多少层
out_len_redce=2**dilation_depth-1  #输出减少了多少
TIME=1000


# In[3]:

model = SunNet(dilation_depth).cuda()
#model.load_state_dict(torch.load("net9_20.pth"))


# In[4]:

data=np.load('data/split321_13_13.npy',allow_pickle=True)#数据长度2883296
data=list(data)[0]
train_xs=data['train_x']
#train_x=torch.tensor(train_x)
#print(train_x.shape)
train_ys=data['train_y']

#train_y=torch.tensor(train_y)
#train_y=train_y.to(torch.float32)
#print(train_y[:300])
#train_y=train_y/300
#print(train_y.shape)
#print(train_y[:300])


# In[5]:

print(type(train_ys))


# In[6]:

#print(max(train_y))


# In[7]:

#train_y=train_y+1


# In[8]:

train_step = torch.optim.Adam(model.parameters(),lr=1e-3, eps=1e-4)#学习率0.01损失无法下降
scheduler = torch.optim.lr_scheduler.MultiStepLR(train_step, milestones=[50,80], gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


# In[9]:

#vis = visdom.Visdom(env='loss')
#绘制loss变化趋势，参数一为Y轴的值，参数二为X轴的值，参数三为窗体名称，参数四为表格名称，参数五为更新选项，从第二个点开始可以更新


# In[10]:
count_ddd = 0
for epoch in range(100):
    count_ddd += 1
    print("----------------------epoch:   ", epoch)
    loss_= []
    scheduler.step()
    #for file_id in [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]:
    for file_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 1-6表示1-6段行为被选为训练集，0作为测试集，编号从0开始，总数为训练集npy文件包含总数
        train_x=train_xs[file_id]
        train_y=train_ys[file_id]
        train_x=torch.tensor(train_x)
        train_y=torch.tensor(train_y)
        for i in range(0,train_x.shape[0]-1000,200):
            x=train_x[i:i+1000].float().unsqueeze(0).to('cuda')
            y=train_y[i:i+1000].to('cuda')
            logits = model(x)#输入格式torch.Size([1, 1000, 25, 3])
            #print(logits.shape)
            #print(y)
            #print(logits.shape,y[out_len_redce:].shape)
            loss = loss_fn(logits, y[out_len_redce:])
            train_step.zero_grad()
            loss.backward()
            train_step.step()
            #print(loss)
            loss_.append(loss.data.cpu().numpy())
    print(epoch,np.mean(loss_))
    #vis.line(Y=np.mean(loss_).reshape(-1), X=np.array([epoch]),win=('train_loss'),opts=dict(title='train_loss'),update='append')
    if count_ddd%10==0:
        torch.save(obj=model.state_dict(), f="models/net9_%d.pth"%(count_ddd))

# In[11]:

#torch.save(obj=model.state_dict(), f="models/njust-pad/net9_10.pth")


# In[12]:

#NJUST-TGN

