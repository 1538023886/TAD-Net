
# coding: utf-8

# In[1]:

#from tqdm import tqdm
import re
import numpy as np


# In[2]:

train_x=[]


# In[3]:

for file_id in range(12,14):  #选中训练集文件范围
    f = open(r"./njust_dataset/dataset_new/{}-action_seq.txt".format(str(file_id)),"r")   #设置文件对象
    line = f.readlines()
    tmp_x=[]
    for i in range(len(line)):
        tmp_x.append(list(map(float,line[i].split(',')[:-1])))
    train_x.append(tmp_x)


# In[4]:

print(len(train_x))


# In[5]:

train_y=[]


# In[6]:

for file_id in range(12,14):
    f = open("./njust_dataset\dataset_new/{}-label_class.txt".format(str(file_id)),"r")   #设置文件对象
    print(file_id)
    line = f.readlines()
    tmp_y=[]
    for i in range(len(line)):
        tmp_y.append(int(line[i]))
    train_y.append(tmp_y)


# In[7]:

for i in range(len(train_x)):
    train_x[i]=np.array(train_x[i]).reshape(len(train_x[i]),25,3)


# In[8]:

train_x=np.array(train_x)


# In[ ]:




# In[9]:

print(train_x[0].shape)


# In[10]:

hashmap={}
hashmap['train_x']=train_x
hashmap['train_y']=train_y


# In[11]:

hashmap=np.array([hashmap])


# In[12]:

np.save('data/split321_13_2.npy', hashmap)
print(".npy saved!!!")

# In[13]:

#len(train_y[6])

