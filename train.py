import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import torch.nn.functional as F

from Poetry import Poetry
from Poetrymodel import *

# Training settings
batch_size = 1
lr = 0.0001


# poetry Dataset

train_dataset = Poetry(root='./dataset/',transform=None)


# Data Loader

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)




#model 

net = PoetryModel(len(train_dataset.get_words_vector()))
optimizer = optim.Adam(net.parameters(),lr=lr)
lossfunc = nn.CrossEntropyLoss()


def main():
	# 如果模型文件存在则尝试加载模型参数
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    for epoch in range(1):
        train(epoch,net,optimizer,lossfunc,train_loader)
        torch.save(net.state_dict(),'./model.pth')
        
if __name__ == '__main__':
	main()
