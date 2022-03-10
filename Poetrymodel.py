import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PoetryModel(nn.Module):
    def __init__(self,words_size,embedding_dim=100,hidden_dim=200):
        super(PoetryModel,self).__init__()
        self.hidden_dim = hidden_dim
        #词向量转化层
        self.embedding = nn.Embedding(words_size,embedding_dim)
        #lstm
        self.lstm = nn.LSTM(embedding_dim,self.hidden_dim,num_layers=3)
        #linear
        self.linear = nn.Linear(self.hidden_dim,words_size)

    def forward(self,input,hidden=None):
        seq_len, batch_size = input.size()
        #print(input.size())
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        embeds = self.embedding(input)
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden
   
def train(epoch,model,optimizer,lossfunc,data_loader):
    loss_meter = []
    f = open("loss",'a')    
    for batch_idx,data in enumerate(data_loader):
        #batch_size*seqlen*wordidx->seqlen*batch_size*wordidx
        batch_size = len(data)
        #print(data.shape)
        data = data.long().transpose(1,0).contiguous()
        input_,target = data[:-1,:],data[1:,:]
        #print(target.shape) 
        output,_ = model(input_)
        #print(output.shape)
        #return

        loss = lossfunc(output,target.view(-1))
        optimizer.zero_grad() #梯度清零
        loss.backward()       #反向传播
        optimizer.step()      #使用optimizer进行梯度下降
        loss_meter.append(loss.data)
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*batch_size , len(data_loader.dataset),
                100*batch_idx*batch_size / len(data_loader.dataset), loss.data))

            f.write("%s\n"%str(sum(loss_meter)/len(loss_meter)))
    f.close()



