import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np

import torch.nn.functional as F

from torch.autograd import Variable


class CNN_for_audio(nn.Module):
    def __init__(self, n_input=1, num_classes=35, stride=16, n_channel=32):
        super().__init__()
        self.n_input=n_input
        self.num_classes=num_classes
        self.stride=n_output
        self.n_channel=n_channel

    def block(self,in_features,out_features,kernel):
        block=nn.Sequential(nn.Conv1d(in_features, out_features, kernel, 4),
                            nn.BatchNorm1d(out_features),
                            nn.Relu,
                            nn.MaxPool1d(4))
        return block

        self.final_layer = nn.Linear(2 * n_channel, num_classes)
  

    def forward(self, x):
        x=self.block(self.n_input,self.n_channel,80,self.stride)(x)
        x=self.block(self.n_channel,self.n_channel,3)(x)
        x=self.block(self.n_channel,2*self.n_channel,3)(x)
        x=self.block(2*self.n_channel,2*self.n_channel,3)(x)
        x=self.final_layer(x)
        output=F.log_softmax(x, dim=2)

        return output


class LSTM_for_baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=10, dropout=0):
        super(LSTM_for_baseline, self).__init__()

        self.num_classes=num_classes
        self.hidden_size = hidden_size
        self.dropout=dropout
        self.num_layers = num_layers

        self.lstm_layer = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.linear3 = nn.Linear(self.hidden_size//2, self.num_classes)
    
    def forward(self, x, ctx):
        x = x.float()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        x, _ = self.lstm_layer(x, (h0,c0)) 

        x=self.linear1(x[:, -1, :])
        x = self.activation(x)
        x=self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x) 
        output=F.log_softmax(x, dim=2)

        return output