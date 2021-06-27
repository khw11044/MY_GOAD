# 데이터가  Transfer 되지않은 데이터 [(7365,23)]
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.modules.linear import Linear



class autoencoder(nn.Module):
    def __init__(self,d):
        super().__init__()
        hidden_dim_1 = d //2 
        hidden_dim_2 = hidden_dim_1 // 2
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden_dim_1),
            #nn.BatchNorm1d(14),
            nn.ReLU(True),

            nn.Linear(hidden_dim_1,hidden_dim_2),
            #nn.BatchNorm1d(7),
            nn.ReLU(True),

            nn.Linear(hidden_dim_2,4),
            #nn.BatchNorm1d(4),
            nn.LeakyReLU(0.1),

            # nn.Linear(256,100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, hidden_dim_2),
            #nn.BatchNorm1d(7),
            nn.ReLU(True),

            nn.Linear(hidden_dim_2,hidden_dim_1),
            #nn.BatchNorm1d(14),
            nn.ReLU(True),

            nn.Linear(hidden_dim_1,d),
            #nn.BatchNorm1d(23),
            nn.Tanh()
            # nn.ReLU(True),

            # nn.Linear(1024, d * d),
            # nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
