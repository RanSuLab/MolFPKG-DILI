import torch.nn as nn
import torch.nn.functional as F

# 处理分子指纹的MLP网络，生成smiles对应分子指纹特征，fc+relu+dropout+fc

class FPNN(nn.Module):
    def __init__(self, input_dim, num_class, dropout):
        super(FPNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.fc4 = nn.Linear(128, num_class)

        self.dropout = dropout
    
    def forward(self, fp_list):

        fpn_out = self.fc1(fp_list)
        fpn_out = self.bn1(fpn_out)
        fpn_out = F.relu(fpn_out)
        # fpn_out = F.tanh(fpn_out)
        fpn_out = F.dropout(fpn_out, self.dropout, training=self.training)

        fpn_out = self.fc2(fpn_out)
        fpn_out = self.bn2(fpn_out)
        fpn_out = F.relu(fpn_out)
        # fpn_out = F.tanh(fpn_out)
        fpn_out = F.dropout(fpn_out, self.dropout, training=self.training)

        fpn_out = self.fc3(fpn_out)
        fpn_out = self.bn3(fpn_out)
        fpn_out = F.relu(fpn_out)
        # fpn_out = F.tanh(fpn_out)
        fpn_out = F.dropout(fpn_out, self.dropout, training=self.training)

        fpn_out = self.fc4(fpn_out)

        # return fpn_out
        return F.log_softmax(fpn_out, dim=-1)
