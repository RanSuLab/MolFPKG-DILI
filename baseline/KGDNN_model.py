import torch.nn as nn
import torch.nn.functional as F

    
class MLP(nn.Module):
    def __init__(self, input_dim, num_class, dropout):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.fc4 = nn.Linear(128, num_class)

        self.dropout = dropout
    
    def forward(self, data_list):

        mlp_out = self.fc1(data_list)
        mlp_out = F.relu(mlp_out)
        mlp_out = self.bn1(mlp_out)
        mlp_out = F.dropout(mlp_out, self.dropout, training=self.training)

        mlp_out = self.fc2(mlp_out)
        mlp_out = F.relu(mlp_out)
        mlp_out = self.bn2(mlp_out)
        mlp_out = F.dropout(mlp_out, self.dropout, training=self.training)

        mlp_out = self.fc3(mlp_out)
        mlp_out = F.relu(mlp_out)
        mlp_out = self.bn3(mlp_out)
        mlp_out = F.dropout(mlp_out, self.dropout, training=self.training)

        mlp_out = self.fc4(mlp_out)

        return F.log_softmax(mlp_out, dim=-1)