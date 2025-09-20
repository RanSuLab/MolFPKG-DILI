import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import RGCNConv, GCNConv, RGATConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import warnings
warnings.filterwarnings("ignore")


class GCN_graph_bn(torch.nn.Module):
    def __init__(self, input_dim, dim1, dim2, dim3, num_classes, dropout):
        super(GCN_graph_bn, self).__init__()

        self.conv1 = GCNConv(input_dim, dim1)
        self.bn1 = nn.BatchNorm1d(num_features=dim1)
        self.conv2 = GCNConv(dim1, dim2)
        self.bn2 = nn.BatchNorm1d(num_features=dim2)
        self.lin1 = Linear(dim2, dim3)
        self.bn3 = nn.BatchNorm1d(num_features=dim3)
        self.lin2 = Linear(dim3, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        x = x.to(torch.device('cuda'))
        edge_index = edge_index.to(torch.device('cuda'))

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        emb_conv1 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # print(x.shape)  # torch.Size([317, 64]), 16张图的节点总数
        x = global_max_pool(x, batch)
        # print(x.shape)  # torch.Size([16, 64]), 16张图对应特征
        emb_conv2 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin1(x)
        x = self.bn3(x)
        x = F.relu(x)
        emb_lin1 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        outEmb = x
        return F.log_softmax(x, dim=-1)


class GAT_graph_bn(torch.nn.Module):
    def __init__(self, input_dim, dim1, dim2, dim3, num_classes, dropout, heads):
        super(GAT_graph_bn, self).__init__()

        self.conv1 = GATConv(input_dim, dim1, heads)
        self.bn1 = nn.BatchNorm1d(num_features=dim1 * heads)
        self.conv2 = GATConv(dim1 * heads, dim2, concat=False)
        self.bn2 = nn.BatchNorm1d(num_features=dim2)
        self.lin1 = Linear(dim2, dim3)
        self.bn3 = nn.BatchNorm1d(num_features=dim3)
        self.lin2 = Linear(dim3, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        x = x.to(torch.device('cuda'))
        edge_index = edge_index.to(torch.device('cuda'))

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        emb_conv1 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = global_max_pool(x, batch)
        emb_conv2 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin1(x)
        x = self.bn3(x)
        x = F.relu(x)
        emb_lin1 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        outEmb = x
        return F.log_softmax(x, dim=-1)


class GraphSAGE_graph_bn(torch.nn.Module):
    def __init__(self, input_dim, dim1, dim2, dim3, num_classes, dropout):
        super(GraphSAGE_graph_bn, self).__init__()

        self.conv1 = SAGEConv(input_dim, dim1)
        self.bn1 = nn.BatchNorm1d(num_features=dim1)
        self.conv2 = SAGEConv(dim1, dim2)
        self.bn2 = nn.BatchNorm1d(num_features=dim2)
        self.lin1 = Linear(dim2, dim3)
        self.bn3 = nn.BatchNorm1d(num_features=dim3)
        self.lin2 = Linear(dim3, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        x = x.to(torch.device('cuda'))
        edge_index = edge_index.to(torch.device('cuda'))

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        emb_conv1 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = global_max_pool(x, batch)
        emb_conv2 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin1(x)
        x = self.bn3(x)
        x = F.relu(x)
        emb_lin1 = x
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin2(x)
        outEmb = x
        return F.log_softmax(x, dim=-1)


















# model-dataloader
class GCN_graph_bn_batch(torch.nn.Module):
    def __init__(self, input_dim, dim1, dim2, dim3, num_classes, dropout):
        super(GCN_graph_bn_batch, self).__init__()

        self.conv1 = GCNConv(input_dim, dim1)
        self.bn1 = nn.BatchNorm1d(num_features=dim1)
        self.conv2 = GCNConv(dim1, dim2)
        self.bn2 = nn.BatchNorm1d(num_features=dim2)
        self.lin1 = Linear(dim2, dim3)
        self.bn3 = nn.BatchNorm1d(num_features=dim3)
        self.lin2 = Linear(dim3, num_classes)
        self.dropout = dropout

    def forward(self, dataloader):

        output, label = [], []
        output = torch.tensor(output).to(torch.device('cuda'))
        label = torch.tensor(label).to(torch.device('cuda'))

        for data in dataloader:

            data = data.to(torch.device('cuda'))
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            emb_conv1 = x
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)

            # print(x.shape)  # torch.Size([317, 64]), 16张图的节点总数
            x = global_max_pool(x, batch)
            # print(x.shape)  # torch.Size([16, 64]), 16张图对应特征
            emb_conv2 = x
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.lin1(x)
            x = self.bn3(x)
            x = F.relu(x)
            emb_lin1 = x
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.lin2(x)
            outEmb = x
            x = F.log_softmax(x, dim=-1)

            # print(type(x)) # <class 'torch.Tensor'>
            # print(x.shape) # torch.Size([16, 2])

            output = torch.cat((output, x), dim=0)
            label = torch.cat((label, data.y), dim=0)

        label = label.to(torch.int64)
        return output, label

