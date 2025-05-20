import torch
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import RGCNConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(2024)

class Attention(torch.nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()

        self.att = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.att(z)
        beta = torch.softmax(w, dim=1)
        res = torch.sum(beta * z, dim=1)
        return res, beta

# 分子指纹使用mlp、分子图使用gcn、注意力拼接特征输入kg，rgcn分类
class MolFPKG_DILI(torch.nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, mlp_output_dim, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, dim1, dim2, dim3, num_classes, dropout):
        super(MolFPKG_DILI, self).__init__()

        # fingerprint network
        self.fp_lin1 = Linear(mlp_input_dim, mlp_hidden_dim)
        self.fp_bn1 = torch.nn.BatchNorm1d(num_features=mlp_hidden_dim)
        self.fp_lin2 = Linear(mlp_hidden_dim, mlp_output_dim)
        self.fp_bn2 = torch.nn.BatchNorm1d(num_features=mlp_output_dim)  

        # mol_graph network
        self.gconv1 = GCNConv(gcn_input_dim, gcn_hidden_dim)
        self.g_bn1 = torch.nn.BatchNorm1d(num_features=gcn_hidden_dim)
        self.gconv2 = GCNConv(gcn_hidden_dim, gcn_output_dim)
        self.g_bn2 = torch.nn.BatchNorm1d(num_features=gcn_output_dim)

        # att feature fusion
        self.attencoder = Attention(gcn_output_dim, 128)

        # kg network
        self.gene_emb = Parameter(torch.randn(5414, gcn_output_dim))
        self.conv1 = RGCNConv(gcn_output_dim, dim1, 8)
        self.kg_bn1 = torch.nn.BatchNorm1d(num_features=dim1)
        self.conv2 = RGCNConv(dim1, dim2, 8)
        self.kg_bn2 = torch.nn.BatchNorm1d(num_features=dim2)
        self.lin1 = Linear(dim2, dim3)
        self.kg_bn3 = torch.nn.BatchNorm1d(num_features=dim3)
        self.lin2 = Linear(dim3, num_classes)

        self.dropout = dropout

    def forward(self, fp_data, graph_dataloader, kg_edge_index, kg_edge_type):

        # mlp
        fp_x = self.fp_lin1(fp_data)
        fp_x = self.fp_bn1(fp_x)
        fp_x = F.relu(fp_x)
        fp_x = F.dropout(fp_x, self.dropout, training=self.training)

        fp_x = self.fp_lin2(fp_x)
        fp_x = self.fp_bn2(fp_x)
        fp_x = F.relu(fp_x)

        # gcn
        graph_output = []
        graph_output = torch.tensor(graph_output)
        for data in graph_dataloader:
            # batch训练，加快速度，注意不打乱

            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            x = self.gconv1(x, edge_index)
            x = self.g_bn1(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.gconv2(x, edge_index)
            x = self.g_bn2(x)
            x = F.relu(x)
            
            x = global_mean_pool(x, batch)
            graph_output = torch.cat((graph_output, x), dim=0)

        # feature fusion
        embedding = torch.stack([graph_output, fp_x], dim=1)
        embedding, att = self.attencoder(embedding)

        # rgcn
        kg_x = embedding
        x = torch.cat((kg_x, self.gene_emb), dim=0)
        x = self.conv1(x, kg_edge_index, kg_edge_type)
        x = self.kg_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, kg_edge_index, kg_edge_type)
        x = self.kg_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin1(x)
        x = self.kg_bn3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.lin2(x)
        output = F.log_softmax(x, dim=-1)
        return output, att



class MolFPKG_DILI_tox(torch.nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, mlp_output_dim, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, dim1, dim2, dim3, num_nodes, num_classes, dropout):
        super(MolFPKG_DILI_tox, self).__init__()

        # fingerprint network
        self.fp_lin1 = Linear(mlp_input_dim, mlp_hidden_dim)
        self.fp_bn1 = torch.nn.BatchNorm1d(num_features=mlp_hidden_dim)
        self.fp_lin2 = Linear(mlp_hidden_dim, mlp_output_dim)
        self.fp_bn2 = torch.nn.BatchNorm1d(num_features=mlp_output_dim)  

        # mol_graph network
        self.gconv1 = GCNConv(gcn_input_dim, gcn_hidden_dim)
        self.g_bn1 = torch.nn.BatchNorm1d(num_features=gcn_hidden_dim)
        self.gconv2 = GCNConv(gcn_hidden_dim, gcn_output_dim)
        self.g_bn2 = torch.nn.BatchNorm1d(num_features=gcn_output_dim)

        # att feature fusion
        self.attencoder = Attention(gcn_output_dim, 128)

        # kg network
        self.gene_emb = Parameter(torch.randn(num_nodes, gcn_output_dim))
        self.conv1 = RGCNConv(gcn_output_dim, dim1, 8)
        self.kg_bn1 = torch.nn.BatchNorm1d(num_features=dim1)
        self.conv2 = RGCNConv(dim1, dim2, 8)
        self.kg_bn2 = torch.nn.BatchNorm1d(num_features=dim2)
        self.lin1 = Linear(dim2, dim3)
        self.kg_bn3 = torch.nn.BatchNorm1d(num_features=dim3)
        self.lin2 = Linear(dim3, num_classes)

        self.dropout = dropout

    def forward(self, fp_data, graph_dataloader, kg_edge_index, kg_edge_type):

        # mlp
        fp_x = self.fp_lin1(fp_data)
        fp_x = self.fp_bn1(fp_x)
        fp_x = F.relu(fp_x)
        fp_x = F.dropout(fp_x, self.dropout, training=self.training)

        fp_x = self.fp_lin2(fp_x)
        fp_x = self.fp_bn2(fp_x)
        fp_x = F.relu(fp_x)

        # gcn
        graph_output = []
        graph_output = torch.tensor(graph_output)
        for data in graph_dataloader:

            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            x = self.gconv1(x, edge_index)
            x = self.g_bn1(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            x = self.gconv2(x, edge_index)
            x = self.g_bn2(x)
            x = F.relu(x)

            x = global_mean_pool(x, batch)
            graph_output = torch.cat((graph_output, x), dim=0)

        # feature fusion
        embedding = torch.stack([graph_output, fp_x], dim=1)
        embedding, att = self.attencoder(embedding)

        # rgcn
        kg_x = embedding
        x = torch.cat((kg_x, self.gene_emb), dim=0)
        x = self.conv1(x, kg_edge_index, kg_edge_type)
        x = self.kg_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, kg_edge_index, kg_edge_type)
        x = self.kg_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lin1(x)
        x = self.kg_bn3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.lin2(x)
        output = F.log_softmax(x, dim=-1)
        return output, att


