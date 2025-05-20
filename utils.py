import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn import metrics, utils
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# 得到五折对应的Data
def get_data_fold5(i, ss, data):

    train_index_list, test_index_list = [], []
    # 划分5折数据中分别对应的drug_list
    for train_index, test_index in ss.split(data.x, data.y):
        train_index_list.append(train_index)
        test_index_list.append(test_index)
        
    train_idx = train_index_list[int(i)-1].tolist()
    test_idx = test_index_list[int(i)-1].tolist()

    return train_idx, test_idx



def evaluate(model, fp_data, graph_dataloader, kg_data, data_idx):

    with torch.no_grad():
        model.eval()
        out, att = model(fp_data, graph_dataloader, kg_data.edge_index, kg_data.edge_type)
        out_test = out.exp()[data_idx]
        pred = out_test.argmax(dim=-1)
        label = kg_data.y[data_idx]

        output_loss = F.nll_loss(out[data_idx], kg_data.y[data_idx])

    return pred.cpu(), label.cpu(), output_loss.item(), att, out_test[:, 1].cpu().detach().numpy()


# 计算混淆矩阵
def compute_confusion_matrix(precited, expected):

    precited = np.asarray(precited)
    expected = np.asarray(expected)
    part = precited ^ expected             # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)             # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    if len(pcount) == 1:
        pcount = np.append(pcount, [0])
    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = pcount[0] - tp                    # 统计TN的个数
    fn = pcount[1] - fp                    # 统计FN的个数
    return tp, fp, tn, fn


# 计算常用指标
def compute_indexes(tp, fp, tn, fn):

    accuracy = (tp+tn) / (tp+tn+fp+fn)         # 准确率
    if tp+fp == 0:
        precision = 0.0000
    else:
        precision = tp / (tp+fp)               # 精确率
    if tp+fn == 0:
        recall = 0.0000
    else:
        recall = tp / (tp+fn)                  # 召回率/敏感性
    if tn+fp == 0:
        spe = 0.0000
    else:
        spe = tn / (tn+fp)                     # 特异性
    if precision+recall == 0:
        F1 = 0.0000
    else:
        F1 = (2*precision*recall) / (precision+recall)    # F1-score
    return accuracy, precision, recall, spe, F1
