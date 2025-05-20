import os
import gc
import time
import utils
import torch
import visdom
import argparse
import numpy as np
import torch.nn.functional as F
from  sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from models import MolFPKG_DILI_tox
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader


import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='MolFPKG_DILI-tox')

parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--model', type=str, default='MolFPKG_DILI_tox',
                    help='Choose model : MolFPKG_DILI_tox')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg', type=float, default=0.01, metavar='R',
                    help='weight decay (default: 0.01)')  # 10e-5
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--opt', type=str, default='Adam',
                    help='optimizers')
parser.add_argument('--viz', type=str, default='MolFPKG_DILI_tox',
                    help='visdom environment: MolFPKG_DILI_tox')
parser.add_argument('--batchSize', type=int, default=16,
                    help='batchsize (default: 16)')

args = parser.parse_args()
torch.manual_seed(args.seed)

def main(args, i, fp_data, data_path, save_res_path, save_model_path, ss):

    print('———————————— 加载数据集 ————————————')
    # load data
    fp_data = torch.from_numpy(fp_data).float()
    graph_data = torch.load(data_path + 'mol_graph.pt')
    graph_dataloader = DataLoader(graph_data, batch_size=args.batchSize, shuffle=False)
    # kg_data = torch.load(data_path + 'graph_drugsim_09.pt')
    kg_data = torch.load(data_path + 'graph.pt')
    train_idx, test_idx = utils.get_data_fold5(i, ss, kg_data)

    train_num, test_num = len(train_idx), len(test_idx)
    print('训练数据集：', train_num, '测试数据集：', test_num)
    print("graph_dataloader len:", len(graph_dataloader))
    print("normal fp len:", len(fp_data[0]))


    mlp_input_dim = len(fp_data[0])
    mlp_hidden_dim = 512
    mlp_output_dim = 256

    gcn_input_dim = 133
    gcn_hidden_dim = 512
    gcn_output_dim = 256

    dim1 = 256
    dim2 = 128
    dim3 = 64


    print('———————————— 初始化模型 ————————————')
    num_classes = 2
    num_nodes = 3684

    if args.model == 'MolFPKG_DILI_tox':
        model = MolFPKG_DILI_tox(mlp_input_dim, mlp_hidden_dim, mlp_output_dim, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, dim1, dim2, dim3, num_nodes, num_classes, args.dropout)
    print("model:{}".format(model))

    print('Model:{}'.format(args.model))
    print('lr:{}'.format(args.lr))
    print('epochs:{}'.format(args.epochs))
    print('dropout:{}'.format(args.dropout))
    print('reg:{}'.format(args.reg))
    print('optimizer:{}'.format(args.opt))
    print('visdom:{}'.format(args.viz))
    print('batchSize:{}'.format(args.batchSize))


    # 定义优化器
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'SGD2':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg, momentum=0.76)


    # 定义早停的参数
    best_loss = float('inf')
    best_acc = 0.0
    patience = 5   # 允许连续多少个epoch的损失上升
    counter = 0  # 记录连续上升的epoch数
    
    '''
        ReduceLROnPlateau自适应调整学习率: 当验证集的 loss 不再下降时，进行学习率调整；
                                        或者监测验证集的 accuracy, 当accuracy 不再上升时，则调整学习率；
        min: 当优化的指标不再下降时, 学习率将减小
        factor: 学习率每次降低多少
        patience: 几个epoch不变时, 才改变学习速率, 默认为10
        verbose: 是否打印出信息
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, verbose=True, patience=5)

    viz = visdom.Visdom(env=args.viz, use_incoming_socket=False)


    print('—————————————————— Start Training ——————————————————')
    dur = list()
    
    model.train() # 表示模型开始训练
    for epoch in range(1, args.epochs + 1):

        # 训练模型
        t0 = time.time()
        optimizer.zero_grad()
        out, att = model(fp_data, graph_dataloader, kg_data.edge_index, kg_data.edge_type)
        loss = F.nll_loss(out[train_idx], kg_data.y[train_idx])
        loss.backward()
        optimizer.step()

        out = out.detach().cpu()
        train_loss = loss.item()

        train_pred_list, train_label_list, non_loss, att, non_pred_probability = utils.evaluate(model, fp_data, graph_dataloader, kg_data, train_idx)
        test_pred_list, test_label_list, test_loss, att, test_pred_probability = utils.evaluate(model, fp_data, graph_dataloader, kg_data, test_idx)

        train_acc = accuracy_score(train_label_list, train_pred_list)
        test_acc = accuracy_score(test_label_list, test_pred_list)

        dur.append(time.time() - t0)
        print(f'Epoch: {epoch:03d}, Time(s) {np.mean(dur):.4f}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


        acc_win = 'Accuracy-'+i
        loss_win = 'Loss-'+i
        if epoch == 1:
            viz.line(
                [[float(train_loss), float(test_loss)]],
                [epoch],
                win=loss_win,
                opts=dict(title=loss_win,
                          legend=['train_loss', 'test_loss']
                          )
            )
            viz.line(
                [[float(train_acc), float(test_acc)]],
                [epoch],
                win=acc_win,
                opts=dict(title=acc_win,
                          legend=['train_acc', 'test_acc']
                          )
            )
        else:
            viz.line(
                [[float(train_loss), float(test_loss)]],
                [epoch],
                win=loss_win,
                update='append' 
            )
            viz.line(
                [[float(train_acc), float(test_acc)]],
                [epoch],
                win=acc_win,
                update='append'
            )



        # 当验证集的 Acc 不再上升时，进行学习率调整
        # scheduler.step(test_acc) # max
        # scheduler.step(test_loss) # min

        # 判断是否早停
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

        gc.collect()

    # 计算指标 test_loss, test_acc, test_pre, test_sen, test_spe, test_f1
    test_pred_list, test_label_list, test_loss, att, test_pred_probability = utils.evaluate(model, fp_data, graph_dataloader, kg_data, test_idx)
    test_acc = accuracy_score(test_label_list, test_pred_list)
    auc_score = roc_auc_score(test_label_list, test_pred_probability) # auc
    
    tp, fp, tn, fn = utils.compute_confusion_matrix(test_pred_list, test_label_list)
    accuracy, precision, recall, specificity, F1 = utils.compute_indexes(tp, fp, tn, fn)
    res = [accuracy, precision, recall, specificity, F1, auc_score]

    print('-----------------------第{}折结果-----------------------------------'.format(i))
    print("att:", att)
    print("y_pred:", test_pred_list)
    print("y_true:", test_label_list)    
    print(f"TP: {tp}、 FP: {fp}、 TN: {tn}、 FN: {fn}")
    print("Accuracy: {:.4f}、 Precision: {:.4f}、 Sensitivity/Recall: {:.4f}、 Specificity: {:.4f}、 F1: {:.4f}、 AUC: {:.4f}".format(accuracy, precision, recall, specificity, F1, auc_score))
    print('-------------------------------------------------------------------')
    print()


    # 保存模型+参数
    torch.save(model, save_model_path + 'res_'+ args.model +'/f' + str(i) + '_model.pth')
    # 写入结果文件
    with open(save_res_path + 'res_'+ args.model + '/' + args.model +'_result.txt', 'a') as test_res:
        test_res.write('Fold' + str(i) + '......\n')
        test_res.write('test acc: ' + str(test_acc) + '\n')
        test_res.write('test   pred: ' + str(test_pred_list) + '\n')
        test_res.write('test labels: ' + str(test_label_list) + '\n')
    
    return res


# 交叉验证
def K_fold_train(data_path, save_res_path, save_model_path, k):

    
    # 5 fold交叉验证，分层划分数据
    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=5)

    res_list = []
    
    maccs_fea_array = np.load('data_tox/maccs_fp_list_tox.npy')
    morgan_fea_array = np.load('data_tox/morgan_fp_list_tox.npy')
    ErG_fea_array = np.load('data_tox/ErG_fp_list_tox.npy')
    fp_data = np.concatenate((maccs_fea_array, morgan_fea_array, ErG_fea_array), axis=1)

    for i in range(k):
        print('----------------------------------------开始第 {} 折训练----------------------------------------'.format(i+1))
        fold_num = str(i + 1)
        res = main(args, fold_num, fp_data, data_path, save_res_path, save_model_path, ss)
 
        res_list.append(res)


    acc_list = [item[0] for item in res_list]

    acc_avg = np.mean([item[0] for item in res_list])
    pre_avg = np.mean([item[1] for item in res_list])
    sen_avg = np.mean([item[2] for item in res_list])
    spe_avg = np.mean([item[3] for item in res_list])
    f1_avg = np.mean([item[4] for item in res_list])
    auc_avg = np.mean([item[5] for item in res_list])


    print('\n')
    print('----------------------------------------------------------------------------------------------')
    print('------------------------The average metrical of {} fold is :-----------------------------------:'.format(k))
    print("avgAcc: {:.4f}、 avgPre: {:.4f}、 avgSen: {:.4f}、 avgSpe: {:.4f}、 avgF1: {:.4f}、 avgAUC: {:.4f}"
          .format(acc_avg, pre_avg, sen_avg, spe_avg, f1_avg, auc_avg))
    print("acc_list: ", acc_list)
    print('----------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------')



    # 写入结果文件
    with open(save_res_path + 'res_'+ args.model + '/' + args.model +'_result.txt', 'a') as test_res:
        test_res.write('The average result: ------------------------------------------------------------\n')
        test_res.write("avgAccuracy: {:.4f}、 avgPrecision: {:.4f}、 avgSensitivity/avgRecall: {:.4f}、 avgSpecificity: {:.4f}、 avgF1: {:.4f}、 avgAUC: {:.4f}"
                       .format(acc_avg, pre_avg, sen_avg, spe_avg, f1_avg, auc_avg))
        test_res.write("acc_list:" + str(acc_list))

        test_res.write(f"\n\n\n")


if __name__ == "__main__":

    
    kfold = 5
    data_path = 'data_tox/'
    save_res_path = 'result_tox/'
    save_model_path = 'model_tox/'
    K_fold_train(data_path, save_res_path, save_model_path, kfold)




