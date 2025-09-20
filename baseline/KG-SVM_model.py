import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import utils
from gensim.models import KeyedVectors


# # 读取数据
# df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')
# word_vectors = KeyedVectors.load("data/node_embedding_800_sim_8.kv")  # node2vec嵌入向量
# df_drug = df_drug[['drug_name', 'label']]

# 强弱毒性
df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity_tox.csv')
word_vectors = KeyedVectors.load("data/node_embedding_800_tox_sim_8.kv")
df_drug = df_drug[['drug_name', 'most_label']]

# 筛选所有drug对应向量，获得dataframe
df_drug['emdedding'] = None  # 初始化新列
for index, row in df_drug.iterrows():
    df_drug.at[index, 'emdedding'] = word_vectors[row['drug_name']]


# 使用StratifiedShuffleSplit进行随机划分
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# 遍历参数组合
# c_list = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
# c_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
c_list = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# c_list = [0.01, 0.05, 0.1, 0.5]
# kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_list = ['rbf']
for c in c_list:

    print('-----------------------------------c = {} -----------------------------------'.format(c))
    for kernel_selected in kernel_list:
        # print('-----------------------------------kernel = {} -----------------------------------'.format(kernel_selected))
        acc_list, res_list = [], []
        count = 0
        # for train_index, test_index in sss.split(df_drug['emdedding'], df_drug['label']):
        for train_index, test_index in sss.split(df_drug['emdedding'], df_drug['most_label']):
            count = count + 1    
            # print('-----------------------------------第 {} 折训练-----------------------------------'.format(count))

            X_train, X_test = df_drug['emdedding'][train_index].tolist(), df_drug['emdedding'][test_index].tolist()
            # y_train, y_test = df_drug['label'][train_index], df_drug['label'][test_index]
            y_train, y_test = df_drug['most_label'][train_index], df_drug['most_label'][test_index]


            # 使用SVM算法进行分类
            # print("-----------------开始训练-----------------")
            clf = SVC(kernel = kernel_selected, C=c)
            clf.fit(X_train, y_train)


            #Test on Training data
            train_result = clf.predict(X_train)
            acc_train = accuracy_score(y_train, train_result)
            # print('Training precision: ', acc_train)
            
            #Test on test data
            test_result = clf.predict(X_test)
            acc_test = accuracy_score(y_test, test_result)
            # print('Test precision: ', acc_test)
            acc_list.append(acc_test)

            # print('y_pred:', test_result.tolist())
            # print('y_true:', y_test.tolist())
            
            tp, fp, tn, fn = utils.compute_confusion_matrix(test_result, y_test)
            accuracy, precision, recall, specificity, F1 = utils.compute_indexes(tp, fp, tn, fn)
            auc_score = roc_auc_score(y_test, test_result) # auc
            res = [accuracy, precision, recall, specificity, F1, auc_score]
            res_list.append(res)

            # print()



        acc_avg = np.mean([item[0] for item in res_list])
        pre_avg = np.mean([item[1] for item in res_list])
        sen_avg = np.mean([item[2] for item in res_list])
        spe_avg = np.mean([item[3] for item in res_list])
        f1_avg = np.mean([item[4] for item in res_list])
        auc_avg = np.mean([item[5] for item in res_list])

        # print()
        # print('----------------------------------------------------------------------------------------------')
        print("avgAcc: {:.4f}、 avgPre: {:.4f}、 avgSen: {:.4f}、 avgSpe: {:.4f}、 avgF1: {:.4f}、 avgAUC: {:.4f}"
                .format(acc_avg, pre_avg, sen_avg, spe_avg, f1_avg, auc_avg))
        print("acc_list: ", acc_list)

        # print()
        # print('----------------------------------------------------------------------------------------------')
        # print('----------------------------------------------------------------------------------------------')

        print()

