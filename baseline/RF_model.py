import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import utils
from sklearn.ensemble import RandomForestClassifier

# # 有无毒性-------------
# # 读取数据
# df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')
# # kg_dilirank_all_toxicity.csv: drug_name,CanonicalSMILES,most_label,less_label,label
 
# # maccs+morgan+ErG
# maccs_fea_array = np.load('data/maccs_fp_list.npy')
# morgan_fea_array = np.load('data/morgan_fp_list.npy')
# ErG_fea_array = np.load('data/ErG_fp_list.npy')

# 强弱毒性-------------
# 读取数据
df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity_tox.csv')
# kg_dilirank_all_toxicity.csv: drug_name,CanonicalSMILES,most_label,less_label,label
 
# maccs+morgan+ErG
maccs_fea_array = np.load('data/maccs_fp_list_tox.npy')
morgan_fea_array = np.load('data/morgan_fp_list_tox.npy')
ErG_fea_array = np.load('data/ErG_fp_list_tox.npy')


fp_array = np.concatenate((maccs_fea_array, morgan_fea_array, ErG_fea_array), axis=1)


# 使用StratifiedShuffleSplit进行随机划分
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# 遍历参数组合
n_list = [120, 300, 500, 800, 1200]

for n in n_list:

    print('-----------------------------------n_estimators = {} -----------------------------------'.format(n))

    acc_list, res_list = [], []
    count = 0
    for train_index, test_index in sss.split(fp_array, df_drug['most_label']):
        count = count + 1    
        # print('-----------------------------------第 {} 折训练-----------------------------------'.format(count))

        X_train, X_test = fp_array[train_index], fp_array[test_index]
        # y_train, y_test = df_drug['label'][train_index], df_drug['label'][test_index]
        y_train, y_test = df_drug['most_label'][train_index], df_drug['most_label'][test_index]


        # 使用RF算法进行分类
        # print("-----------------开始训练-----------------")
        clf = RandomForestClassifier(n_estimators=n)
        clf = clf.fit(X_train, y_train)


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


