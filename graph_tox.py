import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from get_drug_mol_graph import get_mol_adj, get_mol_feature

# load data
df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')
df_drugkg = pd.read_csv('data/kg_dilirank_all_e_t_c.csv')
df_drugsim = pd.read_csv('data/drug_similarity_05.csv')


# PART1：随机选择有毒药物
df_drug_tox_1 = df_drug[df_drug['most_label'] == 1] # 选取强毒药物
df_drug_tox_0 = df_drug[df_drug['less_label'] == 1].sample(n=len(df_drug_tox_1), random_state=114514) # 从弱毒药物中随机选取等额药物
df_drug_tox = pd.concat([df_drug_tox_1, df_drug_tox_0], ignore_index=True)

drug_list_tox = df_drug_tox['drug_name'].tolist()  # 有毒训练数据-drug list
drug_list_tox_index = [df_drug['drug_name'].tolist().index(x) for x in drug_list_tox]  # 在df_drug中的index

df_drug = df_drug_tox.reset_index(drop=True)  # 筛选df_drug
df_drugkg = df_drugkg[df_drugkg['source'].isin(drug_list_tox)] # 筛选df_drugkg
df_drugsim = df_drugsim[df_drugsim['Drug_1'].isin(drug_list_tox) & df_drugsim['Drug_2'].isin(drug_list_tox)] # 筛选df_drugsim

# 输出药物毒性分布
print(df_drug['label'].value_counts())
print(df_drug['most_label'].value_counts())
print(df_drug['less_label'].value_counts())
# 弱毒：153、强毒：153
print('----------------------- PART 1: Select drugs finished! -----------------------------------')




# PART2：根据筛选药物的index获得对应的分子指纹
maccs_fea_array = np.load('data/maccs_fp_list.npy')
morgan_fea_array = np.load('data/morgan_fp_list.npy')
ErG_fea_array = np.load('data/ErG_fp_list.npy')

np.save('data_tox/maccs_fp_list_tox.npy', maccs_fea_array[drug_list_tox_index])
np.save('data_tox/morgan_fp_list_tox.npy', morgan_fea_array[drug_list_tox_index])
np.save('data_tox/ErG_fp_list_tox.npy', ErG_fea_array[drug_list_tox_index])
print('----------------------- PART 2: Get fingerprint finished! -----------------------------------')




# PART3：生成分子图对应graph
data_list = []
for index, row in df_drug.iterrows():
    smiles = row['CanonicalSMILES']
    adj = get_mol_adj(smiles)
    atom_fea = get_mol_feature(smiles)
    node_feature = torch.tensor(atom_fea)

    # 将adj转化为edge_index和edge_type
    sourcenode, destinnode = [], []
    edge_index, edge_type = [], []
    size = adj.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            if adj[i, j] != 0:
                sourcenode.append(i)
                destinnode.append(j)
                edge_type.append(adj[i, j])

    edge_index = torch.tensor([sourcenode + destinnode, destinnode + sourcenode], dtype=torch.long)
    edge_type = torch.tensor(edge_type*2, dtype=torch.long)

    # node labels
    y = row['most_label']
    data = Data(x=node_feature, edge_index=edge_index, edge_type=edge_type, y=y)
    data_list.append(data)
    print(index, '/', len(df_drug))

torch.save(data_list, 'data_tox/mol_graph.pt')
print('----------------------- PART 3: Get drug graph finished! -----------------------------------')




# PART4：生成有毒药物对应的知识图谱

# df_drug_kg新增relation列，标注边类型
df_drugkg['Relation'] = 0  # 新增列
edge_type_list = ['drug', 'has_indication', 'has_gene', 'has_pathway', 'has_target', 'has_enzyme', 'has_transporter', 'has_carrier'] # 用于将str转为对应index
for index, row in df_drugkg.iterrows():
    df_drugkg.loc[index, 'Relation'] = int(edge_type_list.index(row['edge']))

# 将所有str，获取唯一值，编辑索引
unique_drug = df_drugkg['source'].unique()
unique_other = df_drugkg['target'].unique()
unique_node = np.concatenate((unique_drug, unique_other), axis=0).tolist()
unique_index = range(len(unique_node))
print("特征个数: ", len(unique_other))  # 需要在模型中生成的特征个数

df_drugkg['source'] = df_drugkg['source'].replace(unique_node, unique_index)
df_drugkg['target'] = df_drugkg['target'].replace(unique_node, unique_index)

# ------------------------------------------------------------------------------------------
# 没有drug-drug边
sourcenode = df_drugkg['source'].to_list()
destinnode = df_drugkg['target'].to_list()

edge_index = torch.tensor([sourcenode + destinnode, destinnode + sourcenode], dtype=torch.long)
relation_list = df_drugkg['Relation'].to_list()
edge_type = torch.tensor(relation_list*2, dtype=torch.long)

# node labels
y = torch.tensor(df_drug['most_label'].values)

data_drugsim = Data(x=y, edge_index=edge_index, edge_type=edge_type, y=y)
data_drugsim.drug_list = df_drug['drug_name'].tolist()  # 添加drug_list属性的代码
torch.save(data_drugsim, 'data_tox/graph.pt')
# ------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------
# # 包含drug-drug边，注意修改文件
# df_drugsim['Drug_1'] = df_drugsim['Drug_1'].replace(unique_node, unique_index)
# df_drugsim['Drug_2'] = df_drugsim['Drug_2'].replace(unique_node, unique_index)
# sourcenode = df_drugsim['Drug_1'].to_list() + df_drugkg['source'].to_list()
# destinnode = df_drugsim['Drug_2'].to_list() + df_drugkg['target'].to_list()

# edge_index = torch.tensor([sourcenode + destinnode, destinnode + sourcenode], dtype=torch.long)
# relation_list = df_drugsim['Relation'].to_list() + df_drugkg['Relation'].to_list()
# edge_type = torch.tensor(relation_list*2, dtype=torch.long)

# # node labels
# y = torch.tensor(df_drug['most_label'].values)

# data_drugsim = Data(x=y, edge_index=edge_index, edge_type=edge_type, y=y)
# data_drugsim.drug_list = df_drug['drug_name'].tolist()
# torch.save(data_drugsim, 'data_tox/graph_drugsim_05.pt')
# # ------------------------------------------------------------------------------------------

print('----------------------- PART 4: Get drug knowledge graph finished! --------------------------')

