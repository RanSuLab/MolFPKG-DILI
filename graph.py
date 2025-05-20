'''
    1. 读取文件，label、drug_sim（>0.5）、drug_kg
    2. 将drug_kg文件中的边类型改为1 2 3 4
    3. 生成drug节点特征，这里直接读取的生成好的文件
    5. 文件中节点名称使用的str，将其统一进行索引，修改为索引值，便于之后生成Data
    6. 生成Data所需的edge和y
    7. 两种类型的Data，包含drug-drug边、不包含drug-drug边

'''


import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')
df_drugkg = pd.read_csv('data/kg_dilirank_all_e_t_c.csv')

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
y = torch.tensor(df_drug['label'].values)  # 有无毒性

data_drugsim = Data(x=y, edge_index=edge_index, edge_type=edge_type, y=y)
data_drugsim.drug_list = df_drug['drug_name'].tolist()
torch.save(data_drugsim, 'data/graph.pt')
# ------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------
# # 包含drug-drug边，注意修改文件
# df_drugsim = pd.read_csv('data/drug_similarity_05.csv')
# df_drugsim['Drug_1'] = df_drugsim['Drug_1'].replace(unique_node, unique_index)
# df_drugsim['Drug_2'] = df_drugsim['Drug_2'].replace(unique_node, unique_index)

# # edges
# sourcenode = df_drugsim['Drug_1'].to_list() + df_drugkg['source'].to_list()
# destinnode = df_drugsim['Drug_2'].to_list() + df_drugkg['target'].to_list()

# edge_index = torch.tensor([sourcenode + destinnode, destinnode + sourcenode], dtype=torch.long)
# relation_list = df_drugsim['Relation'].to_list() + df_drugkg['Relation'].to_list()
# edge_type = torch.tensor(relation_list*2, dtype=torch.long)

# # node labels
# y = torch.tensor(df_drug['label'].values)  # 有无毒性

# # 包含drug-drug边
# data_drugsim = Data(x=y, edge_index=edge_index, edge_type=edge_type, y=y)
# data_drugsim.drug_list = df_drug['drug_name'].tolist()
# torch.save(data_drugsim, 'data/graph_drugsim_05.pt')
# ------------------------------------------------------------------------------------------


