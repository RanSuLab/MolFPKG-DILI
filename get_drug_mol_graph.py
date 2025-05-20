'''
    smiles生成分子图
'''

import torch
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem

atom_type_max = 100  # 原子类型最大数量
atom_f_dim = 133  # 原子特征维度

# 每种特征取值范围
atom_features_define = {
    'atom_symbol': list(range(atom_type_max)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],}

# 给定键值进行编码，one-hot
def onek_encoding_unk(key,length):
    encoding = [0] * (len(length) + 1)
    index = length.index(key) if key in length else -1
    encoding[index] = 1

    return encoding


# 获取原子的每个特征，在特征取值范围中找到对应index，拼接得到最终特征
def get_atom_feature(atom):
    feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
              onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
              onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
              onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
              onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
              onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
              [1 if atom.GetIsAromatic() else 0] + [atom.GetMass() * 0.01]
    return feature


# 生成分子中原子节点特征——根据smiles生成该drug的分子图节点特征
def get_mol_feature(smiles):

    mol = Chem.MolFromSmiles(smiles)
    atom_num = mol.GetNumAtoms()
    atom_feature = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature.append(get_atom_feature(atom))
    atom_feature = [atom_feature[i] for i in range(atom_num)]

    # print(len(atom_feature[0])) # 133
    # print(len(atom_feature)) # 13
    # shape=(atom_num, 133), 每个原子有133维特征

    return atom_feature

# 生成分子的adj
def get_mol_adj(smiles):

    mol = Chem.MolFromSmiles(smiles)
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
                Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
                Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
                Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
                Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
                Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
                Chem.rdchem.BondType.ZERO]
    for bond in mol.GetBonds():
        adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
        adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())

    return adjacency_matrix


# 读取文件：drug_name,CanonicalSMILES,most_label,less_label,label
df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')


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
    y = row['label']  # 有无毒性

    data = Data(x=node_feature, edge_index=edge_index, edge_type=edge_type, y=y)
    data_list.append(data)
    print(index, '/', len(df_drug))

torch.save(data_list, 'data/mol_graph.pt')

