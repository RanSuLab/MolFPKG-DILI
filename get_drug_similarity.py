'''
    使用RDKit的Chem.MolFromSmiles()函数将CanonicalSMILES转化为分子对象;
    使用AllChem.GetMorganFingerprint()函数计算分子的指纹;
    使用DataStructs.TanimotoSimilarity()函数计算指纹之间的相似性分数，并打印结果。
'''

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')
# df_drug = df_drug.head(10)

fp_list = []
for smiles in df_drug['CanonicalSMILES'].to_list():

    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprint(mol, 2)
    fp_list.append(fp)

# drug-drug similarity
sim_scores = []
for i in range(len(fp_list)-1):
    print(i, '/', len(fp_list)-1)

    for j in range(i+1, len(fp_list)):

        fp1 = fp_list[i]
        fp2 = fp_list[j]
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        # print('drug-', i, '&drug-', j, ' ：', similarity)
        sim_scores.append([df_drug.at[i, 'drug_name'], df_drug.at[j, 'drug_name'], similarity])

df_sim = pd.DataFrame(sim_scores, columns=['Drug_1', 'Drug_2', 'Similarity'])
df_drugsim = df_sim[df_sim['Similarity'] > 0.5].reset_index(drop=True)  # 此处修改相似度阈值
df_drugsim['Relation'] = [0]*df_drugsim.shape[0]  # relation指的是边类型，此处drug-drug为类型0
df_drugsim.to_csv('data/drug_similarity_05.csv', index=False)

print("ok!")