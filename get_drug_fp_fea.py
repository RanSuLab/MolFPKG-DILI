import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from pubchemfp import GetPubChemFPs

# 1. load data
df_drug = pd.read_csv('data/kg_dilirank_all_e_t_c_toxicity.csv')
# fp_type = 'morgan' # 1024
# fp_type = 'maccs' # 167
fp_type = 'ErG' # 441

# 生成分子指纹
fp_list=[]
for i, smiles in enumerate(df_drug['CanonicalSMILES']):
    fp=[]
    mol = Chem.MolFromSmiles(smiles)
    
    if fp_type == 'morgan':
        # morgan指纹就是ECFP，参数(mol, radius半径, nbits)
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp.extend(fp_morgan)
    elif fp_type == 'maccs':
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        fp.extend(fp_maccs)
    elif fp_type == 'ErG':
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        fp.extend(fp_phaErGfp)
    fp_list.append(fp)

fp_array = np.array(fp_list)
# np.save('data/morgan_fp_list.npy', fp_array)
# np.save('data/maccs_fp_list.npy', fp_array)
np.save('data/ErG_fp_list.npy', fp_array)

