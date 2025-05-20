# MolFPKG-DILI

## introduction
Our research introduces MolFPKG-DILI, an advanced model that integrates multiple feature sources—molecular fingerprints, molecular graphs, and expansive knowledge graphs with diverse drug-related data—to enhance prediction accuracy and detail. This model not only predicts the presence of DILI but also discerns varying degrees of liver injury severity. Utilizing attention-based feature fusion and relational graph convolutional networks, MolFPKG-DILI outperforms existing models, offering nuanced insights into DILI levels. This method is anticipated to serve as a dependable tool for toxicity risk assessment in early drug development, reducing the likelihood of development failures attributed to liver toxicity.


## data
1. kg_dilirank_all_e_t_c_toxicity.csv: Contains the names, SMILES, and toxicity labels (No-DILI, Less-DILI and Most-DILI) of 823 drugs. Column names: drug_name, CanonicalSMILES, most_label, less_label, label.
2. kg_dilirank_all_e_t_c.csv: Knowledge graph data consisting of 823 drugs and 7 other entities, containing relationships: drug-indication, drug-target, drug-gene, drug-pathway, drug-transporter, drug-enzyme, drug-carrier.
3. drug_similarity_0x.csv (5 files): Drug-drug relationships at different similarity thresholds, generated using the get_drug_similarity.py file.
4. xxx_fp_list.npy (3 files): Molecular fingerprint features, generated using get_drug_fp_fea.py.
5. mol_graph.pt: Molecular graph features, generated using get_drug_mol_graph.py.
6. graph_drugsim_0x.pt (5 files): Usable knowledge graphs generated from the relationship edges in the knowledge graph, generated using graph.py.
7. graph.py: Knowledge graph that does not include drug-drug relationships, generated using graph.py.


## data-tox
1. xxx_fp_list_tox.npy (3 files): Molecular fingerprint features, generated using graph_tox.py.
mol_graph.pt: Molecular graph features, generated using graph_tox.py.
2. graph_drugsim_0x.pt (5 files): Usable knowledge graphs generated from the relationship edges in the knowledge graph, generated using graph_tox.py.
3. graph.py: Knowledge graph that does not include drug-drug relationships, generated using graph_tox.py.


## code
1. get_drug_similarity.py: Calculates Morgan fingerprints and computes similarities.
2. get_drug_fp_fea.py: Extracts molecular fingerprint features of drugs.
3. get_drug_mol_graph.py: Extracts molecular graph features of drugs.
4. graph.py: Retrieves the knowledge graph of drugs.
5. train.py: Performs toxicity prediction tasks using the MolFPKG-DILI model.
6. models.py: The MolFPKG-DILI model.
7. utils.py: Additional functions used during training.


## code-tox
1. graph_tox.py: Randomly selects toxic drugs and generates corresponding molecular fingerprint features, molecular graph features, and knowledge graph data.
2. train_tox.py: Performs strong and weak toxicity prediction tasks using the MolFPKG-DILI model.

## How to use
1. Use get_drug_similarity.py to generate drug_similarity_0x.csv.
2. Use get_drug_fp_fea.py to generate molecular fingerprint data xxx_fp_list.npy.
3. Use get_drug_mol_graph.py to generate molecular graph data mol_graph.pt.
4. Use graph.py to generate knowledge graph data graph_drugsim_0x.pt.
5. Use train.py to train the model and obtain results.

The process for the toxicity classification task is similar.