# preprocess_data.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

print("Loading data...")
df = pd.read_csv('drugbank_ddi_data.csv')
print(f"✓ Loaded {len(df)} interactions")

# Get unique drugs and their SMILES
print("Building SMILES dictionary...")
drug_smiles = {}

for _, row in df.iterrows():
    if pd.notna(row['Drug1_ID']) and pd.notna(row['Drug1']):
        drug_smiles[row['Drug1_ID']] = row['Drug1']
    if pd.notna(row['Drug2_ID']) and pd.notna(row['Drug2']):
        drug_smiles[row['Drug2_ID']] = row['Drug2']

print(f"✓ Found {len(drug_smiles)} unique drugs with SMILES")

# Convert SMILES to Morgan fingerprints
print("Converting SMILES to fingerprints...")
fp_dict = {}
for drug_id, smiles in tqdm(drug_smiles.items()):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_dict[drug_id] = np.array(fp, dtype=np.float32)
        else:
            fp_dict[drug_id] = np.zeros(2048)
    except:
        fp_dict[drug_id] = np.zeros(2048)

print(f"✓ Generated fingerprints for {len(fp_dict)} drugs")

# Create paired fingerprints for DDI
print("Creating paired drug fingerprints...")
X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    drug1_id = row['Drug1_ID']
    drug2_id = row['Drug2_ID']
    label = row['Y']
    if drug1_id in fp_dict and drug2_id in fp_dict:
        fp_pair = np.concatenate([fp_dict[drug1_id], fp_dict[drug2_id]])
        X.append(fp_pair)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✓ Created {len(X)} paired fingerprints")
print(f"✓ Feature shape: {X.shape}")
print(f"✓ Label distribution: {np.unique(y, return_counts=True)}")

# Save preprocessed data
np.save('X_features.npy', X)
np.save('y_labels.npy', y)
print("✓ Saved features to X_features.npy")
print("✓ Saved labels to y_labels.npy")
