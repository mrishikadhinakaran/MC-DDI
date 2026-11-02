import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from model import MC_DDI_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Adjust labels for zero-based indexing
y -= 1

# Create test set consistent with the original split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=128)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'y_true': y_true, 'y_pred': y_pred
    }

print("=== Loading Models ===")
with open('baseline_results.json', 'r') as f:
    baseline_results = json.load(f)

# Load baseline model
baseline_model = MC_DDI_Model(input_dim=4096, n_classes=86).to(device)
baseline_model.load_state_dict(torch.load('baseline_model.pth'))
baseline_test = evaluate_model(baseline_model, test_loader, device)

# Load self-trained model
mc_ddi_model = MC_DDI_Model(input_dim=4096, n_classes=86).to(device)
mc_ddi_model.load_state_dict(torch.load('mc_ddi_model_self_trained.pth'))
mc_ddi_test = evaluate_model(mc_ddi_model, test_loader, device)

# Print results
print("\n" + "="*60)
print("BASELINE MODEL (WITHOUT PSEUDO-LABELS)")
print("="*60)
for metric, value in {k: v for k, v in baseline_test.items() if k not in ['y_true', 'y_pred']}.items():
    print(f"{metric.upper():15}: {value:.4f}")

print("\n" + "="*60)
print("MC-DDI MODEL (WITH PSEUDO-LABELS)")
print("="*60)
for metric, value in {k: v for k, v in mc_ddi_test.items() if k not in ['y_true', 'y_pred']}.items():
    print(f"{metric.upper():15}: {value:.4f}")

print("\n" + "="*60)
print("IMPROVEMENT OVER BASELINE")
print("="*60)
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    baseline_val = baseline_test[metric]
    mc_ddi_val = mc_ddi_test[metric]
    improvement = ((mc_ddi_val - baseline_val) / baseline_val) * 100
    print(f"{metric.upper():15}: {improvement:+.2f}%")

# Save final results
final_results = {
    'baseline': {k: v for k, v in baseline_test.items() if k not in ['y_true', 'y_pred']},
    'mc_ddi': {k: v for k, v in mc_ddi_test.items() if k not in ['y_true', 'y_pred']},
    'improvements': {
        metric: ((mc_ddi_test[metric] - baseline_test[metric]) / baseline_test[metric]) * 100
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
}

with open('final_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("\n✓ Results saved to final_results.json")
