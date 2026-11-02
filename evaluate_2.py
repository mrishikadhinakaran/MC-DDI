import matplotlib
matplotlib.use('Agg')  # Prevent GUI blocking

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from model import MC_DDI_Model
import sys

print("=== Starting evaluate_2.py ===", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

# Load data
print("Loading data...", flush=True)
X = np.load('X_features.npy')
y = np.load('y_labels.npy')
y -= 1

print(f"Data loaded. Total samples: {len(X)}, Features: {X.shape[1]}", flush=True)

# Create test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=128)

print(f"Test set prepared: {len(X_test)} samples", flush=True)

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch.to(device))
            y_true.extend(y_batch.numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'y_true': y_true, 'y_pred': y_pred}

print("=== Loading Models ===", flush=True)
with open('baseline_results_2.json', 'r') as f:
    baseline_results = json.load(f)

baseline_model = MC_DDI_Model(input_dim=4096, n_classes=86).to(device)
baseline_model.load_state_dict(torch.load('baseline_model_2.pth'))
baseline_test = evaluate_model(baseline_model, test_loader)
print("✓ Baseline model evaluated.", flush=True)

mc_ddi_model = MC_DDI_Model(input_dim=4096, n_classes=86).to(device)
mc_ddi_model.load_state_dict(torch.load('mc_ddi_model_self_trained_2.pth'))
mc_ddi_test = evaluate_model(mc_ddi_model, test_loader)
print("✓ MC-DDI model evaluated.", flush=True)

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

# Save final summary
final_results = {
    'baseline': {k: v for k, v in baseline_test.items() if k not in ['y_true', 'y_pred']},
    'mc_ddi': {k: v for k, v in mc_ddi_test.items() if k not in ['y_true', 'y_pred']},
    'improvements': {
        metric: ((mc_ddi_test[metric] - baseline_test[metric]) / baseline_test[metric]) * 100
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
}
with open('final_results_2.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("\n✓ Results saved to final_results_2.json", flush=True)

# === Visualization Section ===
labels = ['Accuracy', 'Precision', 'Recall', 'F1']
baseline_vals = [baseline_test[k.lower()] for k in labels]
mcddi_vals = [mc_ddi_test[k.lower()] for k in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, baseline_vals, width, label='Baseline', color='skyblue')
plt.bar(x + width/2, mcddi_vals, width, label='MC-DDI', color='lightgreen')
plt.xticks(x, labels)
plt.ylim(0,1.05)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison_2.png', dpi=300)
print("✓ Saved model_comparison_2.png", flush=True)

# Confusion Matrices
for name, result, cmap in [('Baseline', baseline_test, 'Blues'), ('MC-DDI', mc_ddi_test, 'Greens')]:
    cm = confusion_matrix(result['y_true'], result['y_pred'])
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, cmap=cmap, cbar=False)
    plt.title(f'Confusion Matrix - {name} Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.lower()}_2.png', dpi=300)
    print(f"✓ Saved confusion_matrix_{name.lower()}_2.png", flush=True)

print("\n=== Evaluation 2 Complete ===", flush=True)
