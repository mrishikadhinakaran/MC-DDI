import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from model import MC_DDI_Model
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


print("\n=== Loading Preprocessed Data ===")
X = np.load('X_features.npy')
y = np.load('y_labels.npy')


print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")
print(f"Label distribution: {np.unique(y, return_counts=True)}")

# Zero-based label correction: from 1-86 to 0-85
y -= 1

print("\n=== Splitting Data ===")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train label distribution: {np.unique(y_train, return_counts=True)}")
print(f"Val label distribution: {np.unique(y_val, return_counts=True)}")
print(f"Test label distribution: {np.unique(y_test, return_counts=True)}")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

print("\n=== Initializing Model ===")
model = MC_DDI_Model(input_dim=4096, n_classes=86).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


print("\n=== Training Baseline Model ===")
train_losses = []
val_f1_scores = []

for epoch in range(20):
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(loss)
    if (epoch + 1) % 5 == 0:
        val_metrics = evaluate(model, val_loader, device)
        val_f1_scores.append(val_metrics['f1'])
        print(f"Epoch {epoch+1}/20 - Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val F1: {val_metrics['f1']:.4f}")

print("\n=== Test Results ===")
test_metrics = evaluate(model, test_loader, device)
for metric, value in test_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

# Adding suffix _2 to outputs
test_metrics['model_type'] = 'baseline'

with open('baseline_results_2.json', 'w') as f:
    json.dump(test_metrics, f, indent=4)

torch.save(model.state_dict(), 'baseline_model_2.pth')
print("\n✓ Baseline model saved to baseline_model_2.pth")
print("✓ Results saved to baseline_results_2.json")

# Visualization of training progress
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, color='blue', linewidth=2)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

steps = range(5, 21, 5)
plt.subplot(1, 2, 2)
plt.plot(steps, val_f1_scores, color='red', linewidth=2)
plt.title('Validation F1 Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress_2.png', dpi=300)
plt.show()
