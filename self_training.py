import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import MC_DDI_Model
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== Loading Baseline Model ===")
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Correct zero-based labels for multi-class
y -= 1

model = MC_DDI_Model(input_dim=4096, n_classes=86).to(device)
model.load_state_dict(torch.load('baseline_model.pth'))
print("✓ Baseline model loaded")

from sklearn.model_selection import train_test_split
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

print(f"Labeled pool: {len(X_labeled)} samples")
print(f"Unlabeled pool: {len(X_unlabeled)} samples")

def generate_pseudo_labels(model, X_unlabeled, device, confidence_threshold=0.7):
    model.eval()
    X_unlabeled_t = torch.FloatTensor(X_unlabeled).to(device)
    mean_pred, confidence = model.mc_forward(X_unlabeled_t, n_samples=30)
    high_conf_idx = confidence > confidence_threshold
    pseudo_X = X_unlabeled[high_conf_idx]
    pseudo_y = mean_pred[high_conf_idx].argmax(axis=1)
    print(f"Generated {len(pseudo_y)} pseudo-labels (confidence > {confidence_threshold})")
    return pseudo_X, pseudo_y, confidence[high_conf_idx]

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in dataloader:
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
    return total_loss / len(dataloader), correct / total

criterion = nn.CrossEntropyLoss()
X_current_unlabeled = X_unlabeled.copy()

print("\n=== Self-Training with Pseudo-Labels ===")
for round_num in range(3):
    print(f"\n--- Round {round_num + 1}/3 ---")
    pseudo_X, pseudo_y, confidences = generate_pseudo_labels(
        model, X_current_unlabeled, device, confidence_threshold=0.7
    )
    if len(pseudo_y) == 0:
        print("No high-confidence pseudo-labels. Stopping.")
        break
    combined_X = np.concatenate([X_labeled, pseudo_X])
    combined_y = np.concatenate([y_labeled, pseudo_y])
    print(f"Combined training set size: {len(combined_X)}")
    combined_X_t = torch.FloatTensor(combined_X)
    combined_y_t = torch.LongTensor(combined_y)
    combined_dataset = TensorDataset(combined_X_t, combined_y_t)
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(10):
        loss, acc = train_epoch(model, combined_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}, Acc: {acc:.4f}")
    used_indices = np.where(confidences > 0.7)[0]
    X_current_unlabeled = np.delete(X_current_unlabeled, used_indices, axis=0)
    print(f"Remaining unlabeled samples: {len(X_current_unlabeled)}")

torch.save(model.state_dict(), 'mc_ddi_model_self_trained.pth')
print("\n✓ Self-trained model saved to mc_ddi_model_self_trained.pth")
