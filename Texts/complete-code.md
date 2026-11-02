# MC-DDI: Complete Implementation Code

## FILE 1: model.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MC_DDI_Model(nn.Module):
    """
    Monte Carlo Dropout DDI Prediction Model
    Uncertainty-aware drug-drug interaction prediction
    """
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512, 256], 
                 n_classes=2, dropout_p=0.3):
        super(MC_DDI_Model, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))  # MC Dropout
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
    def forward(self, x):
        """Standard forward pass"""
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def mc_forward(self, x, n_samples=30):
        """
        Monte Carlo Dropout forward pass
        Returns: mean predictions and uncertainty (variance)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        import numpy as np
        predictions = np.array(predictions)  # Shape: (n_samples, batch_size, n_classes)
        
        mean_pred = predictions.mean(axis=0)
        variance = predictions.var(axis=0)
        
        # Confidence = 1 - normalized uncertainty
        uncertainty = variance.mean(axis=1)
        confidence = 1 - (uncertainty / (uncertainty.max() + 1e-8))
        
        return mean_pred, confidence
```

## FILE 2: train_baseline.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model import MC_DDI_Model
import json

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load preprocessed data
print("\n=== Loading Preprocessed Data ===")
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")
print(f"Label distribution: {np.unique(y, return_counts=True)}")

# Convert to binary classification if needed
if len(np.unique(y)) > 2:
    # Convert multi-class to binary (interaction or not)
    y_binary = (y > 0).astype(int)
else:
    y_binary = y

# Split data
print("\n=== Splitting Data ===")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train set: {len(X_train)} samples")
print(f"Val set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create dataloaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# Initialize model
print("\n=== Initializing Model ===")
model = MC_DDI_Model(input_dim=4096, n_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
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

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

# Train baseline model
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

# Evaluate baseline on test set
print("\n=== Baseline Test Results ===")
baseline_results = evaluate(model, test_loader, device)
for metric, value in baseline_results.items():
    print(f"{metric.upper()}: {value:.4f}")

# Save results
baseline_results['model_type'] = 'baseline'
with open('baseline_results.json', 'w') as f:
    json.dump(baseline_results, f, indent=4)

# Save model
torch.save(model.state_dict(), 'baseline_model.pth')
print("\n✓ Baseline model saved to baseline_model.pth")
print("✓ Results saved to baseline_results.json")
```

## FILE 3: self_training.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import MC_DDI_Model
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load baseline model and data
print("=== Loading Baseline Model ===")
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Binary classification
y_binary = (y > 0).astype(int)

# Load baseline model
model = MC_DDI_Model(input_dim=4096, n_classes=2).to(device)
model.load_state_dict(torch.load('baseline_model.pth'))
print("✓ Baseline model loaded")

# Split into labeled (training) and pseudo-label candidates
from sklearn.model_selection import train_test_split
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
    X, y_binary, test_size=0.5, random_state=42
)

print(f"Labeled pool: {len(X_labeled)} samples")
print(f"Unlabeled pool: {len(X_unlabeled)} samples")

# Self-training loop
def generate_pseudo_labels(model, X_unlabeled, device, confidence_threshold=0.7):
    """Generate pseudo-labels using MC Dropout"""
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

# Self-training rounds
criterion = nn.CrossEntropyLoss()
X_current_unlabeled = X_unlabeled.copy()

print("\n=== Self-Training with Pseudo-Labels ===")
for round_num in range(3):
    print(f"\n--- Round {round_num + 1}/3 ---")
    
    # Generate pseudo-labels
    pseudo_X, pseudo_y, confidences = generate_pseudo_labels(
        model, X_current_unlabeled, device, confidence_threshold=0.7
    )
    
    if len(pseudo_y) == 0:
        print("No high-confidence pseudo-labels. Stopping.")
        break
    
    # Combine original training data with pseudo-labeled data
    combined_X = np.concatenate([X_labeled, pseudo_X])
    combined_y = np.concatenate([y_labeled, pseudo_y])
    
    print(f"Combined training set size: {len(combined_X)}")
    
    # Create dataloaders
    combined_X_t = torch.FloatTensor(combined_X)
    combined_y_t = torch.LongTensor(combined_y)
    combined_dataset = TensorDataset(combined_X_t, combined_y_t)
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True)
    
    # Re-train model
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning
    
    for epoch in range(10):
        loss, acc = train_epoch(model, combined_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    # Remove used pseudo-labeled samples from unlabeled pool
    used_indices = np.where(confidences > 0.7)[0]
    X_current_unlabeled = np.delete(X_current_unlabeled, used_indices, axis=0)
    print(f"Remaining unlabeled samples: {len(X_current_unlabeled)}")

# Save final model
torch.save(model.state_dict(), 'mc_ddi_model_self_trained.pth')
print("\n✓ Self-trained model saved to mc_ddi_model_self_trained.pth")
```

## FILE 4: evaluate.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import json
from model import MC_DDI_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X = np.load('X_features.npy')
y = np.load('y_labels.npy')
y_binary = (y > 0).astype(int)

# Create test set (consistent with training)
_, X_test, _, y_test = train_test_split(X, y_binary, test_size=0.15, random_state=42)

X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=128)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob
    }

# Load baseline results
print("=== Loading Models ===")
with open('baseline_results.json', 'r') as f:
    baseline_results = json.load(f)

# Evaluate baseline
baseline_model = MC_DDI_Model().to(device)
baseline_model.load_state_dict(torch.load('baseline_model.pth'))
baseline_test = evaluate_model(baseline_model, test_loader, device)

# Evaluate self-trained
mc_ddi_model = MC_DDI_Model().to(device)
mc_ddi_model.load_state_dict(torch.load('mc_ddi_model_self_trained.pth'))
mc_ddi_test = evaluate_model(mc_ddi_model, test_loader, device)

# Print results
print("\n" + "="*60)
print("BASELINE MODEL (WITHOUT PSEUDO-LABELS)")
print("="*60)
for metric, value in {k: v for k, v in baseline_test.items() if k not in ['y_true', 'y_pred', 'y_prob']}.items():
    print(f"{metric.upper():15}: {value:.4f}")

print("\n" + "="*60)
print("MC-DDI MODEL (WITH PSEUDO-LABELS)")
print("="*60)
for metric, value in {k: v for k, v in mc_ddi_test.items() if k not in ['y_true', 'y_pred', 'y_prob']}.items():
    print(f"{metric.upper():15}: {value:.4f}")

print("\n" + "="*60)
print("IMPROVEMENT OVER BASELINE")
print("="*60)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    baseline_val = baseline_test[metric]
    mc_ddi_val = mc_ddi_test[metric]
    improvement = ((mc_ddi_val - baseline_val) / baseline_val) * 100
    print(f"{metric.upper():15}: {improvement:+.2f}%")

# Save final results
final_results = {
    'baseline': {k: v for k, v in baseline_test.items() if k not in ['y_true', 'y_pred', 'y_prob']},
    'mc_ddi': {k: v for k, v in mc_ddi_test.items() if k not in ['y_true', 'y_pred', 'y_prob']},
    'improvements': {
        metric: ((mc_ddi_test[metric] - baseline_test[metric]) / baseline_test[metric]) * 100
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
    }
}

with open('final_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("\n✓ Results saved to final_results.json")
```

## USAGE:

Run in order:
1. `python3 download_data.py` (Phase 2)
2. `python3 preprocess_data.py` (Phase 3)
3. `python3 train_baseline.py` (Phase 4a)
4. `python3 self_training.py` (Phase 4b)
5. `python3 evaluate.py` (Phase 5)
