import torch
import torch.nn as nn
import torch.nn.functional as F

class MC_DDI_Model(nn.Module):
    """
    Monte Carlo Dropout DDI Prediction Model
    Uncertainty-aware drug-drug interaction prediction
    """
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512, 256], n_classes=86, dropout_p=0.3):
        super(MC_DDI_Model, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    def mc_forward(self, x, n_samples=30):
        self.train()
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        import numpy as np
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        variance = predictions.var(axis=0)
        uncertainty = variance.mean(axis=1)
        confidence = 1 - (uncertainty / (uncertainty.max() + 1e-8))
        return mean_pred, confidence
