import numpy as np

y = np.load('y_labels.npy')
unique_labels, counts = np.unique(y, return_counts=True)
print("Label distribution:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")
