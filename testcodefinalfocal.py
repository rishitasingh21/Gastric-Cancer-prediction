import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize
from PIL import Image
import torch.nn.functional as F

# Directories for Test
input_dir = r"E:\rishita\GI"
folders = ['abnormal160', 'normal160']
test_dir = r"E:\rishita\GI"  # Path to test directory
models_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\model"  # Trained model directory
results_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\test\testfinal"  # Folder for saving results

# Hyperparameters
batch_size = 64
num_classes = len(folders)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for Test Data
class GiTestDataset(Dataset):
    def __init__(self, input_dir, folders, subset='test', transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for label, folder in enumerate(folders):
            folder_path = os.path.join(input_dir, folder, subset)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # Return image path as well for tracking

# Data Transformations for Test Data
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.90390341, 0.78627598, 0.9073129], std=[0.12593466, 0.17750887, 0.09886251])
])

# Load Test Dataset
test_dataset = GiTestDataset(input_dir, folders, subset='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN Model (same as training code)
class CustomCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        flattened_size = self._get_flattened_size(input_shape)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)  # Softmax for probabilities
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])
        dummy_output = self.features(dummy_input)
        return dummy_output.view(dummy_output.size(0), -1).shape[1]

# Load the best model
model_path = os.path.join(models_dir, "best_model.pth")
model = CustomCNN(num_classes=num_classes, input_shape=(160, 160)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate on Test Data
# Evaluate the model
all_preds = []
all_labels = []
all_probs = []
all_image_paths = []

with torch.no_grad():
    for inputs, labels, paths in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_image_paths.extend(paths)

# Save predictions
results_df = pd.DataFrame({
    "Image Path": all_image_paths,
    "Actual Label": all_labels,
    "Predicted Label": all_preds
})
probs_df = pd.DataFrame(all_probs, columns=[f"Prob_{cls}" for cls in folders])
results_df = pd.concat([results_df, probs_df], axis=1)
os.makedirs(results_dir, exist_ok=True)
results_csv_path = os.path.join(results_dir, "test_predictions.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Predictions saved to {results_csv_path}.")

# Generate Metrics
# Generate Metrics
cm = confusion_matrix(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds, target_names=folders)
roc_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])  # Use probabilities for positive class

print("Classification Report:")
print(classification_rep)
print(f"ROC AUC: {roc_auc:.4f}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=folders, yticklabels=folders)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(results_dir, "roc_curve.png"))
plt.close()

print("Test evaluation complete. Results saved.")
