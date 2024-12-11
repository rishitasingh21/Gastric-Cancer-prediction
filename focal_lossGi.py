import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Focal Loss Class
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # balancing factor for the class weights
        self.gamma = gamma  # focusing parameter
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)

        # Get probabilities for the target class
        targets = F.one_hot(targets, num_classes=self.num_classes)
        
        # Compute the cross entropy loss
        ce_loss = -targets * torch.log(inputs)
        
        # Compute Focal Loss
        fl_loss = self.alpha * (1 - inputs) ** self.gamma * ce_loss
        
        # Return the mean loss across all examples
        return fl_loss.sum(dim=1).mean()

# Directories
input_dir = r"E:\rishita\GI"
folders = ['abnormal160', 'normal160']
models_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\model"
results_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\train"

# Hyperparameters
batch_size = 64
num_classes = len(folders)
epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class GiDataset(Dataset):
    def __init__(self, input_dir, folders, subset='train_val', transform=None):
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
        return image, label

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.90390341, 0.78627598, 0.9073129], std=[0.12593466, 0.17750887, 0.09886251])
])

# Load Dataset
train_val_dataset = GiDataset(input_dir, folders, subset='train_val', transform=transform)

# Define the CNN Model
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

# Initialize Focal Loss as criterion
criterion = FocalLoss(alpha=0.25, gamma=2, num_classes=num_classes).to(device)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Modify evaluate_model to return precision, recall, and f1-score
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = correct / len(dataloader.dataset)
    # Precision, Recall, and F1-score
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    return total_loss / len(dataloader), accuracy, precision, recall, f1_score, all_preds, all_labels

def plot_metrics(history, fold_no):
    epochs = range(1, len(history['train_loss']) + 1)
    os.makedirs(results_dir, exist_ok=True)

    # Loss Curve
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for Fold {fold_no}")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"loss_curve_fold_{fold_no}.png"))
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['train_acc'], label="Train Accuracy")
    plt.plot(epochs, history['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve for Fold {fold_no}")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"accuracy_curve_fold_{fold_no}.png"))
    plt.close()

# Training and K-Fold Cross Validation
best_model = None
best_val_acc = 0.0
for fold_no, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset), start=1):
    print(f"\n--- Fold {fold_no} ---")

    train_subset = Subset(train_val_dataset, train_idx)
    val_subset = Subset(train_val_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = CustomCNN(num_classes=num_classes, input_shape=(160, 160)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / len(train_loader.dataset)

        val_loss, val_acc, val_precision, val_recall, val_f1_score, val_preds, val_labels = evaluate_model(
            model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1_score:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()  # Save the state_dict of the best model

    # Save model and plot metrics
    torch.save(model.state_dict(), os.path.join(models_dir, f"model_fold_{fold_no}.pth"))
    # Save the best model after all folds

    if best_model is not None:
        torch.save(best_model, os.path.join(models_dir, "best_model.pth"))
        print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
    plot_metrics(history, fold_no)



