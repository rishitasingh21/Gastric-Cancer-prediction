import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image

# Focal Loss Class
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        ce_loss = -targets * torch.log(inputs)
        fl_loss = self.alpha * (1 - inputs) ** self.gamma * ce_loss
        return fl_loss.sum(dim=1).mean()

# Directories
input_dir = r"E:\rishita\GI"
folders = ['abnormal160', 'normal160']
test_dir = r"E:\rishita\GI"
models_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\model"
results_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\test\limex_y_0"
lime_dir = os.path.join(results_dir, "lime_images")  # Folder to save LIME results
os.makedirs(lime_dir, exist_ok=True)

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
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder {folder_path} does not exist.")
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
        return image, label, img_path

# Transformations for Test Data
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.90390341, 0.78627598, 0.9073129], std=[0.12593466, 0.17750887, 0.09886251])
])

test_dataset = GiTestDataset(input_dir, folders, subset='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN Model
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
            nn.Softmax(dim=1)
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

# LIME Integration
explainer = LimeImageExplainer()

def predict(images):
    images = torch.tensor(images).permute(0, 3, 1, 2).to(device).float() / 255.0
    mean = torch.tensor([0.90390341, 0.78627598, 0.9073129], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.12593466, 0.17750887, 0.09886251], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    with torch.no_grad():
        outputs = model(images)
        return outputs.cpu().numpy()
    

# Generate LIME Explanations
for batch_idx, (images, labels, img_paths) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    
    for i in range(images.size(0)):
        image_tensor = images[i].unsqueeze(0)
        label = labels[i].item()
        img_path = img_paths[i]
        img_name = os.path.basename(img_path)
        class_name = folders[label]

        # Convert tensor to numpy for LIME
        image_np = np.array(Image.open(img_path).convert('RGB'))  # Raw image
        
        explanation = explainer.explain_instance(
            image_np,
            predict,
            top_labels=num_classes,
            hide_color=0,
            num_samples=1000
        )

        # Save explanation for each class
        for class_idx in range(num_classes):
            temp, mask = explanation.get_image_and_mask(
                label=class_idx,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )

            # Plot the boundary visualization
            plt.imshow(mark_boundaries(temp, mask))

            # Set the origin of the axes to (0, 0)
            plt.gca().invert_yaxis()  # Invert the y-axis to have (0, 0) at the bottom-left
            plt.xlim(0, temp.shape[1])  # Set x-axis to range from 0 to the width of the image
            plt.ylim(0, temp.shape[0])  # Set y-axis to range from 0 to the height of the image

            # Set the title of the image
            plt.title(f"Explanation for class {folders[class_idx]} - {img_name}")
            
            # Save the explanation image
            save_path = os.path.join(
                lime_dir,
                f"{class_name}_lime_{img_name.split('.')[0]}_class_{class_idx}.png"
            )
            plt.savefig(save_path)
            plt.close()
        
        print(f"Saved LIME explanation for {img_name} in class {class_name}")

print("All LIME explanations have been generated and saved.")
