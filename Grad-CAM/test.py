import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Directories for Test
input_dir = r"E:\rishita\GI"
folders = ['abnormal160', 'normal160']
test_dir = r"E:\rishita\GI"  # Path to test directory
models_dir = r"E:\rishita\code\GI\GiFocal_loss\focal Loss with metrics\model"  # Trained model directory
results_dir = r"E:\rishita\testing\test2"  # Folder for saving results

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

# GradCAM class
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        if self.target_layer is None:
            self.target_layer = self.find_target_layer()

        # Hook for capturing feature maps and gradients (change this to target a layer in 'features')
        self.hook = self.model.features[6].register_forward_hook(self.save_feature_maps)  # Adjust the index for the layer
        self.hook = self.model.features[6].register_full_backward_hook(self.save_gradients)  # Adjust the index for the layer

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_heatmap(self, input_image, class_idx, upsample_size=(160, 160)):
        # Ensure input_image has requires_grad=True
        input_image = input_image.to(device)
        input_image.requires_grad_()

        self.model.zero_grad()
        output = self.model(input_image)

        # Make sure that the final output retains gradients
        output.retain_grad()

        # Backprop to compute gradients
        one_hot = torch.zeros((1, output.size()[-1]), device=input_image.device)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)

        # Pool gradients across the width and height
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)

        # Multiply feature maps by gradients to get the weighted sum
        weighted_sum = torch.sum(pooled_gradients * self.feature_maps, dim=1).squeeze()
        cam = F.relu(weighted_sum)

        # Normalize the CAM
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        # Resize CAM to the original image size
        cam = cam.cpu().detach().numpy()
        cam = cv.resize(cam, upsample_size, interpolation=cv.INTER_LINEAR)

        # Convert CAM to 3-channel image
        cam = np.expand_dims(cam, axis=2)
        cam = np.tile(cam, (1, 1, 3))

        return cam





def overlay_gradCAM(image, cam, alpha=0.5):
    cam = np.uint8(255 * cam)
    cam = cv.applyColorMap(cam, cv.COLORMAP_JET)

    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Overlay the GradCAM with the original image
    overlay = cv.addWeighted(image, 1 - alpha, cam, alpha, 0)

    return overlay

# Load the best model
model_path = os.path.join(models_dir, "best_model.pth")
model = CustomCNN(num_classes=num_classes, input_shape=(160, 160)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize GradCAM with the desired target layer
target_layer = model.features[-4]  # Last convolution layer (or any layer you want)
gradcam = GradCAM(model, target_layer)

# Evaluate the model
all_preds = []
all_labels = []
all_probs = []
all_image_paths = []

for inputs, labels, paths in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    # Forward pass to get predictions
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(outputs, 1)

    # Compute GradCAM for each image
    for i in range(inputs.size(0)):
        input_image = inputs[i].unsqueeze(0)  # Add batch dimension
        class_idx = preds[i].item()

        # Get GradCAM heatmap
        cam = gradcam.compute_heatmap(input_image, class_idx)

        # Convert the tensor to a PIL image for visualization
        image = Image.open(paths[i]).convert('RGB')
        overlay = overlay_gradCAM(image, cam)

        # Save or display the overlay image
        save_path = os.path.join(results_dir, f"{os.path.basename(paths[i])}_gradcam.png")
        cv.imwrite(save_path, overlay)


        # Collect predictions, probabilities, and labels
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_image_paths.extend(paths)

print("Test evaluation complete. Results saved.")
