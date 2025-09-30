import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import glob
import json
from GeoValuator import DATA_DIR
from GeoValuator import MODELS_DIR
from GeoValuator import FIGURES_DIR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import r2_score


############# Path Configuration #############

CHECKPOINT_DATA_NAME = 'download_checkpoint_berlin.json'
MODEL_NAME = "best_regression_model_berlin_aug.pth"
DATA_NAME  = "berlin_images"


############# Functions and Classes #############

def mse_to_euros(normalized_mse, scaler, log_rent):
    """
    Transform MSE from normalized-log scale to approximate RMSE in Euros
    """
    min_log = scaler.data_min_[0]
    max_log = scaler.data_max_[0]
    range_log = max_log - min_log
    mse_log = normalized_mse * (range_log ** 2)
    rmse_log = np.sqrt(mse_log)
    multiplicative_error = np.exp(rmse_log)
    geometric_mean_original = np.exp(np.mean(log_rent))
    rmse_euros_approx = geometric_mean_original * (multiplicative_error - 1)
    return rmse_euros_approx

class StreetViewDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def evaluate_regression_model(model, test_loader):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            labels = labels.unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
        
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(labels.cpu().numpy())
    
    # MSE
    mse = total_loss / len(test_loader.dataset)
    rmse = np.sqrt(mse)
    
    predictions = np.array(predictions).flatten()
    true_values = np.array(true_values).flatten()
    
    # R2 score
    r2 = r2_score(true_values, predictions)
    
    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test R2: {r2:.4f}')
    
    return mse, rmse, r2

# Regression Model
class EfficientNetRegressor(nn.Module):
    def __init__(self, model_name='efficientnet_b3'):
        super(EfficientNetRegressor, self).__init__()
        
        self.backbone = timm.create_model(model_name, 
                                         pretrained=True,
                                         num_classes=0,
                                         global_pool='avg')
        
        # Single output neuron for regression
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)


# Modified training function for regression with MSE tracking
def train_model_regression(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None):
    model = model.to(device)
    train_mses = []
    val_mses = []
    best_val_mse = float('inf')
    
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        running_mse = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.float().to(device)
            labels = labels.unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate MSE
            batch_mse = loss.item()
            running_mse += batch_mse * images.size(0)
        
        # Calculate epoch training metrics
        epoch_train_mse = running_mse / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_mse = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_mse += loss.item() * images.size(0) 
        
        # Calculate epoch validation metrics
        epoch_val_mse = val_mse / len(val_loader.dataset)
        
        # Store metrics
        train_mses.append(epoch_train_mse)
        val_mses.append(epoch_val_mse)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # summary with MSE
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train MSE {epoch_train_mse:.4f}')
        
        # Save best model based on validation los
        if epoch_val_mse < best_val_mse:
            best_val_mse = epoch_val_mse
            torch.save(model.state_dict(), MODELS_DIR)
            print(f'New best model wit Val MSE: {epoch_val_mse:.4f})')
    
    return  train_mses, val_mses

def plot_regression_metrics(train_mses, val_mses, name):
    SAVE_NAME = os.join(FIGURES_DIR, f"{name}_regression_metrics_berlin_aug.pdf")
    plt.figure(figsize=(10, 6))
    
    # Plot MSE
    plt.plot(train_mses, label='Train MSE', linestyle='--')
    plt.plot(val_mses, label='Val MSE', linestyle='--')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300)
    plt.show()


############# Data handling #############

DATA_STRUCTURE_PATH = os.path.join(DATA_DIR, "interim", CHECKPOINT_DATA_NAME)
IMAGE_BASE_DIR = os.path.join(DATA_DIR, "processed", )
MODELS_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

with open(DATA_STRUCTURE_PATH, 'r') as f:
    data = json.load(f)

# Create a district to normalized price mapping
district_price_mapping = {}

for item in data['successful_downloads']:
    district = item['district']
    normalized_price = item['normalized_price']
    district_price_mapping[district] = normalized_price

image_paths = []
image_labels = []
district_names_list = []

# Get all district folders
all_district_folders = [d for d in os.listdir(IMAGE_BASE_DIR) 
                       if os.path.isdir(os.path.join(IMAGE_BASE_DIR, d))]

# Load images and assign the normalized price labels
for district_folder in all_district_folders:
    if district_folder in district_price_mapping:
        average_price_id = district_price_mapping[district_folder]
        district_image_dir = os.path.join(IMAGE_BASE_DIR, district_folder)
        
        # Get all images in this district folder
        pattern = os.path.join(district_image_dir, '*.jpg')
        image_files = glob.glob(pattern)
        
        for image_path in image_files:
            image_paths.append(image_path)
            image_labels.append(average_price_id)
            district_names_list.append(district_folder)

# Create DataFrame
image_df = pd.DataFrame({
    'image_path': image_paths,
    'label': image_labels,
    'district': district_names_list
})

# Group images by label
label_groups = {}
for label in image_df['label'].unique():
    label_groups[label] = image_df[image_df['label'] == label]

# Split each label separately
train_dfs, val_dfs, test_dfs = [], [], []

for label, group in label_groups.items():
    train_val, test = train_test_split(group, test_size=0.0654, random_state=42)
    train, val = train_test_split(train_val, test_size=0.07, random_state=42)
    
    train_dfs.append(train)
    val_dfs.append(val)
    test_dfs.append(test)

train_df = pd.concat(train_dfs).reset_index(drop=True)
val_df = pd.concat(val_dfs).reset_index(drop=True)
test_df = pd.concat(test_dfs).reset_index(drop=True)


# Define transforms for Imagenet B3 (300x300)
INPUT_SIZE = 300

train_transform = transforms.Compose([
    # Randomly crops out a portion of the images
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),

    # Applies a small transform of the image: tilt rotation translation scale shear
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-2, 2), interpolation=transforms.InterpolationMode.BILINEAR),

    # Perspective transformation
    transforms.RandomPerspective(distortion_scale=0.15, p=0.2, interpolation=transforms.InterpolationMode.BILINEAR),

    # Photometric change ranodm brightness, contrast, saturation
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),

    # Occasional camera/quality artifacts
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = StreetViewDataset(train_df, transform=train_transform)
val_dataset = StreetViewDataset(val_df, transform=val_transform)
test_dataset = StreetViewDataset(test_df, transform=val_transform)

# Create dataloaders
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


############# Training #############

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Put model on GPU
model = EfficientNetRegressor(model_name='efficientnet_b3')
model = model.to(device)

print(f"GPU available: {torch.cuda.is_available()}")

# Use MSELoss for regression
criterion = nn.MSELoss()

# Feature Extraction
print("1) Feature Extraction")
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer_stage1 = optim.Adam(model.regressor.parameters(), lr=1e-3)

# Train with metrics
train_losses_s1, val_losses_s1, train_mses_s1, val_mses_s1 = train_model_regression(
    model, train_loader, val_loader, criterion, optimizer_stage1, 
    num_epochs=4
)

# Fine-Tuning
print("2) Fine-Tuning")

for param in model.backbone.parameters():
    param.requires_grad = True

optimizer_stage2 = optim.Adam(
    [{'params': model.regressor.parameters(), 'lr': 1e-4},
     {'params': model.backbone.parameters(), 'lr': 1e-5}]
)

scheduler = StepLR(optimizer_stage2, step_size=5, gamma=0.1)

train_losses_s2, val_losses_s2, train_mses_s2, val_mses_s2 = train_model_regression(
    model, train_loader, val_loader, criterion, optimizer_stage2, 
    num_epochs=5, scheduler=scheduler
)

########## Evaluation ##########

# Plot metrics
print("1) Metrics")
plot_regression_metrics(train_mses_s1, val_mses_s1, "metrics_1")

print("2) Metrics")
plot_regression_metrics(train_mses_s2, val_mses_s2, "metrics_2")

# Combine all metrics for full training view
full_train_losses = train_losses_s1 + train_losses_s2
full_val_losses = val_losses_s1 + val_losses_s2
full_train_mses = train_mses_s1 + train_mses_s2
full_val_mses = val_mses_s1 + val_mses_s2

plot_regression_metrics(full_train_mses, full_val_mses, "metrics_full")

model.load_state_dict(torch.load(MODELS_DIR))
mse, rmse, r2 = evaluate_regression_model(model, test_loader)