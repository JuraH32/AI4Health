import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from data import build_dataloader, build_split_dataloaders
from model import SwinTransformerClassificationModel  # using swin
import time  # add this line at the top along with other imports

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
log_dir = "runs/experiment_1"
num_classes = 3

# Data paths
csv_path = "rsna-breast-cancer-detection/train.csv"
root_dir = "rsna-breast-cancer-detection/train_images"

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Build dataloaders
train_loader, val_loader, test_loader = build_split_dataloaders(
    csv_path, root_dir, batch_size=batch_size, transform=transform, train=True, val_ratio=0.2, test_ratio=0.1
)

# Initialize model, loss function, and optimizer
model = SwinTransformerClassificationModel(num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# Training loop
global_step = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)  # original output shape may be [batch, H, W, 3]
        # Fix: Pool spatial dimensions so outputs become [batch, 3]
        outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, (1, 1)).view(outputs.size(0), -1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1
        if i % 10 == 9:
            avg_loss = running_loss / 10
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {avg_loss:.3f}")
            writer.add_scalar('training loss', avg_loss, global_step)
            running_loss = 0.0

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(val_loader)
    print(f"Validation loss after epoch {epoch+1}: {val_loss_avg:.3f}")
    writer.add_scalar('validation loss', val_loss_avg, epoch)

# Save the model
torch.save(model.state_dict(), "model.pth")
writer.close()
