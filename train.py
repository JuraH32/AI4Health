import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import timm
import torch.nn.functional as F

from data import build_split_dataloaders, MedicalImageDataset
from model import SwinTransformerClassificationModel, SwinMammoClassifier, ConvNeXtClassificationModel

# Hyperparameters
batch_size = 6
learning_rate = 0.001
num_epochs = 10
experiment = "convnextv2"
log_dir = f"runs/{experiment}"
num_classes = 3

# Data paths
root_dir = os.path.join("K:", "rsna-breast-cancer-detection")
csv_path = os.path.join(root_dir, "train.csv")
root_dir = os.path.join(root_dir, "train_images_cropped")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
])

def train_epoch(model, train_loader, criterion, optimizer, device, num_classes, global_step, writer, paired=True):
    model.train()
    running_loss = 0.0
    correct_train_batch = 0
    total_train_batch = 0
    
    total_train = 0
    correct_train = 0
    total_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        if paired:
            mlo, cc = inputs
            mlo_labels, cc_labels = labels
            mlo = mlo.to(device)
            cc = cc.to(device)
            mlo_labels = mlo_labels.to(device).long()
            cc_labels = cc_labels.to(device).long()
            model_input = torch.cat((mlo, cc), dim=0)
            
        else:
            model_input = inputs.to(device)
            labels = labels.to(device).long()
            
        outputs = model(model_input)
        
        if not paired:
            labels = labels.view(-1)
            outputs = outputs.view(-1, num_classes)
            
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct_train_batch += torch.sum(preds == labels).item()
            total_train_batch += labels.size(0)
        else:
            mlo_outputs, cc_outputs = torch.split(outputs, mlo.size(0), dim=0)
            mlo_outputs = mlo_outputs.view(-1, num_classes)
            cc_outputs = cc_outputs.view(-1, num_classes)
            mlo_labels = mlo_labels.view(-1)
            cc_labels = cc_labels.view(-1)

            loss = criterion(mlo_outputs, mlo_labels) + criterion(cc_outputs, cc_labels)
            
            _, mlo_preds = torch.max(mlo_outputs, 1)
            _, cc_preds = torch.max(cc_outputs, 1)
            correct_train_batch += torch.sum(mlo_preds == mlo_labels).item() + torch.sum(cc_preds == cc_labels).item()
            total_train_batch += mlo_labels.size(0) + cc_labels.size(0)
            
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        global_step += 1

        if i % 10 == 9:
            total_loss += running_loss
            total_train += total_train_batch
            correct_train += correct_train_batch
            
            avg_loss = running_loss / 10 / 2
            train_acc_batch = correct_train_batch / total_train_batch
            writer.add_scalar('training loss', avg_loss, global_step)
            writer.add_scalar('training accuracy', train_acc_batch, global_step)
            running_loss = 0.0
            correct_train_batch = 0
            total_train_batch = 0
            
    avg_loss = total_loss / len(train_loader) / 2
    train_acc = correct_train / total_train
    
    return avg_loss, train_acc

def validate(model, val_loader, criterion, device, num_classes, writer, paired=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            
            if paired:
                mlo, cc = inputs
                mlo_labels, cc_labels = labels
                mlo = mlo.to(device)
                cc = cc.to(device)
                mlo_labels = mlo_labels.to(device).long()
                cc_labels = cc_labels.to(device).long()
                model_input = torch.cat((mlo, cc), dim=0)             
            else:
                model_input = inputs.to(device)
                labels = labels.to(device).long()
                
            outputs = model(model_input)
            
            if not paired:
                labels = labels.view(-1)
                outputs = outputs.view(-1, num_classes)
                
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += labels.size(0)
                
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            else:
                mlo_outputs, cc_outputs = torch.split(outputs, mlo.size(0), dim=0)
                mlo_outputs = mlo_outputs.view(-1, num_classes)
                cc_outputs = cc_outputs.view(-1, num_classes)
                mlo_labels = mlo_labels.view(-1)
                cc_labels = cc_labels.view(-1)

                loss = criterion(mlo_outputs, mlo_labels) + criterion(cc_outputs, cc_labels)
                
                _, mlo_preds = torch.max(mlo_outputs, 1)
                _, cc_preds = torch.max(cc_outputs, 1)
                total_correct += torch.sum(mlo_preds == mlo_labels).item() + torch.sum(cc_preds == cc_labels).item()
                total_samples += mlo_labels.size(0) + cc_labels.size(0)
                
                for t, p in zip(mlo_labels.view(-1), mlo_preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                for t, p in zip(cc_labels.view(-1), cc_preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
            total_loss += loss.item()
            
    avg_total_loss = total_loss / len(val_loader) / 2
    total_acc = total_correct / total_samples
    
    return avg_total_loss, total_acc, confusion_matrix

def train_model(csv_path, root_dir, batch_size, transform, num_classes, num_epochs, learning_rate, experiment, model_class, paired=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class(num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    base_log_dir = f"runs/{experiment}"
    log_dir = base_log_dir
    i = 1
    
    while os.path.exists(log_dir):
        log_dir = f"{base_log_dir}_{i}"
        i += 1
    
    writer = SummaryWriter(log_dir=log_dir)
    
    train_loader, val_loader, _ = build_split_dataloaders(
        csv_path, root_dir, batch_size=batch_size, transform=transform, train=True, val_ratio=0.2, test_ratio=0.1, paired=paired
    )
    
    global_step = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, num_classes, global_step, writer, paired)
        
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        epoch_time = time.time() - epoch_start
        val_loss, val_acc, confusion_matrix = validate(model, val_loader, criterion, device, num_classes, writer, paired)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)
        writer.add_image('confusion matrix', confusion_matrix, epoch)
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, validation loss: {val_loss:.3f}, accuracy: {val_acc:.3f}")
        
    torch.save(model.state_dict(), f"model_{experiment}.pth")
    print(f"Model saved to model_{experiment}.pth")
    writer.close()
    
def main():
    train_model(csv_path, root_dir, batch_size, transform, num_classes, num_epochs, learning_rate, experiment, ConvNeXtClassificationModel, paired=True)

if __name__ == "__main__":
    main()