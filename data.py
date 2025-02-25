import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, train=True):
        """
        Args:
            csv_path (str): Path to the CSV file containing the metadata.
            root_dir (str): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied on an image.
            train (bool, optional): Whether the dataset is for training.
        """
        self.metadata = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        patient_id = row['patient_id']
        image_id = row['image_id']
        # Construct the full image path: "train_images/{patient_id}/{image_id}"
        image_path = os.path.join(self.root_dir, str(patient_id), str(image_id))
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        if self.transform:
            image = self.transform(image)
        # For training, assume the 'cancer' column exists as the target label.
        if self.train:
            target = row['cancer']
            return image, target
        return image

def build_dataloader(csv_path, root_dir, batch_size=8, shuffle=True, transform=None, train=True):
    dataset = MedicalImageDataset(csv_path, root_dir, transform=transform, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_split_dataloaders(csv_path, root_dir, batch_size=8, transform=None, train=True, val_ratio=0.0, test_ratio=0.0, random_seed=42):
    """
    Splits the dataset into train, validation, and test sets.
    If val_ratio and test_ratio are 0 then returns a single dataloader using the entire dataset.
    
    Args:
        csv_path (str): Path to the CSV file.
        root_dir (str): Root directory of images.
        batch_size (int): Batch size.
        transform (callable, optional): Transform to apply.
        train (bool): True if the dataset has labels (for training).
        val_ratio (float): Fraction of the data to use for validation.
        test_ratio (float): Fraction of the data to use for testing.
        random_seed (int): Seed for reproducible shuffling.
    
    Returns:
        If splitting is applied: (train_loader, val_loader, test_loader)
        Otherwise, a single DataLoader for the entire dataset.
    """
    dataset = MedicalImageDataset(csv_path, root_dir, transform=transform, train=train)
    dataset_size = len(dataset)
    
    # If no split is defined, then use the whole dataset for training.
    if val_ratio == 0.0 and test_ratio == 0.0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create indices and shuffle them.
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    test_size = int(test_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - test_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader