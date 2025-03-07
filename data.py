import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, train=True, paired=False):
        """
        Args:
            csv_path (str): Path to the CSV file containing the metadata.
            root_dir (str): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied on an image.
            train (bool, optional): Whether the dataset is for training.
        """
        self.metadata = pd.read_csv(csv_path)
        if train:
            self.metadata = self.metadata.dropna(subset=['BIRADS'])
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.paired = paired
        
        # If paired flag is True, apply additional filtering and pairing logic.
        if self.paired:
            # Remove patients that have more than 4 rows.
            # Group by patient_id and keep only those patients with at most 4 rows.
            self.metadata = self.metadata.groupby("patient_id").filter(lambda g: len(g) <= 4)
            
            pairs = []
            # For each patient and each laterality, pair the CC and MLO views.
            for patient_id, patient_group in self.metadata.groupby("patient_id"):
                for laterality, laterality_group in patient_group.groupby("laterality"):
                    # Find CC and MLO entries
                    cc_rows = laterality_group[laterality_group["view"] == "CC"]
                    mlo_rows = laterality_group[laterality_group["view"] == "MLO"]
                    # Only proceed if both views are available.
                    if not cc_rows.empty and not mlo_rows.empty:
                        # For each combination, add a new paired row.
                        for _, cc_row in cc_rows.iterrows():
                            for _, mlo_row in mlo_rows.iterrows():
                                pair = {
                                    "patient_id": patient_id,
                                    "cc_image_id": cc_row["image_id"],
                                    "mlo_image_id": mlo_row["image_id"],
                                    "laterality": laterality,
                                    "cc_birads": cc_row["BIRADS"],
                                    "mlo_birads": mlo_row["BIRADS"]
                                }
                                pairs.append(pair)
            self.metadata = pd.DataFrame(pairs)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if self.paired:
            patient_id = row['patient_id']
            # Construct paths for the paired images.
            cc_path = os.path.join(self.root_dir, str(patient_id), f"{str(row['cc_image_id'])}.jpg")
            mlo_path = os.path.join(self.root_dir, str(patient_id), f"{str(row['mlo_image_id'])}.jpg")
            cc_image = cv2.imread(cc_path, cv2.IMREAD_GRAYSCALE)
            mlo_image = cv2.imread(mlo_path, cv2.IMREAD_GRAYSCALE)
            if cc_image is None:
                raise ValueError(f"CC image not found at path: {cc_path}")
            if mlo_image is None:
                raise ValueError(f"MLO image not found at path: {mlo_path}")
            if self.transform:
                cc_image = self.transform(cc_image)
                mlo_image = self.transform(mlo_image)
            if self.train:
                # You can choose how to represent the target.
                # Here we return a tuple with both BIRADS values.
                target = (row['cc_birads'], row['mlo_birads'])
                return (cc_image, mlo_image), target
            return (cc_image, mlo_image)
        else:
            patient_id = row['patient_id']
            image_id = row['image_id']
            # Construct the full image path: "train_images/{patient_id}/{image_id}.jpg"
            image_path = os.path.join(self.root_dir, str(patient_id), f"{str(image_id)}.jpg")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Image not found at path: {image_path}")
            if self.transform:
                image = self.transform(image)
            if self.train:
                target = row['BIRADS']
                return image, target
            return image

def build_dataloader(csv_path, root_dir, batch_size=8, shuffle=True, transform=None, train=True, paired=False):
    dataset = MedicalImageDataset(csv_path, root_dir, transform=transform, train=train, paired=paired)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_split_dataloaders(csv_path, root_dir, batch_size=8, transform=None, train=True, val_ratio=0.0, test_ratio=0.0, random_seed=42, paired=False):
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
    dataset = MedicalImageDataset(csv_path, root_dir, transform=transform, train=train, paired=paired)
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