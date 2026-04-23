"""
CLIPSeg Dataset Loader
======================
Utility for loading preprocessed CLIPSeg datasets in training pipelines.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class CLIPSegDataset(Dataset):
    """
    PyTorch Dataset for CLIPSeg preprocessed data.
    
    Usage:
        dataset = CLIPSegDataset(
            root_dir="processed",
            split="train",
            transform=None
        )
        
        image, mask, prompt, label = dataset[0]
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_transform=None
    ):
        """
        Args:
            root_dir: Path to processed dataset directory
            split: 'train' or 'valid'
            transform: Optional image transforms
            target_transform: Optional mask transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load metadata
        metadata_path = os.path.join(root_dir, split, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        
        logger.info(f"✓ Loaded {len(self.metadata)} samples from {split} split")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            Tuple of (image, mask, prompt, label)
            - image: np.ndarray (H, W, 3) or torch.Tensor
            - mask: np.ndarray (H, W) or torch.Tensor
            - prompt: str
            - label: str
        """
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, row['image'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, row['mask'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Get prompt and label
        prompt = row['prompt']
        label = row['label']
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask, prompt, label
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.metadata),
            'split': self.split,
            'labels': self.metadata['label'].value_counts().to_dict(),
            'prompts': self.metadata['prompt'].unique().tolist(),
            'mean_height': self.metadata['height'].mean(),
            'mean_width': self.metadata['width'].mean(),
            'max_height': self.metadata['height'].max(),
            'max_width': self.metadata['width'].max(),
        }
        return stats


class CLIPSegDataLoader:
    """
    Convenient data loader wrapper for CLIPSeg datasets.
    
    Usage:
        loader = CLIPSegDataLoader(
            root_dir="processed",
            batch_size=32
        )
        
        train_dataloader = loader.get_train_loader()
        valid_dataloader = loader.get_valid_loader()
    """
    
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 4,
        train_transform=None,
        valid_transform=None,
    ):
        """
        Args:
            root_dir: Path to processed dataset
            batch_size: Batch size for dataloaders
            shuffle_train: Whether to shuffle training data
            num_workers: Number of data loading workers
            train_transform: Image transforms for training
            valid_transform: Image transforms for validation
        """
        from torch.utils.data import DataLoader
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        
        # Create datasets
        self.train_dataset = CLIPSegDataset(
            root_dir=root_dir,
            split='train',
            transform=train_transform
        )
        
        self.valid_dataset = CLIPSegDataset(
            root_dir=root_dir,
            split='valid',
            transform=valid_transform
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    def get_train_loader(self):
        """Get training dataloader."""
        return self.train_loader
    
    def get_valid_loader(self):
        """Get validation dataloader."""
        return self.valid_loader
    
    def get_stats(self) -> Dict:
        """Get statistics for both splits."""
        return {
            'train': self.train_dataset.get_stats(),
            'valid': self.valid_dataset.get_stats()
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_dataset():
    """Basic dataset usage without PyTorch."""
    import cv2
    
    metadata_path = "processed/train/metadata.csv"
    image_dir = "processed/train/images"
    mask_dir = "processed/train/masks"
    
    metadata = pd.read_csv(metadata_path)
    
    for idx, row in metadata.head(5).iterrows():
        image = cv2.imread(os.path.join(image_dir, row['image']))
        mask = cv2.imread(os.path.join(mask_dir, row['mask']), cv2.IMREAD_GRAYSCALE)
        
        print(f"Image {idx}:")
        print(f"  Filename: {row['image']}")
        print(f"  Prompt: {row['prompt']}")
        print(f"  Label: {row['label']}")
        print(f"  Shape: {image.shape}")
        print()


def example_pytorch_dataset():
    """PyTorch Dataset usage."""
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    dataset = CLIPSegDataset(
        root_dir="processed",
        split="train",
        transform=transform
    )
    
    # Get sample
    image, mask, prompt, label = dataset[0]
    
    print(f"Sample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Prompt: {prompt}")
    print(f"  Label: {label}")


def example_pytorch_dataloader():
    """PyTorch DataLoader with batch loading."""
    from torchvision import transforms
    
    # Create loaders
    loader = CLIPSegDataLoader(
        root_dir="processed",
        batch_size=32,
        shuffle_train=True
    )
    
    # Get statistics
    stats = loader.get_stats()
    print("Dataset Statistics:")
    print(f"  Train: {stats['train']['total_samples']} samples")
    print(f"  Valid: {stats['valid']['total_samples']} samples")
    
    # Iterate through batches
    train_loader = loader.get_train_loader()
    for batch_idx, (images, masks, prompts, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images: {images.shape}")
        print(f"  Masks: {masks.shape}")
        print(f"  Prompts: {prompts}")
        print(f"  Labels: {labels}")
        
        if batch_idx >= 2:
            break


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("CLIPSeg Dataset Loader Examples")
    print("="*50)
    
    # Example 1: Basic numpy/cv2 usage
    print("\n1. Basic Dataset (No PyTorch):")
    print("-"*50)
    # example_basic_dataset()  # Uncomment to run
    
    # Example 2: PyTorch Dataset
    print("\n2. PyTorch Dataset:")
    print("-"*50)
    # example_pytorch_dataset()  # Uncomment to run
    
    # Example 3: PyTorch DataLoader with batching
    print("\n3. PyTorch DataLoader with Batching:")
    print("-"*50)
    # example_pytorch_dataloader()  # Uncomment to run
