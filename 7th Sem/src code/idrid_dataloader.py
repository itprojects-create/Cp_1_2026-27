"""
IDRiD Dataset Loader
====================
Custom PyTorch DataLoader for unified IDRiD data with explainability support
"""

import tempfile
from pathlib import Path
from typing import Dict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class IDRiDDataset(Dataset):
    """
    Custom Dataset class for IDRiD
    
    Features:
    - Loads preprocessed .npy files
    - Supports disease grading (DR + DME)
    - Includes localization (OD + Fovea centers)
    - Explainability-aware: includes image metadata
    """
    
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        enhanced: bool = True,
        task: str = "disease_grading",
        augment: bool = True,
        augment_strength: str = "medium"
    ):
        """
        Args:
            metadata_path: Path to unified_metadata.csv
            images_dir: Directory containing preprocessed images
            enhanced: Use CLAHE-enhanced images (True) or original (False)
            task: 'disease_grading' (default) or 'localization'
            augment: Apply data augmentation
            augment_strength: 'light', 'medium', 'heavy'
        """
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = Path(images_dir)
        self.enhanced = enhanced
        self.task = task
        self.augment = augment
        self.augment_strength = augment_strength
        
        # Define augmentation pipeline
        self.augmentation = self._get_augmentation_pipeline()
        
        # DR/DME severity mappings for label encoding
        self.dr_class_weights = self._compute_class_weights('DR_grade')
        self.dme_class_weights = self._compute_class_weights('DME_grade')
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
            - 'image': (C, H, W) tensor
            - 'dr_grade': (0-4) DR severity
            - 'dme_grade': (0-3) DME severity
            - 'dr_label': String label
            - 'dme_label': String label
            - 'brightness': Image brightness (for explainability)
            - 'contrast': Image contrast (for explainability)
            - 'image_name': Original filename
            - For localization task:
              - 'od_center': (x, y) normalized coordinates
              - 'fovea_center': (x, y) normalized coordinates
        """
        row = self.metadata.iloc[idx]
        image_name = row['image_name']
        
        # Load image
        if self.enhanced:
            img_path = self.images_dir / f"{image_name}_enhanced.npy"
        else:
            img_path = self.images_dir / f"{image_name}_preprocessed.npy"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = np.load(str(img_path))  # Shape: (512, 512, 3), range [0, 1]
        
        # Convert to (512, 512, 3) if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Apply augmentation
        if self.augment:
            image = self.augmentation(image=image)['image']
        else:
            # Still convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Prepare output dictionary
        output = {
            'image': image,
            'image_name': image_name,
            'brightness': torch.tensor(row['brightness'], dtype=torch.float32),
            'contrast': torch.tensor(row['contrast'], dtype=torch.float32),
            'saturation': torch.tensor(row['saturation'], dtype=torch.float32),
        }
        
        # Add disease grading labels
        if self.task in ['disease_grading', 'both']:
            output['dr_grade'] = torch.tensor(int(row['DR_grade']), dtype=torch.long)
            output['dme_grade'] = torch.tensor(int(row['DME_grade']), dtype=torch.long)
            output['dr_label'] = str(row['DR_label'])
            output['dme_label'] = str(row['DME_label'])
            output['has_dr'] = torch.tensor(int(row['has_dr']), dtype=torch.float32)
            output['has_dme'] = torch.tensor(int(row['has_dme']), dtype=torch.float32)
        
        # Add localization data
        if self.task in ['localization', 'both']:
            output['od_center'] = torch.tensor(
                [row['od_center_x_normalized'], row['od_center_y_normalized']],
                dtype=torch.float32
            )
            output['fovea_center'] = torch.tensor(
                [row['fovea_center_x_normalized'], row['fovea_center_y_normalized']],
                dtype=torch.float32
            )
            output['od_fovea_distance'] = torch.tensor(
                row['od_fovea_distance'], dtype=torch.float32
            )
        
        return output
    
    def _get_augmentation_pipeline(self):
        """Define augmentation pipeline for fundus images"""
        
        augmentations = {
            'light': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.GaussNoise(p=0.2),
                ToTensorV2()
            ]),
            'medium': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                ToTensorV2()
            ]),
            'heavy': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=45, p=0.7),
                A.ElasticTransform(alpha=50, sigma=10, p=0.5),
                A.GaussNoise(p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
                A.CoarseDropout(max_holes=4, max_height=50, max_width=50, p=0.3),
                ToTensorV2()
            ])
        }
        
        return augmentations.get(self.augment_strength, augmentations['medium'])
    
    def _compute_class_weights(self, grade_column: str) -> torch.Tensor:
        """Compute class weights for imbalanced classification"""
        grades = self.metadata[grade_column].values
        unique, counts = np.unique(grades, return_counts=True)
        num_classes = 5 if grade_column == 'DR_grade' else 4
        
        # Inverse frequency weighting
        weights = len(grades) / (len(unique) * counts)
        
        # Map to tensor
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        for cls, weight in zip(unique, weights):
            class_weights[int(cls)] = weight
        
        return class_weights
    
    def get_class_weights(self, task: str = 'dr') -> torch.Tensor:
        """Get class weights for loss function"""
        if task == 'dr':
            return self.dr_class_weights
        elif task == 'dme':
            return self.dme_class_weights
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_sample_statistics(self) -> Dict:
        """Get dataset statistics for explainability"""
        return {
            'num_samples': len(self),
            'brightness_mean': float(self.metadata['brightness'].mean()),
            'brightness_std': float(self.metadata['brightness'].std()),
            'contrast_mean': float(self.metadata['contrast'].mean()),
            'contrast_std': float(self.metadata['contrast'].std()),
            'dr_distribution': self.metadata['DR_label'].value_counts().to_dict(),
            'dme_distribution': self.metadata['DME_label'].value_counts().to_dict(),
        }


class IDRiDDataModule:
    """
    High-level data module for managing train/val/test splits
    """
    
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = True,
        augment_strength: str = "medium",
        task: str = "disease_grading",
        val_split: float = 0.2
    ):
        """
        Args:
            dataset_root: Root path to preprocessed dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            augment: Apply augmentation
            augment_strength: 'light', 'medium', 'heavy'
            task: 'disease_grading' or 'localization'
            val_split: Validation split from training data
        """
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.augment_strength = augment_strength
        self.task = task
        self.val_split = val_split
        
        # Load train and test metadata
        full_train_meta = pd.read_csv(
            self.dataset_root / "unified" / "train_unified_metadata.csv"
        )
        self.test_meta = pd.read_csv(
            self.dataset_root / "unified" / "test_unified_metadata.csv"
        )
        
        # Create validation split from training data
        val_size = int(len(full_train_meta) * val_split)
        train_indices = np.arange(len(full_train_meta))
        np.random.shuffle(train_indices)
        
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]

        self.train_meta = full_train_meta.iloc[train_indices].reset_index(drop=True)
        self.val_meta = full_train_meta.iloc[val_indices].reset_index(drop=True)
    
    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader"""
        dataset = IDRiDDataset(
            metadata_path=self._save_temp_metadata(self.train_meta),
            images_dir=str(self.dataset_root / "disease_grading" / "train"),
            enhanced=True,
            task=self.task,
            augment=self.augment,
            augment_strength=self.augment_strength
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader"""
        dataset = IDRiDDataset(
            metadata_path=self._save_temp_metadata(self.val_meta),
            images_dir=str(self.dataset_root / "disease_grading" / "train"),
            enhanced=True,
            task=self.task,
            augment=False,  # No augmentation for validation
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get testing DataLoader"""
        dataset = IDRiDDataset(
            metadata_path=self._save_temp_metadata(self.test_meta),
            images_dir=str(self.dataset_root / "disease_grading" / "test"),
            enhanced=True,
            task=self.task,
            augment=False,  # No augmentation for testing
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @staticmethod
    def _save_temp_metadata(df: pd.DataFrame) -> str:
        """Save metadata to temp file"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as temp_file:
            df.to_csv(temp_file.name, index=False)
            return temp_file.name


# Example usage
if __name__ == "__main__":
    # Initialize data module
    data_module = IDRiDDataModule(
        dataset_root="./preprocessed_idrid",
        batch_size=32,
        num_workers=4,
        augment=True,
        augment_strength="medium",
        task="disease_grading"
    )
    
    # Get loaders
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    test_loader = data_module.get_test_loader()
    
    # Iterate through batch
    for batch in train_loader:
        print("Batch loaded successfully!")
        print(f"Image shape: {batch['image'].shape}")
        print(f"DR grades: {batch['dr_grade']}")
        print(f"DME grades: {batch['dme_grade']}")
        print(f"Image names: {batch['image_name']}")
        print(f"Brightness: {batch['brightness']}")
        break
