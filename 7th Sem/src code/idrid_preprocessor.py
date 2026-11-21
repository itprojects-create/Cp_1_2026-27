"""
IDRiD Dataset Preprocessing Pipeline
=====================================
Handles three tasks: Segmentation, Disease Grading, Localization
Includes explainability-focused preprocessing steps
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

class IDRiDPreprocessor:
    """Main preprocessing class for IDRiD dataset"""
    
    def __init__(self, dataset_path: str, output_path: str = "./preprocessed_idrid"):
        """
        Initialize preprocessor
        
        Args:
            dataset_path: Path to IDRiD dataset root
            output_path: Where to save preprocessed data
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Standard image size for consistency
        self.target_size = (512, 512)
        
        # Define DR severity mapping
        self.dr_severity_map = {
            0: "No DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative"
        }
        
        # Define DME severity mapping
        self.dme_severity_map = {
            0: "No DME",
            1: "Mild",
            2: "Moderate",
            3: "Severe"
        }

    def _resolve_dataset_subdir(self, *candidates: str):
        """Return the first matching raw dataset subdirectory."""
        for candidate in candidates:
            path = self.dataset_path / candidate
            if path.exists():
                return path
        return None
        
    # ==================== PART A: SEGMENTATION ====================
    
    def preprocess_segmentation_masks(self):
        """
        Preprocess segmentation masks for lesion detection
        Handles: Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates, Optic Disc
        """
        print("\n" + "="*60)
        print("PROCESSING SEGMENTATION DATA")
        print("="*60)
        
        seg_path = self._resolve_dataset_subdir("1. Segmentation", "A. Segmentation")
        if seg_path is None:
            print(f"⚠️ Segmentation folder not found at {seg_path}")
            return
        
        # Create output directories
        seg_output = self.output_path / "segmentation"
        seg_output.mkdir(parents=True, exist_ok=True)
        
        images_path = seg_path / "1. Original Images"
        masks_path = seg_path / "2. Ground Truth"
        
        # Process training and testing
        for split in ["Training", "Testing"]:
            split_output = seg_output / split.lower()
            split_output.mkdir(parents=True, exist_ok=True)
            
            split_images_dir = images_path / split
            split_masks_dir = masks_path / split
            
            if not split_images_dir.exists():
                print(f"⚠️ {split} images not found")
                continue
                
            print(f"\n📂 Processing {split} segmentation data...")
            image_files = sorted([f for f in split_images_dir.glob("*.jpg")])
            
            metadata = []
            
            for img_file in tqdm(image_files, desc=f"{split} segmentation"):
                img_name = img_file.stem
                
                # Load original image
                original_img = cv2.imread(str(img_file))
                if original_img is None:
                    print(f"❌ Failed to load {img_file}")
                    continue
                
                # Resize image
                resized_img = cv2.resize(original_img, self.target_size, 
                                        interpolation=cv2.INTER_CUBIC)
                
                # Normalize to [0, 1]
                normalized_img = resized_img.astype(np.float32) / 255.0
                
                # Save preprocessed image
                img_output_path = split_output / f"{img_name}_preprocessed.npy"
                np.save(str(img_output_path), normalized_img)
                
                # Process masks for each lesion type
                lesion_types = ["Microaneurysms", "Haemorrhages", 
                               "Hard Exudates", "Soft Exudates"]
                
                mask_data = {"image_name": img_name}
                
                for lesion_type in lesion_types:
                    mask_file = split_masks_dir / f"{img_name}_{lesion_type}.tif"
                    
                    if mask_file.exists():
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, self.target_size, 
                                         interpolation=cv2.INTER_NEAREST)
                        
                        # Normalize mask to [0, 1]
                        mask_normalized = mask.astype(np.float32) / 255.0
                        
                        # Save mask
                        mask_output_path = split_output / \
                            f"{img_name}_{lesion_type.replace(' ', '_')}_mask.npy"
                        np.save(str(mask_output_path), mask_normalized)
                        
                        # Calculate mask statistics for explainability
                        mask_data[f"{lesion_type}_presence"] = int(np.max(mask_normalized) > 0)
                        mask_data[f"{lesion_type}_area_percent"] = \
                            float(np.sum(mask_normalized) / mask_normalized.size * 100)
                    else:
                        mask_data[f"{lesion_type}_presence"] = 0
                        mask_data[f"{lesion_type}_area_percent"] = 0.0
                
                # Process Optic Disc mask
                od_file = split_masks_dir / f"{img_name}_Optic_Disc_and_Cup.tif"
                if od_file.exists():
                    od_mask = cv2.imread(str(od_file), cv2.IMREAD_GRAYSCALE)
                    od_mask = cv2.resize(od_mask, self.target_size, 
                                        interpolation=cv2.INTER_NEAREST)
                    od_normalized = od_mask.astype(np.float32) / 255.0
                    
                    od_output_path = split_output / f"{img_name}_optic_disc_mask.npy"
                    np.save(str(od_output_path), od_normalized)
                    
                    mask_data["optic_disc_presence"] = int(np.max(od_normalized) > 0)
                    mask_data["optic_disc_area_percent"] = \
                        float(np.sum(od_normalized) / od_normalized.size * 100)
                
                metadata.append(mask_data)
            
            # Save metadata
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_csv(split_output / "metadata.csv", index=False)
            print(f"✅ {split} segmentation processed: {len(metadata)} images")
    
    # ==================== PART B: DISEASE GRADING ====================
    
    def preprocess_disease_grading(self):
        """
        Preprocess disease grading data
        Handles: DR Severity (0-4) and DME Severity (0-3)
        """
        print("\n" + "="*60)
        print("PROCESSING DISEASE GRADING DATA")
        print("="*60)
        
        grade_path = self._resolve_dataset_subdir("2. Disease Grading", "B. Disease Grading")
        if grade_path is None:
            print(f"⚠️ Disease Grading folder not found at {grade_path}")
            return
        
        # Create output directory
        grade_output = self.output_path / "disease_grading"
        grade_output.mkdir(parents=True, exist_ok=True)
        
        images_path = grade_path / "1. Original Images"
        
        # Load labels
        train_labels_path = grade_path / "2. Ground Truth" / "train_DR_DME.csv"
        test_labels_path = grade_path / "2. Ground Truth" / "test_DR_DME.csv"
        
        if not train_labels_path.exists() or not test_labels_path.exists():
            print("❌ Label files not found")
            return
        
        # Load CSV files
        train_labels = pd.read_csv(train_labels_path)
        test_labels = pd.read_csv(test_labels_path)
        
        print(f"\n📊 Training samples: {len(train_labels)}")
        print(f"📊 Testing samples: {len(test_labels)}")
        
        # Process each split
        for split_name, labels_df, image_dir in [
            ("train", train_labels, images_path / "Training"),
            ("test", test_labels, images_path / "Testing")
        ]:
            
            split_output = grade_output / split_name
            split_output.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📂 Processing {split_name} disease grading...")
            
            processed_data = []
            
            for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), 
                                desc=f"{split_name} grading"):
                
                img_name = row['Image name'] if 'Image name' in row.index \
                          else row[0]
                dr_grade = int(row['DR']) if 'DR' in row.index else int(row[1])
                dme_grade = int(row['DME']) if 'DME' in row.index else int(row[2])
                
                # Find image file
                img_path = None
                for ext in ['jpg', 'JPG', 'png']:
                    potential_path = image_dir / f"{img_name}.{ext}"
                    if potential_path.exists():
                        img_path = potential_path
                        break
                
                if img_path is None:
                    print(f"⚠️ Image not found: {img_name}")
                    continue
                
                # Load and preprocess image
                original_img = cv2.imread(str(img_path))
                if original_img is None:
                    continue
                
                # Resize
                resized_img = cv2.resize(original_img, self.target_size,
                                        interpolation=cv2.INTER_CUBIC)
                
                # Normalize to [0, 1]
                normalized_img = resized_img.astype(np.float32) / 255.0
                
                # Apply contrast enhancement (CLAHE for fundus images)
                lab_img = cv2.cvtColor((resized_img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
                l_channel = lab_img[:, :, 0]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_clahe = clahe.apply(l_channel)
                lab_img[:, :, 0] = l_clahe
                enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
                enhanced_normalized = enhanced_img.astype(np.float32) / 255.0
                
                # Save preprocessed images
                img_output_path = split_output / f"{img_name}_preprocessed.npy"
                np.save(str(img_output_path), normalized_img)
                
                enhanced_output_path = split_output / f"{img_name}_enhanced.npy"
                np.save(str(enhanced_output_path), enhanced_normalized)
                
                # Calculate image statistics for explainability
                img_brightness = float(np.mean(normalized_img))
                img_contrast = float(np.std(normalized_img))
                img_saturation = float(np.max(normalized_img) - np.min(normalized_img))
                
                processed_data.append({
                    "image_name": img_name,
                    "DR_grade": dr_grade,
                    "DR_label": self.dr_severity_map[dr_grade],
                    "DME_grade": dme_grade,
                    "DME_label": self.dme_severity_map[dme_grade],
                    "brightness": img_brightness,
                    "contrast": img_contrast,
                    "saturation": img_saturation,
                    "has_dr": int(dr_grade > 0),
                    "has_dme": int(dme_grade > 0)
                })
            
            # Save processed metadata
            processed_df = pd.DataFrame(processed_data)
            processed_df.to_csv(split_output / "metadata.csv", index=False)
            
            # Print class distribution
            print(f"\n📊 {split_name.upper()} DR Distribution:")
            dr_counts = processed_df['DR_label'].value_counts().sort_index()
            for label, count in dr_counts.items():
                print(f"   {label}: {count} ({count/len(processed_df)*100:.1f}%)")
            
            print(f"\n📊 {split_name.upper()} DME Distribution:")
            dme_counts = processed_df['DME_label'].value_counts().sort_index()
            for label, count in dme_counts.items():
                print(f"   {label}: {count} ({count/len(processed_df)*100:.1f}%)")
            
            print(f"✅ {split_name} disease grading processed: {len(processed_df)} images")
    
    # ==================== PART C: LOCALIZATION ====================
    
    def preprocess_localization(self):
        """
        Preprocess localization data for Optic Disc and Fovea centers
        """
        print("\n" + "="*60)
        print("PROCESSING LOCALIZATION DATA")
        print("="*60)
        
        loc_path = self._resolve_dataset_subdir("3. Localization", "C. Localization")
        if loc_path is None:
            print(f"⚠️ Localization folder not found at {loc_path}")
            return
        
        # Create output directory
        loc_output = self.output_path / "localization"
        loc_output.mkdir(parents=True, exist_ok=True)
        
        images_path = loc_path / "1. Original Images"
        
        # Load ground truth
        train_od_path = loc_path / "2. Groundtruth" / "train_Optic_Disc_Center_Locations.csv"
        test_od_path = loc_path / "2. Groundtruth" / "test_Optic_Disc_Center_Locations.csv"
        train_fovea_path = loc_path / "2. Groundtruth" / "train_Fovea_Center_Locations.csv"
        test_fovea_path = loc_path / "2. Groundtruth" / "test_Fovea_Center_Locations.csv"
        
        # Load CSV files
        datasets = {}
        for split_name, od_path, fovea_path, image_dir in [
            ("train", train_od_path, train_fovea_path, images_path / "Training"),
            ("test", test_od_path, test_fovea_path, images_path / "Testing")
        ]:
            
            if not od_path.exists() or not fovea_path.exists():
                print(f"⚠️ {split_name} localization files not found")
                continue
            
            split_output = loc_output / split_name
            split_output.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📂 Processing {split_name} localization data...")
            
            # Load labels
            od_labels = pd.read_csv(od_path)
            fovea_labels = pd.read_csv(fovea_path)
            
            # Merge on image name
            merged_labels = od_labels.merge(fovea_labels, 
                                           on=['Image name'], 
                                           suffixes=('_OD', '_Fovea'))
            
            processed_data = []
            
            for idx, row in tqdm(merged_labels.iterrows(), total=len(merged_labels),
                                desc=f"{split_name} localization"):
                
                img_name = row['Image name']
                
                # Extract coordinates (adjust column names based on actual CSV)
                od_x = float(row['OD Center X']) if 'OD Center X' in row.index else float(row[1])
                od_y = float(row['OD Center Y']) if 'OD Center Y' in row.index else float(row[2])
                fovea_x = float(row['Fovea Center X']) if 'Fovea Center X' in row.index else float(row[1])
                fovea_y = float(row['Fovea Center Y']) if 'Fovea Center Y' in row.index else float(row[2])
                
                # Find image
                img_path = None
                for ext in ['jpg', 'JPG', 'png']:
                    potential_path = image_dir / f"{img_name}.{ext}"
                    if potential_path.exists():
                        img_path = potential_path
                        break
                
                if img_path is None:
                    continue
                
                # Load image
                original_img = cv2.imread(str(img_path))
                if original_img is None:
                    continue
                
                h_orig, w_orig = original_img.shape[:2]
                
                # Resize image
                resized_img = cv2.resize(original_img, self.target_size,
                                        interpolation=cv2.INTER_CUBIC)
                
                # Scale coordinates to new image size
                scale_x = self.target_size[0] / w_orig
                scale_y = self.target_size[1] / h_orig
                
                od_x_scaled = od_x * scale_x
                od_y_scaled = od_y * scale_y
                fovea_x_scaled = fovea_x * scale_x
                fovea_y_scaled = fovea_y * scale_y
                
                # Normalize to [0, 1]
                od_x_norm = od_x_scaled / self.target_size[0]
                od_y_norm = od_y_scaled / self.target_size[1]
                fovea_x_norm = fovea_x_scaled / self.target_size[0]
                fovea_y_norm = fovea_y_scaled / self.target_size[1]
                
                # Normalize image to [0, 1]
                normalized_img = resized_img.astype(np.float32) / 255.0
                
                # Save preprocessed image
                img_output_path = split_output / f"{img_name}_preprocessed.npy"
                np.save(str(img_output_path), normalized_img)
                
                # Create heatmap for visualization (for explainability)
                heatmap = np.zeros(self.target_size, dtype=np.float32)
                
                # Draw Gaussian blobs at landmark locations
                od_x_pixel = int(od_x_scaled)
                od_y_pixel = int(od_y_scaled)
                fovea_x_pixel = int(fovea_x_scaled)
                fovea_y_pixel = int(fovea_y_scaled)
                
                # Ensure within bounds
                od_x_pixel = np.clip(od_x_pixel, 0, self.target_size[0]-1)
                od_y_pixel = np.clip(od_y_pixel, 0, self.target_size[1]-1)
                fovea_x_pixel = np.clip(fovea_x_pixel, 0, self.target_size[0]-1)
                fovea_y_pixel = np.clip(fovea_y_pixel, 0, self.target_size[1]-1)
                
                # Create Gaussian
                y_grid, x_grid = np.mgrid[0:self.target_size[1], 0:self.target_size[0]]
                
                # OD heatmap
                od_dist = np.sqrt((x_grid - od_x_pixel)**2 + (y_grid - od_y_pixel)**2)
                od_heatmap = np.exp(-(od_dist**2) / (2 * 30**2))
                
                # Fovea heatmap
                fovea_dist = np.sqrt((x_grid - fovea_x_pixel)**2 + (y_grid - fovea_y_pixel)**2)
                fovea_heatmap = np.exp(-(fovea_dist**2) / (2 * 20**2))
                
                # Save heatmaps
                od_heatmap_path = split_output / f"{img_name}_od_heatmap.npy"
                np.save(str(od_heatmap_path), od_heatmap)
                
                fovea_heatmap_path = split_output / f"{img_name}_fovea_heatmap.npy"
                np.save(str(fovea_heatmap_path), fovea_heatmap)
                
                # Calculate distance between OD and Fovea (important for glaucoma risk)
                od_fovea_distance = np.sqrt((od_x_norm - fovea_x_norm)**2 + 
                                           (od_y_norm - fovea_y_norm)**2)
                
                processed_data.append({
                    "image_name": img_name,
                    "od_center_x_normalized": float(od_x_norm),
                    "od_center_y_normalized": float(od_y_norm),
                    "fovea_center_x_normalized": float(fovea_x_norm),
                    "fovea_center_y_normalized": float(fovea_y_norm),
                    "od_fovea_distance": float(od_fovea_distance)
                })
            
            # Save metadata
            processed_df = pd.DataFrame(processed_data)
            processed_df.to_csv(split_output / "metadata.csv", index=False)
            print(f"✅ {split_name} localization processed: {len(processed_df)} images")
    
    # ==================== INTEGRATION & VALIDATION ====================
    
    def create_unified_dataset(self):
        """
        Create unified dataset combining all three tasks
        """
        print("\n" + "="*60)
        print("CREATING UNIFIED DATASET")
        print("="*60)
        
        unified_output = self.output_path / "unified"
        unified_output.mkdir(parents=True, exist_ok=True)
        
        # Load all metadata files
        grade_train = pd.read_csv(self.output_path / "disease_grading" / "train" / "metadata.csv")
        grade_test = pd.read_csv(self.output_path / "disease_grading" / "test" / "metadata.csv")
        
        loc_train = pd.read_csv(self.output_path / "localization" / "train" / "metadata.csv")
        loc_test = pd.read_csv(self.output_path / "localization" / "test" / "metadata.csv")
        
        for split_name, grade_df, loc_df in [
            ("train", grade_train, loc_train),
            ("test", grade_test, loc_test)
        ]:
            
            # Merge disease grading and localization data
            merged = grade_df.merge(loc_df, on='image_name', how='inner')
            
            # Save unified metadata
            merged.to_csv(unified_output / f"{split_name}_unified_metadata.csv", index=False)
            
            print(f"✅ Unified {split_name} dataset: {len(merged)} samples")
    
    def generate_data_statistics(self):
        """
        Generate statistics report for explainability analysis
        """
        print("\n" + "="*60)
        print("GENERATING DATA STATISTICS")
        print("="*60)
        
        stats_output = self.output_path / "statistics"
        stats_output.mkdir(parents=True, exist_ok=True)
        
        # Load unified metadata
        train_meta = pd.read_csv(self.output_path / "unified" / "train_unified_metadata.csv")
        test_meta = pd.read_csv(self.output_path / "unified" / "test_unified_metadata.csv")
        
        stats = {
            "dataset_info": {
                "total_training_samples": len(train_meta),
                "total_testing_samples": len(test_meta),
                "total_samples": len(train_meta) + len(test_meta),
                "train_test_split": f"{len(train_meta)/(len(train_meta)+len(test_meta))*100:.1f}% / {len(test_meta)/(len(train_meta)+len(test_meta))*100:.1f}%"
            },
            "DR_distribution": {
                "training": train_meta['DR_label'].value_counts().to_dict(),
                "testing": test_meta['DR_label'].value_counts().to_dict()
            },
            "DME_distribution": {
                "training": train_meta['DME_label'].value_counts().to_dict(),
                "testing": test_meta['DME_label'].value_counts().to_dict()
            },
            "image_quality": {
                "train_brightness_mean": float(train_meta['brightness'].mean()),
                "train_brightness_std": float(train_meta['brightness'].std()),
                "train_contrast_mean": float(train_meta['contrast'].mean()),
                "train_contrast_std": float(train_meta['contrast'].std()),
                "test_brightness_mean": float(test_meta['brightness'].mean()),
                "test_brightness_std": float(test_meta['brightness'].std()),
                "test_contrast_mean": float(test_meta['contrast'].mean()),
                "test_contrast_std": float(test_meta['contrast'].std())
            }
        }
        
        # Save statistics
        with open(stats_output / "data_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n📊 DATA STATISTICS:")
        print(json.dumps(stats, indent=2))
        
        return stats
    
    def run_complete_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*80)
        print(" "*20 + "IDRiD DATASET PREPROCESSING PIPELINE")
        print("="*80)
        
        self.preprocess_segmentation_masks()
        self.preprocess_disease_grading()
        self.preprocess_localization()
        self.create_unified_dataset()
        self.generate_data_statistics()
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE!")
        print(f"📁 Output saved to: {self.output_path}")
        print("="*80)


if __name__ == "__main__":
    # Example usage
    # Update this path to your actual IDRiD dataset location
    dataset_root = "/path/to/IDRiD_dataset"
    output_root = "./preprocessed_idrid"
    
    preprocessor = IDRiDPreprocessor(dataset_root, output_root)
    preprocessor.run_complete_pipeline()
