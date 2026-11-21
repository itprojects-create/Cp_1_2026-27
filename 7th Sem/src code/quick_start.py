"""
IDRiD Quick Start Guide
=======================
Run this script to:
1. Preprocess the IDRiD dataset
2. Verify preprocessing quality
3. Load data with PyTorch DataLoader
4. Visualize samples with metadata
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== CONFIGURATION ====================

# Update these paths to match your system
IDRID_DATASET_PATH = "./data_capstone"  # ← CHANGE THIS
OUTPUT_PATH = "./preprocessed_idrid"
BATCH_SIZE = 32
NUM_WORKERS = 4

# ==================== STEP 1: RUN PREPROCESSING ====================

def step1_run_preprocessing():
    """Step 1: Preprocess the dataset"""
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING IDRiD DATASET")
    print("=" * 80)

    from idrid_preprocessor import IDRiDPreprocessor

    dataset_path = Path(IDRID_DATASET_PATH)
    if not dataset_path.exists():
        print(f"Dataset path not found: {IDRID_DATASET_PATH}")
        print("Please update IDRID_DATASET_PATH in this script")
        return False

    expected_folders = [
        ("1. Segmentation", "A. Segmentation"),
        ("2. Disease Grading", "B. Disease Grading"),
        ("3. Localization", "C. Localization"),
    ]

    for primary_name, alternate_name in expected_folders:
        if (dataset_path / primary_name).exists():
            print(f"Found: {primary_name}")
        elif (dataset_path / alternate_name).exists():
            print(f"Found: {alternate_name}")
        else:
            print(f"Missing folder: {primary_name} or {alternate_name}")

    print("\nStarting preprocessing...")
    preprocessor = IDRiDPreprocessor(str(dataset_path), OUTPUT_PATH)

    try:
        preprocessor.run_complete_pipeline()
        print("\nPreprocessing complete!")
        return True
    except Exception as e:
        print(f"\nPreprocessing failed: {e}")
        return False

# ==================== STEP 2: VERIFY PREPROCESSING ====================

def step2_verify_preprocessing():
    """Step 2: Verify preprocessing quality"""
    print("\n" + "="*80)
    print("STEP 2: VERIFYING PREPROCESSING QUALITY")
    print("="*80)
    
    output_path = Path(OUTPUT_PATH)
    
    # Check directories
    print("\n📂 Checking output structure...")
    expected_dirs = [
        "disease_grading/train",
        "disease_grading/test",
        "localization/train",
        "localization/test",
        "segmentation/training",
        "segmentation/testing",
        "unified",
        "statistics"
    ]
    
    for dir_name in expected_dirs:
        dir_path = output_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✓ {dir_name}: {file_count} files")
        else:
            print(f"✗ {dir_name}: NOT FOUND")
    
    # Load and display statistics
    print("\n📊 Dataset Statistics:")
    stats_file = output_path / "statistics" / "data_statistics.json"
    
    if stats_file.exists():
        import json
        with open(stats_file) as f:
            stats = json.load(f)
        
        print(f"\nTotal samples: {stats['dataset_info']['total_samples']}")
        print(f"Train/Test split: {stats['dataset_info']['train_test_split']}")
        
        print("\nDR Distribution (Training):")
        for label, count in stats['DR_distribution']['training'].items():
            print(f"  {label}: {count}")
        
        print("\nDME Distribution (Training):")
        for label, count in stats['DME_distribution']['training'].items():
            print(f"  {label}: {count}")
        
        print("\nImage Quality Metrics (Training):")
        print(f"  Brightness: {stats['image_quality']['train_brightness_mean']:.3f} ± "
              f"{stats['image_quality']['train_brightness_std']:.3f}")
        print(f"  Contrast: {stats['image_quality']['train_contrast_mean']:.3f} ± "
              f"{stats['image_quality']['train_contrast_std']:.3f}")
    
    # Load and check metadata
    print("\n✓ Verifying metadata files...")
    metadata_files = [
        "disease_grading/train/metadata.csv",
        "disease_grading/test/metadata.csv",
        "unified/train_unified_metadata.csv",
        "unified/test_unified_metadata.csv"
    ]
    
    for meta_file in metadata_files:
        meta_path = output_path / meta_file
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            print(f"✓ {meta_file}: {len(df)} rows")
        else:
            print(f"✗ {meta_file}: NOT FOUND")


# ==================== STEP 3: LOAD DATA ====================

def step3_load_data():
    """Step 3: Load data with DataLoader"""
    print("\n" + "="*80)
    print("STEP 3: LOADING DATA WITH PYTORCH DATALOADER")
    print("="*80)
    
    from idrid_dataloader import IDRiDDataModule
    
    try:
        # Initialize data module
        print("\n📂 Initializing data module...")
        data_module = IDRiDDataModule(
            dataset_root=OUTPUT_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            augment=True,
            augment_strength="medium",
            task="disease_grading",
            val_split=0.15
        )
        
        # Get loaders
        print("\n🔄 Creating data loaders...")
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_val_loader()
        test_loader = data_module.get_test_loader()
        
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        print(f"✓ Test loader: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None, None, None


# ==================== STEP 4: VISUALIZE SAMPLES ====================

def step4_visualize_samples(train_loader):
    """Step 4: Visualize preprocessed samples"""
    print("\n" + "="*80)
    print("STEP 4: VISUALIZING PREPROCESSED SAMPLES")
    print("="*80)
    
    if train_loader is None:
        print("❌ No data loader provided")
        return
    
    # Get first batch
    print("\n🖼️ Loading first batch...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        elif hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Visualize images
    print("\n📸 Visualizing sample images...")
    images = batch['image']  # Shape: (B, 3, 512, 512)
    dr_labels = batch['dr_label']
    brightness = batch['brightness']
    contrast = batch['contrast']
    image_names = batch['image_name']
    
    # Display grid
    num_samples = min(4, images.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        # Clip to valid range
        img = np.clip(img, 0, 1)
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(
            f"{image_names[i]}\n"
            f"DR: {dr_labels[i]} | "
            f"Bright: {brightness[i]:.2f} | "
            f"Contrast: {contrast[i]:.2f}",
            fontsize=9
        )
        ax.axis('off')
    
    plt.tight_layout()
    output_file = Path(OUTPUT_PATH) / "sample_batch.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    plt.close()
    
    # Print batch statistics
    print(f"\n📊 Batch Statistics:")
    print(f"  Mean brightness: {brightness.mean():.3f}")
    print(f"  Mean contrast: {contrast.mean():.3f}")
    print(f"  DR grade distribution: {dict(zip(dr_labels, [1]*len(dr_labels)))}")


# ==================== STEP 5: VERIFY AUGMENTATION ====================

def step5_test_augmentation(train_loader):
    """Step 5: Test augmentation pipeline"""
    print("\n" + "="*80)
    print("STEP 5: TESTING AUGMENTATION PIPELINE")
    print("="*80)
    
    if train_loader is None:
        print("❌ No data loader provided")
        return
    
    print("\n🔄 Testing augmentation consistency...")
    
    # Get same sample multiple times (should be different due to augmentation)
    dataset = train_loader.dataset
    sample_idx = 0
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    # Get same sample 6 times (with augmentation)
    for i in range(6):
        sample = dataset[sample_idx]
        img = sample['image'].numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Augmentation #{i+1}", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("Same Sample with Different Augmentations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(OUTPUT_PATH) / "augmentation_examples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved augmentation examples to: {output_file}")
    plt.close()


# ==================== STEP 6: CLASS WEIGHT ANALYSIS ====================

def step6_class_weight_analysis(train_loader):
    """Step 6: Analyze class weights for loss function"""
    print("\n" + "="*80)
    print("STEP 6: CLASS WEIGHT ANALYSIS")
    print("="*80)
    
    if train_loader is None:
        print("❌ No data loader provided")
        return
    
    dataset = train_loader.dataset
    
    # Get class weights
    dr_weights = dataset.get_class_weights('dr')
    dme_weights = dataset.get_class_weights('dme')
    
    dr_labels = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }
    
    dme_labels = {
        0: "No DME",
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }
    
    print("\n⚖️ DR Class Weights (for loss function):")
    for i, weight in enumerate(dr_weights):
        print(f"  Class {i} ({dr_labels[i]}): {weight:.3f}")
    
    print("\n⚖️ DME Class Weights:")
    for i, weight in enumerate(dme_weights):
        if i < len(dme_labels):
            print(f"  Class {i} ({dme_labels[i]}): {weight:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # DR weights
    ax = axes[0]
    ax.bar(range(len(dr_weights)), dr_weights.numpy())
    ax.set_xlabel("DR Grade")
    ax.set_ylabel("Weight")
    ax.set_title("DR Class Weights")
    ax.set_xticks(range(len(dr_weights)))
    ax.set_xticklabels([dr_labels[i] for i in range(len(dr_weights))], rotation=45)
    
    # DME weights
    ax = axes[1]
    ax.bar(range(len(dme_weights)), dme_weights.numpy())
    ax.set_xlabel("DME Grade")
    ax.set_ylabel("Weight")
    ax.set_title("DME Class Weights")
    ax.set_xticks(range(len(dme_weights)))
    ax.set_xticklabels([dme_labels[i] for i in range(len(dme_weights))], rotation=45)
    
    plt.tight_layout()
    output_file = Path(OUTPUT_PATH) / "class_weights.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved class weights visualization to: {output_file}")
    plt.close()


# ==================== MAIN EXECUTION ====================

def main():
    """Run complete pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "IDRiD PREPROCESSING & DATA LOADING")
    print(" "*15 + "Complete Quick Start Guide for Explainable AI")
    print("="*80)
    
    # Step 1: Preprocess
    if not step1_run_preprocessing():
        print("\n❌ Preprocessing failed. Exiting.")
        return
    
    # Step 2: Verify
    step2_verify_preprocessing()
    
    # Step 3: Load data
    train_loader, val_loader, test_loader = step3_load_data()
    if train_loader is None:
        print("\n❌ Data loading failed. Exiting.")
        return
    
    # Step 4: Visualize
    step4_visualize_samples(train_loader)
    
    # Step 5: Test augmentation
    step5_test_augmentation(train_loader)
    
    # Step 6: Analyze class weights
    step6_class_weight_analysis(train_loader)
    
    # Summary
    print("\n" + "="*80)
    print("✅ COMPLETE! Your preprocessed dataset is ready for training.")
    print("="*80)
    print("\nNext steps:")
    print("1. Use train_loader for model training")
    print("2. Use val_loader for validation during training")
    print("3. Use test_loader for final evaluation")
    print("\nExample training loop:")
    print("""
    for epoch in range(num_epochs):
        for batch in train_loader:
            images = batch['image']
            dr_labels = batch['dr_grade']
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, dr_labels, 
                           weight=dataset.get_class_weights('dr'))
            
            # Backward pass
            loss.backward()
            optimizer.step()
    """)
    print("\nFor Explainability:")
    print("- Save image names and predictions")
    print("- Compare Grad-CAM with brightness/contrast from batch metadata")
    print("- Use SHAP to compute feature importance across image quality metrics")
    print("="*80)


if __name__ == "__main__":
    main()
