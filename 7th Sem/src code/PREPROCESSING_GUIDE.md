# IDRiD Dataset Preprocessing Guide
## For Explainable AI in Diabetic Retinopathy Detection

---

## 📋 Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Why Preprocessing Matters for Explainability](#why-preprocessing)
3. [Detailed Preprocessing Steps](#preprocessing-steps)
4. [Data Augmentation Strategy](#augmentation)
5. [Class Imbalance Handling](#class-imbalance)
6. [Quality Control](#quality-control)
7. [Integration with Explainable AI](#explainability-integration)

---

## Dataset Overview

The IDRiD dataset consists of **three complementary tasks**:

### Task A: Segmentation (81 images)
- **Purpose**: Identify and localize retinal lesions
- **Data**:
  - Original fundus images (RGB JPG)
  - Ground truth masks for lesion types:
    - Microaneurysms (MA) - smallest, earliest sign
    - Hemorrhages (HE) - blood vessel ruptures
    - Hard Exudates (EX) - yellow-white deposits
    - Soft Exudates (SE) - cotton-wool spots
    - Optic Disc (OD) - reference landmark

**Why it matters for Explainability:**
- Pixel-level annotations let us validate where the model focuses
- Can use segmentation masks to check if Grad-CAM highlights match actual lesions

### Task B: Disease Grading (516 images)
- **Purpose**: Classify DR severity level
- **Data**:
  - Original fundus images (RGB JPG)
  - Labels: DR severity (0-4) + DME severity (0-3)
  - 413 training, 103 testing

**DR Severity Grades:**
- 0: No DR - No abnormalities
- 1: Mild DR - Only microaneurysms
- 2: Moderate DR - MA + other lesions, not reaching severe
- 3: Severe DR - Multiple/extensive lesions
- 4: Proliferative DR - Neovascularization (most dangerous)

**DME Severity Grades:**
- 0: No DME
- 1: Mild DME - Within 1 disc diameter of fovea
- 2: Moderate DME - 1-2 disc diameters from fovea
- 3: Severe DME - Involving/threatening fovea

**Why it matters for Explainability:**
- This is your PRIMARY classification task
- Class imbalance needs attention (fewer severe cases)
- Image quality varies → explains prediction uncertainty

### Task C: Localization (516 images)
- **Purpose**: Locate anatomical landmarks
- **Data**:
  - Original fundus images
  - Optic Disc center coordinates (X, Y)
  - Fovea center coordinates (X, Y)

**Why it matters for Explainability:**
- Identifies regions of interest for the model
- OD-Fovea distance is clinically relevant
- Can overlay landmark heatmaps on Grad-CAM for validation

---

## Why Preprocessing Matters for Explainability

### Problem 1: Raw Images Have Inconsistent Quality
**Issue**: Different fundus cameras, lighting conditions, patient populations
```
Raw image variance:
- Brightness: Highly variable (20-95% of max intensity)
- Contrast: Wildly different across dataset
- Color balance: Slight variations
- Artifact types: Dust, reflections, poor focus
```

**Solution**: Standardize without losing diagnostic information
- Resize to uniform (512×512) for consistent receptive fields
- Normalize to [0, 1] range for stable training
- Apply CLAHE enhancement (Contrast-Limited Adaptive Histogram Equalization)
  - Enhances local contrast without over-amplifying noise
  - Mimics what ophthalmologists do when examining images

**For Explainability:**
- Consistent preprocessing → consistent model behavior
- Brightness/contrast features become meaningful for attribution
- Easier to compare Grad-CAM across different image types

---

### Problem 2: Lesion-Level Information Is Lost in Classification
**Issue**: Classification (0-4) doesn't tell which lesions caused the grade
```
Example:
Image A: Grade 2 due to 5 microaneurysms
Image B: Grade 2 due to hemorrhage + exudates
→ Same label, different visual patterns
```

**Solution**: Preserve lesion annotations during preprocessing
- Keep segmentation masks alongside images
- Track lesion presence/area percentages
- Use for validation of explanations (Intersection over Union)

**For Explainability:**
- Generate "expected" regions that should light up in Grad-CAM
- Validate that explanations focus on clinically relevant areas
- Compare model's lesion focus vs. actual lesion locations

---

### Problem 3: Class Imbalance Skews Model Confidence
**Issue**: Proliferative DR (grade 4) is rare (~5% of cases)
```
Grade distribution (typical):
- No DR: 40%
- Mild: 25%
- Moderate: 20%
- Severe: 10%
- Proliferative: 5%

→ Model becomes less confident on rare classes
→ Predictions for severe cases are less reliable
```

**Solution**: Compute class weights & use in loss function
```python
weight[class_i] = total_samples / (num_classes × samples_in_class_i)

Proliferative gets ~4× higher loss weight
→ Model pays more attention to getting it right
```

**For Explainability:**
- Document class weights in results section
- Explain why confidence is lower for rare classes
- Use SHAP to show feature importance per class

---

### Problem 4: Landmarks Need Spatial Context
**Issue**: Optic Disc and Fovea locations matter for diagnosis
```
Clinical significance:
- OD position: Reference for abnormality mapping
- Fovea position: Center of vision, must preserve
- OD-Fovea distance: Indicates optic nerve health
- Macular region: Most critical for vision
```

**Solution**: Create Gaussian heatmaps around landmarks
- Soft localization (not binary) to avoid boundary issues
- Normalized coordinates for scale-invariance
- Pre-compute OD-Fovea distance as feature

**For Explainability:**
- Model should focus more on macular region (around fovea)
- Can overlay landmark heatmaps on predictions
- Validate that "severe DR" predictions involve foveal damage

---

## Detailed Preprocessing Steps

### Step 1: Image Resizing
```
Raw image: Variable sizes (1400×960 to 2048×1360)
            ↓
Resized:   512×512 (standard size)

Why 512?
- Large enough to preserve fine lesion details (MA are tiny)
- Small enough for GPU memory (fit larger batches)
- Power of 2 (good for CNN architectures)
- Consistent with medical imaging conventions
```

### Step 2: Normalization
```
Raw pixel values: [0, 255] (uint8)
                  ↓
Normalized:       [0, 1] (float32)

Formula: pixel_normalized = pixel_raw / 255.0

Why float32 and [0, 1]?
- PyTorch uses float32 by default
- [0, 1] range is standard for neural networks
- Prevents overflow in gradients
- Makes batch normalization stable
```

### Step 3: CLAHE Enhancement (for disease grading only)
```
Purpose: Enhance local contrast like ophthalmologists do
         when examining images under magnification

Steps:
1. Convert BGR → LAB color space
   (L=lightness, A=green-red, B=yellow-blue)
   
2. Apply CLAHE only to L (lightness) channel
   - Preserves color information
   - Focuses on contrast enhancement
   
3. Clip limit = 2.0 (prevents over-sharpening)
   Tile size = 8×8 (local region size)
   
4. Convert back to BGR

Result: Microaneurysms and exudates become more visible
        WITHOUT amplifying artifacts

Enhanced image uses for:
- Better feature extraction by EfficientNet
- More robust to image quality variations
```

### Step 4: Mask Processing (Segmentation Task)
```
Raw mask:    TIF file with 0=background, 255=lesion
             Size: Original image dimensions
             ↓
Resized:     Bilinear interpolation to 512×512
             (NEAREST for masks, not CUBIC)
             ↓
Normalized:  [0, 1] range
             ↓
Saved:       Separate .npy file per lesion type

Why separate files?
- Memory efficient (load only needed masks)
- Easy to compute per-lesion statistics
- Allows flexible loss weighting per lesion

Statistics computed:
- Presence: 1 if any lesion pixels, 0 otherwise
- Area %: (lesion_pixels / total_pixels) × 100
  → Used for severity validation
```

### Step 5: Coordinate Normalization (Localization)
```
Raw coordinates: Pixels in original image size
                 e.g., OD_X=756, OD_Y=480 (in 1400×960 image)
                 ↓
Scaled:          Scaled to 512×512
                 OD_X_scaled = 756 × (512/1400)
                 OD_Y_scaled = 480 × (512/960)
                 ↓
Normalized:      [0, 1] range
                 OD_X_norm = OD_X_scaled / 512
                 OD_Y_norm = OD_Y_scaled / 512
                 
Result: Model-agnostic coordinates for any image size

Additional computed:
- OD-Fovea distance = sqrt((OD_x - Fovea_x)² + (OD_y - Fovea_y)²)
  → Captures spatial relationship
```

### Step 6: Heatmap Generation
```
Purpose: Visualize landmarks for Grad-CAM validation

Process:
1. Create 512×512 empty heatmap
2. Place Gaussian blob at OD center
   - σ = 30 pixels (radius of optic disc)
   - Formula: exp(-(distance² / (2 × σ²)))
   
3. Place Gaussian blob at Fovea center
   - σ = 20 pixels (smaller than OD)
   
4. Save as .npy file

Use in explainability:
- Overlay on Grad-CAM to see if attention aligns with landmarks
- Compute correlation between Grad-CAM and heatmap
```

---

## Data Augmentation Strategy

### Why Augment?
```
Training set: Only 413 images
Deep learning model: Millions of parameters
→ Severe overfitting risk

Solution: Create variations without changing truth labels
```

### Three Levels of Augmentation

#### Light Augmentation (for stable training)
```python
- Horizontal flip (50%)
  → Retinas can appear on left or right
  
- Rotate ±15° (50%)
  → Camera angle varies in practice
  
- Gaussian noise (20%)
  → Sensor noise simulation
```

#### Medium Augmentation (default, recommended)
```python
- Horizontal + Vertical flip
  → Real variations in imaging angle
  
- Rotate ±30°
  → More camera angle variation
  
- Elastic transform (30%)
  → Simulates eyeball movement during image capture
  
- Gaussian noise (30%)
  
- Brightness/Contrast ±20% (40%)
  → Camera settings vary
  
- Color jitter (30%)
  → Lighting conditions change
```

#### Heavy Augmentation (for overfitting prevention)
```python
- All of above, plus:
- Rotate ±45° (70%)
- Elastic transform (50%, stronger)
- Brightness/Contrast ±30% (50%)
- Color jitter (40%, stronger)
- Coarse dropout: Remove 50×50 patches (30%)
  → Simulates image artifacts/cataracts
```

### Important: Medical Imaging Considerations
```
❌ DON'T flip vertically without care
   (retina has anatomical top/bottom, affects grading)
   
✓ DO use medical imaging augmentation libraries
   (albumentations designed for this)
   
⚠️ Validate that augmentation doesn't change ground truth
   (example: 90° rotation changes DR grade if anatomy changes)
```

---

## Class Imbalance Handling

### The Problem
```
DR Severity distribution (typical):
No DR:        40% → 166 samples
Mild:         25% →  103 samples
Moderate:     20% →   83 samples
Severe:       10% →   41 samples
Proliferative: 5% →   20 samples

Model behavior:
- Learns "No DR" well (lots of examples)
- Struggles with "Proliferative" (few examples)
- Biased toward predicting "No DR"
```

### Solution 1: Class Weights
```python
# In dataset loader
weight = total_samples / (num_classes × count_per_class)

For our example:
No DR weight:        413 / (5 × 166) = 0.50
Mild weight:         413 / (5 × 103) = 0.80
Moderate weight:     413 / (5 × 83) = 0.99
Severe weight:       413 / (5 × 41) = 2.01
Proliferative weight: 413 / (5 × 20) = 4.13

In loss function:
CE_loss = -weight[class] × log(p[class])

Effect: Proliferative misclassification costs 8× more than No DR
```

### Solution 2: Focal Loss (optional)
```python
# Reduces loss for easy examples, focuses on hard ones
Focal_Loss = -α × (1 - p_t)^γ × log(p_t)

γ = 2.0 (focus parameter)

When model is wrong on easy class: γ=2, high loss
When model is right on hard class: γ=2, but (1-p_t) small, lower loss

Effect: Prioritizes learning from difficult minority classes
```

### Solution 3: Data Resampling (optional)
```
Oversampling: Duplicate minority class samples
Undersampling: Remove majority class samples
SMOTE: Generate synthetic minority samples

Trade-off:
- Oversampling: Risk of overfitting on minority class
- Undersampling: Lose information from majority class
- SMOTE: Requires careful parameter tuning

Recommended: Use class weights (simplest, works well)
```

---

## Quality Control

### Image Quality Metrics
```
Computed during preprocessing:

1. Brightness: Mean pixel intensity [0, 1]
   - Low brightness: Poor visibility
   - High brightness: Overexposed
   - Ideal: 0.35-0.65
   
2. Contrast: Std deviation of pixel intensity
   - Low contrast: Blurry/washed out
   - High contrast: Over-sharpened
   - Ideal: 0.15-0.35
   
3. Saturation: Max - Min pixel intensity
   - Low saturation: Grayscale-like
   - High saturation: Very colorful
   - Ideal: 0.40-0.90

Action:
- Log these statistics
- Flag outliers (quality issues)
- Use in explainability (explain low confidence on poor quality)
```

### Sanity Checks
```
After preprocessing, verify:

✓ All images: (512, 512, 3) shape
✓ All images: [0, 1] range
✓ All masks: (512, 512) single channel
✓ All coordinates: [0, 1] normalized
✓ No NaN/Inf values
✓ Metadata matches image counts
✓ Class distribution as expected

Implementation:
- Dataset loader validates on first access
- Raises exceptions if checks fail
- Prevents silent bugs downstream
```

---

## Explainability Integration

### 1. Metadata for Attribution
```python
# What gets saved with each image:
{
    "image_name": "IDRiD_001",
    "DR_grade": 2,
    "DR_label": "Moderate",
    "DME_grade": 0,
    "brightness": 0.42,
    "contrast": 0.18,
    "saturation": 0.65,
    "has_dr": 1,
    "has_dme": 0,
    "microaneurysm_presence": 1,
    "microaneurysm_area_percent": 0.8,
    "hemorrhage_presence": 1,
    "hemorrhage_area_percent": 1.2,
    ...
}

Use in explainability:
- SHAP can compute feature importance for "brightness"
- Understand: "Low contrast images → less confident predictions?"
- Generate explanations like: "Model detected microaneurysms 
  despite poor image quality (contrast=0.12)"
```

### 2. Lesion-Based Validation
```python
# After model generates Grad-CAM:

For each test image:
1. Get Grad-CAM heatmap from model
2. Get actual lesion masks (segmentation ground truth)
3. Compute overlap: IoU = |CAM ∩ Lesion| / |CAM ∪ Lesion|

Interpretation:
- IoU > 0.5: Model focuses on actual lesions ✓
- IoU < 0.3: Model uses spurious features ✗
- 0.3-0.5: Partial alignment (room for improvement)

Report:
- Mean IoU per lesion type
- Identify which lesion types model struggles to explain
- Use to improve model transparency
```

### 3. Landmark-Based Validation
```python
# After model generates attention:

1. Create expected attention heatmap from landmarks
   - Gaussian peaks at OD and fovea centers
   
2. Compare model's Grad-CAM with expected heatmap
3. Compute correlation coefficient

High correlation: Model attends to anatomically relevant regions
Low correlation: Model uses global patterns (less interpretable)

Possible findings:
- Model correctly focuses on macular region for severe DR
- Model uses optic disc as false signal for grading
- Model's confidence varies with landmark visibility
```

### 4. Class-Wise Explanations
```python
# Different explanations per DR severity:

No DR explanations:
- Focus: Absence of lesions
- Grad-CAM: Diffuse, low intensity
- Confidence: High (confident in absence)

Proliferative DR explanations:
- Focus: Neovascularization areas
- Grad-CAM: High intensity clusters
- Confidence: Lower (rare, harder to predict)
- Additional: Compare with SHAP for feature importance

Use in report:
"Model's confidence decreases with rarity of class, 
 but explanations remain anatomically valid"
```

---

## Practical Preprocessing Checklist

### Before Running Preprocessor
```
□ Download IDRiD dataset (ensure all folders present)
□ Verify folder structure:
  - 1. Segmentation/
  - 2. Disease Grading/
  - 3. Localization/
□ Check storage space: ~1-2 GB for preprocessed data
□ Install required packages: OpenCV, pandas, numpy, PIL
```

### After Preprocessing
```
□ Verify output folders created correctly
□ Check statistics report (data_statistics.json)
□ Spot-check preprocessed images visually
□ Verify class distribution matches expected
□ Check for missing files/metadata entries
□ Compare image statistics with expected ranges
```

### Before Training Model
```
□ Load dataset with DataLoader
□ Verify batch loading works
□ Check batch shapes: (B, 3, 512, 512)
□ Verify labels are correct tensors
□ Test augmentation pipeline
□ Confirm class weights computed correctly
```

---

## Final Notes for Explainability

### Record Everything
```python
# In your preprocessing logs, save:

preprocessing_config = {
    "idrid_version": "dataset_version_used",
    "target_size": (512, 512),
    "normalization": "[0, 1] range",
    "enhancement": "CLAHE (clip=2.0, tile=8x8)",
    "augmentation": "medium",
    "class_weights": "computed",
    "segmentation_masks": "included",
    "landmark_heatmaps": "included",
    "statistics": data_statistics,
}

Save as preprocessing_metadata.json
→ Reference in paper/report
→ Enables reproducibility
```

### Connect to Model
```
In your model training report, include:

"Preprocessing Impact on Explainability:
- Images normalized to [0, 1] for gradient stability
- CLAHE enhancement improves lesion visibility
  → Easier for model to learn and explain
- Segmentation masks enable IoU validation
  → Verify Grad-CAM focuses on actual lesions
- Brightness/contrast recorded per image
  → SHAP can identify quality-confidence relationship
- Landmark heatmaps overlay with explanations
  → Show model respects anatomical relevance"
```

---

## References

Porwal, A., Gour, N., Garg, S., et al. (2020). 
Indian Diabetic Retinopathy Image Dataset (IDRiD). 
IEEE DataPort. https://doi.org/10.21203/rs.3.rs-72636/v1

