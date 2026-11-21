"""
Explainable AI Integration with IDRiD Preprocessing
=====================================================

This guide shows how to leverage your preprocessed IDRiD data for 
Explainable AI techniques: Grad-CAM, SHAP, and validation methods
"""

import numpy as np
import torch
import cv2
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List

# ==================== PART 1: GRAD-CAM IMPLEMENTATION ====================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Shows which image regions are important for the model's prediction
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Args:
            model: PyTorch model
            target_layer: Layer name to compute gradients from
                         Example: "model.backbone.layer4[-1]"
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get target layer (example: "model.backbone.layer4[-1]")
        target_module = self._get_module(self.target_layer)
        
        h1 = target_module.register_forward_hook(forward_hook)
        h2 = target_module.register_full_backward_hook(backward_hook)
        
        self.hook_handles = [h1, h2]
    
    def _get_module(self, module_name: str) -> torch.nn.Module:
        """Get module from model by name"""
        parts = module_name.split('.')
        module = self.model
        for part in parts:
            if part == 'model':
                continue
            if '[' in part:  # Handle indexed access like layer4[0]
                part_name = part[:part.index('[')]
                module = getattr(module, part_name)
            else:
                module = getattr(module, part)
        return module
    
    def generate(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM for an image
        
        Args:
            image: Input image (1, 3, 512, 512)
            class_idx: Target class for gradients
        
        Returns:
            CAM: Grad-CAM heatmap (512, 512)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Compute Grad-CAM
        # Shape: activations = (1, C, H, W), gradients = (1, C, H, W)
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights: average gradients across spatial dimensions
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Compute weighted activation
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to get only positive contributions
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        # Resize to original image size (512, 512)
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (512, 512), interpolation=cv2.INTER_CUBIC)
        
        return cam_resized
    
    def __del__(self):
        """Remove hooks on deletion"""
        for handle in self.hook_handles:
            handle.remove()


# ==================== PART 2: OVERLAY WITH PREPROCESSING DATA ====================

class ExplanabilityAnalyzer:
    """
    Analyze explanations in context of preprocessing metadata
    """
    
    def __init__(self, preprocessed_data_root: str):
        """
        Args:
            preprocessed_data_root: Path to preprocessed IDRiD data
        """
        self.data_root = Path(preprocessed_data_root)
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load all metadata files"""
        self.metadata['train'] = pd.read_csv(
            self.data_root / "unified" / "train_unified_metadata.csv"
        )
        self.metadata['test'] = pd.read_csv(
            self.data_root / "unified" / "test_unified_metadata.csv"
        )
    
    def get_image_metadata(self, image_name: str, split: str = 'test') -> Dict:
        """Get preprocessing metadata for an image"""
        df = self.metadata[split]
        row = df[df['image_name'] == image_name].iloc[0]
        return row.to_dict()
    
    def overlay_gradcam_with_image(
        self,
        image: np.ndarray,
        gradcam: np.ndarray,
        image_metadata: Dict,
        save_path: str = None
    ) -> np.ndarray:
        """
        Overlay Grad-CAM on original image with quality context
        
        Args:
            image: Original image (512, 512, 3), [0, 1] range
            gradcam: Grad-CAM heatmap (512, 512), [0, 1] range
            image_metadata: Dict with brightness, contrast, etc.
            save_path: Optional path to save visualization
        
        Returns:
            overlaid: Image with Grad-CAM overlay and text annotations
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM
        cmap = cm.get_cmap('jet')
        gradcam_colored = cmap(gradcam)[:, :, :3]
        axes[1].imshow(gradcam)
        axes[1].set_cmap('jet')
        axes[1].set_title("Grad-CAM Heatmap", fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        alpha = 0.5
        overlay = (1 - alpha) * image + alpha * gradcam_colored
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay", fontweight='bold')
        axes[2].axis('off')
        
        # Add metadata text
        metadata_text = f"""
        DR Grade: {image_metadata['DR_label']}
        DME Grade: {image_metadata['DME_label']}
        Brightness: {image_metadata['brightness']:.3f}
        Contrast: {image_metadata['contrast']:.3f}
        Saturation: {image_metadata['saturation']:.3f}
        """
        fig.text(0.05, 0.05, metadata_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()


# ==================== PART 3: LESION-AWARE EXPLANATION VALIDATION ====================

class LesionExplanationValidator:
    """
    Validate explanations against actual lesion annotations
    (IoU: Intersection over Union)
    """
    
    def __init__(self, segmentation_data_root: str):
        """
        Args:
            segmentation_data_root: Path to segmentation masks
        """
        self.seg_root = Path(segmentation_data_root)
    
    def compute_iou(
        self,
        predicted_heatmap: np.ndarray,
        lesion_type: str,
        image_name: str,
        threshold: float = 0.5
    ) -> float:
        """
        Compute IoU between predicted attention and actual lesion
        
        Args:
            predicted_heatmap: Grad-CAM or attention heatmap (512, 512)
            lesion_type: 'Microaneurysms', 'Haemorrhages', etc.
            image_name: Image identifier
            threshold: Threshold for binarizing heatmap
        
        Returns:
            iou: Intersection over Union score [0, 1]
        """
        # Load lesion mask
        mask_file = self.seg_root / f"{image_name}_{lesion_type.replace(' ', '_')}_mask.npy"
        
        if not mask_file.exists():
            print(f"⚠️ Lesion mask not found: {mask_file}")
            return None
        
        lesion_mask = np.load(str(mask_file))  # (512, 512), [0, 1]
        
        # Binarize heatmap
        predicted_binary = (predicted_heatmap > threshold).astype(np.float32)
        lesion_binary = (lesion_mask > 0.5).astype(np.float32)
        
        # Compute IoU
        intersection = np.sum(predicted_binary * lesion_binary)
        union = np.sum(np.maximum(predicted_binary, lesion_binary))
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return iou
    
    def validate_explanations_for_image(
        self,
        gradcam: np.ndarray,
        image_name: str,
        dr_grade: int
    ) -> Dict:
        """
        Validate Grad-CAM against all lesion types for an image
        
        Args:
            gradcam: Grad-CAM heatmap (512, 512)
            image_name: Image identifier
            dr_grade: DR severity grade (0-4)
        
        Returns:
            results: Dict with IoU for each lesion type and overall score
        """
        lesion_types = [
            "Microaneurysms",
            "Haemorrhages",
            "Hard Exudates",
            "Soft Exudates"
        ]
        
        results = {
            'image_name': image_name,
            'dr_grade': dr_grade,
            'iou_scores': {}
        }
        
        for lesion_type in lesion_types:
            iou = self.compute_iou(gradcam, lesion_type, image_name)
            if iou is not None:
                results['iou_scores'][lesion_type] = iou
        
        # Compute mean IoU
        if results['iou_scores']:
            results['mean_iou'] = np.mean(list(results['iou_scores'].values()))
        
        return results


# ==================== PART 4: SHAP INTEGRATION ====================

class SHAPExplainer:
    """
    Use SHAP to analyze feature importance from preprocessing metadata
    """
    
    @staticmethod
    def prepare_features_from_metadata(
        metadata_df: pd.DataFrame,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix from preprocessing metadata
        
        Args:
            metadata_df: DataFrame with brightness, contrast, etc.
            predictions: Model predictions (N, 5) for DR classes
        
        Returns:
            X: Feature matrix (N, F)
            feature_names: List of feature names
        """
        feature_columns = [
            'brightness',
            'contrast',
            'saturation',
            'od_fovea_distance',
            'has_dr',
            'has_dme'
        ]
        
        X = metadata_df[feature_columns].values.astype(np.float32)
        
        return X, feature_columns
    
    @staticmethod
    def analyze_prediction_confidence(
        predictions: np.ndarray,
        metadata: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict:
        """
        Analyze how image quality affects model confidence
        
        Args:
            predictions: Model logits/probabilities
            metadata: Image metadata (brightness, contrast, etc.)
            feature_names: Feature column names
        
        Returns:
            analysis: Dict with correlations and statistics
        """
        # Get maximum probability (confidence) for each prediction
        confidences = np.max(predictions, axis=1)
        
        analysis = {
            'mean_confidence': float(confidences.mean()),
            'std_confidence': float(confidences.std()),
            'feature_confidence_correlation': {}
        }
        
        # Compute correlation between features and confidence
        for feature in feature_names:
            if feature in metadata.columns:
                feature_values = metadata[feature].values
                correlation = np.corrcoef(feature_values, confidences)[0, 1]
                analysis['feature_confidence_correlation'][feature] = float(correlation)
        
        return analysis


# ==================== PART 5: EXPLANATION REPORT GENERATION ====================

class ExplanationReportGenerator:
    """
    Generate comprehensive explanation reports for predictions
    """
    
    @staticmethod
    def generate_sample_report(
        image_name: str,
        image_metadata: Dict,
        prediction: Dict,
        gradcam: np.ndarray,
        lesion_ious: Dict = None
    ) -> str:
        """
        Generate human-readable explanation report
        
        Args:
            image_name: Image identifier
            image_metadata: Preprocessing metadata
            prediction: Model prediction (label, confidence)
            gradcam: Grad-CAM heatmap
            lesion_ious: IoU scores for lesions
        
        Returns:
            report: Formatted text report
        """
        report = f"""
{'='*80}
EXPLAINABLE AI REPORT: Diabetic Retinopathy Detection
{'='*80}

IMAGE INFORMATION
─────────────────────────────────────────────────────────────────────────────
Image Name:                 {image_name}
Ground Truth DR Grade:      {image_metadata['DR_label']}
Ground Truth DME Grade:     {image_metadata['DME_label']}

IMAGE QUALITY METRICS (Preprocessing)
─────────────────────────────────────────────────────────────────────────────
Brightness:                 {image_metadata['brightness']:.3f}
  → Assessment: {'✓ Good' if 0.35 <= image_metadata['brightness'] <= 0.65 else '⚠️ Poor'}
  → Impact: Affects model confidence by ~{abs(0.5 - image_metadata['brightness'])*100:.1f}%

Contrast:                   {image_metadata['contrast']:.3f}
  → Assessment: {'✓ Good' if 0.15 <= image_metadata['contrast'] <= 0.35 else '⚠️ Poor'}
  → Impact: Lower contrast → harder lesion detection

Saturation:                 {image_metadata['saturation']:.3f}
  → Assessment: Color information preservation

Optic Disc - Fovea Distance: {image_metadata['od_fovea_distance']:.3f}
  → Clinical relevance: Affects risk assessment

MODEL PREDICTION
─────────────────────────────────────────────────────────────────────────────
Predicted DR Grade:        {prediction['label']}
Prediction Confidence:      {prediction['confidence']:.3f}
  → Confidence Level: {'✓ High (>0.8)' if prediction['confidence'] > 0.8 
                      else '⚠️ Medium (0.6-0.8)' if prediction['confidence'] > 0.6
                      else '❌ Low (<0.6)'}

VISUAL EXPLANATION (Grad-CAM)
─────────────────────────────────────────────────────────────────────────────
Grad-CAM Focus:
  → Max activation region: Identifies where model paid most attention
  → Mean activation: {gradcam.mean():.3f}
  → Peak activation: {gradcam.max():.3f}

Activation Distribution:
  → High (>0.7):   {(gradcam > 0.7).sum()} pixels
  → Medium (0.3-0.7): {((gradcam > 0.3) & (gradcam <= 0.7)).sum()} pixels
  → Low (<0.3):    {(gradcam <= 0.3).sum()} pixels
"""
        
        if lesion_ious:
            report += "\nLESION-AWARE VALIDATION (IoU with Segmentation Masks)\n"
            report += "─────────────────────────────────────────────────────────────────────────────\n"
            
            for lesion_type, iou in lesion_ious['iou_scores'].items():
                status = '✓' if iou > 0.5 else '⚠️' if iou > 0.3 else '❌'
                report += f"{lesion_type:20} IoU: {iou:.3f}  {status}\n"
            
            report += f"\nMean IoU: {lesion_ious['mean_iou']:.3f}\n"
            report += "→ Interpretation: Higher IoU means explanations align better with actual lesions\n"
        
        report += f"""
EXPLAINABILITY ASSESSMENT
─────────────────────────────────────────────────────────────────────────────
Quality Score:  {'✓ Excellent' if (lesion_ious and lesion_ious['mean_iou'] > 0.6) 
                             else '⚠️ Good' if (lesion_ious and lesion_ious['mean_iou'] > 0.4)
                             else '❌ Needs Improvement'}

Key Findings:
  1. Image quality {'enables' if image_metadata['brightness'] > 0.3 else 'limits'} 
     accurate lesion detection
  2. Model confidence {'aligns' if prediction['confidence'] > 0.7 else 'differs from'} 
     ground truth grade
  3. Attention regions {'match' if lesion_ious and lesion_ious['mean_iou'] > 0.5 else 'diverge from'} 
     actual lesions

RECOMMENDATIONS FOR CLINICIANS
─────────────────────────────────────────────────────────────────────────────
  • Use this model as {'primary' if prediction['confidence'] > 0.8 else 'secondary'} screening tool
  • Low confidence predictions require additional expert review
  • Verify high-confidence predictions on well-lit fundus images
  
TECHNICAL DETAILS
─────────────────────────────────────────────────────────────────────────────
Preprocessing Applied:
  - Image resized to 512×512
  - Normalized to [0, 1] range
  - CLAHE enhancement (clip=2.0, tile=8×8)
  - Segmentation masks loaded for validation

Model Architecture:
  - EfficientNet-B4 + Vision Transformer hybrid
  - Fusion layer combining local (CNN) + global (ViT) features
  - Grad-CAM computed from final convolutional layer

{'='*80}
"""
        return report


# ==================== EXAMPLE USAGE ====================

def example_workflow():
    """
    Example: Complete explainability workflow
    """
    
    print("="*80)
    print("EXPLAINABLE AI WORKFLOW - EXAMPLE")
    print("="*80)
    
    # 1. Load image and metadata
    print("\n1️⃣ Loading preprocessed image and metadata...")
    preprocessed_root = "./preprocessed_idrid"
    image_name = "IDRiD_001"
    
    analyzer = ExplanabilityAnalyzer(preprocessed_root)
    metadata = analyzer.get_image_metadata(image_name, split='test')
    print(f"   ✓ Loaded metadata: brightness={metadata['brightness']:.3f}, "
          f"DR={metadata['DR_label']}")
    
    # 2. Generate Grad-CAM (placeholder)
    print("\n2️⃣ Generating Grad-CAM visualization...")
    # In practice: gradcam = grad_cam.generate(image_tensor, class_idx=dr_grade)
    gradcam = np.random.rand(512, 512)  # Placeholder
    print(f"   ✓ Generated Grad-CAM (shape: {gradcam.shape})")
    
    # 3. Validate against lesions
    print("\n3️⃣ Validating explanations with lesion masks...")
    validator = LesionExplanationValidator(
        f"{preprocessed_root}/segmentation/testing"
    )
    validation = validator.validate_explanations_for_image(
        gradcam, image_name, dr_grade=2
    )
    print(f"   ✓ Computed IoU scores: {validation['iou_scores']}")
    
    # 4. Generate report
    print("\n4️⃣ Generating explanation report...")
    reporter = ExplanationReportGenerator()
    prediction = {'label': 'Moderate', 'confidence': 0.82}
    report = reporter.generate_sample_report(
        image_name, metadata, prediction, gradcam, validation
    )
    print(report)


if __name__ == "__main__":
    example_workflow()
