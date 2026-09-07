"""
evaluate_gradcam_fidelity.py
======================================================
Empirical Quantitative Evaluation Framework for Grad-CAM Explanation Fidelity
in Wafer Defect Classification using Real Test Dataset Wafer Maps.

Metrics Computed:
  1. Intersection over Union (IoU) between Grad-CAM heatmap & Die-Segmented Defect Mask
  2. Dice Coefficient
  3. Pointing Game Accuracy (Hit Rate: peak activation in defect die region)
  4. Inside-Mask Energy Ratio (Fraction of activation energy on defect dies)

Author: AI Research Team
======================================================
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

import config
import gradcam
import predictor
import utils

def extract_defect_die_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Extract die-level empirical defect mask by segmenting non-background active defect pixels.
    Wafer background is dark (~0 intensity or die pattern background).
    Defect patterns manifest as distinct intensity highlights or die clusters.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image_np * 255).astype(np.uint8)
        
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    r_max = int(min(w, h) * 0.48)
    
    # Circular wafer region mask
    wafer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wafer_mask, (cx, cy), r_max, 255, -1)
    
    # Segment active defect die pixels inside wafer boundary
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    defect_mask = cv2.bitwise_and(thresh, wafer_mask)
    
    return (defect_mask > 0).astype(np.float32)

def compute_fidelity_metrics(heatmap: np.ndarray, gt_mask: np.ndarray, tau: float = 0.5) -> dict:
    """
    Compute IoU, Dice Coefficient, Pointing Game Hit, and Inside-Mask Energy Ratio.
    """
    if np.max(heatmap) > 0:
        norm_heatmap = heatmap / np.max(heatmap)
    else:
        norm_heatmap = heatmap
        
    bin_heatmap = (norm_heatmap >= tau).astype(np.float32)
    bin_gt = (gt_mask > 0.5).astype(np.float32)
    
    intersection = np.sum(bin_heatmap * bin_gt)
    union = np.sum(bin_heatmap) + np.sum(bin_gt) - intersection
    
    iou = float(intersection / union) if union > 0 else (1.0 if np.sum(bin_gt) == 0 else 0.0)
    dice = float((2.0 * intersection) / (np.sum(bin_heatmap) + np.sum(bin_gt))) if (np.sum(bin_heatmap) + np.sum(bin_gt)) > 0 else (1.0 if np.sum(bin_gt) == 0 else 0.0)
    
    # Pointing Game Hit Rate
    peak_idx = np.unravel_index(np.argmax(norm_heatmap), norm_heatmap.shape)
    hit = 1.0 if bin_gt[peak_idx] > 0.5 else 0.0
    
    # Energy Ratio
    total_energy = np.sum(norm_heatmap)
    inside_energy = np.sum(norm_heatmap * bin_gt)
    energy_ratio = float(inside_energy / total_energy) if total_energy > 0 else 0.0
    
    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "pointing_hit": float(hit),
        "energy_ratio": round(energy_ratio, 4)
    }

def evaluate_model_fidelity(model, class_names, num_samples_per_class=50):
    results = {}
    
    for cname in class_names:
        if cname == "none":
            continue
            
        class_dir = config.TEST_IMAGES_DIR / cname
        if not class_dir.exists():
            continue
            
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        if not image_files:
            continue
            
        sample_files = image_files[:num_samples_per_class]
        class_id = list(config.CLASS_NAMES.values()).index(cname)
        
        ious, dices, hits, energies = [], [], [], []
        
        for img_path in sample_files:
            try:
                # Load & preprocess image
                pil_img = cv2.imread(str(img_path))
                if pil_img is None:
                    continue
                pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(pil_img, config.IMAGE_SIZE)
                norm_img = resized_img.astype(np.float32) / 255.0
                
                # Empirical defect die mask
                gt_mask = extract_defect_die_mask(norm_img)
                
                # Model input
                input_tensor = np.expand_dims(norm_img, axis=0)
                
                # Grad-CAM heatmap
                heatmap = gradcam.make_gradcam_heatmap(
                    model=model,
                    preprocessed_input=input_tensor,
                    class_index=class_id,
                )
                heatmap_resized = cv2.resize(heatmap, config.IMAGE_SIZE)
                
                metrics = compute_fidelity_metrics(heatmap_resized, gt_mask)
                ious.append(metrics["iou"])
                dices.append(metrics["dice"])
                hits.append(metrics["pointing_hit"])
                energies.append(metrics["energy_ratio"])
            except Exception as exc:
                continue

        if ious:
            results[cname] = {
                "sample_count": len(ious),
                "mean_iou": round(float(np.mean(ious)), 4),
                "mean_dice": round(float(np.mean(dices)), 4),
                "pointing_accuracy_pct": round(float(np.mean(hits)) * 100.0, 2),
                "mean_energy_ratio": round(float(np.mean(energies)), 4)
            }
            print(f"  Class: {cname:<12} | Samples: {len(ious):<3} | IoU: {np.mean(ious):.4f} | Dice: {np.mean(dices):.4f} | Hit: {np.mean(hits)*100:.1f}% | Energy: {np.mean(energies):.4f}")

    return results

def main():
    print("=" * 75)
    print("EMPIRICAL QUANTITATIVE GRAD-CAM FIDELITY EVALUATION ON REAL TEST SET")
    print("=" * 75)

    model_path = config.MODEL_PATH
    if not model_path.exists():
        print(f"Error: Champion model not found at {model_path}")
        return

    print(f"Loading champion model from {model_path.name}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    class_names = [c for c in config.CLASS_NAMES.values() if c != "none"]

    print("\nEvaluating empirical Grad-CAM fidelity across test wafer images...")
    fidelity_results = evaluate_model_fidelity(model, class_names, num_samples_per_class=40)

    results_dir = config.BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_json = results_dir / "gradcam_fidelity_metrics.json"
    with open(output_json, "w") as f:
        json.dump(fidelity_results, f, indent=4)

    print(f"\nEmpirical fidelity metrics saved successfully to {output_json}")
    print("=" * 75)

if __name__ == "__main__":
    main()
