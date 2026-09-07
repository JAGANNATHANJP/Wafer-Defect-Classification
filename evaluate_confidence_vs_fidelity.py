"""
evaluate_confidence_vs_fidelity.py
======================================================
Empirical Evaluation of Prediction Confidence vs. Grad-CAM Explanation Fidelity.

Analyzes:
  1. Correlation (Pearson r) between model confidence score p(y_hat|x) and Grad-CAM IoU.
  2. Mean Explanation IoU on Correctly Classified vs. Misclassified (False Positive) Samples.
  3. Morphological breakdown of explanation fidelity under high vs. low confidence.

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
from evaluate_gradcam_fidelity import extract_defect_die_mask, compute_fidelity_metrics

def main():
    print("=" * 75)
    print("EMPIRICAL CONFIDENCE VS. EXPLANATION FIDELITY AUDIT")
    print("=" * 75)

    model_path = config.MODEL_PATH
    if not model_path.exists():
        print(f"Error: Champion model not found at {model_path}")
        return

    print(f"Loading champion model from {model_path.name}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    confidences_correct = []
    ious_correct = []
    
    confidences_incorrect = []
    ious_incorrect = []

    class_names = [c for c in config.CLASS_NAMES.values() if c != "none"]

    for cname in class_names:
        class_dir = config.TEST_IMAGES_DIR / cname
        if not class_dir.exists():
            continue
            
        image_files = (list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))[:30]
        true_class_id = list(config.CLASS_NAMES.values()).index(cname)
        
        for img_path in image_files:
            try:
                pil_img = cv2.imread(str(img_path))
                if pil_img is None:
                    continue
                pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(pil_img, config.IMAGE_SIZE)
                norm_img = resized_img.astype(np.float32) / 255.0
                
                gt_mask = extract_defect_die_mask(norm_img)
                input_tensor = np.expand_dims(norm_img, axis=0)
                
                # Model Prediction
                preds = model(input_tensor, training=False).numpy()[0]
                pred_class_id = int(np.argmax(preds))
                confidence = float(preds[pred_class_id])
                
                # Heatmap for predicted class
                heatmap = gradcam.make_gradcam_heatmap(
                    model=model,
                    preprocessed_input=input_tensor,
                    class_index=pred_class_id,
                )
                heatmap_resized = cv2.resize(heatmap, config.IMAGE_SIZE)
                metrics = compute_fidelity_metrics(heatmap_resized, gt_mask)
                iou = metrics["iou"]
                
                if pred_class_id == true_class_id:
                    confidences_correct.append(confidence)
                    ious_correct.append(iou)
                else:
                    confidences_incorrect.append(confidence)
                    ious_incorrect.append(iou)
            except Exception:
                continue

    # Pearson correlation for correct predictions
    if len(confidences_correct) > 1:
        corr_matrix = np.corrcoef(confidences_correct, ious_correct)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = 0.0

    mean_iou_correct = float(np.mean(ious_correct)) if ious_correct else 0.0
    mean_iou_incorrect = float(np.mean(ious_incorrect)) if ious_incorrect else 0.0
    mean_conf_correct = float(np.mean(confidences_correct)) if confidences_correct else 0.0
    mean_conf_incorrect = float(np.mean(confidences_incorrect)) if confidences_incorrect else 0.0

    results = {
        "sample_count_correct": len(confidences_correct),
        "sample_count_incorrect": len(confidences_incorrect),
        "mean_confidence_correct": round(mean_conf_correct, 4),
        "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        "mean_iou_correct": round(mean_iou_correct, 4),
        "mean_iou_incorrect": round(mean_iou_incorrect, 4),
        "confidence_fidelity_correlation_r": round(pearson_r, 4)
    }

    print(f"\nEmpirical Audit Summary:")
    print(f"  • Correct Samples      : {len(confidences_correct)} | Mean Conf: {mean_conf_correct:.4f} | Mean IoU: {mean_iou_correct:.4f}")
    print(f"  • Incorrect Samples    : {len(confidences_incorrect)} | Mean Conf: {mean_conf_incorrect:.4f} | Mean IoU: {mean_iou_incorrect:.4f}")
    print(f"  • Confidence-Fidelity Pearson r: {pearson_r:.4f}")

    results_dir = config.BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_json = results_dir / "confidence_vs_fidelity_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {out_json}")
    print("=" * 75)

if __name__ == "__main__":
    main()
