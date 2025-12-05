"""
Script test model nâng cao với visualization và metrics chi tiết
Sử dụng: python test_model_advanced.py
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from data.loader_tf2 import build_dataset
from configs.config_v2 import IMAGE_SIZE, NUM_CLASSES
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def test_with_visualization():
    """Test model với visualization và metrics chi tiết"""
    
    print("=" * 60)
    print("🔬 ADVANCED MODEL TESTING")
    print("=" * 60)
    
    # 1. Load model
    model_path = input("Nhập path đến model (Enter = best_model.h5): ").strip()
    if not model_path:
        model_path = "trained_models/best_model.h5"
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded!")
    
    # 2. Load test dataset
    print(f"\n📊 Loading test dataset...")
    test_paths = ['tfdata/tct/test.tfrecord']
    
    if not Path(test_paths[0]).exists():
        print(f"❌ Không tìm thấy test data")
        return
    
    test_ds = build_dataset(
        test_paths, 
        image_size=IMAGE_SIZE, 
        batch_size=8,
        is_training=False
    )
    
    def extract_label(img, tgt):
        return img, tgt['labels'][:, 0]
    
    test_ds = test_ds.map(extract_label).take(100)  # 800 images
    
    # 3. Collect predictions and ground truth
    print(f"\n⚙️  Running predictions...")
    all_predictions = []
    all_true_labels = []
    all_images = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_predictions.extend(np.argmax(preds, axis=1))
        all_true_labels.extend(labels.numpy().astype(int))
        all_images.extend(images.numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_images = np.array(all_images)
    
    print(f"✅ Processed {len(all_predictions)} images")
    
    # 4. Calculate metrics
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = np.mean(all_predictions == all_true_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print("\nPer-class metrics:")
    print(classification_report(
        all_true_labels, 
        all_predictions,
        target_names=[f"Class {i}" for i in range(NUM_CLASSES)],
        zero_division=0
    ))
    
    # 5. Confusion Matrix
    print("\n📈 Generating confusion matrix...")
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f"C{i}" for i in range(NUM_CLASSES)],
                yticklabels=[f"C{i}" for i in range(NUM_CLASSES)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = Path('test_results_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"✅ Saved to: {cm_path}")
    plt.close()
    
    # 6. Visualize sample predictions
    print("\n🖼️  Generating sample predictions visualization...")
    
    # Get correct and incorrect predictions
    correct_mask = all_predictions == all_true_labels
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # Plot correct predictions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('✅ Correct Predictions (Sample)', fontsize=16, color='green')
    
    for i, ax in enumerate(axes.flat):
        if i < len(correct_indices):
            idx = correct_indices[i]
            ax.imshow(all_images[idx])
            ax.set_title(f"True: {all_true_labels[idx]}, Pred: {all_predictions[idx]}", 
                        color='green')
            ax.axis('off')
    
    plt.tight_layout()
    correct_path = Path('test_results_correct_predictions.png')
    plt.savefig(correct_path, dpi=150)
    print(f"✅ Correct predictions saved to: {correct_path}")
    plt.close()
    
    # Plot incorrect predictions
    if len(incorrect_indices) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('❌ Incorrect Predictions (Sample)', fontsize=16, color='red')
        
        for i, ax in enumerate(axes.flat):
            if i < len(incorrect_indices):
                idx = incorrect_indices[i]
                ax.imshow(all_images[idx])
                ax.set_title(f"True: {all_true_labels[idx]}, Pred: {all_predictions[idx]}", 
                            color='red')
                ax.axis('off')
        
        plt.tight_layout()
        incorrect_path = Path('test_results_incorrect_predictions.png')
        plt.savefig(incorrect_path, dpi=150)
        print(f"✅ Incorrect predictions saved to: {incorrect_path}")
        plt.close()
    
    # 7. Class distribution
    print("\n📊 Analyzing class distribution...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # True labels distribution
    unique_true, counts_true = np.unique(all_true_labels, return_counts=True)
    ax1.bar(unique_true, counts_true, color='skyblue')
    ax1.set_title('True Label Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Predicted labels distribution
    unique_pred, counts_pred = np.unique(all_predictions, return_counts=True)
    ax2.bar(unique_pred, counts_pred, color='salmon')
    ax2.set_title('Predicted Label Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    dist_path = Path('test_results_class_distribution.png')
    plt.savefig(dist_path, dpi=150)
    print(f"✅ Class distribution saved to: {dist_path}")
    plt.close()
    
    # 8. Save detailed results
    results = {
        'model_path': str(model_path),
        'total_images': len(all_predictions),
        'overall_accuracy': float(accuracy),
        'num_correct': int(np.sum(correct_mask)),
        'num_incorrect': int(np.sum(incorrect_mask)),
        'per_class_accuracy': {},
        'confusion_matrix': cm.tolist()
    }
    
    # Per-class accuracy
    for i in range(NUM_CLASSES):
        mask = all_true_labels == i
        if np.sum(mask) > 0:
            class_acc = np.mean(all_predictions[mask] == all_true_labels[mask])
            results['per_class_accuracy'][f'class_{i}'] = {
                'accuracy': float(class_acc),
                'total_samples': int(np.sum(mask)),
                'correct': int(np.sum(all_predictions[mask] == all_true_labels[mask]))
            }
    
    results_path = Path('test_results_detailed.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {results_path}")
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETED!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  📄 {results_path}")
    print(f"  📊 {cm_path}")
    print(f"  🖼️  {correct_path}")
    if len(incorrect_indices) > 0:
        print(f"  🖼️  {incorrect_path}")
    print(f"  📈 {dist_path}")
    print("=" * 60)


if __name__ == '__main__':
    test_with_visualization()
