"""
Script đơn giản để test model đã train
Sử dụng: python test_model_simple.py
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from data.loader_tf2 import build_dataset
from configs.config_v2 import IMAGE_SIZE, NUM_CLASSES
import json

def test_model_on_dataset():
    """Test model trên test dataset"""
    
    # 1. Load model
    print("=" * 60)
    print("🔍 TESTING MODEL")
    print("=" * 60)
    
    model_path = input("Nhập path đến model (Enter = best_model.h5): ").strip()
    if not model_path:
        model_path = "trained_models/best_model.h5"
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        print("\n💡 Hướng dẫn:")
        print("   1. Download model từ Google Drive")
        print("   2. Copy vào folder: trained_models/")
        print("   3. Run lại script này")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # 2. Load test dataset
    print(f"\n📊 Loading test dataset...")
    test_paths = ['tfdata/tct/test.tfrecord']
    
    if not Path(test_paths[0]).exists():
        print(f"❌ Không tìm thấy test data: {test_paths[0]}")
        print("\n💡 Đảm bảo bạn có file tfdata/tct/test.tfrecord")
        return
    
    test_ds = build_dataset(
        test_paths, 
        image_size=IMAGE_SIZE, 
        batch_size=4,
        is_training=False
    )
    
    # Map to extract labels
    def extract_label(img, tgt):
        # Get first label for each image (simplified classification)
        return img, tgt['labels'][:, 0]
    
    test_ds = test_ds.map(extract_label).take(50)  # Take 50 batches = 200 images
    
    # 3. Evaluate
    print(f"\n⚙️  Evaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    
    test_loss = results[0]
    test_accuracy = results[1]
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 60)
    
    # 4. Save results
    results_dict = {
        'model_path': str(model_path),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'num_batches': 50,
        'batch_size': 4,
        'total_images_tested': 200
    }
    
    output_path = Path('test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # 5. Sample predictions
    print(f"\n🔮 Running sample predictions...")
    for images, labels in test_ds.take(1):
        predictions = model.predict(images, verbose=0)
        
        print("\nSample predictions (first 4 images):")
        print("-" * 60)
        for i in range(min(4, len(images))):
            pred_class = np.argmax(predictions[i])
            pred_conf = predictions[i][pred_class]
            true_class = int(labels[i].numpy())
            
            status = "✅" if pred_class == true_class else "❌"
            print(f"{status} Image {i+1}:")
            print(f"   True class:      {true_class}")
            print(f"   Predicted class: {pred_class}")
            print(f"   Confidence:      {pred_conf:.2%}")
            print()
    
    print("=" * 60)
    print("✅ Testing completed!")
    print("=" * 60)


def test_single_image():
    """Test model trên 1 ảnh đơn lẻ"""
    
    print("=" * 60)
    print("🖼️  TESTING ON SINGLE IMAGE")
    print("=" * 60)
    
    # Load model
    model_path = input("Nhập path đến model (Enter = best_model.h5): ").strip()
    if not model_path:
        model_path = "trained_models/best_model.h5"
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded!")
    
    # Load image
    from PIL import Image
    
    image_path = input("\nNhập path đến ảnh test: ").strip()
    if not Path(image_path).exists():
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return
    
    print(f"\n🖼️  Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    
    print(f"   Original size: {img.size}")
    print(f"   Resized to: {IMAGE_SIZE}")
    print(f"   Array shape: {img_array.shape}")
    
    # Predict
    print(f"\n🔮 Predicting...")
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULT")
    print("=" * 60)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence:      {confidence:.2%}")
    print("=" * 60)
    
    # Show top-3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    print("\nTop 3 predictions:")
    for idx in top3_indices:
        print(f"  Class {idx}: {predictions[0][idx]:.2%}")
    
    print("\n✅ Done!")


def main():
    print("\n" + "=" * 60)
    print("🧪 MODEL TESTING TOOL")
    print("=" * 60)
    print("\nChọn chế độ test:")
    print("  1. Test trên test dataset (evaluation)")
    print("  2. Test trên 1 ảnh đơn lẻ")
    print("  0. Exit")
    
    choice = input("\nNhập lựa chọn (1/2/0): ").strip()
    
    if choice == '1':
        test_model_on_dataset()
    elif choice == '2':
        test_single_image()
    elif choice == '0':
        print("Bye! 👋")
    else:
        print("❌ Lựa chọn không hợp lệ!")


if __name__ == '__main__':
    main()
