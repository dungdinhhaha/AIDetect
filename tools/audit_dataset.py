import tensorflow as tf
import numpy as np
from collections import defaultdict
import json
import os

def audit_dataset(tfrecord_path, dataset_name='train'):
    """Comprehensive dataset audit"""
    
    print(f"\n{'='*60}")
    print(f"AUDITING: {dataset_name}")
    print(f"{'='*60}")
    
    stats = {
        'total_images': 0,
        'total_boxes': 0,
        'class_distribution': defaultdict(int),
        'box_sizes': [],
        'images_per_num_boxes': defaultdict(int),
        'issues': {
            'empty_images': [],
            'tiny_boxes': [],
            'huge_boxes': [],
            'invalid_boxes': []
        }
    }
    
    try:
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        for idx, record in enumerate(dataset):
            # Parse
            features = {
                'img': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                'gtboxes_and_label': tf.io.FixedLenFeature([], tf.string),
            }
            parsed = tf.io.parse_single_example(record, features)
            
            # Decode boxes
            gtboxes_and_label = tf.io.decode_raw(parsed['gtboxes_and_label'], tf.int32)
            gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])
            
            boxes = gtboxes_and_label[:, :4].numpy()
            labels = gtboxes_and_label[:, 4].numpy()
            
            height = int(parsed['img_height'].numpy())
            width = int(parsed['img_width'].numpy())
            
            stats['total_images'] += 1
            num_boxes = len(boxes)
            stats['total_boxes'] += num_boxes
            stats['images_per_num_boxes'][num_boxes] += 1
            
            # Empty images
            if num_boxes == 0:
                stats['issues']['empty_images'].append(idx)
            
            # Check each box
            for i, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Class distribution
                stats['class_distribution'][int(label)] += 1
                
                # Box size
                stats['box_sizes'].append((w, h))
                
                # Tiny boxes
                if w < 10 or h < 10:
                    stats['issues']['tiny_boxes'].append((idx, i, box.tolist()))
                
                # Huge boxes
                if w > width * 0.8 or h > height * 0.8:
                    stats['issues']['huge_boxes'].append((idx, i, box.tolist()))
                
                # Invalid boxes
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    stats['issues']['invalid_boxes'].append((idx, i, box.tolist()))
            
            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} images...", end='\r')
        
        print(f"  Processed {stats['total_images']} images - DONE!       ")
        
    except Exception as e:
        print(f"❌ Error reading {tfrecord_path}: {e}")
        return None
    
    # Calculate metrics
    if stats['total_images'] > 0:
        avg_boxes_per_image = stats['total_boxes'] / stats['total_images']
    else:
        avg_boxes_per_image = 0
    
    # Class imbalance
    class_counts = list(stats['class_distribution'].values())
    if class_counts and min(class_counts) > 0:
        imbalance_ratio = max(class_counts) / min(class_counts)
    else:
        imbalance_ratio = float('inf')
    
    # Print report
    print(f"\n📊 BASIC STATS:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total boxes: {stats['total_boxes']}")
    print(f"  Avg boxes/image: {avg_boxes_per_image:.2f}")
    
    print(f"\n📦 CLASS DISTRIBUTION:")
    total = sum(stats['class_distribution'].values())
    for class_id in sorted(stats['class_distribution'].keys()):
        count = stats['class_distribution'][class_id]
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  Class {class_id:2d}: {count:5d} ({pct:5.2f}%) {bar}")
    
    print(f"\n⚖️ CLASS IMBALANCE:")
    print(f"  Ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 10:
        print(f"  🔴 CRITICAL IMBALANCE - Need urgent balancing!")
    elif imbalance_ratio > 5:
        print(f"  ⚠️ SEVERE IMBALANCE - Need balancing!")
    elif imbalance_ratio > 3:
        print(f"  ⚠️ MODERATE IMBALANCE - Consider balancing")
    else:
        print(f"  ✅ GOOD balance")
    
    print(f"\n⚠️ ISSUES FOUND:")
    total_issues = 0
    for issue_type, items in stats['issues'].items():
        count = len(items)
        total_issues += count
        if count > 0:
            print(f"  {issue_type}: {count}")
    
    if total_issues == 0:
        print(f"  ✅ No issues found!")
    
    # Box size distribution
    if stats['box_sizes']:
        box_sizes = np.array(stats['box_sizes'])
        print(f"\n📏 BOX SIZE STATS:")
        print(f"  Width:  Mean={box_sizes[:,0].mean():.1f}, Std={box_sizes[:,0].std():.1f}, Min={box_sizes[:,0].min():.0f}, Max={box_sizes[:,0].max():.0f}")
        print(f"  Height: Mean={box_sizes[:,1].mean():.1f}, Std={box_sizes[:,1].std():.1f}, Min={box_sizes[:,1].min():.0f}, Max={box_sizes[:,1].max():.0f}")
    
    # Save report
    report_file = f'dataset_audit_{dataset_name}.json'
    stats_json = {
        'dataset_name': dataset_name,
        'total_images': int(stats['total_images']),
        'total_boxes': int(stats['total_boxes']),
        'avg_boxes_per_image': float(avg_boxes_per_image),
        'class_distribution': {int(k): int(v) for k, v in stats['class_distribution'].items()},
        'imbalance_ratio': float(imbalance_ratio) if imbalance_ratio != float('inf') else None,
        'issues_count': {k: len(v) for k, v in stats['issues'].items()},
        'issues_details': {
            'empty_images': stats['issues']['empty_images'][:10],  # First 10 only
            'tiny_boxes': stats['issues']['tiny_boxes'][:10],
            'huge_boxes': stats['issues']['huge_boxes'][:10],
            'invalid_boxes': stats['issues']['invalid_boxes'][:10]
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    print(f"\n✅ Report saved to {report_file}")
    
    return stats


if __name__ == '__main__':
    # Check if TFRecord files exist
    train_path = 'tfdata/tct/train.tfrecord'
    test_path = 'tfdata/tct/test.tfrecord'
    
    print("\n" + "="*60)
    print("DATASET AUDIT TOOL")
    print("="*60)
    
    if os.path.exists(train_path):
        train_stats = audit_dataset(train_path, 'train')
    else:
        print(f"\n❌ Train file not found: {train_path}")
        train_stats = None
    
    if os.path.exists(test_path):
        test_stats = audit_dataset(test_path, 'test')
    else:
        print(f"\n❌ Test file not found: {test_path}")
        test_stats = None
    
    # Summary
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    
    if train_stats:
        print(f"\n✅ Train: {train_stats['total_images']} images, {train_stats['total_boxes']} boxes")
    if test_stats:
        print(f"✅ Test:  {test_stats['total_images']} images, {test_stats['total_boxes']} boxes")
    
    if train_stats and test_stats:
        total_images = train_stats['total_images'] + test_stats['total_images']
        total_boxes = train_stats['total_boxes'] + test_stats['total_boxes']
        print(f"\n📊 TOTAL: {total_images} images, {total_boxes} boxes")
        
        # Check split ratio
        train_ratio = train_stats['total_images'] / total_images * 100
        test_ratio = test_stats['total_images'] / total_images * 100
        print(f"\n📂 SPLIT RATIO:")
        print(f"  Train: {train_ratio:.1f}%")
        print(f"  Test:  {test_ratio:.1f}%")
        
        if 75 <= train_ratio <= 85:
            print(f"  ✅ Good split ratio!")
        else:
            print(f"  ⚠️ Consider 80/20 split")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if train_stats:
        # Check class imbalance
        class_counts = list(train_stats['class_distribution'].values())
        if class_counts and min(class_counts) > 0:
            imbalance = max(class_counts) / min(class_counts)
            
            if imbalance > 5:
                print("\n🔴 HIGH PRIORITY: Fix class imbalance")
                print(f"   - Current ratio: {imbalance:.1f}x")
                print(f"   - Target: < 3x")
                print(f"   - Action: Oversample minority classes or use Focal Loss")
        
        # Check data issues
        total_issues = sum(len(v) for v in train_stats['issues'].values())
        if total_issues > train_stats['total_images'] * 0.05:  # >5% issues
            print("\n⚠️ MEDIUM PRIORITY: Clean data quality issues")
            print(f"   - Total issues: {total_issues}")
            print(f"   - Percentage: {total_issues/train_stats['total_images']*100:.1f}%")
            print(f"   - Action: Review and fix annotations")
        
        # Check dataset size
        if train_stats['total_images'] < 5000:
            print("\n⚠️ MEDIUM PRIORITY: Dataset size is small")
            print(f"   - Current: {train_stats['total_images']} images")
            print(f"   - Recommended: 10,000+ images")
            print(f"   - Action: Collect more data OR use advanced augmentation")
    
    print("\n✅ Audit completed!")
