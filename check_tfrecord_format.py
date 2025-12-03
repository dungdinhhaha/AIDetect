"""Script to inspect TFRecord format"""
import tensorflow as tf
import sys

def inspect_tfrecord(tfrecord_path):
    """Print the feature keys and types in a TFRecord file"""
    print(f"\n📋 Inspecting: {tfrecord_path}\n")
    
    # Read first record
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("Feature keys found:")
        print("-" * 50)
        
        for key, feature in example.features.feature.items():
            # Determine feature type
            if feature.bytes_list.value:
                ftype = "bytes_list"
                sample = f"(length: {len(feature.bytes_list.value[0])} bytes)"
            elif feature.float_list.value:
                ftype = "float_list"
                sample = f"(values: {list(feature.float_list.value[:5])}...)"
            elif feature.int64_list.value:
                ftype = "int64_list"
                sample = f"(values: {list(feature.int64_list.value[:5])}...)"
            else:
                ftype = "unknown"
                sample = ""
            
            print(f"  '{key}': {ftype} {sample}")
        
        print("-" * 50)
        print("✓ Inspection complete")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        tfrecord_path = sys.argv[1]
    else:
        tfrecord_path = '/content/drive/MyDrive/content/data/tct/train.tfrecord'
    
    inspect_tfrecord(tfrecord_path)
