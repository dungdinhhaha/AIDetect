"""Check TFRecord image format without TensorFlow"""
import struct

tfrecord_path = r'd:\ComparisonDetector\tfdata\tct\train.tfrecord'

# Read first record manually
with open(tfrecord_path, 'rb') as f:
    # Read record length
    length_bytes = f.read(8)
    length = struct.unpack('<Q', length_bytes)[0]
    
    # Read record data
    record_bytes = f.read(length)
    
    # Find image data pattern
    # Look for 'img' key followed by bytes
    img_start = record_bytes.find(b'img')
    if img_start > 0:
        # Skip ahead to find actual image data
        search_pos = img_start + 20
        chunk = record_bytes[search_pos:search_pos+100]
        
        print("First 100 bytes after 'img' key:")
        print(chunk[:100].hex())
        print()
        
        # Check common image formats
        if b'\xff\xd8\xff' in chunk:
            print("✓ Found JPEG signature (FF D8 FF)")
        elif b'\x89PNG' in chunk:
            print("✓ Found PNG signature (89 50 4E 47)")
        elif b'BM' in chunk[:10]:
            print("✓ Found BMP signature")
        else:
            print("⚠ No standard image format found - may be raw pixel data")
            print("Checking if it's raw bytes...")
            
            # Look for dimensions
            if b'img_height' in record_bytes:
                print("✓ Found img_height field")
            if b'img_width' in record_bytes:
                print("✓ Found img_width field")
