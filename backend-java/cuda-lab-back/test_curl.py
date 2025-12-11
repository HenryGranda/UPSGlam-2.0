"""
Test script to verify the /filters endpoint with binary image data.
This simulates what curl does when sending --data-binary.

Tests all available filters including the new boomerang and ups_logo.
"""
import requests
from pathlib import Path

def test_filter_endpoint(filter_name="gaussian", test_image="husky.jpg"):
    """
    Test a specific filter endpoint.
    
    Args:
        filter_name: Name of the filter to test
        test_image: Path to test image
    """
    # Test with a small test image
    test_image_path = Path(test_image)
    
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        print("Please create a test image or update the path")
        return
    
    # Read image bytes
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
    
    print(f"\n{'='*60}")
    print(f"🔬 Testing filter: {filter_name}")
    print(f"{'='*60}")
    print(f"📷 Image size: {len(image_bytes)} bytes")
    
    # Test endpoint (adjust port as needed)
    url = f"http://localhost:5000/filters/{filter_name}"
    
    print(f"🌐 URL: {url}")
    
    try:
        response = requests.post(
            url,
            data=image_bytes,  # Send raw bytes
            headers={"Content-Type": "image/jpeg"}
        )
        
        print(f"📊 Status: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        print(f"📏 Response size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            # Determine output file extension
            content_type = response.headers.get('Content-Type', '')
            ext = '.gif' if 'gif' in content_type else '.jpg'
            
            # Save result
            base_name = test_image_path.stem
            output_path = f"{base_name}_{filter_name}_test{ext}"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Image saved to: {output_path}")
            
            # Verify it's a valid image
            from PIL import Image
            import io
            try:
                img = Image.open(io.BytesIO(response.content))
                print(f"✅ Valid image: {img.format} {img.size} {img.mode}")
            except Exception as e:
                print(f"❌ Invalid image: {e}")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the server running?")
        print("   Try: python -m uvicorn app:app --host 0.0.0.0 --port 5000")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_all_filters(test_image="husky.jpg"):
    """Test all available filters"""
    filters = [
        "gaussian",
        "box_blur",
        "prewitt",
        "laplacian",
        "ups_logo",
        "ups_color",
        "boomerang"
    ]
    
    print("\n" + "="*60)
    print("🚀 TESTING ALL FILTERS")
    print("="*60)
    
    for filter_name in filters:
        test_filter_endpoint(filter_name, test_image)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific filter
        filter_name = sys.argv[1]
        test_image = sys.argv[2] if len(sys.argv) > 2 else "husky.jpg"
        test_filter_endpoint(filter_name, test_image)
    else:
        # Test all filters
        test_all_filters()
        
    print("\n💡 Usage:")
    print("  python test_curl.py                    # Test all filters")
    print("  python test_curl.py <filter>           # Test specific filter")
    print("  python test_curl.py <filter> <image>   # Test filter with custom image")
    print("\n📝 Available filters:")
    print("  - gaussian, box_blur, prewitt, laplacian")
    print("  - ups_logo (with aura effects)")
    print("  - ups_color")
    print("  - boomerang (animated GIF)")
