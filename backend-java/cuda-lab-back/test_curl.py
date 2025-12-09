"""
Test script to verify the /filters endpoint with binary image data.
This simulates what curl does when sending --data-binary.
"""
import requests
from pathlib import Path

def test_filter_endpoint():
    # Test with a small test image
    test_image_path = Path("husky.jpg")
    
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        print("Please create a test image or update the path")
        return
    
    # Read image bytes
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
    
    print(f"📷 Image size: {len(image_bytes)} bytes")
    
    # Test endpoint (adjust port as needed)
    url = "http://localhost:5000/filters/gaussian"
    
    print(f"🌐 Testing: {url}")
    
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
            # Save result
            output_path = "husky_gaussian_test.jpg"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Image saved to: {output_path}")
            
            # Verify it's a valid JPEG
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

if __name__ == "__main__":
    test_filter_endpoint()
