# test_image_generator.py
# Genera una imagen de prueba válida en base64

import base64
import io
from PIL import Image
import numpy as np

# Crear una imagen de prueba 64x64 con un patrón simple
def create_test_image():
    # Crear un gradiente simple
    img_array = np.zeros((64, 64), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img_array[i, j] = (i + j) % 256
    
    img = Image.fromarray(img_array, mode='L')
    
    # Convertir a base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    b64_bytes = base64.b64encode(buffer.read())
    b64_str = b64_bytes.decode("utf-8")
    
    return f"data:image/png;base64,{b64_str}"

if __name__ == "__main__":
    test_image_b64 = create_test_image()
    print("Imagen de prueba generada:")
    print(test_image_b64[:100] + "...")
    
    # Crear un JSON de prueba completo
    test_request = {
        "image_base64": test_image_b64,
        "filter": {
            "type": "box_blur",
            "mask_size": 3
        },
        "cuda_config": {
            "block_dim": [8, 8],
            "grid_dim": [8, 8]
        }
    }
    
    import json
    print("\n" + "="*60)
    print("JSON completo para prueba:")
    print("="*60)
    print(json.dumps(test_request, indent=2)[:500] + "...")
    
    # Guardar en archivo
    with open("test_request.json", "w") as f:
        json.dump(test_request, f, indent=2)
    
    print("\n✅ Guardado en test_request.json")
