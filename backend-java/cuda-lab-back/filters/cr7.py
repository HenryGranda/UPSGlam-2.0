import numpy as np
import cv2
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cuda_kernels import _initialize_cuda, compile_cuda_kernel_to_ptx

import pycuda.driver as cuda


# ============================================================
#  KERNEL CUDA: ALPHA BLEND SOBRE REGIÓN (CARA)
# ============================================================
KERNEL_SOURCE = r"""
extern "C" {

__global__ void alpha_blend_face(
    uchar4* frame,
    const uchar4* mask,
    int fx, int fy,
    int width,
    int height,
    int mw,
    int mh
){
    int mx = blockIdx.x * blockDim.x + threadIdx.x;
    int my = blockIdx.y * blockDim.y + threadIdx.y;

    if (mx >= mw || my >= mh) return;

    int fx_pos = fx + mx;
    int fy_pos = fy + my;

    if (fx_pos < 0 || fy_pos < 0 || fx_pos >= width || fy_pos >= height)
        return;

    int frame_idx = fy_pos * width + fx_pos;
    int mask_idx  = my * mw + mx;

    uchar4 f = frame[frame_idx];
    uchar4 m = mask[mask_idx];

    float alpha = m.w / 255.0f;
    float inv   = 1.0f - alpha;

    uchar4 out;
    out.x = (unsigned char)(f.x * inv + m.x * alpha);
    out.y = (unsigned char)(f.y * inv + m.y * alpha);
    out.z = (unsigned char)(f.z * inv + m.z * alpha);
    out.w = 255;

    frame[frame_idx] = out;
}

} // extern "C"
"""


# ============================================================
#  CLASE PRINCIPAL DEL FILTRO
# ============================================================
class FaceMaskFilter:
    def __init__(self):
        # Initialize CUDA context
        _initialize_cuda()
        
        # Compile using nvcc directly to avoid auto-detection issues
        ptx_code = compile_cuda_kernel_to_ptx(KERNEL_SOURCE, arch="sm_89")
        self.module = cuda.module_from_buffer(ptx_code.encode())
        self.kernel = self.module.get_function("alpha_blend_face")

        # Clasificador de caras (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def ensure_rgba(img: np.ndarray) -> np.ndarray:
        """
        Asegura BGRA (4 canales). Si viene BGR, agrega alpha=255.
        """
        if img is None:
            raise ValueError("Imagen recibida es None")

        if img.ndim != 3:
            raise ValueError("Imagen debe tener 3 dimensiones (H, W, C)")

        if img.shape[2] == 4:
            return img

        if img.shape[2] == 3:
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
            return np.concatenate((img, alpha), axis=2)

        raise ValueError("Imagen debe tener 3 o 4 canales")

    def detect_faces(self, bgr_img: np.ndarray):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )
        return faces

    def apply(self, base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """
        Aplica overlay (BGRA) MÁS GRANDE que la cara detectada.
        Retorna BGRA.
        """
        SCALE_FACTOR = 1.6  # 👈 AJUSTA ESTO (1.4 – 2.0 suele ir bien)

        base_bgr = np.ascontiguousarray(base_bgr)
        base_bgra = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2BGRA)
        base_bgra = self.ensure_rgba(base_bgra)

        overlay_rgba = self.ensure_rgba(np.ascontiguousarray(overlay_rgba))

        h, w, _ = base_bgra.shape
        faces = self.detect_faces(base_bgr)

        if len(faces) == 0:
            return base_bgra

        # Subir frame UNA sola vez a GPU
        flat_frame = np.ascontiguousarray(base_bgra.reshape(-1))
        d_frame = cuda.mem_alloc(flat_frame.nbytes)
        cuda.memcpy_htod(d_frame, flat_frame)

        block = (16, 16, 1)

        for (x, y, fw, fh) in faces:
            # ==================================================
            # 1️⃣ Escalar overlay MÁS GRANDE que la cara
            # ==================================================
            new_w = int(fw * SCALE_FACTOR)
            new_h = int(fh * SCALE_FACTOR)

            resized_mask = cv2.resize(
                overlay_rgba,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            # ==================================================
            # 2️⃣ Recentrar overlay para tapar la cara
            # ==================================================
            new_x = int(x - (new_w - fw) / 2)
            new_y = int(y - (new_h - fh) / 2)

            mh, mw, _ = resized_mask.shape

            flat_mask = np.ascontiguousarray(resized_mask.reshape(-1))
            d_mask = cuda.mem_alloc(flat_mask.nbytes)
            cuda.memcpy_htod(d_mask, flat_mask)

            grid = (
                (mw + block[0] - 1) // block[0],
                (mh + block[1] - 1) // block[1]
            )

            self.kernel(
                d_frame,
                d_mask,
                np.int32(new_x),   # 👈 posición ajustada
                np.int32(new_y),
                np.int32(w),
                np.int32(h),
                np.int32(mw),
                np.int32(mh),
                block=block,
                grid=grid
            )

            d_mask.free()

        cuda.memcpy_dtoh(flat_frame, d_frame)
        d_frame.free()

        return flat_frame.reshape((h, w, 4))



# ============================================================
#  CAPTURA DE CÁMARA (SPACE captura)
# ============================================================
def capture_photo_from_camera(cam_index=0, window="CAM"):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Windows-friendly
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        raise RuntimeError("❌ No pude abrir la cámara. Prueba cam_index=1 o revisa permisos.")

    print("📷 Cámara abierta. Controles:")
    print("  - SPACE: tomar foto")
    print("  - Q o ESC: salir")

    frame = None
    while True:
        ok, img = cap.read()
        if not ok:
            continue

        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            frame = img.copy()
            break
        if key in (ord('q'), 27):  # q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


# ============================================================
#  MAIN
# ============================================================
def main():
    # ✅ TU RUTA DIRECTA (sin assets)
    OVERLAY_PATH = r"C:\Users\EleXc\Music\upsGLAM\UPSGlam-2.0\backend-java\cuda-lab-back\tests\face_mask.png"

    photo = capture_photo_from_camera(cam_index=0)
    if photo is None:
        print("👋 Saliste sin tomar foto.")
        return

    cv2.imwrite("captura.png", photo)

    overlay = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise RuntimeError(f"❌ Overlay no encontrado en: {OVERLAY_PATH}")

    filt = FaceMaskFilter()
    result_bgra = filt.apply(photo, overlay)

    # guardar resultado visible (BGR)
    result_bgr = cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2BGR)
    out_name = f"face_mask_{uuid.uuid4().hex}.png"
    cv2.imwrite(out_name, result_bgr)

    print("✅ Listo:")
    print("  - captura.png")
    print(f"  - {out_name}")


def apply_cr7_bytes(image_bytes: bytes) -> bytes:
    """
    Apply CR7 face mask filter to image bytes and return image bytes.
    
    Args:
        image_bytes: Input image as bytes (JPEG/PNG)
    
    Returns:
        bytes: JPEG image with CR7 face mask applied
    """
    import os
    
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image")
    
    # Load face mask overlay
    current_dir = os.path.dirname(os.path.abspath(__file__))
    overlay_path = os.path.join(current_dir, "assets", "face_mask.png")
    
    if not os.path.exists(overlay_path):
        raise FileNotFoundError(f"Face mask not found at: {overlay_path}")
    
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise RuntimeError(f"Could not load face mask from: {overlay_path}")
    
    # Apply filter
    filter_instance = FaceMaskFilter()
    result_bgra = filter_instance.apply(img_bgr, overlay)
    
    # Convert BGRA to BGR for JPEG encoding
    result_bgr = cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2BGR)
    
    # Encode to JPEG with high quality
    _, encoded = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return encoded.tobytes()


if __name__ == "__main__":
    main()