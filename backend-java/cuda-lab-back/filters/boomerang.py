# filters/boomerang.py
"""
Boomerang Filter - Creative Filter with Multiple Ball Trails

Applies effect showing multiple balls at different positions (like a boomerang effect).
Returns a single static image with crisp, textured balls showing motion trail.
"""

import numpy as np
import cv2
from pathlib import Path

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
extern "C" {

__global__ void draw_texture_balls(
    const unsigned char* bg,
    unsigned char* out,
    int width,
    int height,
    int channels,
    const float* positions,
    int num_balls,
    float radius,
    const unsigned char* tex,
    int tw,
    int th
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    float r = bg[idx + 2];
    float g = bg[idx + 1];
    float b = bg[idx + 0];

    float radius2 = radius * radius;

    // Draw all balls (checking from back to front for proper layering)
    for (int i = 0; i < num_balls; i++)
    {
        float cx = positions[2*i + 0];
        float cy = positions[2*i + 1];

        float dx = x - cx;
        float dy = y - cy;
        float dist2 = dx*dx + dy*dy;

        if (dist2 < radius2)
        {
            // Calculate texture coordinates (bilinear interpolation for smoothness)
            float u = (dx + radius) / (2.0f * radius);
            float v = (dy + radius) / (2.0f * radius);
            
            // Clamp to texture bounds
            u = fminf(fmaxf(u, 0.0f), 1.0f);
            v = fminf(fmaxf(v, 0.0f), 1.0f);
            
            // Map to texture coordinates
            float tx_f = u * (tw - 1);
            float ty_f = v * (th - 1);
            
            int tx = (int)tx_f;
            int ty = (int)ty_f;
            
            // Ensure bounds
            tx = min(max(tx, 0), tw - 1);
            ty = min(max(ty, 0), th - 1);

            int tidx = (ty * tw + tx) * 4;

            unsigned char tr = tex[tidx + 2];
            unsigned char tg = tex[tidx + 1];
            unsigned char tb = tex[tidx + 0];
            unsigned char ta = tex[tidx + 3];

            float alpha = ta / 255.0f;

            // Alpha blending
            r = (1.0f - alpha) * r + alpha * tr;
            g = (1.0f - alpha) * g + alpha * tg;
            b = (1.0f - alpha) * b + alpha * tb;
        }
    }

    out[idx + 2] = (unsigned char)(r);
    out[idx + 1] = (unsigned char)(g);
    out[idx + 0] = (unsigned char)(b);
}

} // extern
"""


class BoomerangFilter:
    """Boomerang filter - shows multiple balls at different positions creating a trail effect"""
    
    def __init__(self):
        self.module = SourceModule(KERNEL_CODE)
        self.draw_texture_kernel = self.module.get_function("draw_texture_balls")
    
    def generate_ball_positions(self, num_positions, width, height, radius):
        """
        Generate multiple ball positions to create trail effect.
        Positions follow a curved path like a boomerang.
        """
        positions = []
        
        # Create a boomerang-like curved path
        center_x = width / 2
        center_y = height / 2
        
        # Arc parameters for boomerang effect
        for i in range(num_positions):
            # Create an arc/curve pattern
            t = i / (num_positions - 1) * np.pi * 1.5  # 270 degree arc
            
            # Parametric curve for boomerang shape
            offset_x = np.cos(t) * (width * 0.3)
            offset_y = np.sin(t) * (height * 0.3) - (t / np.pi) * height * 0.1
            
            x = center_x + offset_x
            y = center_y + offset_y
            
            # Ensure within bounds
            x = max(radius, min(width - radius, x))
            y = max(radius, min(height - radius, y))
            
            positions.extend([x, y])
        
        return np.array(positions, dtype=np.float32)
    
    def apply(self, img_bgr, num_balls=8):
        """
        Apply boomerang effect to BGR image - static image with ball trail
        
        Args:
            img_bgr: Input image in BGR format
            num_balls: Number of balls to show in the trail (default 8)
        
        Returns:
            numpy.ndarray: Output image with ball trail effect
        """
        h, w, ch = img_bgr.shape
        
        # Calculate ball radius (larger for better visibility)
        radius = int(0.08 * min(w, h))
        
        # Load texture with high quality
        assets_dir = Path(__file__).parent / "assets"
        tex_path = assets_dir / "sonrisa.png"
        
        if not tex_path.exists():
            raise RuntimeError(f"❌ Texture not found: {tex_path}")
        
        tex = cv2.imread(str(tex_path), cv2.IMREAD_UNCHANGED)
        if tex is None:
            raise RuntimeError(f"❌ Could not load texture: {tex_path}")
        
        # Resize texture with high-quality interpolation for crisp result
        target_size = int(2 * radius)
        tex = cv2.resize(tex, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        tex = tex.astype(np.uint8)
        
        # Ensure texture has alpha channel
        if tex.shape[2] == 3:
            alpha = np.full((tex.shape[0], tex.shape[1], 1), 255, dtype=np.uint8)
            tex = np.concatenate((tex, alpha), axis=2)

        # Generate ball positions
        positions = self.generate_ball_positions(num_balls, w, h, radius)

        # Allocate GPU memory
        d_bg = cuda.mem_alloc(img_bgr.nbytes)
        d_out = cuda.mem_alloc(img_bgr.nbytes)
        d_tex = cuda.mem_alloc(tex.nbytes)
        d_pos = cuda.mem_alloc(positions.nbytes)

        # Copy data to GPU
        cuda.memcpy_htod(d_bg, img_bgr)
        cuda.memcpy_htod(d_tex, tex)
        cuda.memcpy_htod(d_pos, positions)

        # Set up CUDA grid
        block2D = (16, 16, 1)
        grid2D = ((w + 15) // 16, (h + 15) // 16, 1)

        # Execute kernel
        self.draw_texture_kernel(
            d_bg, d_out,
            np.int32(w),
            np.int32(h),
            np.int32(ch),
            d_pos,
            np.int32(num_balls),
            np.float32(radius),
            d_tex,
            np.int32(tex.shape[1]),
            np.int32(tex.shape[0]),
            block=block2D,
            grid=grid2D
        )

        # Copy result back
        out = np.empty_like(img_bgr)
        cuda.memcpy_dtoh(out, d_out)

        # Cleanup
        d_bg.free()
        d_out.free()
        d_tex.free()
        d_pos.free()

        return out


def apply_boomerang_bytes(image_bytes: bytes, num_balls: int = 8) -> bytes:
    """
    Apply boomerang filter to image bytes and return image bytes.
    
    Args:
        image_bytes: Input image as bytes (JPEG/PNG)
        num_balls: Number of balls in the trail (default 8)
    
    Returns:
        bytes: JPEG image with boomerang trail effect
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image")
    
    # Apply filter
    filter_instance = BoomerangFilter()
    result = filter_instance.apply(img_bgr, num_balls=num_balls)
    
    # Encode to JPEG with high quality
    _, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return encoded.tobytes()
