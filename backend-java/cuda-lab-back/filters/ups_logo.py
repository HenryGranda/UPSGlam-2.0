# filters/ups_logo.py
"""
UPS Logo Filter - Creative Filter with Aura Effects

Applies UPS logo overlay with dynamic aura, particles, and color effects using PyCUDA.
This is a high-quality creative filter incorporating UPS branding.
"""

import numpy as np
import cv2
import uuid
from pathlib import Path

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_SOURCE = r"""
extern "C" {

// -------------------- helpers --------------------
__device__ float hash2D(int x, int y, float time) {
    unsigned int seed = (unsigned int)(x * 374761393u + y * 668265263u);
    seed = (seed ^ (seed >> 13)) * 1274126177u;
    seed = (seed ^ (seed >> 7))  * 1597334677u;
    seed ^= (unsigned int)(time * 4096.0f);
    return (seed & 0x00FFFFFF) / 16777216.0f;
}

__device__ float luminance(const uchar4 &p) {
    return (0.2126f*p.x + 0.7152f*p.y + 0.0722f*p.z) / 255.0f;
}

__device__ float3 f3_add(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 f3_mul(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}


// ============================================================
//   KERNEL PRINCIPAL UPS LOGO AURA
// ============================================================
__global__ void ups_logo_overlay_aura(
    const uchar4 *input,
    uchar4 *output,
    const uchar4 *overlay,
    int width,
    int height,
    int overlayWidth,
    int overlayHeight,
    float time,
    float threshold,
    float waveAmplitude,
    float waveFrequency,
    float glowStrength,
    float particleDensity,
    float auraBlend,
    float haloStrength,
    float haloFalloff,
    float overlayPosX,
    float overlayPosY,
    float overlayTargetWidth,
    float overlayTargetHeight,
    float overlayOscillationPx,
    float overlayOscillationSpeed,
    float overlayTintStrength
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y*width + x;

    uchar4 px = input[idx];
    float lum = luminance(px);

    float mask = lum > threshold ? 1.0f : 0.0f;

    float neigh = 0.0f;
    for(int oy=-1; oy<=1; oy++)
    for(int ox=-1; ox<=1; ox++){
        int nx = min(max(x+ox,0), width-1);
        int ny = min(max(y+oy,0), height-1);
        neigh += luminance(input[ny*width + nx]);
    }
    neigh /= 9.0f;

    float glowVal = fabsf(lum - neigh) * glowStrength;
    float auraMask = fminf(1.0f, mask + glowVal);

    float nx = (float)x/(width-1);
    float ny = (float)y/(height-1);

    float wave = __sinf((nx+ny)*waveFrequency + time) * waveAmplitude;

    int sx = min(max(int((nx+wave)*(width-1)),0), width-1);
    int sy = min(max(int((ny+wave)*(height-1)),0), height-1);

    uchar4 wpx = input[sy*width + sx];

    float3 baseC = make_float3(px.x/255.0f, px.y/255.0f, px.z/255.0f);
    float3 warpC = make_float3(wpx.x/255.0f, wpx.y/255.0f, wpx.z/255.0f);

    const float3 brown = make_float3(0.227f,0.173f,0.102f);
    const float3 gold  = make_float3(0.949f,0.663f,0.0f);

    float pal = 0.5f*(sinf(time + nx*10 - ny*10)+1.0f);

    float3 auraC = f3_add(f3_mul(brown,1.0f-pal), f3_mul(gold,pal));

    float pnoise = hash2D(x,y,time)*particleDensity;
    if(pnoise - floorf(pnoise) > 0.97f){
        float3 spark = make_float3(1.0f,0.95f,0.7f);
        auraC = f3_add(f3_mul(auraC,0.5f), f3_mul(spark,0.5f));
    }

    float aA = fminf(fmaxf(auraMask*auraBlend,0.0f),1.0f);

    float3 final = f3_add(f3_mul(warpC,(1.0f-aA)), f3_mul(auraC,aA));

    float dynTop = overlayPosY + sinf(time*overlayOscillationSpeed)*overlayOscillationPx;
    float dynBottom = dynTop + overlayTargetHeight;
    float dynLeft = overlayPosX;
    float dynRight = overlayPosX + overlayTargetWidth;

    float dxh = 0.0f;
    if (x < dynLeft) dxh = dynLeft-x;
    else if(x > dynRight) dxh = x-dynRight;

    float dyh = 0.0f;
    if (y < dynTop) dyh = dynTop-y;
    else if(y > dynBottom) dyh = y-dynBottom;

    float dist = sqrtf(dxh*dxh + dyh*dyh);
    float halo = expf(-dist*haloFalloff)*haloStrength;

    final = f3_add(final, f3_mul(gold,halo));

    if (x>=dynLeft && x<=dynRight && y>=dynTop && y<=dynBottom){
        float u = (x-dynLeft)/overlayTargetWidth;
        float v = (y-dynTop) /overlayTargetHeight;

        u = fminf(fmaxf(u,0.0f),1.0f);
        v = fminf(fmaxf(v,0.0f),1.0f);

        int ox = int(u*(overlayWidth-1));
        int oy = int(v*(overlayHeight-1));

        uchar4 op = overlay[oy*overlayWidth + ox];

        float oa = op.w/255.0f;
        float3 oc = make_float3(op.x/255.0f, op.y/255.0f, op.z/255.0f);

        float tint = 0.5f*(sinf(time*0.7f + u*8 - v*6)+1.0f);
        float3 tinted = f3_add(
            f3_mul(oc,1.0f-overlayTintStrength),
            f3_mul(f3_add(f3_mul(brown,1.0f-tint),f3_mul(gold,tint)), overlayTintStrength)
        );

        final = f3_add(
            f3_mul(final, (1.0f-oa)),
            f3_mul(tinted, oa)
        );
    }

    uchar4 outp;
    outp.x = (unsigned char)(fminf(1.0f,fmaxf(0.0f,final.x))*255);
    outp.y = (unsigned char)(fminf(1.0f,fmaxf(0.0f,final.y))*255);
    outp.z = (unsigned char)(fminf(1.0f,fmaxf(0.0f,final.z))*255);
    outp.w = 255;
    output[idx] = outp;
}

} // extern
"""


class UPSLogoAuraFilter:
    """UPS Logo filter with aura effects"""

    def __init__(self):
        self.module = SourceModule(KERNEL_SOURCE, options=["-use_fast_math"])
        self.kernel = self.module.get_function("ups_logo_overlay_aura")

    @staticmethod
    def ensure_rgba(img):
        """Ensure image has 4 channels (RGBA)"""
        if img.shape[2] == 4:
            return img
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate((img, alpha), axis=2)

    def apply(self, base_img, overlay_img, aura_params=None, overlay_params=None):
        """
        Apply UPS Logo filter with aura effects
        
        Args:
            base_img: Base image (BGR)
            overlay_img: Overlay logo image (BGRA)
            aura_params: Aura effect parameters
            overlay_params: Overlay positioning parameters
        
        Returns:
            Filtered image (RGBA)
        """
        aura_params = aura_params or {}
        overlay_params = overlay_params or {}

        base = self.ensure_rgba(base_img)
        overlay = self.ensure_rgba(overlay_img)

        h, w, _ = base.shape

        target_w = overlay_params.get("target_width", overlay.shape[1])
        target_h = overlay_params.get("target_height", overlay.shape[0])

        left = float(overlay_params.get("left", (w - target_w) // 2))
        top  = float(overlay_params.get("top",  h // 4))

        overlay_resized = cv2.resize(overlay, (target_w, target_h))

        flat_in  = np.ascontiguousarray(base.reshape(-1))
        flat_ov  = np.ascontiguousarray(overlay_resized.reshape(-1))
        flat_out = np.empty_like(flat_in)

        d_in  = cuda.mem_alloc(flat_in.nbytes)
        d_ov  = cuda.mem_alloc(flat_ov.nbytes)
        d_out = cuda.mem_alloc(flat_out.nbytes)

        cuda.memcpy_htod(d_in, flat_in)
        cuda.memcpy_htod(d_ov, flat_ov)

        block = (32,16,1)
        grid = ((w+31)//32, (h+15)//16, 1)

        self.kernel(
            d_in, d_out, d_ov,
            np.int32(w), np.int32(h),
            np.int32(target_w), np.int32(target_h),
            np.float32(aura_params.get("time",1.0)),
            np.float32(aura_params.get("threshold",0.35)),
            np.float32(aura_params.get("wave_amplitude",0.02)),
            np.float32(aura_params.get("wave_frequency",14.0)),
            np.float32(aura_params.get("glow_strength",2.5)),
            np.float32(aura_params.get("particle_density",5.0)),
            np.float32(aura_params.get("aura_blend",0.65)),
            np.float32(aura_params.get("halo_strength",0.45)),
            np.float32(aura_params.get("halo_falloff",0.06)),
            np.float32(left), np.float32(top),
            np.float32(target_w), np.float32(target_h),
            np.float32(overlay_params.get("oscillation_px",0.0)),
            np.float32(overlay_params.get("oscillation_speed",0.0)),
            np.float32(overlay_params.get("tint_strength",0.3)),
            block=block, grid=grid
        )

        cuda.memcpy_dtoh(flat_out, d_out)

        d_in.free()
        d_ov.free()
        d_out.free()

        return flat_out.reshape((h,w,4))


def apply_ups_logo_bytes(image_bytes: bytes) -> bytes:
    """
    Apply UPS Logo filter to image bytes and return filtered image bytes.
    
    IMPORTANTE: La imagen de Don Bosco es el FONDO (grande), 
    y la imagen del usuario es el OVERLAY (pequeño superpuesto).
    
    Args:
        image_bytes: Input image as bytes (JPEG/PNG) - será el overlay pequeño
    
    Returns:
        bytes: Filtered image as JPEG bytes with Don Bosco background
    """
    # Decode user's image from bytes - this will be the SMALL overlay
    nparr = np.frombuffer(image_bytes, np.uint8)
    user_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if user_img is None:
        raise ValueError("Could not decode image")
    
    # Load Don Bosco image - this will be the LARGE background
    assets_dir = Path(__file__).parent / "assets"
    donbosco_path = assets_dir / "filtro_don_bosco.png"
    donbosco_img = cv2.imread(str(donbosco_path), cv2.IMREAD_UNCHANGED)

    if donbosco_img is None:
        raise RuntimeError(f"Don Bosco background not found: {donbosco_path}")

    # Apply filter: Don Bosco as BASE (background), user image as OVERLAY (small on top)
    filt = UPSLogoAuraFilter()

    # Get Don Bosco dimensions (this is the output size)
    h_bg, w_bg = donbosco_img.shape[0], donbosco_img.shape[1]
    
    # User image will be small overlay (about 1/3 of background)
    overlay_w = w_bg // 3
    overlay_h = h_bg // 3

    # ============================================================
    # AJUSTAR POSICIÓN DE LA FOTO DEL USUARIO (overlay pequeño)
    # ============================================================
    # "left": Posición horizontal (0.0=izquierda, 1.0=derecha)
    #         Ejemplo: 0.60 = 60% hacia la derecha
    #
    # "top": Posición vertical (0.0=arriba, 1.0=abajo)
    #        Ejemplo: 0.65 = 65% hacia abajo (más abajo que antes)
    #        Valores sugeridos:
    #        - 0.30 = arriba
    #        - 0.50 = centro
    #        - 0.65 = abajo (ACTUAL)
    #        - 0.75 = muy abajo
    # ============================================================
    
    result = filt.apply(
        donbosco_img,      # BASE: Don Bosco background (large)
        user_img,          # OVERLAY: User's image (small, on top)
        aura_params={"time": 1.0},
        overlay_params={
            "target_width": overlay_w,
            "target_height": overlay_h,
            "left": w_bg * 0.60,    # Horizontal: 60% a la derecha
            "top":  h_bg * 0.65     # Vertical: 65% hacia abajo (CAMBIA ESTE VALOR)
        }
    )

    # Convert RGBA to BGR for JPEG
    rgb = cv2.cvtColor(result[:,:,:3], cv2.COLOR_RGB2BGR)
    
    # Encode to JPEG bytes
    _, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return encoded.tobytes()
