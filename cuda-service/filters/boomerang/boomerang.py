import numpy as np
import cv2
import imageio
import uuid
import os
from pathlib import Path

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

kernel_code = r"""
extern "C" {

__global__ void update_balls(
    float* pos,
    float* vel,
    int num,
    float dt,
    float width,
    float height,
    float radius
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    float x = pos[2*i + 0];
    float y = pos[2*i + 1];
    float vx = vel[2*i + 0];
    float vy = vel[2*i + 1];

    x += vx * dt;
    y += vy * dt;

    if (x - radius < 0) { x = radius; vx = -vx; }
    if (x + radius > width) { x = width - radius; vx = -vx; }
    if (y - radius < 0) { y = radius; vy = -vy; }
    if (y + radius > height) { y = height - radius; vy = -vy; }

    pos[2*i + 0] = x;
    pos[2*i + 1] = y;
    vel[2*i + 0] = vx;
    vel[2*i + 1] = vy;
}



__global__ void draw_texture_balls(
    const unsigned char* bg,
    unsigned char* out,
    int width,
    int height,
    int channels,
    const float* pos,
    int num,
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

    for (int i = 0; i < num; i++)
    {
        float cx = pos[2*i + 0];
        float cy = pos[2*i + 1];

        float dx = x - cx;
        float dy = y - cy;
        float dist2 = dx*dx + dy*dy;

        if (dist2 < radius2)
        {
            float u = (dx + radius) * (tw - 1) / (2*radius);
            float v = (dy + radius) * (th - 1) / (2*radius);

            int tx = (int)u;
            int ty = (int)v;

            int tidx = (ty * tw + tx) * 4;

            unsigned char tr = tex[tidx + 2];
            unsigned char tg = tex[tidx + 1];
            unsigned char tb = tex[tidx + 0];
            unsigned char ta = tex[tidx + 3];

            float a = ta / 255.0f;

            r = (1.0f - a)*r + a*tr;
            g = (1.0f - a)*g + a*tg;
            b = (1.0f - a)*b + a*tb;
        }
    }

    out[idx + 2] = (unsigned char)(r);
    out[idx + 1] = (unsigned char)(g);
    out[idx + 0] = (unsigned char)(b);
}

} // extern
"""

mod = SourceModule(kernel_code)
update_balls_kernel = mod.get_function("update_balls")
draw_texture_kernel = mod.get_function("draw_texture_balls")


def init_balls(num, width, height, radius):
    pos = []
    vel = []

    for _ in range(num):
        x = np.random.uniform(radius, width - radius)
        y = np.random.uniform(radius, height - radius)
        vx = np.random.uniform(-200, 200)
        vy = np.random.uniform(-200, 200)
        pos += [x, y]
        vel += [vx, vy]

    pos = np.array(pos, dtype=np.float32)
    vel = np.array(vel, dtype=np.float32)

    d_pos = cuda.mem_alloc(pos.nbytes)
    d_vel = cuda.mem_alloc(vel.nbytes)

    cuda.memcpy_htod(d_pos, pos)
    cuda.memcpy_htod(d_vel, vel)

    return d_pos, d_vel, pos, vel


def simulate_boomerang(img, num_balls=3, frames=30):
    h, w, ch = img.shape
    radius = int(0.06 * min(w, h))
    dt = 0.04

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    tex_path = assets_dir / "sonrisa.png"
    tex = cv2.imread(tex_path, cv2.IMREAD_UNCHANGED)

    if tex is None:
        raise RuntimeError("❌ No se encontró la textura sonrisa.png en filters/assets")

    tex = cv2.resize(tex, (2*radius, 2*radius), interpolation=cv2.INTER_AREA)
    tex = tex.astype(np.uint8)

    d_tex = cuda.mem_alloc(tex.nbytes)
    cuda.memcpy_htod(d_tex, tex)

    d_pos, d_vel, _, _ = init_balls(num_balls, w, h, radius)

    d_bg = cuda.mem_alloc(img.nbytes)
    d_out = cuda.mem_alloc(img.nbytes)

    threads = 32
    blocks = (num_balls + threads - 1) // threads

    block2D = (16, 16, 1)
    grid2D = ((w + 15) // 16, (h + 15) // 16, 1)

    frames_out = []

    for _ in range(frames):
        update_balls_kernel(
            d_pos, d_vel,
            np.int32(num_balls),
            np.float32(dt),
            np.float32(w),
            np.float32(h),
            np.float32(radius),
            block=(threads, 1, 1),
            grid=(blocks, 1, 1)
        )

        bg_copy = img.copy()
        out = np.empty_like(img)

        cuda.memcpy_htod(d_bg, bg_copy)

        draw_texture_kernel(
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

        cuda.memcpy_dtoh(out, d_out)
        frames_out.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    out_name = f"boomerang_{uuid.uuid4().hex}.png"
    out_path = os.path.join("/tmp", out_name)
    cv2.imwrite(out_path, cv2.cvtColor(frames_out[-1], cv2.COLOR_RGB2BGR))

    gif_bytes = imageio.mimsave(imageio.RETURN_BYTES, frames_out, format="GIF", duration=0.06)

    d_pos.free()
    d_vel.free()
    d_bg.free()
    d_out.free()
    d_tex.free()

    return gif_bytes, out_path
