import numpy as np
import cv2
import imageio
import uuid
import os

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
