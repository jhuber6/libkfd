#include "shader.h"

struct Uniforms {
  unsigned *framebuffer;
  unsigned width;
  unsigned height;
  unsigned pitch;
  float time;
  unsigned frame;
};

static float sd_box(float2 point, float2 box) {
  float2 q = abs(point) - box;
  return length(max(q, (float2){0.0f, 0.0f})) + min(max(q.x, q.y), 0.0f);
}

__gpu_kernel void fragment(struct Uniforms u) {
  unsigned x = __gpu_thread_id_x() + __gpu_block_id_x() * __gpu_num_threads_x();
  unsigned y = __gpu_thread_id_y() + __gpu_block_id_y() * __gpu_num_threads_y();
  if (x >= u.width || y >= u.height)
    return;
  float2 resolution = {(float)u.width, (float)u.height};
  float2 pos = {(float)x, (float)y};
  float2 uv = (pos * 2.0f - resolution) / resolution.x;

  float phase = frac(u.time * 2.0f);
  float3 background = {0.05f, 0.02f, 0.08f};
  float3 rgb = {};
  const int N = 50;
  for (int i = 0; i < N; i++) {
    float depth = (float)(i + 1) + phase;
    float s = 1.0f - depth * (1.0f / N);
    s = s / (depth * 0.15f);
    float angle = depth * 0.10f;
    float2 box = {s, s};
    float d = sd_box(rotate(uv, angle), box);
    float fade = smoothstep(0.0f, 0.15f, s);
    float3 color = {0.5f + 0.5f * sin(depth * 0.5f),
                    0.5f + 0.5f * sin(depth * 0.5f + 2.1f),
                    0.5f + 0.5f * sin(depth * 0.5f + 4.2f)};
    float glow = min(0.001f / abs(d), 2.0f);
    rgb += glow * color * fade;
  }
  rgb += background * ((1.0f - length(uv)) * 5.0f);

  pixel(u, x, y) = pack_argb(rgb);
}
