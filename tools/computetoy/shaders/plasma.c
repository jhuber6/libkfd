#include <gpuintrin.h>

typedef float __attribute__((ext_vector_type(2))) float2;
typedef float __attribute__((ext_vector_type(3))) float3;
typedef float __attribute__((ext_vector_type(4))) float4;

#define MATH(FN, TY)                                                           \
  [[clang::overloadable]] static TY FN(TY x) {                                 \
    return __builtin_elementwise_##FN(x);                                      \
  }
MATH(sin, float)
MATH(cos, float)
MATH(sqrt, float)
MATH(sin, float3)

static float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

static float clamp(float x, float min, float max) {
  return x < min ? min : (x > max ? max : x);
}

static unsigned pack_argb(float4 c) {
  return (unsigned)(clamp(c.a, 0.0f, 1.0f) * 255.0f) << 24u |
         (unsigned)(clamp(c.r, 0.0f, 1.0f) * 255.0f) << 16u |
         (unsigned)(clamp(c.g, 0.0f, 1.0f) * 255.0f) << 8u |
         (unsigned)(clamp(c.b, 0.0f, 1.0f) * 255.0f);
}

struct Uniforms {
  unsigned *framebuffer;
  unsigned width;
  unsigned height;
  float time;
  unsigned frame;
};

#define PI 3.14159265f
#define TAU 6.28318530f

__gpu_kernel void fragment(struct Uniforms u) {
  unsigned x = __gpu_thread_id_x() + __gpu_block_id_x() * __gpu_num_threads_x();
  unsigned y = __gpu_thread_id_y() + __gpu_block_id_y() * __gpu_num_threads_y();
  if (x >= u.width || y >= u.height)
    return;

  float2 uv = {(float)x / (float)u.width, (float)y / (float)u.height};
  float t = u.time;

  float v1 = sin(uv.x * 10.0f + t);
  float v2 = sin(10.0f * (uv.x * sin(t * 0.5f) + uv.y * cos(t * 0.33f)) + t);
  float2 c = uv - 0.5f + 0.5f * (float2){sin(t * 0.2f), cos(t * 0.15f)};
  float v3 = sin(sqrt(dot(c, c) * 100.0f + 1.0f) + t);
  float v = (v1 + v2 + v3) * 0.333f;

  float3 phase = v * PI + (float3){0.0f, TAU / 3.0f, 2.0f * TAU / 3.0f};
  float3 rgb = sin(phase) * 0.5f + 0.5f;

  u.framebuffer[y * u.width + x] =
      pack_argb((float4){rgb.r, rgb.g, rgb.b, 1.0f});
}
