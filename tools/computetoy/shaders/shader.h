//===-- tools/computetoy/shader/shader.h - HLSL-like utilities ----*- C -*-===//
//
// Basic functions and types that provide an interface similar to HLSL in a
// compute kernel.
//
//===----------------------------------------------------------------------===//

#ifndef COMPUTETOY_SHADER_H
#define COMPUTETOY_SHADER_H

#include <gpuintrin.h>

//===----------------------------------------------------------------------===//
// Vector types
//===----------------------------------------------------------------------===//

typedef float __attribute__((ext_vector_type(2))) float2;
typedef float __attribute__((ext_vector_type(3))) float3;
typedef float __attribute__((ext_vector_type(4))) float4;

typedef int __attribute__((ext_vector_type(2))) int2;
typedef int __attribute__((ext_vector_type(3))) int3;
typedef int __attribute__((ext_vector_type(4))) int4;

typedef unsigned __attribute__((ext_vector_type(2))) uint2;
typedef unsigned __attribute__((ext_vector_type(3))) uint3;
typedef unsigned __attribute__((ext_vector_type(4))) uint4;

//===----------------------------------------------------------------------===//
// Vector construction helpers
//===----------------------------------------------------------------------===//

[[clang::overloadable]] static float3 vec3(float2 xy, float z) {
  return (float3){xy.x, xy.y, z};
}
[[clang::overloadable]] static float3 vec3(float x, float2 yz) {
  return (float3){x, yz.x, yz.y};
}

[[clang::overloadable]] static float4 vec4(float3 xyz, float w) {
  return (float4){xyz.x, xyz.y, xyz.z, w};
}
[[clang::overloadable]] static float4 vec4(float x, float3 yzw) {
  return (float4){x, yzw.x, yzw.y, yzw.z};
}
[[clang::overloadable]] static float4 vec4(float2 xy, float2 zw) {
  return (float4){xy.x, xy.y, zw.x, zw.y};
}
[[clang::overloadable]] static float4 vec4(float2 xy, float z, float w) {
  return (float4){xy.x, xy.y, z, w};
}
[[clang::overloadable]] static float4 vec4(float x, float2 yz, float w) {
  return (float4){x, yz.x, yz.y, w};
}
[[clang::overloadable]] static float4 vec4(float x, float y, float2 zw) {
  return (float4){x, y, zw.x, zw.y};
}

//===----------------------------------------------------------------------===//
// Elementwise builtin overloads
//===----------------------------------------------------------------------===//

#define OVERLOAD_F1(FN)                                                        \
  [[clang::overloadable]] static float FN(float x) {                           \
    return __builtin_elementwise_##FN(x);                                      \
  }                                                                            \
  [[clang::overloadable]] static float2 FN(float2 x) {                         \
    return __builtin_elementwise_##FN(x);                                      \
  }                                                                            \
  [[clang::overloadable]] static float3 FN(float3 x) {                         \
    return __builtin_elementwise_##FN(x);                                      \
  }                                                                            \
  [[clang::overloadable]] static float4 FN(float4 x) {                         \
    return __builtin_elementwise_##FN(x);                                      \
  }                                                                            \
  static_assert(1)

#define OVERLOAD_F2(FN)                                                        \
  [[clang::overloadable]] static float FN(float x, float y) {                  \
    return __builtin_elementwise_##FN(x, y);                                   \
  }                                                                            \
  [[clang::overloadable]] static float2 FN(float2 x, float2 y) {               \
    return __builtin_elementwise_##FN(x, y);                                   \
  }                                                                            \
  [[clang::overloadable]] static float3 FN(float3 x, float3 y) {               \
    return __builtin_elementwise_##FN(x, y);                                   \
  }                                                                            \
  [[clang::overloadable]] static float4 FN(float4 x, float4 y) {               \
    return __builtin_elementwise_##FN(x, y);                                   \
  }                                                                            \
  static_assert(1)

#define OVERLOAD_F3(FN)                                                        \
  [[clang::overloadable]] static float FN(float x, float y, float z) {         \
    return __builtin_elementwise_##FN(x, y, z);                                \
  }                                                                            \
  [[clang::overloadable]] static float2 FN(float2 x, float2 y, float2 z) {     \
    return __builtin_elementwise_##FN(x, y, z);                                \
  }                                                                            \
  [[clang::overloadable]] static float3 FN(float3 x, float3 y, float3 z) {     \
    return __builtin_elementwise_##FN(x, y, z);                                \
  }                                                                            \
  [[clang::overloadable]] static float4 FN(float4 x, float4 y, float4 z) {     \
    return __builtin_elementwise_##FN(x, y, z);                                \
  }                                                                            \
  static_assert(1)

OVERLOAD_F1(sin);
OVERLOAD_F1(cos);
OVERLOAD_F1(exp);
OVERLOAD_F1(exp2);
OVERLOAD_F1(log);
OVERLOAD_F1(log2);
OVERLOAD_F1(log10);
OVERLOAD_F1(floor);
OVERLOAD_F1(ceil);
OVERLOAD_F1(round);
OVERLOAD_F1(trunc);
OVERLOAD_F1(sqrt);
OVERLOAD_F1(abs);
OVERLOAD_F2(pow);
OVERLOAD_F2(fmod);
OVERLOAD_F2(copysign);
OVERLOAD_F3(fma);

#undef OVERLOAD_F1
#undef OVERLOAD_F2
#undef OVERLOAD_F3

//===----------------------------------------------------------------------===//
// min / max / abs for all numeric types
//===----------------------------------------------------------------------===//

#define DEF_EACH_F(M)                                                          \
  M(float);                                                                    \
  M(float2);                                                                   \
  M(float3);                                                                   \
  M(float4)
#define DEF_EACH_I(M)                                                          \
  M(int);                                                                      \
  M(int2);                                                                     \
  M(int3);                                                                     \
  M(int4)
#define DEF_EACH_U(M)                                                          \
  M(unsigned);                                                                 \
  M(uint2);                                                                    \
  M(uint3);                                                                    \
  M(uint4)

#define DEF_IABS(T)                                                            \
  [[clang::overloadable]] static T abs(T x) {                                  \
    return __builtin_elementwise_abs(x);                                       \
  }
DEF_EACH_I(DEF_IABS);
#undef DEF_IABS

#define DEF_FMINMAX(T)                                                         \
  [[clang::overloadable]] static T min(T x, T y) {                             \
    return __builtin_elementwise_minnum(x, y);                                 \
  }                                                                            \
  [[clang::overloadable]] static T max(T x, T y) {                             \
    return __builtin_elementwise_maxnum(x, y);                                 \
  }
DEF_EACH_F(DEF_FMINMAX);
#undef DEF_FMINMAX

#define DEF_IMINMAX(T)                                                         \
  [[clang::overloadable]] static T min(T x, T y) {                             \
    return __builtin_elementwise_min(x, y);                                    \
  }                                                                            \
  [[clang::overloadable]] static T max(T x, T y) {                             \
    return __builtin_elementwise_max(x, y);                                    \
  }
DEF_EACH_I(DEF_IMINMAX);
DEF_EACH_U(DEF_IMINMAX);
#undef DEF_IMINMAX

//===----------------------------------------------------------------------===//
// Derived scalar/vector functions
//===----------------------------------------------------------------------===//

#define DEF_TAN(T)                                                             \
  [[clang::overloadable]] static T tan(T x) { return sin(x) / cos(x); }
DEF_EACH_F(DEF_TAN);
#undef DEF_TAN

#define DEF_CLAMP(T)                                                           \
  [[clang::overloadable]] static T clamp(T x, T lo, T hi) {                    \
    return min(max(x, lo), hi);                                                \
  }
DEF_EACH_F(DEF_CLAMP);
DEF_EACH_I(DEF_CLAMP);
DEF_EACH_U(DEF_CLAMP);
#undef DEF_CLAMP

#define DEF_SATURATE(T)                                                        \
  [[clang::overloadable]] static T saturate(T x) {                             \
    return clamp(x, (T)0.0f, (T)1.0f);                                         \
  }
DEF_EACH_F(DEF_SATURATE);
#undef DEF_SATURATE

#define DEF_FRAC(T)                                                            \
  [[clang::overloadable]] static T frac(T x) { return x - floor(x); }
DEF_EACH_F(DEF_FRAC);
#undef DEF_FRAC

#define DEF_RSQRT(T)                                                           \
  [[clang::overloadable]] static T rsqrt(T x) { return (T)1.0f / sqrt(x); }
DEF_EACH_F(DEF_RSQRT);
#undef DEF_RSQRT

#define DEF_LERP(T)                                                            \
  [[clang::overloadable]] static T lerp(T a, T b, T t) {                       \
    return a + t * (b - a);                                                    \
  }
DEF_EACH_F(DEF_LERP);
#undef DEF_LERP

// Scalar interpolant with vector endpoints.
[[clang::overloadable]] static float2 lerp(float2 a, float2 b, float t) {
  return a + t * (b - a);
}
[[clang::overloadable]] static float3 lerp(float3 a, float3 b, float t) {
  return a + t * (b - a);
}
[[clang::overloadable]] static float4 lerp(float4 a, float4 b, float t) {
  return a + t * (b - a);
}

#define DEF_STEP(T)                                                            \
  [[clang::overloadable]] static T step(T edge, T x) {                         \
    return x >= edge ? (T)1.0f : (T)0.0f;                                      \
  }
DEF_EACH_F(DEF_STEP);
#undef DEF_STEP

#define DEF_SMOOTHSTEP(T)                                                      \
  [[clang::overloadable]] static T smoothstep(T lo, T hi, T x) {               \
    T t = saturate((x - lo) / (hi - lo));                                      \
    return t * t * ((T)3.0f - (T)2.0f * t);                                    \
  }
DEF_EACH_F(DEF_SMOOTHSTEP);
#undef DEF_SMOOTHSTEP

#undef DEF_EACH_F
#undef DEF_EACH_I
#undef DEF_EACH_U

//===----------------------------------------------------------------------===//
// Vector geometry functions
//===----------------------------------------------------------------------===//

[[clang::overloadable]] static float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
[[clang::overloadable]] static float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
[[clang::overloadable]] static float dot(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static float3 cross(float3 a, float3 b) {
  return (float3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x};
}

[[clang::overloadable]] static float length(float2 v) {
  return sqrt(dot(v, v));
}
[[clang::overloadable]] static float length(float3 v) {
  return sqrt(dot(v, v));
}
[[clang::overloadable]] static float length(float4 v) {
  return sqrt(dot(v, v));
}

[[clang::overloadable]] static float distance(float2 a, float2 b) {
  return length(a - b);
}
[[clang::overloadable]] static float distance(float3 a, float3 b) {
  return length(a - b);
}
[[clang::overloadable]] static float distance(float4 a, float4 b) {
  return length(a - b);
}

[[clang::overloadable]] static float2 normalize(float2 v) {
  return v * rsqrt(dot(v, v));
}
[[clang::overloadable]] static float3 normalize(float3 v) {
  return v * rsqrt(dot(v, v));
}
[[clang::overloadable]] static float4 normalize(float4 v) {
  return v * rsqrt(dot(v, v));
}

[[clang::overloadable]] static float2 reflect(float2 i, float2 n) {
  return i - 2.0f * dot(n, i) * n;
}
[[clang::overloadable]] static float3 reflect(float3 i, float3 n) {
  return i - 2.0f * dot(n, i) * n;
}
[[clang::overloadable]] static float4 reflect(float4 i, float4 n) {
  return i - 2.0f * dot(n, i) * n;
}

//===----------------------------------------------------------------------===//
// Utility
//===----------------------------------------------------------------------===//

[[clang::overloadable]] static unsigned pack_argb(float4 c) {
  c = saturate(c);
  return (unsigned)(c.a * 255.0f) << 24u | (unsigned)(c.r * 255.0f) << 16u |
         (unsigned)(c.g * 255.0f) << 8u | (unsigned)(c.b * 255.0f);
}
[[clang::overloadable]] static unsigned pack_argb(float3 c) {
  return pack_argb(vec4(c, 1.0));
}

#define pixel(u, x, y) ((u).framebuffer[(y) * (u).pitch + (x)])

#endif // COMPUTETOY_SHADER_H
