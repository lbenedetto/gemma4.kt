#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Dot product: f32 x f32
float ggml_bridge_dot_f32(int n, const float* x, const float* y);

// Dot product: f32 (from MemorySegment) x f32
float ggml_bridge_dot_f32_mem(int n, const void* x, const float* y);

// Dot product: f16 x f32 (converts f32 to f16 internally)
float ggml_bridge_dot_f16_f32(int n, const void* x_f16, const float* y_f32);

// Dot product: bf16 x f32 (converts f32 to bf16 internally)
float ggml_bridge_dot_bf16_f32(int n, const void* x_bf16, const float* y_f32);

// Dot products: quantized x f32 (converts f32 to q8 internally)
float ggml_bridge_dot_q4_0_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_q4_1_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_q5_1_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_q8_0_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_q4_K_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_q5_K_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_q6_K_f32(int n, const void* x, const float* y);
float ggml_bridge_dot_mxfp4_f32(int n, const void* x, const float* y);

#ifdef __cplusplus
}
#endif
