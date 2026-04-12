#include "ggml_bridge.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-quants.h"

// Forward declarations from ggml-cpu
extern void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern void ggml_vec_dot_bf16(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);

// Temp buffer for dequantized f32 weights and f16/bf16 conversions.
// 128KB = 32768 floats, enough for typical embedding dims up to 32768.
#define TMP_BUF_SIZE (128 * 1024)
static _Thread_local char tmp_buf[TMP_BUF_SIZE];

float ggml_bridge_dot_f32(int n, const float* x, const float* y) {
    float result;
    ggml_vec_dot_f32(n, &result, 0, x, 0, y, 0, 1);
    return result;
}

float ggml_bridge_dot_f32_mem(int n, const void* x, const float* y) {
    float result;
    ggml_vec_dot_f32(n, &result, 0, (const float*)x, 0, y, 0, 1);
    return result;
}

float ggml_bridge_dot_f16_f32(int n, const void* x_f16, const float* y_f32) {
    ggml_fp16_t* tmp = (ggml_fp16_t*)tmp_buf;
    ggml_fp32_to_fp16_row(y_f32, tmp, n);
    float result;
    ggml_vec_dot_f16(n, &result, 0, (ggml_fp16_t*)x_f16, 0, tmp, 0, 1);
    return result;
}

float ggml_bridge_dot_bf16_f32(int n, const void* x_bf16, const float* y_f32) {
    ggml_bf16_t* tmp = (ggml_bf16_t*)tmp_buf;
    ggml_fp32_to_bf16_row(y_f32, tmp, n);
    float result;
    ggml_vec_dot_bf16(n, &result, 0, (ggml_bf16_t*)x_bf16, 0, tmp, 0, 1);
    return result;
}

// Dequantize weights to f32, then f32 x f32 dot product.
// Matches the JVM approach: no activation quantization error.
#define DEQUANT_DOT(name, dequant_fn) \
float ggml_bridge_dot_##name##_f32_dequant(int n, const void* x, const float* y) { \
    float* tmp = (float*)tmp_buf; \
    dequant_fn(x, tmp, n); \
    float result; \
    ggml_vec_dot_f32(n, &result, 0, tmp, 0, y, 0, 1); \
    return result; \
}

DEQUANT_DOT(q4_0, dequantize_row_q4_0)
DEQUANT_DOT(q4_1, dequantize_row_q4_1)
DEQUANT_DOT(q5_1, dequantize_row_q5_1)
DEQUANT_DOT(q8_0, dequantize_row_q8_0)
DEQUANT_DOT(q4_K, dequantize_row_q4_K)
DEQUANT_DOT(q5_K, dequantize_row_q5_K)
DEQUANT_DOT(q6_K, dequantize_row_q6_K)
DEQUANT_DOT(mxfp4, dequantize_row_mxfp4)
