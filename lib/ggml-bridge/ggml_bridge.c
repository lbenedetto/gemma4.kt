#include "ggml_bridge.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

// Forward declarations from ggml-cpu
extern void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern void ggml_vec_dot_bf16(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);

extern void quantize_row_q8_0(const float * x, void * y, int64_t k);
extern void quantize_row_q8_1(const float * x, void * y, int64_t k);
extern void quantize_row_q8_K(const float * x, void * y, int64_t k);

extern void ggml_vec_dot_q4_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q4_1_q8_1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q5_1_q8_1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q5_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_mxfp4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// Thread-local buffer for q8 quantization.
// Sized for embedding dims up to 32768:
//   q8_0: 32768/32 * 34 = 34816 bytes
//   q8_1: 32768/32 * 36 = 36864 bytes
//   q8_K: 32768/256 * 292 = 37376 bytes
// Also used for f16/bf16 conversion: 32768 * 2 = 65536 bytes
#define Q8_BUF_SIZE (128 * 1024)
static _Thread_local char q8_buf[Q8_BUF_SIZE];

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
    ggml_fp16_t* tmp = (ggml_fp16_t*)q8_buf;
    ggml_fp32_to_fp16_row(y_f32, tmp, n);
    float result;
    ggml_vec_dot_f16(n, &result, 0, (ggml_fp16_t*)x_f16, 0, tmp, 0, 1);
    return result;
}

float ggml_bridge_dot_bf16_f32(int n, const void* x_bf16, const float* y_f32) {
    ggml_bf16_t* tmp = (ggml_bf16_t*)q8_buf;
    ggml_fp32_to_bf16_row(y_f32, tmp, n);
    float result;
    ggml_vec_dot_bf16(n, &result, 0, (ggml_bf16_t*)x_bf16, 0, tmp, 0, 1);
    return result;
}

float ggml_bridge_dot_q4_0_f32(int n, const void* x, const float* y) {
    quantize_row_q8_0(y, q8_buf, n);
    float result;
    ggml_vec_dot_q4_0_q8_0(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_q4_1_f32(int n, const void* x, const float* y) {
    quantize_row_q8_1(y, q8_buf, n);
    float result;
    ggml_vec_dot_q4_1_q8_1(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_q5_1_f32(int n, const void* x, const float* y) {
    quantize_row_q8_1(y, q8_buf, n);
    float result;
    ggml_vec_dot_q5_1_q8_1(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_q8_0_f32(int n, const void* x, const float* y) {
    quantize_row_q8_0(y, q8_buf, n);
    float result;
    ggml_vec_dot_q8_0_q8_0(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_q4_K_f32(int n, const void* x, const float* y) {
    quantize_row_q8_K(y, q8_buf, n);
    float result;
    ggml_vec_dot_q4_K_q8_K(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_q5_K_f32(int n, const void* x, const float* y) {
    quantize_row_q8_K(y, q8_buf, n);
    float result;
    ggml_vec_dot_q5_K_q8_K(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_q6_K_f32(int n, const void* x, const float* y) {
    quantize_row_q8_K(y, q8_buf, n);
    float result;
    ggml_vec_dot_q6_K_q8_K(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}

float ggml_bridge_dot_mxfp4_f32(int n, const void* x, const float* y) {
    quantize_row_q8_0(y, q8_buf, n);
    float result;
    ggml_vec_dot_mxfp4_q8_0(n, &result, 0, x, 0, q8_buf, 0, 1);
    return result;
}
