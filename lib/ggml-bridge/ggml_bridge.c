#include "ggml_bridge.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-quants.h"

// Forward declarations from ggml-cpu
extern void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern void ggml_vec_dot_bf16(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);

// Chunked dequantize+dot: dequantize in L1-cache-friendly chunks,
// then dot each chunk immediately while it's still hot in cache.
// This avoids writing the entire dequantized row to memory.
#define CHUNK_SIZE 256  // 256 floats = 1KB, fits comfortably in L1
static _Thread_local float chunk_buf[CHUNK_SIZE];

// For f16/bf16 conversion (needs full-row buffer since ggml_vec_dot_f16
// reads both sides as f16)
#define CONV_BUF_SIZE (128 * 1024)
static _Thread_local char conv_buf[CONV_BUF_SIZE];

// Helper: chunked dequant+dot for any quant type
// dequant_fn signature: void(const void* src, float* dst, int64_t k)
// block_size: number of elements per block
// type_size: bytes per block
typedef void (*dequant_fn_t)(const void *, float *, int64_t);

static inline float chunked_dequant_dot(
    dequant_fn_t dequant_fn,
    int block_size,
    int type_size,
    int n,
    const void* x,
    const float* y
) {
    float result = 0.0f;
    const char* xb = (const char*)x;
    int i = 0;

    // Process in chunks aligned to block_size
    // CHUNK_SIZE is 256, which is a multiple of all block sizes (32, 256)
    const int chunk = (CHUNK_SIZE / block_size) * block_size;

    while (i + chunk <= n) {
        dequant_fn(xb, chunk_buf, chunk);
        float chunk_result;
        ggml_vec_dot_f32(chunk, &chunk_result, 0, chunk_buf, 0, y + i, 0, 1);
        result += chunk_result;
        xb += (chunk / block_size) * type_size;
        i += chunk;
    }

    // Handle remaining elements (must still be block-aligned)
    if (i < n) {
        int remaining = n - i;
        dequant_fn(xb, chunk_buf, remaining);
        float chunk_result;
        ggml_vec_dot_f32(remaining, &chunk_result, 0, chunk_buf, 0, y + i, 0, 1);
        result += chunk_result;
    }

    return result;
}

// === f32 dot products ===

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

// === f16/bf16 dot products ===

float ggml_bridge_dot_f16_f32(int n, const void* x_f16, const float* y_f32) {
    ggml_fp16_t* tmp = (ggml_fp16_t*)conv_buf;
    ggml_fp32_to_fp16_row(y_f32, tmp, n);
    float result;
    ggml_vec_dot_f16(n, &result, 0, (ggml_fp16_t*)x_f16, 0, tmp, 0, 1);
    return result;
}

float ggml_bridge_dot_bf16_f32(int n, const void* x_bf16, const float* y_f32) {
    ggml_bf16_t* tmp = (ggml_bf16_t*)conv_buf;
    ggml_fp32_to_bf16_row(y_f32, tmp, n);
    float result;
    ggml_vec_dot_bf16(n, &result, 0, (ggml_bf16_t*)x_bf16, 0, tmp, 0, 1);
    return result;
}

// === Quantized dot products: chunked dequantize + f32 dot ===

float ggml_bridge_dot_q4_0_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q4_0, QK4_0, sizeof(block_q4_0), n, x, y);
}

float ggml_bridge_dot_q4_1_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q4_1, QK4_1, sizeof(block_q4_1), n, x, y);
}

float ggml_bridge_dot_q5_1_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q5_1, QK5_1, sizeof(block_q5_1), n, x, y);
}

float ggml_bridge_dot_q8_0_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q8_0, QK8_0, sizeof(block_q8_0), n, x, y);
}

float ggml_bridge_dot_q4_K_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q4_K, QK_K, sizeof(block_q4_K), n, x, y);
}

float ggml_bridge_dot_q5_K_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q5_K, QK_K, sizeof(block_q5_K), n, x, y);
}

float ggml_bridge_dot_q6_K_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_q6_K, QK_K, sizeof(block_q6_K), n, x, y);
}

float ggml_bridge_dot_mxfp4_f32(int n, const void* x, const float* y) {
    return chunked_dequant_dot((dequant_fn_t)dequantize_row_mxfp4, QK_MXFP4, sizeof(block_mxfp4), n, x, y);
}
