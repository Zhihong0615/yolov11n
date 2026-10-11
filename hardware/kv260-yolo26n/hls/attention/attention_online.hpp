#pragma once

// Pretrained YOLO26n PSA attention cut: Q[2][400][32], K[2][32][400],
// V[2][64][400], O[2][64][400], all contiguous FP32 in ONNX axis order.
namespace psa {
constexpr int kHeads = 2;
constexpr int kTokens = 400;
constexpr int kQkDim = 32;
constexpr int kValueDim = 64;
constexpr int kQueryTile = 8;
constexpr int kKeyTile = 32;
constexpr int kQueryElements = kHeads * kTokens * kQkDim;
constexpr int kKeyElements = kHeads * kQkDim * kTokens;
constexpr int kValueElements = kHeads * kValueDim * kTokens;
}

extern "C" void psa_attention_online(
    const float* q, const float* k, const float* v, float* output);

// Second design point: one head's K/V held in banked URAM across all queries.
extern "C" void psa_attention_cached(
    const float* q, const float* k, const float* v, float* output);

// Third design point: serial product loading, then a balanced FP32 reduction.
extern "C" void psa_attention_cached_tree(
    const float* q, const float* k, const float* v, float* output);

extern "C" void psa_attention_cached_tree8(
    const float* q, const float* k, const float* v, float* output);

extern "C" void psa_attention_cached_tree8_value4(
    const float* q, const float* k, const float* v, float* output);
