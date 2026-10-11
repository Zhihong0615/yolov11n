#include "attention_online.hpp"

#include <hls_math.h>
#include <cmath>

namespace {
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 64;
inline float attention_exp(float x) {
#ifdef __SYNTHESIS__
  return hls::expf(x);
#else
  return std::exp(x);
#endif
}
}

extern "C" void psa_attention_cached_tree8(
    const float* q, const float* k, const float* v, float* output) {
#pragma HLS INTERFACE mode=m_axi port=q offset=slave bundle=gmem_q depth=25600 max_read_burst_length=32
#pragma HLS INTERFACE mode=m_axi port=k offset=slave bundle=gmem_k depth=25600 max_read_burst_length=64
#pragma HLS INTERFACE mode=m_axi port=v offset=slave bundle=gmem_v depth=51200 max_read_burst_length=64
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem_o depth=51200 max_write_burst_length=16
#pragma HLS INTERFACE mode=s_axilite port=q bundle=control
#pragma HLS INTERFACE mode=s_axilite port=k bundle=control
#pragma HLS INTERFACE mode=s_axilite port=v bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

  // One head at a time: 150 KiB K+V logical state, banked into at least
  // eight URAM288 blocks before HLS padding/replication. Placement unproved.
  float k_cache[psa::kQkDim][psa::kTokens];
  float v_cache[psa::kValueDim][psa::kTokens];
#pragma HLS ARRAY_PARTITION variable=k_cache cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=v_cache cyclic factor=4 dim=1
#pragma HLS BIND_STORAGE variable=k_cache type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=v_cache type=ram_2p impl=uram
  float q_tile[kQueryTile][psa::kQkDim];
#pragma HLS ARRAY_PARTITION variable=q_tile cyclic factor=8 dim=2
  float scores[kQueryTile][kKeyTile];
  float max_score[kQueryTile];
  float normalizer[kQueryTile];
  float accumulator[kQueryTile][psa::kValueDim];

  for (int head = 0; head < psa::kHeads; ++head) {
    for (int dim = 0; dim < psa::kQkDim; ++dim) {
      for (int key = 0; key < psa::kTokens; ++key) {
#pragma HLS PIPELINE II=1
        k_cache[dim][key] =
            k[(head * psa::kQkDim + dim) * psa::kTokens + key];
      }
    }
    for (int dim = 0; dim < psa::kValueDim; ++dim) {
      for (int key = 0; key < psa::kTokens; ++key) {
#pragma HLS PIPELINE II=1
        v_cache[dim][key] =
            v[(head * psa::kValueDim + dim) * psa::kTokens + key];
      }
    }

    for (int query_start = 0; query_start < psa::kTokens;
         query_start += kQueryTile) {
      for (int row = 0; row < kQueryTile; ++row) {
        max_score[row] = -3.402823466e38f;
        normalizer[row] = 0.0f;
        for (int dim = 0; dim < psa::kQkDim; ++dim) {
#pragma HLS PIPELINE II=1
          q_tile[row][dim] =
              q[(head * psa::kTokens + query_start + row) *
                    psa::kQkDim + dim];
        }
        for (int dim = 0; dim < psa::kValueDim; ++dim) {
#pragma HLS PIPELINE II=1
          accumulator[row][dim] = 0.0f;
        }
      }

      for (int key_start = 0; key_start < psa::kTokens;
           key_start += kKeyTile) {
        for (int row = 0; row < kQueryTile; ++row) {
          for (int col = 0; col < kKeyTile; ++col) {
            float product[psa::kQkDim];
            float reduce16[16];
            float reduce8[8];
            float reduce4[4];
            float reduce2[2];
#pragma HLS ARRAY_PARTITION variable=product complete
#pragma HLS ARRAY_PARTITION variable=reduce16 complete
#pragma HLS ARRAY_PARTITION variable=reduce8 complete
#pragma HLS ARRAY_PARTITION variable=reduce4 complete
#pragma HLS ARRAY_PARTITION variable=reduce2 complete
            for (int dim = 0; dim < psa::kQkDim; ++dim) {
#pragma HLS UNROLL factor=8
              product[dim] = key_start + col < psa::kTokens
                                 ? q_tile[row][dim] *
                                       k_cache[dim][key_start + col]
                                 : 0.0f;
            }
            for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
              reduce16[i] = product[i] + product[i + 16];
            }
            for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
              reduce8[i] = reduce16[i] + reduce16[i + 8];
            }
            for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
              reduce4[i] = reduce8[i] + reduce8[i + 4];
            }
            for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
              reduce2[i] = reduce4[i] + reduce4[i + 2];
            }
            scores[row][col] = reduce2[0] + reduce2[1];
          }
        }
        for (int row = 0; row < kQueryTile; ++row) {
          float next_max = max_score[row];
          for (int col = 0; col < kKeyTile; ++col) {
#pragma HLS PIPELINE II=1
            if (key_start + col < psa::kTokens &&
                scores[row][col] > next_max) {
              next_max = scores[row][col];
            }
          }
          const float alpha = normalizer[row] == 0.0f
                                  ? 0.0f
                                  : attention_exp(max_score[row] - next_max);
          normalizer[row] *= alpha;
          for (int dim = 0; dim < psa::kValueDim; ++dim) {
#pragma HLS PIPELINE II=1
            accumulator[row][dim] *= alpha;
          }
          for (int col = 0; col < kKeyTile; ++col) {
            if (key_start + col >= psa::kTokens) continue;
            const float weight = attention_exp(scores[row][col] - next_max);
            normalizer[row] += weight;
            for (int dim = 0; dim < psa::kValueDim; ++dim) {
#pragma HLS PIPELINE II=1
              accumulator[row][dim] += weight * v_cache[dim][key_start + col];
            }
          }
          max_score[row] = next_max;
        }
      }
      for (int dim = 0; dim < psa::kValueDim; ++dim) {
        for (int row = 0; row < kQueryTile; ++row) {
#pragma HLS PIPELINE II=1
          output[(head * psa::kValueDim + dim) * psa::kTokens +
                 query_start + row] =
              accumulator[row][dim] / normalizer[row];
        }
      }
    }
  }
}
