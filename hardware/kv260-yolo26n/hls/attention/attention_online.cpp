#include "attention_online.hpp"

#include <hls_math.h>
#include <hls_stream.h>
#include <cmath>

namespace {

inline float attention_exp(float x) {
#ifdef __SYNTHESIS__
  return hls::expf(x);
#else
  return std::exp(x);
#endif
}

void load_tiles(const float* q, const float* k, const float* v,
                hls::stream<float>& q_stream,
                hls::stream<float>& k_stream,
                hls::stream<float>& v_stream) {
  for (int head = 0; head < psa::kHeads; ++head) {
    for (int query_start = 0; query_start < psa::kTokens;
         query_start += psa::kQueryTile) {
      for (int row = 0; row < psa::kQueryTile; ++row) {
        for (int dim = 0; dim < psa::kQkDim; ++dim) {
#pragma HLS PIPELINE II=1
          q_stream.write(q[(head * psa::kTokens + query_start + row) *
                               psa::kQkDim + dim]);
        }
      }
      for (int key_start = 0; key_start < psa::kTokens;
           key_start += psa::kKeyTile) {
        // Channel-major K/V allows each inner loop to form a 32-word DDR burst.
        for (int dim = 0; dim < psa::kQkDim; ++dim) {
          for (int col = 0; col < psa::kKeyTile; ++col) {
#pragma HLS PIPELINE II=1
            const int key = key_start + col;
            k_stream.write(key < psa::kTokens
                               ? k[(head * psa::kQkDim + dim) *
                                       psa::kTokens + key]
                               : 0.0f);
          }
        }
        for (int dim = 0; dim < psa::kValueDim; ++dim) {
          for (int col = 0; col < psa::kKeyTile; ++col) {
#pragma HLS PIPELINE II=1
            const int key = key_start + col;
            v_stream.write(key < psa::kTokens
                               ? v[(head * psa::kValueDim + dim) *
                                       psa::kTokens + key]
                               : 0.0f);
          }
        }
      }
    }
  }
}

void compute_tiles(hls::stream<float>& q_stream,
                   hls::stream<float>& k_stream,
                   hls::stream<float>& v_stream, float* output) {
  float q_tile[psa::kQueryTile][psa::kQkDim];
  float k_tile[psa::kQkDim][psa::kKeyTile];
  float v_tile[psa::kValueDim][psa::kKeyTile];
  float scores[psa::kQueryTile][psa::kKeyTile];
  float max_score[psa::kQueryTile];
  float normalizer[psa::kQueryTile];
  float accumulator[psa::kQueryTile][psa::kValueDim];

  for (int head = 0; head < psa::kHeads; ++head) {
    for (int query_start = 0; query_start < psa::kTokens;
         query_start += psa::kQueryTile) {
      for (int row = 0; row < psa::kQueryTile; ++row) {
        max_score[row] = -3.402823466e38f;
        normalizer[row] = 0.0f;
        for (int dim = 0; dim < psa::kQkDim; ++dim) {
#pragma HLS PIPELINE II=1
          q_tile[row][dim] = q_stream.read();
        }
        for (int dim = 0; dim < psa::kValueDim; ++dim) {
#pragma HLS PIPELINE II=1
          accumulator[row][dim] = 0.0f;
        }
      }

      for (int key_start = 0; key_start < psa::kTokens;
           key_start += psa::kKeyTile) {
        for (int dim = 0; dim < psa::kQkDim; ++dim) {
          for (int col = 0; col < psa::kKeyTile; ++col) {
#pragma HLS PIPELINE II=1
            k_tile[dim][col] = k_stream.read();
          }
        }
        for (int dim = 0; dim < psa::kValueDim; ++dim) {
          for (int col = 0; col < psa::kKeyTile; ++col) {
#pragma HLS PIPELINE II=1
            v_tile[dim][col] = v_stream.read();
          }
        }

        // Scores remain in a bounded 8x32 tile; no 400x400 matrix is made.
        for (int row = 0; row < psa::kQueryTile; ++row) {
          for (int col = 0; col < psa::kKeyTile; ++col) {
            float dot = 0.0f;
            for (int dim = 0; dim < psa::kQkDim; ++dim) {
#pragma HLS PIPELINE II=1
              dot += q_tile[row][dim] * k_tile[dim][col];
            }
            scores[row][col] = dot;
          }
        }

        for (int row = 0; row < psa::kQueryTile; ++row) {
          float next_max = max_score[row];
          for (int col = 0; col < psa::kKeyTile; ++col) {
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
          for (int col = 0; col < psa::kKeyTile; ++col) {
            if (key_start + col >= psa::kTokens) continue;
            const float weight = attention_exp(scores[row][col] - next_max);
            normalizer[row] += weight;
            for (int dim = 0; dim < psa::kValueDim; ++dim) {
#pragma HLS PIPELINE II=1
              accumulator[row][dim] += weight * v_tile[dim][col];
            }
          }
          max_score[row] = next_max;
        }
      }

      // Channel-major output makes each query tile a short contiguous burst.
      for (int dim = 0; dim < psa::kValueDim; ++dim) {
        for (int row = 0; row < psa::kQueryTile; ++row) {
#pragma HLS PIPELINE II=1
          output[(head * psa::kValueDim + dim) * psa::kTokens +
                 query_start + row] =
              accumulator[row][dim] / normalizer[row];
        }
      }
    }
  }
}

}  // namespace

extern "C" void psa_attention_online(
    const float* q, const float* k, const float* v, float* output) {
#pragma HLS INTERFACE mode=m_axi port=q offset=slave bundle=gmem_q depth=25600 max_read_burst_length=32
#pragma HLS INTERFACE mode=m_axi port=k offset=slave bundle=gmem_k depth=25600 max_read_burst_length=32
#pragma HLS INTERFACE mode=m_axi port=v offset=slave bundle=gmem_v depth=51200 max_read_burst_length=32
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem_o depth=51200 max_write_burst_length=32
#pragma HLS INTERFACE mode=s_axilite port=q bundle=control
#pragma HLS INTERFACE mode=s_axilite port=k bundle=control
#pragma HLS INTERFACE mode=s_axilite port=v bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
#pragma HLS DATAFLOW
  hls::stream<float> q_stream("q_stream");
  hls::stream<float> k_stream("k_stream");
  hls::stream<float> v_stream("v_stream");
#pragma HLS STREAM variable=q_stream depth=64
#pragma HLS STREAM variable=k_stream depth=64
#pragma HLS STREAM variable=v_stream depth=64
  load_tiles(q, k, v, q_stream, k_stream, v_stream);
  compute_tiles(q_stream, k_stream, v_stream, output);
}
