#include <ap_int.h>
#include <hls_stream.h>

// Exact spatial/channel shape of /model.2/m.0/cv1/conv in the frozen
// six-class YOLO26n ONNX. This is a stand-alone, DPU-free PL experiment.
namespace {
constexpr int H = 160;
constexpr int W = 160;
constexpr int CI = 16;
constexpr int CO = 8;
constexpr int T = 16;
constexpr int N = H / T;
constexpr int PARAM_WEIGHT_WORDS = 9 * CI;
constexpr int PARAM_BIAS_WORDS = CO / 2;
constexpr int PARAM_MULT_WORDS = CO / 2;
constexpr int PARAM_SHIFT_WORDS = CO / 2;
constexpr int PARAM_LUT_WORDS = 256;

typedef ap_uint<128> input_word;
typedef ap_uint<64> output_word;

ap_int<8> byte_at(input_word word, int n) {
#pragma HLS INLINE
  return ap_int<8>(word.range(8 * n + 7, 8 * n));
}

ap_int<8> weight_at(output_word word, int n) {
#pragma HLS INLINE
  return ap_int<8>(word.range(8 * n + 7, 8 * n));
}

ap_int<8> saturate8(ap_int<64> x) {
#pragma HLS INLINE
  if (x > 127) return 127;
  if (x < -128) return -128;
  return ap_int<8>(x);
}

// Q31 multiplier followed by a per-output-channel right shift. Rounds
// half away from zero. Calibrated multipliers/LUTs must be supplied for
// a real layer; this stand-alone contract does not claim model accuracy.
ap_int<8> requant(ap_int<32> acc, ap_int<32> mult, ap_uint<6> shift) {
#pragma HLS INLINE
  ap_int<64> product = ap_int<64>(acc) * ap_int<64>(mult);
  if (shift == 0) return saturate8(product);
  ap_int<64> half = ap_int<64>(1) << (shift - 1);
  ap_int<64> rounded;
  if (product >= 0)
    rounded = (product + half) >> shift;
  else
    rounded = -(((-product) + half) >> shift);
  return saturate8(rounded);
}

void load_tiles(const input_word* input, hls::stream<input_word>& tiles) {
  for (int ty = 0; ty < N; ++ty) {
    for (int tx = 0; tx < N; ++tx) {
      for (int y = 0; y < T + 2; ++y) {
        for (int x = 0; x < T + 2; ++x) {
#pragma HLS PIPELINE II=1
          const int iy = ty * T + y - 1;
          const int ix = tx * T + x - 1;
          input_word value = 0;
          if (iy >= 0 && iy < H && ix >= 0 && ix < W)
            value = input[iy * W + ix];
          tiles.write(value);
        }
      }
    }
  }
}

void compute_tiles(const output_word* params,
                   hls::stream<input_word>& tiles,
                   hls::stream<output_word>& results) {
  output_word weight[9][CI];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
  ap_int<32> bias[CO];
  ap_int<32> multiplier[CO];
  ap_uint<6> shift[CO];
  ap_int<8> activation_lut[CO][256];
#pragma HLS ARRAY_PARTITION variable=bias complete
#pragma HLS ARRAY_PARTITION variable=multiplier complete
#pragma HLS ARRAY_PARTITION variable=shift complete
#pragma HLS ARRAY_PARTITION variable=activation_lut complete dim=1

  for (int tap = 0; tap < 9; ++tap) {
    for (int ic = 0; ic < CI; ++ic) {
#pragma HLS PIPELINE II=1
      weight[tap][ic] = params[tap * CI + ic];
    }
  }
  for (int pair = 0; pair < CO / 2; ++pair) {
#pragma HLS PIPELINE II=1
    output_word b = params[PARAM_WEIGHT_WORDS + pair];
    output_word m = params[PARAM_WEIGHT_WORDS + PARAM_BIAS_WORDS + pair];
    output_word s = params[PARAM_WEIGHT_WORDS + PARAM_BIAS_WORDS +
                           PARAM_MULT_WORDS + pair];
    bias[2 * pair] = ap_int<32>(b.range(31, 0));
    bias[2 * pair + 1] = ap_int<32>(b.range(63, 32));
    multiplier[2 * pair] = ap_int<32>(m.range(31, 0));
    multiplier[2 * pair + 1] = ap_int<32>(m.range(63, 32));
    shift[2 * pair] = ap_uint<6>(s.range(5, 0));
    shift[2 * pair + 1] = ap_uint<6>(s.range(37, 32));
  }
  for (int q = 0; q < PARAM_LUT_WORDS; ++q) {
#pragma HLS PIPELINE II=1
    output_word packed = params[PARAM_WEIGHT_WORDS + PARAM_BIAS_WORDS +
                                PARAM_MULT_WORDS + PARAM_SHIFT_WORDS + q];
    for (int oc = 0; oc < CO; ++oc) {
#pragma HLS UNROLL
      activation_lut[oc][q] = weight_at(packed, oc);
    }
  }

  input_word tile[T + 2][T + 2];
#pragma HLS BIND_STORAGE variable=tile type=ram_2p impl=bram
  for (int tile_id = 0; tile_id < N * N; ++tile_id) {
    for (int y = 0; y < T + 2; ++y) {
      for (int x = 0; x < T + 2; ++x) {
#pragma HLS PIPELINE II=1
        tile[y][x] = tiles.read();
      }
    }

    for (int y = 0; y < T; ++y) {
      for (int x = 0; x < T; ++x) {
        ap_int<32> acc[CO];
#pragma HLS ARRAY_PARTITION variable=acc complete
        for (int oc = 0; oc < CO; ++oc) {
#pragma HLS UNROLL
          acc[oc] = bias[oc];
        }
        for (int tap = 0; tap < 9; ++tap) {
#pragma HLS PIPELINE II=1
          const int ky = tap / 3;
          const int kx = tap % 3;
          input_word pixel = tile[y + ky][x + kx];
          for (int oc = 0; oc < CO; ++oc) {
#pragma HLS UNROLL
            ap_int<24> product[CI];
#pragma HLS ARRAY_PARTITION variable=product complete
            for (int ic = 0; ic < CI; ++ic) {
#pragma HLS UNROLL
              product[ic] = byte_at(pixel, ic) *
                            weight_at(weight[tap][ic], oc);
            }
            ap_int<32> sum = 0;
            for (int ic = 0; ic < CI; ++ic) {
#pragma HLS UNROLL
              sum += product[ic];
            }
            acc[oc] += sum;
          }
        }
        output_word packed = 0;
        for (int oc = 0; oc < CO; ++oc) {
#pragma HLS UNROLL
          ap_int<8> q = requant(acc[oc], multiplier[oc], shift[oc]);
          ap_uint<8> index = ap_uint<8>(q) ^ 0x80;
          packed.range(8 * oc + 7, 8 * oc) = activation_lut[oc][index];
        }
        results.write(packed);
      }
    }
  }
}

void store_tiles(output_word* output, hls::stream<output_word>& results) {
  for (int ty = 0; ty < N; ++ty) {
    for (int tx = 0; tx < N; ++tx) {
      for (int y = 0; y < T; ++y) {
        for (int x = 0; x < T; ++x) {
#pragma HLS PIPELINE II=1
          output[(ty * T + y) * W + tx * T + x] = results.read();
        }
      }
    }
  }
}
}  // namespace

extern "C" void conv3x3_int8_stream(const input_word* input,
                                     const output_word* params,
                                     output_word* output) {
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem_input depth=25600 max_read_burst_length=64
#pragma HLS INTERFACE mode=m_axi port=params offset=slave bundle=gmem_params depth=412 max_read_burst_length=64
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem_output depth=25600 max_write_burst_length=64
#pragma HLS INTERFACE mode=s_axilite port=input bundle=control
#pragma HLS INTERFACE mode=s_axilite port=params bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
#pragma HLS DATAFLOW
  hls::stream<input_word> tiles("tile_pixels");
  hls::stream<output_word> results("result_pixels");
// One complete 18x18x16 tile may queue while the local BRAM tile is
// consumed. This is a bounded two-tile producer/consumer buffer.
#pragma HLS STREAM variable=tiles depth=324
#pragma HLS STREAM variable=results depth=64
  load_tiles(input, tiles);
  compute_tiles(params, tiles, results);
  store_tiles(output, results);
}
