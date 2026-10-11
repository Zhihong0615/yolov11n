#include <ap_int.h>

#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void conv3x3_int8_stream(const ap_uint<128>* input,
                                     const ap_uint<64>* params,
                                     ap_uint<64>* output);

namespace {
constexpr int H = 160;
constexpr int W = 160;
constexpr int CI = 16;
constexpr int CO = 8;
constexpr int PARAM_WORDS = 9 * CI + 3 * CO / 2 + 256;

int8_t input_value(int y, int x, int ic) {
  return static_cast<int8_t>(((y * 7 + x * 11 + ic * 13) % 31) - 15);
}

int8_t weight_value(int tap, int ic, int oc) {
  return static_cast<int8_t>(((tap * 3 + ic * 5 + oc * 7) % 7) - 3);
}

int32_t bias_value(int oc) { return (oc - 4) * 19; }
int32_t multiplier_value(int oc) { return 1 << (22 + (oc % 3)); }
int shift_value(int oc) { return 30 + (oc & 1); }

int8_t lut_value(int oc, int8_t q) {
  // Mix identity and ReLU; the LUT could instead be calibrated SiLU.
  return (oc & 1) && q < 0 ? 0 : q;
}

int8_t saturate8(int64_t x) {
  if (x > 127) return 127;
  if (x < -128) return -128;
  return static_cast<int8_t>(x);
}

int8_t requant(int32_t acc, int oc) {
  int64_t product = int64_t(acc) * multiplier_value(oc);
  int shift = shift_value(oc);
  int64_t half = int64_t(1) << (shift - 1);
  int64_t rounded = product >= 0 ? (product + half) >> shift
                                  : -(((-product) + half) >> shift);
  return saturate8(rounded);
}

int8_t golden(int y, int x, int oc) {
  int32_t acc = bias_value(oc);
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      const int iy = y + ky - 1;
      const int ix = x + kx - 1;
      if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
      for (int ic = 0; ic < CI; ++ic)
        acc += int32_t(input_value(iy, ix, ic)) *
               int32_t(weight_value(ky * 3 + kx, ic, oc));
    }
  }
  return lut_value(oc, requant(acc, oc));
}
}  // namespace

int main() {
  std::vector<ap_uint<128>> input(H * W);
  std::vector<ap_uint<64>> params(PARAM_WORDS);
  std::vector<ap_uint<64>> output(H * W, 0);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ap_uint<128> p = 0;
      for (int ic = 0; ic < CI; ++ic)
        p.range(8 * ic + 7, 8 * ic) =
            static_cast<uint8_t>(input_value(y, x, ic));
      input[y * W + x] = p;
    }
  }
  for (int tap = 0; tap < 9; ++tap) {
    for (int ic = 0; ic < CI; ++ic) {
      ap_uint<64> p = 0;
      for (int oc = 0; oc < CO; ++oc)
        p.range(8 * oc + 7, 8 * oc) =
            static_cast<uint8_t>(weight_value(tap, ic, oc));
      params[tap * CI + ic] = p;
    }
  }
  for (int pair = 0; pair < CO / 2; ++pair) {
    ap_uint<64> b = 0, m = 0, s = 0;
    b.range(31, 0) = static_cast<uint32_t>(bias_value(2 * pair));
    b.range(63, 32) = static_cast<uint32_t>(bias_value(2 * pair + 1));
    m.range(31, 0) = static_cast<uint32_t>(multiplier_value(2 * pair));
    m.range(63, 32) = static_cast<uint32_t>(multiplier_value(2 * pair + 1));
    s.range(31, 0) = shift_value(2 * pair);
    s.range(63, 32) = shift_value(2 * pair + 1);
    params[144 + pair] = b;
    params[148 + pair] = m;
    params[152 + pair] = s;
  }
  for (int index = 0; index < 256; ++index) {
    const int8_t q = static_cast<int8_t>(index ^ 0x80);
    ap_uint<64> p = 0;
    for (int oc = 0; oc < CO; ++oc)
      p.range(8 * oc + 7, 8 * oc) =
          static_cast<uint8_t>(lut_value(oc, q));
    params[156 + index] = p;
  }

  conv3x3_int8_stream(input.data(), params.data(), output.data());
  int mismatches = 0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int oc = 0; oc < CO; ++oc) {
        const int8_t actual = static_cast<int8_t>(
            static_cast<uint8_t>(output[y * W + x].range(8 * oc + 7, 8 * oc)));
        const int8_t expected = golden(y, x, oc);
        if (actual != expected) {
          if (mismatches < 8)
            std::printf("mismatch y=%d x=%d oc=%d actual=%d expected=%d\n",
                        y, x, oc, int(actual), int(expected));
          ++mismatches;
        }
      }
    }
  }
  std::printf("full_shape_outputs=%d mismatches=%d\n", H * W * CO,
              mismatches);
  return mismatches ? 1 : 0;
}
