// Reusable KV260 ARM BGR -> exact DPU INT8 NHWC letterbox boundary.
// OpenCV owns only the resize. Padding, RGB swap and quantization are fused
// into one write to the final DPU input slot. No model layer runs here.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

#include <opencv2/imgproc.hpp>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
constexpr int kSize = 640;
using Clock = std::chrono::steady_clock;

struct Context {
  explicit Context(int fix_point) : fix(fix_point) {
    const float scale = std::ldexp(1.0f, fix_point);
    for (int value = 0; value < 256; ++value) {
      const float normalized = static_cast<float>(value) * (1.0f / 255.0f);
      const long rounded = std::lrintf(normalized * scale);
      lut[value] = static_cast<int8_t>(std::clamp(rounded, -128L, 127L));
    }
  }
  int fix;
  int8_t lut[256]{};
  cv::Mat resized;
};

#if defined(__aarch64__)
// For 0 <= fix <= 7, round(uint8 * 2^fix / 255) has no half-way ties.
// floor((n + 127) / 255) equals ((v + (v >> 8) + 1) >> 8) for
// 0 <= v <= 32767. The latter has an exact 16-lane NEON implementation.
uint8x8_t quantize8(uint8x8_t pixels, uint8x8_t scale) {
  uint16x8_t value = vmlal_u8(vdupq_n_u16(127), pixels, scale);
  value = vaddq_u16(value, vshrq_n_u16(value, 8));
  value = vaddq_u16(value, vdupq_n_u16(1));
  return vmin_u8(vshrn_n_u16(value, 8), vdup_n_u8(127));
}
#endif

void quantize_rgb_row(const uint8_t* src, int8_t* dst, int width,
                      const Context& context) {
  int x = 0;
#if defined(__aarch64__)
  if (context.fix >= 0 && context.fix <= 7) {
    const uint8x8_t scale = vdup_n_u8(static_cast<uint8_t>(1 << context.fix));
    for (; x + 16 <= width; x += 16) {
      const uint8x16x3_t bgr = vld3q_u8(src + 3 * x);
      uint8x16x3_t rgb;
      rgb.val[0] = vcombine_u8(quantize8(vget_low_u8(bgr.val[2]), scale),
                               quantize8(vget_high_u8(bgr.val[2]), scale));
      rgb.val[1] = vcombine_u8(quantize8(vget_low_u8(bgr.val[1]), scale),
                               quantize8(vget_high_u8(bgr.val[1]), scale));
      rgb.val[2] = vcombine_u8(quantize8(vget_low_u8(bgr.val[0]), scale),
                               quantize8(vget_high_u8(bgr.val[0]), scale));
      vst3q_u8(reinterpret_cast<uint8_t*>(dst) + 3 * x, rgb);
    }
  }
#endif
  for (; x < width; ++x) {
    dst[3 * x + 0] = context.lut[src[3 * x + 2]];
    dst[3 * x + 1] = context.lut[src[3 * x + 1]];
    dst[3 * x + 2] = context.lut[src[3 * x + 0]];
  }
}

uint64_t ns(Clock::time_point a, Clock::time_point b) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}
}  // namespace

extern "C" {
struct LetterboxResult {
  double gain;
  int left;
  int top;
  int resized_width;
  int resized_height;
  uint64_t resize_ns;
  uint64_t pad_ns;
  uint64_t convert_ns;
};

void* kv260_lb_create(int fix_point) noexcept {
  if (fix_point < -31 || fix_point > 31) return nullptr;
  try {
    return new Context(fix_point);
  } catch (...) {
    return nullptr;
  }
}

void kv260_lb_destroy(void* opaque) noexcept { delete static_cast<Context*>(opaque); }

int kv260_lb_prepare(void* opaque, const uint8_t* bgr, int width, int height,
                     int row_stride_bytes, int8_t* output,
                     LetterboxResult* result) noexcept {
  if (!opaque || !bgr || !output || !result || width < 1 || height < 1 ||
      row_stride_bytes < width * 3) return -1;
  try {
    auto* context = static_cast<Context*>(opaque);
    const double gain = std::min(static_cast<double>(kSize) / height,
                                 static_cast<double>(kSize) / width);
    const int new_width = static_cast<int>(std::nearbyint(width * gain));
    const int new_height = static_cast<int>(std::nearbyint(height * gain));
    const double dw = (kSize - new_width) / 2.0;
    const double dh = (kSize - new_height) / 2.0;
    const int left = static_cast<int>(std::nearbyint(dw - 0.1));
    const int right = static_cast<int>(std::nearbyint(dw + 0.1));
    const int top = static_cast<int>(std::nearbyint(dh - 0.1));
    const int bottom = static_cast<int>(std::nearbyint(dh + 0.1));
    if (new_width < 1 || new_height < 1 || left < 0 || top < 0 ||
        left + new_width + right != kSize || top + new_height + bottom != kSize)
      return -2;

    const cv::Mat source(height, width, CV_8UC3, const_cast<uint8_t*>(bgr),
                         static_cast<size_t>(row_stride_bytes));
    const auto start = Clock::now();
    const cv::Mat* pixels = &source;
    if (width != new_width || height != new_height) {
      context->resized.create(new_height, new_width, CV_8UC3);
      cv::resize(source, context->resized, cv::Size(new_width, new_height),
                 0, 0, cv::INTER_LINEAR);
      pixels = &context->resized;
    }
    const auto resized_end = Clock::now();

    std::memset(output, static_cast<unsigned char>(context->lut[114]),
                static_cast<size_t>(kSize) * kSize * 3);
    const auto padded_end = Clock::now();
    for (int y = 0; y < new_height; ++y) {
      const auto* src = pixels->ptr<uint8_t>(y);
      auto* dst = output + (static_cast<size_t>(top + y) * kSize + left) * 3;
      quantize_rgb_row(src, dst, new_width, *context);
    }
    const auto converted_end = Clock::now();
    *result = LetterboxResult{gain, left, top, new_width, new_height,
                              ns(start, resized_end), ns(resized_end, padded_end),
                              ns(padded_end, converted_end)};
    return 0;
  } catch (const cv::Exception&) {
    return -3;
  } catch (...) {
    return -4;
  }
}
}  // extern "C"
