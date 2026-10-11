// Final-boundary decoder for six native INT8 YOLO26n raw heads on KV260 ARM.
// All neural operators have already run in the single DPU leaf. This routine
// only selects top scores, decodes ltrb boxes, filters and undoes LetterBox.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace {
constexpr int kAnchors = 8400;
constexpr int kClasses = 6;
constexpr int kMaximum = 300;
constexpr int kPairs = kMaximum * kClasses;
constexpr std::array<int, 3> kWidths{80, 40, 20};
constexpr std::array<int, 3> kStrides{8, 16, 32};
constexpr std::array<int, 3> kOffsets{0, 6400, 8000};

inline float sigmoid(float raw) {
  return 1.0f / (1.0f + std::exp(-std::clamp(raw, -80.0f, 80.0f)));
}

inline float clamp_image(float value, int size) {
  return std::clamp(value, 0.0f, static_cast<float>(size));
}

template <size_t N, class Better>
void first_sorted(std::array<int, N>& indices, int count, Better better) {
  if (count < static_cast<int>(N))
    std::nth_element(indices.begin(), indices.begin() + count, indices.end(), better);
  std::sort(indices.begin(), indices.begin() + count, better);
}
}  // namespace

extern "C" {
struct FastDecodeSixResult {
  float rows[kMaximum * 6];
  int count;
  // topk, below floor, invalid x at floor, invalid y at floor,
  // invalid either at floor, retained.
  int stats[6];
};

int kv260_decode_six_offsets(const int8_t* const class_maps[3],
                             const int8_t* const box_maps[3],
                             const int class_fix[3], const int box_fix[3],
                             int clip8, float confidence, int image_height,
                             int image_width, double gain, int left, int top,
                             const float score_logit_offsets[kClasses],
                             FastDecodeSixResult* out) noexcept {
  if (!class_maps || !box_maps || !class_fix || !box_fix || !out ||
      image_height < 1 || image_width < 1 || !(gain > 0.0) ||
      confidence < 0.0f || confidence > 1.0f) return -1;
  for (int s = 0; s < 3; ++s) {
    if (!class_maps[s] || !box_maps[s] || class_fix[s] < -31 ||
        class_fix[s] > 31 || box_fix[s] < -31 || box_fix[s] > 31)
      return -2;
  }
  if (score_logit_offsets) {
    for (int c = 0; c < kClasses; ++c)
      if (!std::isfinite(score_logit_offsets[c])) return -4;
  }
  try {
    std::array<float, kAnchors> anchor_scores{};
    std::array<int, kAnchors> anchor_indices{};
    const float class_scales[3] = {std::ldexp(1.0f, -class_fix[0]),
                                   std::ldexp(1.0f, -class_fix[1]),
                                   std::ldexp(1.0f, -class_fix[2])};
    const float box_scales[3] = {std::ldexp(1.0f, -box_fix[0]),
                                 std::ldexp(1.0f, -box_fix[1]),
                                 std::ldexp(1.0f, -box_fix[2])};
    const float shift = clip8 ? 8.0f : 0.0f;
    for (int scale = 0; scale < 3; ++scale) {
      const int size = kWidths[scale] * kWidths[scale];
      for (int local = 0; local < size; ++local) {
        const int8_t* logits = class_maps[scale] + local * kClasses;
        int8_t maximum = logits[0];
        for (int c = 1; c < kClasses; ++c)
          maximum = std::max(maximum, logits[c]);
        anchor_scores[kOffsets[scale] + local] =
            sigmoid(static_cast<float>(maximum) * class_scales[scale] - shift);
      }
    }
    std::iota(anchor_indices.begin(), anchor_indices.end(), 0);
    const auto better_anchor = [&](int a, int b) {
      if (anchor_scores[a] != anchor_scores[b])
        return anchor_scores[a] > anchor_scores[b];
      return a < b;
    };
    first_sorted(anchor_indices, kMaximum, better_anchor);

    std::array<float, kPairs> pair_scores{};
    std::array<int, kPairs> pair_indices{};
    for (int a = 0; a < kMaximum; ++a) {
      const int index = anchor_indices[a];
      const int scale = index < kOffsets[1] ? 0 : (index < kOffsets[2] ? 1 : 2);
      const int local = index - kOffsets[scale];
      const int8_t* logits = class_maps[scale] + local * kClasses;
      for (int c = 0; c < kClasses; ++c)
        pair_scores[a * kClasses + c] =
            sigmoid(static_cast<float>(logits[c]) * class_scales[scale] - shift);
    }
    std::iota(pair_indices.begin(), pair_indices.end(), 0);
    const auto better_pair = [&](int a, int b) {
      if (pair_scores[a] != pair_scores[b]) return pair_scores[a] > pair_scores[b];
      return a < b;
    };
    first_sorted(pair_indices, kMaximum, better_pair);

    *out = FastDecodeSixResult{};
    out->stats[0] = kMaximum;
    const float inverse_gain = 1.0f / static_cast<float>(gain);
    for (int selected = 0; selected < kMaximum; ++selected) {
      const int pair = pair_indices[selected];
      float score = pair_scores[pair];
      const int label = pair % kClasses;
      if (score_logit_offsets && score_logit_offsets[label] != 0.0f) {
        // The ordinary two-stage TopK above is deliberately unchanged.
        // Only the selected row's score is calibrated at the final boundary.
        const int anchor = anchor_indices[pair / kClasses];
        const int scale = anchor < kOffsets[1] ? 0 : (anchor < kOffsets[2] ? 1 : 2);
        const int local = anchor - kOffsets[scale];
        const int8_t raw_logit = class_maps[scale][local * kClasses + label];
        score = sigmoid(static_cast<float>(raw_logit) * class_scales[scale] - shift +
                        score_logit_offsets[label]);
      }
      if (score < confidence) {
        ++out->stats[1];
        continue;
      }
      const int anchor = anchor_indices[pair / kClasses];
      const int scale = anchor < kOffsets[1] ? 0 : (anchor < kOffsets[2] ? 1 : 2);
      const int local = anchor - kOffsets[scale];
      const int width = kWidths[scale];
      const int8_t* boxes = box_maps[scale] + 4 * local;
      const float center_x = static_cast<float>(local % width) + 0.5f;
      const float center_y = static_cast<float>(local / width) + 0.5f;
      const float stride = static_cast<float>(kStrides[scale]);
      const float x1 = (center_x - static_cast<float>(boxes[0]) * box_scales[scale]) * stride;
      const float y1 = (center_y - static_cast<float>(boxes[1]) * box_scales[scale]) * stride;
      const float x2 = (center_x + static_cast<float>(boxes[2]) * box_scales[scale]) * stride;
      const float y2 = (center_y + static_cast<float>(boxes[3]) * box_scales[scale]) * stride;
      const bool invalid_x = x2 < x1;
      const bool invalid_y = y2 < y1;
      out->stats[2] += invalid_x;
      out->stats[3] += invalid_y;
      if (invalid_x || invalid_y) {
        ++out->stats[4];
        continue;
      }
      float* row = out->rows + out->count * 6;
      row[0] = clamp_image((x1 - static_cast<float>(left)) * inverse_gain, image_width);
      row[1] = clamp_image((y1 - static_cast<float>(top)) * inverse_gain, image_height);
      row[2] = clamp_image((x2 - static_cast<float>(left)) * inverse_gain, image_width);
      row[3] = clamp_image((y2 - static_cast<float>(top)) * inverse_gain, image_height);
      row[4] = score;
      row[5] = static_cast<float>(label);
      ++out->count;
    }
    out->stats[5] = out->count;
    return 0;
  } catch (...) {
    return -3;
  }
}

int kv260_decode_six(const int8_t* const class_maps[3],
                     const int8_t* const box_maps[3],
                     const int class_fix[3], const int box_fix[3],
                     int clip8, float confidence, int image_height,
                     int image_width, double gain, int left, int top,
                     FastDecodeSixResult* out) noexcept {
  return kv260_decode_six_offsets(class_maps, box_maps, class_fix, box_fix,
                                  clip8, confidence, image_height, image_width,
                                  gain, left, top, nullptr, out);
}
}  // extern "C"
