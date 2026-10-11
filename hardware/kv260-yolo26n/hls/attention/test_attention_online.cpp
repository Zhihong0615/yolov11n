#include "attention_online.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

static void reference(const std::vector<float>& q, const std::vector<float>& k,
                      const std::vector<float>& v, std::vector<float>& out) {
  std::vector<double> score(psa::kTokens);
  for (int head = 0; head < psa::kHeads; ++head) {
    for (int query = 0; query < psa::kTokens; ++query) {
      double max_score = -std::numeric_limits<double>::infinity();
      for (int key = 0; key < psa::kTokens; ++key) {
        double dot = 0.0;
        for (int dim = 0; dim < psa::kQkDim; ++dim) {
          dot += static_cast<double>(q[(head * psa::kTokens + query) *
                                        psa::kQkDim + dim]) *
                 k[(head * psa::kQkDim + dim) * psa::kTokens + key];
        }
        score[key] = dot;
        max_score = std::max(max_score, dot);
      }
      double denominator = 0.0;
      for (int key = 0; key < psa::kTokens; ++key) {
        score[key] = std::exp(score[key] - max_score);
        denominator += score[key];
      }
      for (int dim = 0; dim < psa::kValueDim; ++dim) {
        double numerator = 0.0;
        for (int key = 0; key < psa::kTokens; ++key) {
          numerator +=
              score[key] * v[(head * psa::kValueDim + dim) * psa::kTokens + key];
        }
        out[(head * psa::kValueDim + dim) * psa::kTokens + query] =
            static_cast<float>(numerator / denominator);
      }
    }
  }
}

static bool run_case(int kind) {
  std::vector<float> q(psa::kQueryElements);
  std::vector<float> k(psa::kKeyElements);
  std::vector<float> v(psa::kValueElements);
  std::vector<float> actual(psa::kValueElements, std::nanf(""));
  std::vector<float> expected(psa::kValueElements, std::nanf(""));
  std::mt19937 gen(49253 + kind);
  std::uniform_real_distribution<float> small(-0.4f, 0.4f);
  for (size_t i = 0; i < q.size(); ++i) q[i] = small(gen);
  for (size_t i = 0; i < k.size(); ++i) k[i] = small(gen);
  for (size_t i = 0; i < v.size(); ++i) v[i] = small(gen);
  if (kind == 1) {
    std::fill(q.begin(), q.end(), 0.0f);
    std::fill(k.begin(), k.end(), 0.0f);
  } else if (kind == 2) {
    std::fill(q.begin(), q.end(), 1.0f);
    std::fill(k.begin(), k.end(), 1.0f);
  } else if (kind == 3) {
    std::fill(q.begin(), q.end(), 0.0f);
    std::fill(k.begin(), k.end(), 0.0f);
    for (int head = 0; head < psa::kHeads; ++head) {
      for (int query = 0; query < psa::kTokens; ++query) {
        q[(head * psa::kTokens + query) * psa::kQkDim] = 30.0f;
      }
      for (int key = 0; key < psa::kTokens; ++key) {
        k[head * psa::kQkDim * psa::kTokens + key] =
            key % 2 ? -30.0f : 30.0f;
      }
    }
  } else if (kind == 4) {
    std::fill(q.begin(), q.end(), 0.0f);
    std::fill(k.begin(), k.end(), 0.0f);
    for (int head = 0; head < psa::kHeads; ++head) {
      for (int query = 0; query < psa::kTokens; ++query) {
        q[(head * psa::kTokens + query) * psa::kQkDim] = -50.0f;
      }
      for (int key = 0; key < psa::kTokens; ++key) {
        k[head * psa::kQkDim * psa::kTokens + key] =
            key == psa::kTokens - 1 ? -50.0f : 50.0f;
      }
    }
  }

  reference(q, k, v, expected);
#if defined(TEST_TREE8_VALUE4)
  psa_attention_cached_tree8_value4(q.data(), k.data(), v.data(), actual.data());
#elif defined(TEST_TREE8)
  psa_attention_cached_tree8(q.data(), k.data(), v.data(), actual.data());
#elif defined(TEST_TREE)
  psa_attention_cached_tree(q.data(), k.data(), v.data(), actual.data());
#elif defined(TEST_CACHED)
  psa_attention_cached(q.data(), k.data(), v.data(), actual.data());
#else
  psa_attention_online(q.data(), k.data(), v.data(), actual.data());
#endif
  double max_abs = 0.0;
  double max_rel = 0.0;
  int invalid = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      ++invalid;
      continue;
    }
    const double err = std::abs(static_cast<double>(actual[i]) - expected[i]);
    max_abs = std::max(max_abs, err);
    max_rel = std::max(max_rel, err / std::max(1e-3, std::abs(
                                                   static_cast<double>(expected[i]))));
  }
  std::printf("case=%d outputs=%zu invalid=%d max_abs=%.9g max_rel=%.9g\n",
              kind, actual.size(), invalid, max_abs, max_rel);
  std::fflush(stdout);
  return invalid == 0 && max_abs <= 5e-4;
}

int main() {
  bool okay = true;
  for (int kind = 0; kind < 5; ++kind) okay = run_case(kind) && okay;
  std::puts(okay ? "PASS" : "FAIL");
  return okay ? EXIT_SUCCESS : EXIT_FAILURE;
}
