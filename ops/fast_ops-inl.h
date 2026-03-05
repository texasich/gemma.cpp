// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Include guard for non-SIMD code.
#ifndef THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_INL_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_INL_H_

#include <stddef.h>

#include "ops/ops.h"
#include "util/threading_context.h"
#include "util/zones.h"
#include "hwy/base.h"

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_INL_H_

// Include guard for (potentially) SIMD code.
#if defined(THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_OPS_FAST_OPS_TOGGLE
#endif

#include "compression/compress-inl.h"
#include "hwy/contrib/math/fast_math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// We use the tanh approximation for gelu (also used in training).
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//         = 0.5 * x * (1 + tanh(x * (sqrt(2/π) + sqrt(2/π) * 0.044715 * x^2)))
//         = 0.5 * x * (1 + tanh(x * (0.79788 + 0.035677 * x^2)))
//         = x * (0.5 + 0.5 * tanh(x * (0.79788 + 0.035677 * x^2))))
//
// This uses hn::FastTanh from
// third_party/highway/hwy/contrib/math/fast_math-inl.h
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> FastGelu(D d, hn::Vec<D> v) {
  const hn::Vec<D> kMul = hn::Set(d, 0.03567740813636141f);
  const hn::Vec<D> kSqrt2OverPi = hn::Set(d, 0.797884560804236f);
  const hn::Vec<D> kHalf = hn::Set(d, 0.5f);

  const hn::Vec<D> v2 = hn::Mul(v, v);
  const hn::Vec<D> arg = hn::Mul(v, hn::MulAdd(kMul, v2, kSqrt2OverPi));
  const hn::Vec<D> cdf = hn::MulAdd(kHalf, hn::FastTanh(d, arg), kHalf);
  return hn::Mul(v, cdf);
}

// Fast approximation of sigmoid(x) = 1 / (1 + exp(-x))
// Derived from FastTanh by substituting x/2.
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> FastSigmoid(D d, hn::Vec<D> val) {
  using T = hn::TFromD<D>;

  // Abs(val) and preserve sign for later for symmetric rational approximation
  auto y = hn::Abs(val);

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  hn::Vec<D> a, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4)) {
    // Coefficients for P(y/2) ~ index using CF algo
    const auto k0 = hn::Set(d, static_cast<T>(-0.1145426548151546));
    const auto k1 = hn::Set(d, static_cast<T>(3.4556654973457404));
    const auto k2 = hn::Set(d, static_cast<T>(-0.6278480784875462));
    const auto k3 = hn::Set(d, static_cast<T>(0.04331384030062471));

    // Index calculation: idx = P(y/2)
    // Estrin's scheme
    // k0 + y * k1 + y^2 * (k2 + y * k3)
    const auto y2 = hn::Mul(y, y);
    const auto p01 = hn::MulAdd(k1, y, k0);
    const auto p23 = hn::MulAdd(k3, y, k2);
    auto idx_poly = hn::MulAdd(y2, p23, p01);

    // Convert to integer index
    using DI = hn::RebindToSigned<D>;
    auto idx_i = hn::ConvertTo(DI(), idx_poly);

    // Clamp index to 7
    idx_i = hn::Min(idx_i, hn::Set(DI(), 7));

    HWY_ALIGN static constexpr T arr_a[8] = {
        static_cast<T>(-1435.326650329326),
        static_cast<T>(-96.9456723845743),
        static_cast<T>(-18.628915468855695),
        static_cast<T>(-5.90191111348809),
        static_cast<T>(-2.356433838423728),
        static_cast<T>(-1.0464246812594584),
        static_cast<T>(-0.4801959711368016),
        static_cast<T>(-0.2132727031175401)};
    HWY_ALIGN static constexpr T arr_c[8] = {static_cast<T>(-316.5640994591445),
                                            static_cast<T>(-49.14374182730444),
                                            static_cast<T>(-15.69264419046708),
                                            static_cast<T>(-6.949871926785674),
                                            static_cast<T>(-3.513259738716989),
                                            static_cast<T>(-1.839177585570145),
                                            static_cast<T>(-0.9298342163526662),
                                            static_cast<T>(-0.426230503963466)};
    HWY_ALIGN static constexpr T arr_d[8] = {
        static_cast<T>(-5676.517069241468), static_cast<T>(-363.0662559912978),
        static_cast<T>(-60.61589604370584), static_cast<T>(-14.306713103378062),
        static_cast<T>(-2.725237489187118), static_cast<T>(0.7890752292798894),
        static_cast<T>(1.8089988725725492), static_cast<T>(1.9956027601801545)};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      auto idx = hn::IndicesFromVec(d, idx_i);
      hn::CappedTag<T, 8> d8;
      a = hn::TableLookupLanes(hn::ResizeBitCast(d, hn::Load(d8, arr_a)), idx);
      c = hn::TableLookupLanes(hn::ResizeBitCast(d, hn::Load(d8, arr_c)), idx);
      d_coef =
          hn::TableLookupLanes(hn::ResizeBitCast(d, hn::Load(d8, arr_d)), idx);
    } else {
      auto idx = hn::IndicesFromVec(d, idx_i);
      hn::FixedTag<T, 4> d4;
      a = hn::TwoTablesLookupLanes(d, hn::Load(d4, arr_a),
                                   hn::Load(d4, arr_a + 4), idx);
      c = hn::TwoTablesLookupLanes(d, hn::Load(d4, arr_c),
                                   hn::Load(d4, arr_c + 4), idx);
      d_coef = hn::TwoTablesLookupLanes(d, hn::Load(d4, arr_d),
                                        hn::Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Thresholds for intervals
    const auto t0 = hn::Set(d, static_cast<T>(0.3434497447432422));
    const auto t1 = hn::Set(d, static_cast<T>(0.6955976007186494));
    const auto t2 = hn::Set(d, static_cast<T>(1.1068914127668934));
    const auto t3 = hn::Set(d, static_cast<T>(1.608648163822941));
    const auto t4 = hn::Set(d, static_cast<T>(2.269039121646492));
    const auto t5 = hn::Set(d, static_cast<T>(3.288402547357102));
    const auto t6 = hn::Set(d, static_cast<T>(5.271780018997146));

    // Start with highest index (7)
    a = hn::Set(d, static_cast<T>(-0.2132727031175401));
    c = hn::Set(d, static_cast<T>(-0.426230503963466));
    d_coef = hn::Set(d, static_cast<T>(1.9956027601801545));

    // If y < t6 (idx 6)
    auto mask = hn::Lt(y, t6);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-0.4801959711368016)),
                       a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-0.9298342163526662)),
                       c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(1.8089988725725492)), d_coef);

    // If y < t5 (idx 5)
    mask = hn::Lt(y, t5);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-1.0464246812594584)),
                       a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-1.839177585570145)), c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(0.7890752292798894)), d_coef);

    // If y < t4 (idx 4)
    mask = hn::Lt(y, t4);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-2.356433838423728)), a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-3.513259738716989)), c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(-2.725237489187118)), d_coef);

    // If y < t3 (idx 3)
    mask = hn::Lt(y, t3);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-5.90191111348809)), a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-6.949871926785674)), c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(-14.306713103378062)), d_coef);

    // If y < t2 (idx 2)
    mask = hn::Lt(y, t2);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-18.628915468855695)),
                       a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-15.69264419046708)), c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(-60.61589604370584)), d_coef);

    // If y < t1 (idx 1)
    mask = hn::Lt(y, t1);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-96.9456723845743)), a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-49.14374182730444)), c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(-363.0662559912978)), d_coef);

    // If y < t0 (idx 0)
    mask = hn::Lt(y, t0);
    a = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-1435.326650329326)), a);
    c = hn::IfThenElse(mask, hn::Set(d, static_cast<T>(-316.5640994591445)), c);
    d_coef = hn::IfThenElse(
        mask, hn::Set(d, static_cast<T>(-5676.517069241468)), d_coef);
  }

  // Math: 0.5 * tanh(y/2) = (ay + 1.0)/(cy + d_coef)
  auto num = hn::MulAdd(a, y, hn::Set(d, static_cast<T>(1.0)));
  auto den = hn::MulAdd(c, y, d_coef);

  auto approx = hn::Div(num, den);

  const auto half = hn::Set(d, static_cast<T>(0.5));
  // Clamp the approx value to 0.5
  approx = hn::Min(approx, half);
  // sigmoid(x) = 0.5 + sign(x) * (0.5 * tanh(|x|/2))
  return hn::Add(half, hn::CopySign(approx, val));
}

// Activation already has a profiler zone.
template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void FastGelu(T* HWY_RESTRICT x,
                                                   size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  DecompressAndCompressInplace(
      DF(), x, size, [](DF d, VF v) HWY_ATTR -> VF { return FastGelu(d, v); });
}

template <typename T>
static HWY_NOINLINE HWY_MAYBE_UNUSED void FastSigmoid(T* HWY_RESTRICT x,
                                                      size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  using VF = hn::Vec<DF>;
  DecompressAndCompressInplace(DF(), x, size, [](DF d, VF v) HWY_ATTR -> VF {
    return FastSigmoid(d, v);
  });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
