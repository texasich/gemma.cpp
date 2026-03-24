#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "compression/compress.h"
#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/kv_cache.h"
#include "ops/matmul.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// Note: HWY_DISABLED_TARGETS needs to be defined the same everywhere.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "util/basics.h"
#include "util/mat.h"
#include "util/threading_context.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/tiled_attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "gemma/attention.h"
#include "gemma/flash_attention.h"  // includes highway.h
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

static HWY_INLINE void MergeOnlineSoftmax(
    const float* HWY_RESTRICT other_att_out, const float other_softmax_max,
    const float other_softmax_d, int qkv_dim,
    float* HWY_RESTRICT accumulator_att_out, float& accumulator_softmax_max,
    float& accumulator_softmax_d) {
  if (other_softmax_d == 0.0f) {
    return;
  }
  if (accumulator_softmax_d == 0.0f) {
    memcpy(accumulator_att_out, other_att_out,
           qkv_dim * sizeof(*accumulator_att_out));
    accumulator_softmax_max = other_softmax_max;
    accumulator_softmax_d = other_softmax_d;
    return;
  }
  const float m_new = std::max(accumulator_softmax_max, other_softmax_max);
  const float exp_l = std::exp(accumulator_softmax_max - m_new);
  const float exp_r = std::exp(other_softmax_max - m_new);
  const float d_new = accumulator_softmax_d * exp_l + other_softmax_d * exp_r;
  const float d_new_inv = 1.0f / d_new;
  const float c1 = accumulator_softmax_d * exp_l * d_new_inv;
  const float c2 = other_softmax_d * exp_r * d_new_inv;
  MulByConst(c1, accumulator_att_out, qkv_dim);
  MulByConstAndAdd(c2, other_att_out, accumulator_att_out, qkv_dim);
  accumulator_softmax_max = m_new;
  accumulator_softmax_d = d_new;
}

template <typename T>
T AbsMaxOfSpan(hwy::Span<const T> span) {
  hn::ScalableTag<T> dt;
  using VT = hn::Vec<decltype(dt)>;
  VT max_vec = hn::Set(dt, 0.0f);
  const size_t lanes = hn::Lanes(dt);
  size_t i = 0;
  // Process full vectors using LoadU.
  for (; i + lanes <= span.size(); i += lanes) {
    const VT vec = hn::Abs(hn::LoadU(dt, span.data() + i));
    max_vec = hn::Max(max_vec, vec);
  }
  // Process remaining elements using LoadN.
  const size_t remaining = span.size() - i;
  if (HWY_UNLIKELY(remaining != 0)) {
    const VT vec = hn::Abs(hn::LoadN(dt, span.data() + i, remaining));
    max_vec = hn::Max(max_vec, vec);
  }
  return hn::ReduceMax(dt, max_vec);
}

// Forked from ComputeQKV. But it stores the K/V in the tiled format
// KV_T is type stored in the KV cache (typically float or BF16).
template <typename KV_T>
static HWY_INLINE void ComputeQKVTransposedTile(
    size_t num_tokens, const size_t layer_idx, const LayerWeightsPtrs& layer,
    AttentionImpl attention_impl, AttentionActivationsPtrs& activations,
    const QBatch& qbatch, const int flags, MatMulEnv& env) {
  PROFILER_ZONE("Gen.Attention.QKVTiled");
  const hwy::Divisor div_qbatch(qbatch.Size());
  const size_t num_interleaved = num_tokens * div_qbatch.GetDivisor();
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t kv_heads = layer_config.kv_heads;

  // The original qkv_einsum_w has shape [(heads + kv_heads * 2), qkv_dim,
  // model_dim], which we reshaped to (heads + kv_heads * 2) * qkv_dim rows.
  // This computes Q and stores it in activations.q.
  // The original qkv_einsum_w has shape [(heads + kv_heads * 2), qkv_dim,
  // model_dim], which we reshaped to (heads + kv_heads * 2) * qkv_dim rows.
  // This computes Q and stores it in activations.q.
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w1,
             /*add=*/nullptr, env, activations.q);

  // Compute the combined KV output from pre_att_rms_out.
  // The output shape is [num_interleaved, kv_heads * 2 * qkv_dim].
  const size_t kv_out_cols = kv_heads * 2 * qkv_dim;
  hwy::AlignedFreeUniquePtr<float[]> kv_out_mem =
      hwy::AllocateAligned<float>(num_interleaved * kv_out_cols);
  float* kv_out_data = kv_out_mem.get();
  MatPtrT<float> kv_out_mat("kv_out", Extents2D(num_interleaved, kv_out_cols));
  kv_out_mat.SetPtr(kv_out_data, kv_out_cols);
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2,
             /*add=*/nullptr, env, kv_out_mat);

  // Apply positional encodings and store K/V in tiled format.
  hwy::Divisor div_kv_heads(kv_heads);

  bool is_transposed_qs =
      attention_impl == AttentionImpl::kFlashTransposedQsBF16
      || attention_impl == AttentionImpl::kFlashTransposedQsInt16;

  hn::ScalableTag<float> df;
  static hwy::Divisor tile_size_divisor(KVCache::kTileSize);
  ParallelFor(
      Parallelism::kFlat, kv_heads * qbatch.Size(), env.ctx,
      /*cluster_idx=*/0, Callers::kAttComputeQKV,
      [&](size_t task, size_t worker) HWY_ATTR {
        const size_t kv_head = div_kv_heads.Remainder(task);
        const size_t query_idx = div_kv_heads.Divide(task);
        CompressPerThread tls;
        size_t current_token_idx = 0;
        float* k_tile_vec = activations.k_tile_vec.Row(task);
        float* v_tile_vec = activations.v_tile_vec.Row(task);
        HWY_ALIGN float k_f32[kMaxQKVDim];
        const size_t start_pos = qbatch.Pos(query_idx);
        const bool is_global_layer =
            activations.config.IsGlobalLayer(layer_idx);
        std::vector<MatPtr> kv_ptrs =
            qbatch.KV(query_idx).cache->GetPointers(
                layer_idx, kv_head, kv_heads, start_pos, is_global_layer);
        size_t tile_offset = 0;
        if (!is_global_layer) {
          tile_offset = start_pos / KVCache::kTileSize;
        }

        while (current_token_idx < num_tokens) {
          const size_t pos = start_pos + current_token_idx;
          const size_t pos_mod = activations.div_seq_len.Remainder(pos);
          const size_t tile_idx = tile_size_divisor.Divide(pos_mod);
          const size_t relative_tile_idx = tile_idx - tile_offset;
          KV_T* tile_ptr;
          int kv_ptr_idx = 0;
          size_t absolute_rows = 0;
          while (absolute_rows + kv_ptrs[kv_ptr_idx].Rows() <=
                 relative_tile_idx) {
            absolute_rows += kv_ptrs[kv_ptr_idx].Rows();
            kv_ptr_idx++;
          }
          tile_ptr = HWY_RCAST_ALIGNED(
              KV_T*,
              kv_ptrs[kv_ptr_idx].RowBytes(relative_tile_idx - absolute_rows));
          PackedSpan<KV_T> tile_packed_span{tile_ptr,
                                            2 * qkv_dim * KVCache::kTileSize};

          DecompressAndZeroPad(df, tile_packed_span, 0, k_tile_vec,
                               qkv_dim * KVCache::kTileSize);
          DecompressAndZeroPad(df, tile_packed_span,
                               qkv_dim * KVCache::kTileSize, v_tile_vec,
                               qkv_dim * KVCache::kTileSize);

          size_t token_in_tile_idx = current_token_idx;
          while (token_in_tile_idx < num_tokens) {
            const size_t current_pos =
                qbatch.Pos(query_idx) + token_in_tile_idx;
            const size_t current_pos_mod =
                activations.div_seq_len.Remainder(current_pos);
            if (tile_size_divisor.Divide(current_pos_mod) != tile_idx) {
              break;  // Moved to next tile
            }

            const float* kv_row =
                kv_out_data +
                (token_in_tile_idx * qbatch.Size() + query_idx) * kv_out_cols;
            const float* k_values = kv_row + kv_head * 2 * qkv_dim;
            const float* v_values = kv_row + kv_head * 2 * qkv_dim + qkv_dim;
            hwy::CopyBytes(k_values, k_f32, qkv_dim * sizeof(float));
            if (layer.key_norm_scale.HasPtr()) {
              CallUpcasted(&layer.key_norm_scale, [&](const auto* weights_t) {
                RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, k_f32,
                               qkv_dim, env.ctx, worker);
              });
            }
            PositionalEncodingQK(
                k_f32, layer_idx, activations, env.ctx, worker,
                current_pos ,
                /*mul=*/1.0f);

            const size_t in_tile_idx = current_pos_mod % KVCache::kTileSize;
            // `v_cache_values` is a pointer to the V data that will be
            // compressed and stored in the KV cache. By default, it points to
            // the raw `v_values`.
            const float* v_cache_values = v_values;
            // `v_buf` is a temporary buffer used only when quantizing V values
            // to int8_t.
            HWY_ALIGN float v_buf[kMaxQKVDim];

            if constexpr (IsInt8<KV_T>()) {
              BF16* scales_ptr = HWY_RCAST_ALIGNED(
                  BF16*, tile_ptr + 2 * qkv_dim * KVCache::kTileSize);

              auto scale_and_store = [&](float* values, int dim,
                                         size_t scale_idx) HWY_ATTR {
                const float max_abs =
                    AbsMaxOfSpan(hwy::Span<const float>(values, dim));
                float scale = max_abs / 127.0f;
                if (scale == 0.0f) scale = 1.0f;
                scales_ptr[scale_idx] = hwy::ConvertScalarTo<BF16>(scale);
                const float inv_scale = 1.0f / scale;
                const hn::Vec<decltype(df)> v_inv_scale =
                    hn::Set(df, inv_scale);
                const size_t lanes = hn::Lanes(df);
                size_t i = 0;
                for (; i + lanes <= dim; i += lanes) {
                  hn::StoreU(hn::Mul(hn::LoadU(df, values + i), v_inv_scale),
                             df, values + i);
                }
                if (HWY_UNLIKELY(i < dim)) {
                  hn::StoreN(
                      hn::Mul(hn::LoadN(df, values + i, dim - i), v_inv_scale),
                      df, values + i, dim - i);
                }
              };

              // K Scaling
              scale_and_store(k_f32, qkv_dim, in_tile_idx);

              // V Scaling: Copy `v_values` to `v_buf`, scale `v_buf` in-place,
              // and then update `v_cache_values` to point to `v_buf`.
              hwy::CopyBytes(v_values, v_buf, qkv_dim * sizeof(float));
              scale_and_store(v_buf, qkv_dim, KVCache::kTileSize + in_tile_idx);
              v_cache_values = v_buf;
            }

            if (is_transposed_qs) {
              const int in_tile_idx_mod_2 = in_tile_idx % 2;
              for (int dim = 0; dim < qkv_dim; dim += 2) {
                const int dim_mod_2 = dim % 2;
                // Pack k's in pairs in preparation for BF16 dot product.
                // See flash_attention.cc
                // QDotKTilexUpTo4TransposedKDoubleWidthBF16
                k_tile_vec[(dim - dim_mod_2) * KVCache::kTileSize +
                           in_tile_idx * 2] = k_f32[dim];
                k_tile_vec[(dim - dim_mod_2) * KVCache::kTileSize +
                           in_tile_idx * 2 + 1] = k_f32[dim + 1];
                // Pack v's in pairs
                v_tile_vec[(in_tile_idx - in_tile_idx_mod_2) * qkv_dim +
                           dim * 2 + in_tile_idx_mod_2] = v_cache_values[dim];
                v_tile_vec[(in_tile_idx - in_tile_idx_mod_2) * qkv_dim +
                           (dim + 1) * 2 + in_tile_idx_mod_2] =
                    v_cache_values[dim + 1];
              }

            } else {
              for (int i = 0; i < qkv_dim; ++i) {
                k_tile_vec[i * KVCache::kTileSize + in_tile_idx] = k_f32[i];
              }
              Compress(v_cache_values, qkv_dim, tls, tile_packed_span,
                       qkv_dim * (KVCache::kTileSize + in_tile_idx));
            }

            token_in_tile_idx++;
          }
          Compress(k_tile_vec, qkv_dim * KVCache::kTileSize, tls,
                   tile_packed_span, 0);
          if (is_transposed_qs) {
            Compress(v_tile_vec, qkv_dim * KVCache::kTileSize, tls,
                     tile_packed_span, qkv_dim * KVCache::kTileSize);
          }
          current_token_idx = token_in_tile_idx;
        }
      });
}

// Transposes queries
// Input: vector of pointers to subsequent queries. (allows for arbitrary
// strides)
// qkv_dim: dimension of query
// allocator: aligned allocator to use for temporary storage
//
// Output: Pointer to contiguous memory with shape (qkv_dim,
// queries.size())
void TransposeStridedQueries(
    hwy::Span<float*> queries, int qkv_dim,
    hwy::Span<float> transposed_queries) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  using DI = hn::ScalableTag<int32_t>;
  const DI di;
  using VI = hn::Vec<DI>;
  const size_t lanes = hn::Lanes(df);
  const size_t num_queries = queries.size();
  const size_t num_queries_rounded_up = hwy::RoundUpTo(num_queries, lanes);
  std::vector<int32_t, hwy::AlignedAllocator<int32_t>> query_offsets(
      num_queries_rounded_up);
  for (size_t i = 0; i < num_queries; ++i) {
    query_offsets[i] = queries[i] - queries[0];
  }
  for (size_t i = num_queries; i < num_queries_rounded_up; ++i) {
    // last offset is the same so gather doesn't read out of bounds
    query_offsets[i] = query_offsets[num_queries - 1];
  }

  for (size_t i = 0; i < qkv_dim; i++) {
    size_t j = 0;
    if (num_queries >= lanes) {
      for (; j <= num_queries-lanes; j += lanes) {
        const VI offsets = hn::LoadU(di, query_offsets.data() + j);
        VF x = hn::GatherIndex(df, queries[0] + i, offsets);
        hn::StoreU(x, df, transposed_queries.data() + i * num_queries + j);
      }
    }
    if (j < num_queries) {
      const VI offsets = hn::LoadU(di, query_offsets.data() + j);
      VF x = hn::GatherIndex(df, queries[0] + i, offsets);
      hn::StoreN(x, df, transposed_queries.data() + i * num_queries + j,
                 num_queries - j);
    }
  }
}

std::pair<AlignedFloatVector, std::vector<float*>> TransposeQueriesToGroupsOf4(
    hwy::Span<float*> queries_ptrs, int qkv_dim) {
  int num_queries = queries_ptrs.size();
  int num_groups = hwy::DivCeil(num_queries, 4);
  AlignedFloatVector transposed_queries(num_groups * 4 * qkv_dim);
  std::vector<float*> transposed_queries_ptrs;
  for (int group_idx = 0; group_idx < num_groups; ++group_idx){
    int group_size = std::min(4, num_queries - group_idx * 4);
    transposed_queries_ptrs.push_back(transposed_queries.data() +
                                      group_idx * qkv_dim * 4);
    TransposeStridedQueries(
        hwy::Span<float*>(queries_ptrs.data() + group_idx * 4,
                          group_size),
        qkv_dim,
        hwy::Span<float>(transposed_queries_ptrs.back(), qkv_dim * group_size));
  }
  return std::make_pair(std::move(transposed_queries),
                        std::move(transposed_queries_ptrs));
}

template <typename OutT>
static HWY_INLINE void TransposeStridedQueriesBF16orInt16(
    hwy::Span<const float*> queries, int qkv_dim,
    hwy::Span<OutT> transposed_queries, hwy::Span<float> q_scales) {
  namespace hn = hwy::HWY_NAMESPACE;
  using DF = hn::ScalableTag<float>;
  const DF df;
  using VF = hn::Vec<DF>;
  // doubles to avoid moving between int/float domains when gathering
  using DF64 = hn::ScalableTag<double>;
  const DF64 dd64;
  using DI64 = hn::ScalableTag<int64_t>;
  const DI64 di64;
  using VI64 = hn::Vec<DI64>;
  auto d_out = hn::Rebind<OutT, decltype(df)>();
  const size_t lanes = hn::Lanes(df);
  const size_t half_lanes = lanes / 2;
  const size_t num_queries = queries.size();
  const size_t num_numbers_to_gather = num_queries * 2;
  const size_t num_queries_rounded_up = hwy::RoundUpTo(num_queries, half_lanes);
  const size_t num_scales_rounded_up =
      hwy::RoundUpTo(num_numbers_to_gather, lanes);

  // We store scales twice so we will be able to just load them without a need
  // to duplicate for multiplication
  AlignedFloatVector inverted_q_scales_doubled(num_scales_rounded_up);

  if constexpr (IsInt16<OutT>()) {
    // compute microscales
    for (size_t i = 0; i < num_queries; ++i) {
      float max_abs = AbsMaxOfSpan(hwy::Span<const float>(queries[i], qkv_dim));
      float scale = max_abs == 0.0f ? 1.0f : 32767.0f / max_abs;
      inverted_q_scales_doubled[2 * i] = scale;
      inverted_q_scales_doubled[2 * i + 1] = scale;
      q_scales[i] = 1.0f / scale;
    }
  }

  std::vector<int64_t, hwy::AlignedAllocator<int64_t>> query_offsets(
      num_queries_rounded_up);
  for (size_t i = 0; i < num_queries; ++i) {
    query_offsets[i] = (queries[i] - queries[0]) / 2;
  }
  for (size_t i = num_queries; i < num_queries_rounded_up; ++i) {
    // last offset is the same so gather doesn't read out of bounds
    query_offsets[i] = query_offsets[num_queries > 0 ? num_queries - 1 : 0];
  }

  const double* queries_0_double = HWY_RCAST_ALIGNED(const double*, queries[0]);

  // Lambda to handle the scaling and demotion for Int16 types.
  auto process_values = [&]() HWY_ATTR {
    if constexpr (IsInt16<OutT>()) {
      return [&](VF& x, size_t j) HWY_ATTR {
        VF scales = hn::Load(df, inverted_q_scales_doubled.data() + j * 2);
        x = hn::Mul(x, scales);
        return hn::DemoteTo(d_out, hn::NearestInt(x));
      };
    } else {
      return [&](VF& x, size_t j) HWY_ATTR { return hn::DemoteTo(d_out, x); };
    }
  }();

  for (size_t i = 0; i < qkv_dim; i += 2) {
    size_t j = 0;
    if (num_queries >= half_lanes) {
      for (; j <= num_queries - half_lanes; j += half_lanes) {
        const VI64 offsets = hn::LoadU(di64, query_offsets.data() + j);
        auto x64 = hn::GatherIndex(dd64, queries_0_double + i / 2, offsets);
        VF x = hn::BitCast(df, x64);
        if constexpr (IsInt16<OutT>()) {
          auto demoted = process_values(x, j);
          hn::Store(demoted, d_out,
                    transposed_queries.data() + i * num_queries + j * 2);
        } else if constexpr (IsBF16<OutT>()) {
          auto demoted = hn::DemoteTo(d_out, x);
          hn::Store(demoted, d_out,
                    transposed_queries.data() + i * num_queries + j * 2);
        } else {
          static_assert(false, "Unsupported type");
        }
      }
    }
    if (j < num_queries) {
      const VI64 offsets = hn::LoadU(di64, query_offsets.data() + j);
      auto x64 = hn::GatherIndex(dd64, queries_0_double + i / 2, offsets);
      VF x = hn::BitCast(df, x64);
      if constexpr (IsInt16<OutT>()) {
        auto demoted = process_values(x, j);
        hn::StoreN(demoted, d_out,
                   transposed_queries.data() + i * num_queries + j * 2,
                   num_numbers_to_gather - j * 2);
      } else if constexpr (IsBF16<OutT>()) {
        auto demoted = hn::DemoteTo(d_out, x);
        hn::StoreN(demoted, d_out,
                   transposed_queries.data() + i * num_queries + j * 2,
                   num_numbers_to_gather - j * 2);
      } else {
        static_assert(false, "Unsupported type");
      }
    }
  }
}

// Transposed queries data, vector of pointers to transposed queries, vector of
// scales
template <typename OutT>
std::tuple<std::vector<OutT, hwy::AlignedAllocator<OutT>>, std::vector<OutT*>,
           AlignedFloatVector>
TransposeQueriesToGroupsOfNBF16orInt16(hwy::Span<float*> queries_ptrs,
                                       int qkv_dim, size_t group_size) {
  size_t num_queries = queries_ptrs.size();
  size_t num_groups = hwy::DivCeil(num_queries, group_size);
  std::vector<OutT, hwy::AlignedAllocator<OutT>> transposed_queries(
      num_groups * group_size * qkv_dim);
  std::vector<OutT*> transposed_queries_ptrs;
  AlignedFloatVector q_scales(num_groups * 4);
  for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    size_t current_group_size =
        std::min(group_size, num_queries - group_idx * group_size);
    transposed_queries_ptrs.push_back(transposed_queries.data() +
                                      group_idx * qkv_dim * group_size);
    TransposeStridedQueriesBF16orInt16(
        hwy::Span<const float*>(
            const_cast<const float**>(queries_ptrs.data() +
                                      group_idx * group_size),
            current_group_size),
        qkv_dim,
        hwy::Span<OutT>(transposed_queries_ptrs.back(),
                        qkv_dim * current_group_size),
        hwy::Span<float>(q_scales.data() + group_idx * group_size,
                         current_group_size));
  }
  return std::make_tuple(std::move(transposed_queries),
                         std::move(transposed_queries_ptrs),
                         std::move(q_scales));
}

std::pair<AlignedBF16Vector, std::vector<BF16*>>
TransposeTransposedQueriesAndPackIntoBF16(hwy::Span<float*> queries_ptrs,
                                          int qkv_dim, int num_queries) {
  constexpr int kMaxGroupSize = 4;
  int num_groups = queries_ptrs.size();
  AlignedBF16Vector transposed_queries(num_groups * kMaxGroupSize * qkv_dim);
  std::vector<BF16*> transposed_queries_ptrs;
  transposed_queries_ptrs.reserve(num_groups);
  for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
    int group_size =
        std::min(kMaxGroupSize, num_queries - group_idx * kMaxGroupSize);
    transposed_queries_ptrs.push_back(transposed_queries.data() +
                                      group_idx * qkv_dim * kMaxGroupSize);
    for (int dim_idx = 0; dim_idx < qkv_dim; dim_idx += 2) {
      for (int query_idx = 0; query_idx < group_size; ++query_idx) {
        transposed_queries_ptrs.back()[dim_idx * group_size + query_idx * 2] =
            hwy::ConvertScalarTo<BF16>(
                queries_ptrs[group_idx][dim_idx * group_size + query_idx]);
        transposed_queries_ptrs
            .back()[dim_idx * group_size + query_idx * 2 + 1] =
            hwy::ConvertScalarTo<BF16>(
                queries_ptrs[group_idx]
                            [(dim_idx + 1) * group_size + query_idx]);
      }
    }
  }
  return std::make_pair(std::move(transposed_queries),
                        std::move(transposed_queries_ptrs));
}

template <typename T>
static HWY_INLINE void MaybeResizeMatStorage(MatStorageT<T>& mat_storage,
                                             int rows, int cols,
                                             const char* name,
                                             const Allocator& allocator) {
  if (mat_storage.Rows() != rows || mat_storage.Cols() != cols) {
    mat_storage = MatStorageT<T>(name, Extents2D(rows, cols), allocator,
                                 MatPadding::kOdd);
  }
}

// clang-format off
// Schedules TiledFlashAttention for all heads, tokens and batch.
// Returns partial results in the same order as queries in `activations.q`.
// Might not work yet for prefix lm.
// To help understanding how to use this function below is description of how
// parameters are used:
//
// attention_impl - Used to determine attention kernel to use.
// num_query_tokens - number of tokens/timesteps in processed in a single batch
// it will influence how many queries kvs are evaluated against.
// num_kv_tokens - number of tokens/timesteps in kv cache
// layer_idx - layer index
// layer - used to get kv_heads, heads, qkv_dim
// activations - reads: activations.q queries, att_cap, IsGlobalLayer
// qbatch - kv cache, Pos / EndPrefix
// ctx - threading context
// clang-format on
void LocalAttentionForAllHeadsTokensAndBatch(
    AttentionImpl attention_impl, const size_t num_query_tokens,
    const size_t layer_idx, const LayerWeightsPtrs& layer,
    AttentionActivationsPtrs& activations, QBatch& qbatch,
    ThreadingContext& ctx) {
  const size_t heads_per_kv_head =
      layer.layer_config.heads / layer.layer_config.kv_heads;

  int core_count = ctx.pools.MaxWorkers();
  int task_multiplier = 1;
  while (qbatch.Size() * layer.layer_config.kv_heads * task_multiplier <
         core_count * 2) {
    task_multiplier++;
  }
  // Finding the smallest context we need to attend to avoid unnecessary
  // overhead when sub-splitting doesn't make sense. This check overestimates
  // context sizes because it ignores [local] layer sizes and explicit
  // qbatch.Prefix settings.
  size_t min_pos = qbatch.Pos(0);
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    min_pos = std::min(min_pos, qbatch.Pos(qi));
  }
  if (min_pos / task_multiplier < num_query_tokens) {
    // In case where min_pos / task_multiplier < num_tokens
    // To make sure we don't over count tokens or read out of bounds code
    // requires quite a bit more involved logic.
    // Also there is not much point to splitting the work into more tasks, when
    // amount of work is small.
    task_multiplier = 1;
  }
  [[maybe_unused]] int num_tasks = qbatch.Size() * layer.layer_config.kv_heads;
  [[maybe_unused]] int num_sub_tasks =
      qbatch.Size() * layer.layer_config.kv_heads * task_multiplier;
  HWY_DASSERT_M(activations.q.Rows() == num_query_tokens * qbatch.Size(),
                "qbatch size mismatch");
  int qkv_dim = layer.layer_config.qkv_dim;

  // sizes of all should be in sync
  if (num_sub_tasks > activations.sub_task_att_out->size()) {
    activations.sub_task_att_out->resize(num_sub_tasks);
    activations.sub_task_exp_denominator_sums->resize(num_sub_tasks);
    activations.sub_task_max_logits->resize(num_sub_tasks);
  }
  std::vector<int> skip_sub_task(num_sub_tasks, 0);

  // This loop parallelizes over qbatch, kv_head and substrings of context
  // tokens. Each parallel invocation handles all query tokens of the given
  // qbatch.
  ParallelFor(
      Parallelism::kHierarchical, num_sub_tasks, ctx,
      /*cluster_idx=*/0, Callers::kFlashAttention,
      [&](size_t task_idx, size_t worker) HWY_ATTR {
        size_t main_task_idx = task_idx / task_multiplier;
        size_t sub_task_idx = task_idx % task_multiplier;
        size_t current_qbatch_idx =
            main_task_idx / layer.layer_config.kv_heads;
        size_t kv_head_idx = main_task_idx % layer.layer_config.kv_heads;
        // First and last context token we will attend to.
        size_t global_start_context_pos = StartPos(
            qbatch.Pos(current_qbatch_idx), activations.config, layer_idx);
        // Keep in mind this is overestimation because some timesteps might not
        // need all tokens due to causal mask.
        // We will use it to determine how to divide work between sub tasks
        // and make sure PrefixEnd is taken into account
        size_t start_context_pos = global_start_context_pos;
        size_t last_context_pos =
            qbatch.Pos(current_qbatch_idx) + num_query_tokens - 1;
        // In some models, context is limited to some prefix - make sure we take
        // that into account.
        const size_t prefix_end = qbatch.PrefixEnd(current_qbatch_idx);
        if (prefix_end > 0 && prefix_end - 1 > last_context_pos) {
          last_context_pos = prefix_end - 1;
        }
        size_t total_num_context_tokens =
            last_context_pos - start_context_pos + 1;
        size_t context_tokens_per_sub_task =
            hwy::DivCeil(total_num_context_tokens, task_multiplier);
        // Restrict tokens to attend to the substring of context tokens that
        // this subtask is responsible for.
        start_context_pos =
            start_context_pos + context_tokens_per_sub_task * sub_task_idx;
        if (start_context_pos > last_context_pos) {
          skip_sub_task[task_idx] = 1;
          return;
        }
        last_context_pos =
            std::min(last_context_pos,
                     start_context_pos + context_tokens_per_sub_task - 1);
        // pre-initialize memory [to avoid racy resizes laters].
        int num_queries = num_query_tokens * heads_per_kv_head;
        std::vector<float*> queries_ptrs;
        queries_ptrs.reserve(num_queries);
        for (int token_idx = 0; token_idx < num_query_tokens; ++token_idx) {
          for (int q_head_idx = 0; q_head_idx < heads_per_kv_head;
               ++q_head_idx) {
            queries_ptrs.push_back(
                activations.q.Row(token_idx * qbatch.Size() +
                                  current_qbatch_idx) +
                (kv_head_idx * heads_per_kv_head + q_head_idx) * qkv_dim);
          }
        }
        hwy::Span<float*> queries_ptrs_span(queries_ptrs.data(),
                                            queries_ptrs.size());

        MatStorageT<float>& att_out =
            activations.sub_task_att_out->at(task_idx);
        AlignedFloatVector& exp_denominator_sums =
            activations.sub_task_exp_denominator_sums->at(task_idx);
        AlignedFloatVector& max_logits =
            activations.sub_task_max_logits->at(task_idx);
        MaybeResizeMatStorage(att_out, num_queries, qkv_dim, "att_out",
                              ctx.allocator);
        for (int i = 0; i < num_queries; ++i) {
          hwy::ZeroBytes(att_out.Row(i),
                         att_out.Cols() * sizeof(decltype(att_out.Row(i)[0])));
        }

        int num_queries_rounded_to_8 = hwy::RoundUpTo(num_queries, 8);
        exp_denominator_sums.resize(num_queries_rounded_to_8);
        max_logits.resize(num_queries_rounded_to_8);
        for (int i = 0; i < num_queries_rounded_to_8; ++i) {
          exp_denominator_sums[i] = 0.0f;
          max_logits[i] = -std::numeric_limits<float>::max() / 2.0f;
        }
        // Get pointers to the KVCache tiles, starting at global_start_pos
        // Returns multiple matrices for non-contiguous memory, for example as a
        // result of the wraparound in local layers.
        std::vector<MatPtr> kv_ptrs =
            qbatch.KV(current_qbatch_idx)
                .cache->GetPointers(
                    layer_idx, kv_head_idx, layer.layer_config.kv_heads,
                    global_start_context_pos,
                    activations.config.IsGlobalLayer(layer_idx));

        std::vector<size_t, hwy::AlignedAllocator<size_t>> start_pos_per_query;
        std::vector<size_t, hwy::AlignedAllocator<size_t>> last_pos_per_query;
        start_pos_per_query.reserve(num_queries);
        last_pos_per_query.reserve(num_queries);
        // Position of the first token in the first tile whose pointer was
        // returned above. Allows for handling of token positions relative to
        // the KV tiles returned above.
        size_t rounded_down_global_start_pos =
            hwy::RoundDownTo(global_start_context_pos, KVCache::kTileSize);
        for (int token_idx = 0; token_idx < num_query_tokens; ++token_idx) {
          int64_t global_query_pos =
              qbatch.Pos(current_qbatch_idx) + token_idx;
          // Intersect context to attend to for this specific query token
          // to the context tokens of the current subtask.
          int64_t query_last_context_pos = std::min(
              static_cast<int64_t>(last_context_pos), global_query_pos);
          // This max is to not go into negative values, for the same reason we
          // use int64_t and not size_t here.
          int64_t query_start_context_pos = std::max(
              global_query_pos -
                  static_cast<int64_t>(
                      activations.config.attention_window_sizes[layer_idx]) +
                  1,
              static_cast<int64_t>(start_context_pos));

          // Turn token position into KV-tile relative token positions.
          query_last_context_pos -= rounded_down_global_start_pos;
          query_start_context_pos -= rounded_down_global_start_pos;
          for (int q_head_idx = 0; q_head_idx < heads_per_kv_head;
               ++q_head_idx) {
            start_pos_per_query.push_back(query_start_context_pos);
            last_pos_per_query.push_back(query_last_context_pos);
          }
        }

        if (attention_impl == AttentionImpl::kFlashTransposedQsBF16) {
          // pack transposed queries into BF16
          auto [transposed_queries, transposed_queries_ptrs, _] =
              TransposeQueriesToGroupsOfNBF16orInt16<BF16>(
                  queries_ptrs_span, qkv_dim, /*group_size=*/4);
          hwy::Span<const BF16*> queries_span(
              const_cast<const BF16**>(transposed_queries_ptrs.data()),
              transposed_queries_ptrs.size());
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsBF16(
              kv_ptrs, num_queries, queries_span,
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data());
        } else if (attention_impl == AttentionImpl::kFlashTransposedQsInt16) {
          auto [transposed_queries, transposed_queries_ptrs, q_scales] =
              TransposeQueriesToGroupsOfNBF16orInt16<int16_t>(
                  queries_ptrs_span, qkv_dim, /*group_size=*/4);
          hwy::Span<const int16_t*> queries_span(
              const_cast<const int16_t**>(transposed_queries_ptrs.data()),
              transposed_queries_ptrs.size());
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogitsInt16(
              kv_ptrs, num_queries, queries_span, q_scales,
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data());
        } else {
          auto [transposed_queries, transposed_queries_ptrs] =
              TransposeQueriesToGroupsOf4(queries_ptrs_span, qkv_dim);
          DispatchTileFlashAttentionReturnExpSumsAndMaxLogits(
              kv_ptrs, num_queries,
              hwy::Span<const float*>(
                  const_cast<const float**>(transposed_queries_ptrs.data()),
                  transposed_queries_ptrs.size()),
              hwy::Span<const size_t>(start_pos_per_query),
              hwy::Span<const size_t>(last_pos_per_query),
              activations.config.att_cap, att_out, exp_denominator_sums.data(),
              max_logits.data());
        }
      });

  // This loop takes results from separate subtasks (subsequence of kv) and
  // merges them into single att_out over whole kv sequence.
  ParallelFor(
      Parallelism::kFlat, num_tasks, ctx,
      /*cluster_idx=*/0, Callers::kFlashAttention,
      [&](size_t main_task_idx, size_t worker) HWY_ATTR {
        size_t current_qbatch_idx = main_task_idx / layer.layer_config.kv_heads;
        size_t kv_head_idx = main_task_idx % layer.layer_config.kv_heads;
        for (int token_idx = 0; token_idx < num_query_tokens; ++token_idx) {
          for (int head_in_group_idx = 0; head_in_group_idx < heads_per_kv_head;
               ++head_in_group_idx) {
            const size_t batch_index =
                current_qbatch_idx * num_query_tokens + token_idx;
            const size_t q_head_idx =
                kv_head_idx * heads_per_kv_head + head_in_group_idx;
            const size_t att_out_row_idx =
                token_idx * heads_per_kv_head + head_in_group_idx;
            const size_t activations_att_out_start_idx = q_head_idx * qkv_dim;
            auto& att_out_0 = activations.sub_task_att_out->at(
                main_task_idx * task_multiplier + 0);
            auto& exp_denominator_sums_0 =
                activations.sub_task_exp_denominator_sums->at(
                    main_task_idx * task_multiplier + 0);
            auto& max_logits_0 = activations.sub_task_max_logits->at(
                main_task_idx * task_multiplier + 0);

            hwy::CopyBytes(att_out_0.Row(att_out_row_idx),
                           activations.att_out.Row(batch_index) +
                               activations_att_out_start_idx,
                           qkv_dim * sizeof(float));
            activations.softmax_d.Row(batch_index)[q_head_idx] =
                exp_denominator_sums_0[token_idx * heads_per_kv_head +
                                       head_in_group_idx];
            activations.softmax_max.Row(batch_index)[q_head_idx] =
                max_logits_0[token_idx * heads_per_kv_head + head_in_group_idx];
            for (int sub_task_idx = 1; sub_task_idx < task_multiplier;
                 ++sub_task_idx) {
              int task_idx = main_task_idx * task_multiplier + sub_task_idx;
              if (skip_sub_task[task_idx] == 1) {
                continue;
              }
              auto& att_out = activations.sub_task_att_out->at(task_idx);
              auto& exp_denominator_sums =
                  activations.sub_task_exp_denominator_sums->at(task_idx);
              auto& max_logits = activations.sub_task_max_logits->at(task_idx);
              MergeOnlineSoftmax(
                  att_out.Row(att_out_row_idx),
                  max_logits[token_idx * heads_per_kv_head + head_in_group_idx],
                  exp_denominator_sums[token_idx * heads_per_kv_head +
                                       head_in_group_idx],
                  qkv_dim,
                  activations.att_out.Row(batch_index) +
                      activations_att_out_start_idx,
                  activations.softmax_max.Row(batch_index)[q_head_idx],
                  activations.softmax_d.Row(batch_index)[q_head_idx]);
            }
          }
        }
      });
}

void TiledAttention(AttentionImpl attention_impl, size_t num_tokens,
                    size_t layer_idx, const LayerWeightsPtrs& layer,
                    AttentionActivationsPtrs& activations, QBatch& qbatch,
                    MatMulEnv& env, int flags) {
  static const auto zone = env.ctx.profiler.AddZone(
      "Gen.TiledAttention", hwy::ProfilerFlags::kInclusive);
  PROFILER_ZONE3(env.ctx.profiler, hwy::Profiler::Thread(), zone);

  const LayerConfig& layer_config = layer.layer_config;

  HWY_DASSERT_M((layer_config.heads % layer_config.kv_heads) == 0,
                "query heads must be a multiple of key-value heads");
  (void)layer_config;  // only used in HWY_DASSERT

  if (qbatch.KV(0).cache->compact_kv_cache_ptr.GetType() == Type::kBF16) {
    ComputeQKVTransposedTile<BF16>(num_tokens, layer_idx, layer, attention_impl,
                                   activations, qbatch, flags, env);
  } else if (qbatch.KV(0).cache->compact_kv_cache_ptr.GetType() == Type::kF32) {
    ComputeQKVTransposedTile<float>(num_tokens, layer_idx, layer,
                                    attention_impl, activations, qbatch, flags,
                                    env);
  } else if (qbatch.KV(0).cache->compact_kv_cache_ptr.GetType() ==
             Type::kInt8) {
    ComputeQKVTransposedTile<int8_t>(num_tokens, layer_idx, layer,
                                     attention_impl, activations, qbatch, flags,
                                     env);
  } else {
    HWY_ABORT(
        "Unsupported KV cache type: %d",
        static_cast<int>(qbatch.KV(0).cache->compact_kv_cache_ptr.GetType()));
  }
  RMSNormAndPositionalEncoding(num_tokens, qbatch, activations.q,
                               layer.query_norm_scale, layer_idx, activations,
                               env.ctx);
  LocalAttentionForAllHeadsTokensAndBatch(attention_impl, num_tokens, layer_idx,
                                          layer, activations, qbatch, env.ctx);
  SumHeads(layer, activations, env);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
