#include "vllm/model_executor/models/ministral3.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/qwen3_5_common.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using namespace dense_attn;
using vt::DType;
using vt::Tensor;
using v1::CommonAttentionMetadata;

bool IsQueryScalingEnabled() {
  return Ministral3QueryScalingEnabled();
}

void CheckQueryScaling(const std::vector<int32_t>& positions,
                       const HfConfig& config) {
  if (!IsQueryScalingEnabled()) return;
  const int64_t original =
      config.rope_parameters.original_max_position_embeddings.value_or(16384);
  for (int32_t position : positions) {
    if (position >= original)
      throw std::runtime_error(
          "Ministral-3 long-context query scaling refused: missing "
          "CPU-first row-scale op for positions >= original_max_position_embeddings");
  }
}

DBuf MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& config,
              const Tensor& hidden, int64_t tokens) {
  const int64_t intermediate = config.intermediate_size;
  layers::UnquantizedMlpGateUpMethod gate_up(&w.gate_up_proj, intermediate);
  DBuf activation = gate_up.Apply(d, hidden);
  DBuf output(d, DType::kBF16, {tokens, config.hidden_size});
  Tensor down = ResidentWeight(d, w.down_proj);
  vt::MatmulBT(d.q, output.t(), activation.t(), down);
  return output;
}

DBuf AttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& config,
               const Tensor& hidden, const StepInputs& si,
               const CommonAttentionMetadata& meta, const PagedKvCache& kv,
               int64_t tokens, const Tensor& yarn_cache) {
  const int64_t qdim = config.num_attention_heads * config.head_dim;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  VT_CHECK(w.q_norm.Empty() && w.k_norm.Empty(),
           "ministral3: q/k norms are not supported by this text tower");
  VT_CHECK(w.qkv_bias.Empty(),
           "ministral3: attention bias is not supported by this text tower");
  VT_CHECK(kv.num_kv_heads == config.num_key_value_heads &&
               kv.head_size == config.head_dim,
           "ministral3: KV cache shape does not match text config");

  DBuf qkv(d, DType::kBF16, {tokens, qdim + 2 * kdim});
  Tensor qkv_weight = ResidentWeight(d, w.qkv_proj);
  vt::MatmulBT(d.q, qkv.t(), hidden, qkv_weight);
  DBuf q(d, DType::kBF16, {tokens, qdim});
  DBuf k(d, DType::kBF16, {tokens, kdim});
  DBuf v(d, DType::kBF16, {tokens, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  Tensor q_heads = Reshape(q.t(), {tokens, config.num_attention_heads,
                                   config.head_dim});
  Tensor k_heads = Reshape(k.t(), {tokens, config.num_key_value_heads,
                                   config.head_dim});
  if (yarn_cache.rank != 0) {
    Tensor k_view = k_heads;
    vt::RopeFromCache(d.q, q_heads, &k_view, si.positions.t(), yarn_cache,
                      vt::RopeArgs{.rotary_dim = static_cast<int>(config.rotary_dim),
                                   .is_neox_style = true});
  } else {
    vt::RopeNeox(d.q, q_heads, k_heads, si.positions.t(), MakeRopeArgs(config));
  }

  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, k_heads, Reshape(v.t(), {tokens, config.num_key_value_heads,
                                                    config.head_dim}),
                      k_cache, v_cache, si.slot_mapping.t());

  DBuf attention(d, DType::kBF16,
                 {tokens, config.num_attention_heads, config.head_dim});
  vt::PagedAttentionArgs args{
      1.0F / std::sqrt(static_cast<float>(config.head_dim)), meta.causal};
  args.query_start_loc_host = meta.query_start_loc.data();
  args.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, attention.t(), q_heads, k_cache, v_cache,
                     si.block_table.t(), si.seq_lens.t(), si.query_start_loc.t(),
                     args);

  DBuf output(d, DType::kBF16, {tokens, config.hidden_size});
  Tensor output_weight = ResidentWeight(d, w.o_proj);
  vt::MatmulBT(d.q, output.t(),
               Reshape(attention.t(), {tokens, config.hidden_size}),
               output_weight);
  return output;
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& meta,
                 const std::vector<PagedKvCache>& kv,
                 const Ministral3Weights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t tokens = static_cast<int64_t>(token_ids.size());
  const int64_t hidden_size = config.hidden_size;
  const int64_t vocab_size = config.vocab_size;
  CheckQueryScaling(positions, config);
  VT_CHECK(kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "ministral3: one full-attention KV cache per layer is required");

  DBuf hidden(d, DType::kBF16, {tokens, hidden_size});
  Tensor embedding = ResidentWeight(d, weights.dense.embed_tokens);
  DBuf ids(d, DType::kI32, {tokens}, token_ids.data());
  vt::Embedding(d.q, hidden.t(), embedding, ids.t());

  StepInputs si = BuildStepInputs(d, positions, meta, config);
  Tensor yarn;
  if (!weights.rope_cos_sin_yarn.Empty())
    yarn = ResidentWeight(d, weights.rope_cos_sin_yarn);

  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const auto& layer_weights = weights.dense.layers[static_cast<size_t>(layer)];
    Tensor input_norm = ResidentWeight(d, layer_weights.input_layernorm,
                                        {hidden_size});
    DBuf normed(d, DType::kBF16, {tokens, hidden_size});
    vt::RmsNorm(d.q, normed.t(), hidden.t(), input_norm,
                vt::RmsNormArgs{static_cast<float>(config.rms_norm_eps), false});
    DBuf attention = AttnBlock(
        d, layer_weights.attn, config, normed.t(), si, meta,
        kv[static_cast<size_t>(layer)], tokens, yarn);
    Tensor post_norm = ResidentWeight(d, layer_weights.post_attention_layernorm,
                                      {hidden_size});
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(
          d.q, normed.t(), attention.t(), post_norm, &hidden.t(),
          vt::kFusedAddRmsNormStd, static_cast<float>(config.rms_norm_eps));
    } else {
      vt::RmsNorm(
          d.q, normed.t(), attention.t(), post_norm,
          vt::RmsNormArgs{static_cast<float>(config.rms_norm_eps), false},
          &hidden.t());
    }
    DBuf mlp = MlpBlock(d, layer_weights.mlp, config, normed.t(), tokens);
    hidden = std::move(mlp);
  }

  Tensor final_norm = ResidentWeight(d, weights.dense.final_norm, {hidden_size});
  DBuf normalized(d, DType::kBF16, {tokens, hidden_size});
  vt::RmsNorm(d.q, normalized.t(), hidden.t(), final_norm,
              vt::RmsNormArgs{static_cast<float>(config.rms_norm_eps), false});

  Tensor source = normalized.t();
  DBuf gathered(d, DType::kBF16, {1, 1});
  if (!logits_indices.empty() &&
      static_cast<int64_t>(logits_indices.size()) < tokens) {
    gathered = DBuf(d, DType::kBF16,
                    {static_cast<int64_t>(logits_indices.size()), hidden_size});
    const size_t row_bytes =
        static_cast<size_t>(hidden_size) * vt::SizeOf(DType::kBF16);
    for (size_t i = 0; i < logits_indices.size(); ++i)
      d.b.Copy(d.q, static_cast<char*>(gathered.ptr()) + i * row_bytes,
               static_cast<const char*>(normalized.t().data) +
                   static_cast<size_t>(logits_indices[i]) * row_bytes,
               row_bytes);
    source = gathered.t();
  }
  const int64_t output_rows = source.shape[0];
  DBuf logits(d, DType::kF32, {output_rows, vocab_size});
  Tensor lm_head = ResidentWeight(d, weights.dense.lm_head);
  vt::Matmul(d.q, logits.t(), source, lm_head);
  return logits;
}

ForwardLogits Wrap(Dev d, DBuf&& logits, int64_t rows, int64_t vocab) {
  ForwardLogits out;
  out.rows = rows;
  out.vocab = vocab;
  out.device_tensor = logits.t();
  const size_t bytes = logits.alloc_bytes();
  void* data = logits.Release();
  out.device_storage =
      std::shared_ptr<void>(data, [bytes](void* p) { Pool().Put(bytes, p); });
  (void)d;
  return out;
}

}  // namespace

std::vector<float> Ministral3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& meta, const std::vector<PagedKvCache>& kv,
    const Ministral3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf logits = ForwardBody(d, token_ids, positions, meta, kv, weights, config,
                             logits_indices);
  std::vector<float> host(static_cast<size_t>(logits.t().shape[0]) *
                          static_cast<size_t>(config.vocab_size));
  logits.Download(d, host.data());
  return host;
}

ForwardLogits Ministral3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& meta, const std::vector<PagedKvCache>& kv,
    const Ministral3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf logits = ForwardBody(d, token_ids, positions, meta, kv, weights, config,
                             logits_indices);
  return Wrap(d, std::move(logits), logits.t().shape[0], config.vocab_size);
}

}  // namespace vllm
