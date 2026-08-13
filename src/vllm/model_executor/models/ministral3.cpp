#include "vllm/model_executor/models/ministral3.h"

#include "vllm/model_executor/models/qwen3.h"

namespace vllm {

std::vector<float> Ministral3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Ministral3Weights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  return Qwen3DenseModel::Forward(token_ids, positions, attn_meta, attn_kv,
                                  weights, config, queue, logits_indices);
}

ForwardLogits Ministral3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Ministral3Weights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  return Qwen3DenseModel::ForwardDevice(token_ids, positions, attn_meta, attn_kv,
                                        weights, config, queue, logits_indices);
}

}  // namespace vllm
