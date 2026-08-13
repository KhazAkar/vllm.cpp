#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/mistral.h"

namespace vllm {

struct Ministral3Weights {
  MistralWeights dense;
  OwnedTensor rope_cos_sin_yarn;
};
class Ministral3Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Ministral3Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Ministral3Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

Ministral3Weights LoadMinistral3Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

void ParseMinistral3Config(const HfConfig& config);

v1::KVCacheConfig MakeMinistral3KVCache(const HfConfig& config,
                                        int block_size, int num_blocks);

double Ministral3QueryScale(int64_t position, double beta,
                            int64_t original_max_position_embeddings);

bool Ministral3QueryScalingEnabled();

}  // namespace vllm
