#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/ministral3.h"
#include "vllm/model_executor/models/qwen3_5_common.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/v1/kv_cache_dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kMinistral3Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Ministral3LoadedModel final : public LoadedModel {
 public:
  Ministral3LoadedModel(const ModelRegistration& registration,
                        Ministral3Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const Ministral3Weights& weights() const { return weights_; }

 private:
  Ministral3Weights weights_;
};

std::unique_ptr<LoadedModel> LoadMinistral3(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors)
    throw std::runtime_error(
        "Ministral-3 does not support GGUF: missing Ministral-3 GGUF loader");
  if (source.safetensors == nullptr)
    throw std::runtime_error("Ministral-3 safetensors source is empty");
  ParseMinistral3Config(config);
  return std::make_unique<Ministral3LoadedModel>(
      registration, LoadMinistral3Weights(*source.safetensors, config));
}

void PrepareMinistral3(LoadedModel&, const HfConfig&, vt::Queue&) {}

ForwardLogits ForwardMinistral3(LoadedModel& model,
                                const ModelForwardInput& input) {
  if (input.mm.has_value())
    throw std::runtime_error(
        "Ministral-3 image inputs refused: vision tower and multimodal "
        "projector are not implemented");
  CheckMinistral3QueryScaling(input.positions, input.config);
  auto& m = static_cast<Ministral3LoadedModel&>(model);
  if (input.gather_logits)
    return Ministral3Model::ForwardDevice(
        input.token_ids, input.positions, input.attn_meta, input.attn_kv,
        m.weights(), input.config,
        input.queue, input.logits_indices);
  return HostLogits(
      Ministral3Model::Forward(input.token_ids, input.positions, input.attn_meta,
                               input.attn_kv, m.weights(), input.config, input.queue,
                               input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kMinistral3Factory{
    .parse_config = &ParseMinistral3Config,
    .load_weights = &LoadMinistral3,
    .prepare = &PrepareMinistral3,
    .forward = &ForwardMinistral3,
    .make_kv_cache = &MakeMinistral3KVCache,
    .is_dense_model = true,
};

}  // namespace

REGISTER_VLLM_MODEL(ministral3_causal, "Ministral3ForCausalLM",
                    kMinistral3Factory, kMinistral3Info)
REGISTER_VLLM_MODEL(ministral3_conditional, "Mistral3ForConditionalGeneration",
                    kMinistral3Factory, kMinistral3Info)

void ParseMinistral3Config(const HfConfig& config) {
  VT_CHECK(config.rope_parameters.rope_type == "yarn",
           "Ministral-3 text tower requires YaRN rope_parameters");
  VT_CHECK(config.rope_parameters.factor.has_value() &&
               config.rope_parameters.original_max_position_embeddings.has_value(),
           "Ministral-3 YaRN requires factor and original_max_position_embeddings");
  VT_CHECK(config.rope_parameters.llama_4_scaling_beta.has_value(),
           "Ministral-3 requires rope_parameters.llama_4_scaling_beta");
  VT_CHECK(config.torch_dtype == "bfloat16" || config.torch_dtype == "bf16" ||
               config.torch_dtype.empty(),
           "Ministral-3 spike supports BF16 checkpoints only; use the "
           "Ministral-3-*-Base-2512 or *-BF16 mirror");
}

v1::KVCacheConfig MakeMinistral3KVCache(const HfConfig& config,
                                        int block_size, int num_blocks) {
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, static_cast<int>(config.num_key_value_heads),
          static_cast<int>(config.head_dim), v1::ResolveKvCacheDType()));
  return kv;
}

}  // namespace vllm
