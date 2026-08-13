#include "vllm/model_executor/models/ministral3.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  return it != doc.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

std::string Prefix(const std::unordered_map<std::string, const SafetensorsFile*>&
                       where) {
  if (where.count("model.language_model.embed_tokens.weight") != 0)
    return "model.language_model.";
  return "model.";
}

void ClassifyTensorNames(
    const std::unordered_map<std::string, const SafetensorsFile*>& where,
    const std::string& prefix) {
  const std::string& text_prefix = prefix;
  for (const auto& [name, unused] : where) {
    (void)unused;
    if (name.rfind("model.vision_tower.", 0) == 0 ||
        name.rfind("model.multi_modal_projector.", 0) == 0)
      continue;
    if (name == "lm_head.weight" ||
        name.rfind(text_prefix + "embed_tokens.", 0) == 0 ||
        name.rfind(text_prefix + "norm.", 0) == 0 ||
        name.rfind(text_prefix + "layers.", 0) == 0)
      continue;
    throw std::runtime_error(
        "ministral3: unclassified checkpoint tensor: " + name);
  }
}

}  // namespace

Ministral3Weights LoadMinistral3Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const std::string p = Prefix(where);
  ClassifyTensorNames(where, p);
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    const auto it = where.find(name);
    VT_CHECK(it != where.end(), "ministral3: tensor not found: " + name);
    return it->second->Get(name);
  };
  VT_CHECK(config.num_hidden_layers > 0,
           "ministral3: num_hidden_layers must be positive");

  Ministral3Weights out;
  auto& w = out;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.attention_bias = RawBool(config.raw, "attention_bias", false);
  if (config.torch_dtype.find("float8") != std::string::npos ||
      config.torch_dtype.find("fp8") != std::string::npos)
    throw std::runtime_error(
        "Ministral-3 FP8 checkpoints are refused: missing FP8 loader; use "
        "Ministral-3-*-Base-2512 or *-BF16 mirrors");
  w.embed_tokens = LoadBf16Direct(get, p + "embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, p + "norm.weight");
  if (!w.tie_word_embeddings)
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");

  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string b = p + "layers." + std::to_string(layer) + ".";
    const std::string sa = b + "self_attn.";
    const std::string mlp = b + "mlp.";
    Qwen3DenseLayerWeights lw;
    lw.input_layernorm = LoadBf16Direct(get, b + "input_layernorm.weight");
    lw.post_attention_layernorm =
        LoadBf16Direct(get, b + "post_attention_layernorm.weight");
    lw.attn.qkv_proj = LoadMergedBf16RawNK(
        get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
    lw.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
    if (w.attention_bias)
      lw.attn.qkv_bias = LoadMergedBf16RawNK(
          get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
    lw.mlp.gate_up_proj = LoadMergedBf16RawNK(
        get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
    lw.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
    w.layers.push_back(std::move(lw));
  }
  const auto rope = get_rope(config.head_dim, config.max_position_embeddings,
                             /*is_neox_style=*/true, config.rope_parameters,
                             vt::DType::kBF16);
  const vt::Tensor cache = rope->cos_sin_cache();
  w.rope_cos_sin_yarn =
      MakeOwned(vt::DType::kBF16, {cache.shape[0], cache.shape[1]});
  std::memcpy(w.rope_cos_sin_yarn.bytes.data(), cache.data,
              w.rope_cos_sin_yarn.bytes.size());
  return out;
}

double Ministral3QueryScale(int64_t position, double beta,
                            int64_t original_max_position_embeddings) {
  VT_CHECK(position >= 0, "ministral3: position must be non-negative");
  VT_CHECK(beta >= 0.0 && original_max_position_embeddings > 0,
           "ministral3: invalid query-scale parameters");
  return 1.0 + beta * std::log1p(
      static_cast<double>(position / original_max_position_embeddings));
}

bool Ministral3QueryScalingEnabled() {
  const char* e = std::getenv("VT_MINISTRAL3_QUERY_SCALE");
  return e == nullptr || e[0] != '0';
}

void CheckMinistral3QueryScaling(const std::vector<int32_t>& positions,
                                 const HfConfig& config) {
  if (!Ministral3QueryScalingEnabled()) return;
  const int64_t original =
      config.rope_parameters.original_max_position_embeddings.value_or(16384);
  for (int32_t position : positions) {
    if (position >= original)
      throw std::runtime_error(
          "Ministral-3 long-context query scaling refused: missing "
          "CPU-first row-scale op for positions >= "
          "original_max_position_embeddings");
  }
}

}  // namespace vllm
