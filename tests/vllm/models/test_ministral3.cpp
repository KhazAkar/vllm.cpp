#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include <doctest/doctest.h>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/models/ministral3.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, uint32_t seed,
                           float scale = 0.08f) {
  vllm::OwnedTensor out;
  out.dtype = vt::DType::kBF16;
  out.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < out.rank; ++i) {
    out.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  out.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  auto* values = reinterpret_cast<uint16_t*>(out.bytes.data());
  for (int64_t i = 0; i < numel; ++i) values[i] = vt::F32ToBF16(dist(rng));
  return out;
}

vllm::HfConfig TinyConfig() {
  vllm::HfConfig config;
  config.num_hidden_layers = 2;
  config.hidden_size = 32;
  config.num_attention_heads = 4;
  config.num_key_value_heads = 2;
  config.head_dim = 8;
  config.rotary_dim = 8;
  config.intermediate_size = 64;
  config.rms_norm_eps = 1e-5;
  config.rope_theta = 1000000.0;
  config.max_position_embeddings = 64;
  config.vocab_size = 31;
  return config;
}

vllm::MistralWeights TinyWeights(const vllm::HfConfig& config) {
  const int64_t h = config.hidden_size;
  const int64_t qdim = config.num_attention_heads * config.head_dim;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  vllm::MistralWeights weights;
  weights.tie_word_embeddings = false;
  weights.embed_tokens = MakeBf16({config.vocab_size, h}, 1);
  weights.final_norm = MakeBf16({h}, 2, 0.5f);
  weights.lm_head = MakeBf16({h, config.vocab_size}, 3);
  uint32_t seed = 10;
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeBf16({h}, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({h}, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, h}, seed++);
    lw.attn.o_proj = MakeBf16({h, qdim}, seed++);
    lw.mlp.gate_up_proj = MakeBf16({2 * config.intermediate_size, h}, seed++);
    lw.mlp.down_proj = MakeBf16({h, config.intermediate_size}, seed++);
    weights.layers.push_back(std::move(lw));
  }
  return weights;
}

struct CachePool {
  std::vector<std::vector<float>> storage;
  std::vector<vllm::PagedKvCache> kv;

  explicit CachePool(const vllm::HfConfig& config) {
    constexpr int64_t block_size = 8;
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
      storage.emplace_back(static_cast<size_t>(
          2 * block_size * config.num_key_value_heads * config.head_dim));
      vllm::PagedKvCache cache;
      cache.data = storage.back().data();
      cache.dtype = vt::DType::kF32;
      cache.num_blocks = 1;
      cache.block_size = block_size;
      cache.num_kv_heads = config.num_key_value_heads;
      cache.head_size = config.head_dim;
      kv.push_back(cache);
    }
  }
};

vllm::v1::CommonAttentionMetadata Meta() {
  vllm::v1::CommonAttentionMetadata meta;
  meta.num_reqs = 1;
  meta.num_actual_tokens = 3;
  meta.query_start_loc = {0, 3};
  meta.query_start_loc_cpu = meta.query_start_loc;
  meta.seq_lens = {3};
  meta.seq_lens_cpu = meta.seq_lens;
  meta.max_query_len = 3;
  meta.max_seq_len = 3;
  meta.block_table_num_cols = 1;
  meta.block_table_tensor = {0};
  meta.slot_mapping = {0, 1, 2};
  meta.causal = true;
  return meta;
}

vllm::OwnedTensor BuildYarnCache(const vllm::HfConfig& config) {
  const auto rope = vllm::get_rope(
      config.head_dim, config.max_position_embeddings, true,
      config.rope_parameters, vt::DType::kBF16);
  const vt::Tensor cache = rope->cos_sin_cache();
  vllm::OwnedTensor out;
  out.dtype = vt::DType::kBF16;
  out.rank = 2;
  out.shape[0] = cache.shape[0];
  out.shape[1] = cache.shape[1];
  out.bytes.resize(static_cast<size_t>(cache.shape[0] * cache.shape[1]) *
                   vt::SizeOf(vt::DType::kBF16));
  std::memcpy(out.bytes.data(), cache.data, out.bytes.size());
  return out;
}

vt::Queue CpuQueue() {
  return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
}

}  // namespace

TEST_CASE("Ministral-3 query scale mirrors Transformers floor semantics") {
  CHECK(vllm::Ministral3QueryScale(0, 0.1, 16384) == doctest::Approx(1.0));
  CHECK(vllm::Ministral3QueryScale(16383, 0.1, 16384) == doctest::Approx(1.0));
  CHECK(vllm::Ministral3QueryScale(16384, 0.1, 16384) ==
        doctest::Approx(1.0 + 0.1 * std::log(2.0)));
  CHECK(vllm::Ministral3QueryScale(32768, 0.1, 16384) ==
        doctest::Approx(1.0 + 0.1 * std::log(3.0)));
}

TEST_CASE("Ministral-3 architectures resolve") {
  CHECK(vllm::ModelRegistry::Resolve(
            std::vector<std::string>{"Ministral3ForCausalLM"})
            .architecture == "Ministral3ForCausalLM");
  CHECK(vllm::ModelRegistry::Resolve(
            std::vector<std::string>{"Mistral3ForConditionalGeneration"})
            .architecture == "Mistral3ForConditionalGeneration");
}

TEST_CASE("Ministral-3 nested text configuration preserves real 3B/8B fields") {
  const std::string path = "/tmp/ministral3-test-config.json";
  std::ofstream out(path);
  out << R"json({
    "architectures": ["Mistral3ForConditionalGeneration"],
    "model_type": "mistral3",
    "text_config": {
      "model_type": "ministral3",
      "hidden_size": 4096,
      "num_hidden_layers": 34,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "intermediate_size": 14336,
      "vocab_size": 131072,
      "rms_norm_eps": 1e-5,
      "max_position_embeddings": 262144,
      "torch_dtype": "bfloat16",
      "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 1000000,
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "llama_4_scaling_beta": 0.1
      }
    },
    "vision_config": {"model_type": "pixtral"}
  })json";
  out.close();
  const vllm::HfConfig config = vllm::LoadHfConfig(path);
  CHECK(config.model_type == "mistral3");
  CHECK(config.hidden_size == 4096);
  CHECK(config.num_hidden_layers == 34);
  CHECK(config.head_dim == 128);
  CHECK(config.rope_parameters.rope_type == "yarn");
  REQUIRE(config.rope_parameters.llama_4_scaling_beta.has_value());
  CHECK(*config.rope_parameters.llama_4_scaling_beta == doctest::Approx(0.1));
}

TEST_CASE("Ministral-3 plain path matches shared dense path exactly") {
  const vllm::HfConfig config = TinyConfig();
  const vllm::MistralWeights weights = TinyWeights(config);
  const std::vector<int32_t> tokens = {2, 5, 7};
  const std::vector<int32_t> positions = {0, 1, 2};
  CachePool shared_cache(config);
  CachePool ministral_cache(config);
  const auto meta = Meta();
  vt::Queue shared_queue = CpuQueue();
  vt::Queue ministral_queue = CpuQueue();
  const auto shared = vllm::MistralModel::Forward(
      tokens, positions, meta, shared_cache.kv, weights, config, shared_queue);
  const auto ministral = vllm::Ministral3Model::Forward(
      tokens, positions, meta, ministral_cache.kv, weights, config,
      ministral_queue);
  REQUIRE(shared.size() == ministral.size());
  CHECK(std::memcmp(shared.data(), ministral.data(),
                    shared.size() * sizeof(float)) == 0);
}

TEST_CASE("Ministral-3 YaRN cache changes the shared RoPE branch") {
  vllm::HfConfig config = TinyConfig();
  config.rope_parameters.rope_type = "yarn";
  config.rope_parameters.factor = 4.0;
  config.rope_parameters.original_max_position_embeddings = 16;
  config.rope_parameters.beta_fast = 32.0;
  config.rope_parameters.beta_slow = 1.0;
  config.rope_parameters.llama_4_scaling_beta = 0.1;
  vllm::MistralWeights plain = TinyWeights(config);
  vllm::MistralWeights yarn = plain;
  yarn.rope_cos_sin_yarn = BuildYarnCache(config);
  const std::vector<int32_t> tokens = {2, 5, 7};
  const std::vector<int32_t> positions = {0, 1, 2};
  CachePool plain_cache(config);
  CachePool yarn_cache(config);
  const auto meta = Meta();
  vt::Queue plain_queue = CpuQueue();
  vt::Queue yarn_queue = CpuQueue();
  const auto no_cache = vllm::MistralModel::Forward(
      tokens, positions, meta, plain_cache.kv, plain, config, plain_queue);
  const auto with_cache = vllm::Ministral3Model::Forward(
      tokens, positions, meta, yarn_cache.kv, yarn, config, yarn_queue);
  REQUIRE(no_cache.size() == with_cache.size());
  bool differs = false;
  for (size_t i = 0; i < no_cache.size(); ++i) {
    if (no_cache[i] != with_cache[i]) {
      differs = true;
      break;
    }
  }
  CHECK(differs);
}
