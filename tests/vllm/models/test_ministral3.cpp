#include <cmath>
#include <fstream>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/ministral3.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

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
