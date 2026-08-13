# Ministral-3 text-tower implementation spike

## Traceability

- Upstream issue: #387 (an outside contributor is offering the Mistral3 row).
- Matrix row: `MODEL-MM-mistral3-mistral3-for-conditional-generation`.
- Branch: `ministral-spike-implementation`.
- Upstream vLLM source pin: `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## Scope

This spike adds a self-registering text-only implementation for released
Ministral-3 checkpoints.  It resolves both `Ministral3ForCausalLM` text-only
checkpoints and the released wrapper architecture
`Mistral3ForConditionalGeneration`, descending through `text_config` and
running the dense language tower.  It supports BF16 safetensors generation
only.

Vision inputs are refused because the Ministral-3 vision tower and multimodal
projector are not implemented in vllm.cpp.  GGUF is refused because this
architecture has no GGUF loader arm.  FP8 is refused because the dense FP8
loader arm is not implemented; users must use the BF16
`Ministral-3-*-Base-2512` or `*-BF16` mirrors.

Token parity against a real Ministral-3 checkpoint is owed: this environment
has no AMD GPU, no NVIDIA oracle, and no large Ministral safetensors artifact
is downloaded.

## Upstream anchors

The text-tower shape and weight ownership mirror the pinned
`vllm/model_executor/models/mistral.py`, especially `MistralAttention.forward`.
The wrapper descent and prefix mapping mirror
`vllm/model_executor/models/mistral3.py`:
`Mistral3ForConditionalGeneration.hf_to_vllm_mapper` and
`init_vllm_registered_model(hf_config=config.text_config)`.
The pinned registry maps `Ministral3ForCausalLM` and `MistralForCausalLM` to
`mistral.py` (`registry.py:168-169`).

Transformers
`models/ministral3/modeling_ministral3.py` is the behavior reference for the
per-position query scale:

```text
1 + beta * log(1 + floor(position_id / original_max_position_embeddings))
```

It scales query only, after RoPE and before attention.  The real checkpoints
store `beta=0.1` in `rope_parameters["llama_4_scaling_beta"]` and use
`original_max_position_embeddings=16384` as the divisor.

## Design

`LoadHfConfig` already descends into `text_config` for text dimensions while
retaining top-level wrapper architecture metadata.  `RopeParameters` gains the
typed `llama_4_scaling_beta` field.  The existing `YaRNScalingRotaryEmbedding`
is selected by the existing rope factory, preserving the checkpoint's factor,
original context, beta-fast/slow, theta, and attention-factor values.

The new Ministral model files reuse the dense QKV, KV-cache, MLP, residual, and
LM-head machinery.  Weight mapping accepts the wrapper prefixes
`model.language_model.*` and `lm_head.*`; vision tower and multimodal projector
tensors are classified as deliberate skips rather than generic unmapped
tensors.  The model is untied and BF16-only.

Query scaling is transformers-faithful by default and has an environment-gated
opt-out for A/B comparison.  Positions below 16384 need no multiply because
their exact scale is 1.0.  Positions at or above 16384 currently refuse with a
named missing row-scale operation rather than silently producing unscaled
queries; adding a CPU-first row-scale op is an explicit follow-up owed for
long-context execution.  The implementation records the pinned-vLLM
discrepancy explicitly: pinned `mistral.py` reads a top-level
`config.llama_4_scaling` dictionary, while these checkpoints carry only the
nested `rope_parameters["llama_4_scaling_beta"]`.  Therefore the pinned vLLM
path is expected not to apply the scale for these checkpoints, whereas
Transformers does.  This is an open oracle question, not silently resolved
parity evidence.

## Risks and known gaps

- The query scale is a per-position, per-query-row operation not expressible by
  the existing scalar/column-vector operation.  Short contexts below 16384
  skip it exactly; long contexts are explicitly refused with a named missing
  row-scale operation.  A CPU-first row-vector multiply operation remains owed
  for long-context execution and the CUDA arm is owed thereafter.
- No image processor, vision encoder, projector, image-token path, GGUF path,
  or FP8 path is included.
- No real checkpoint load or token parity result is claimed in this spike.
- Large 3B/8B/14B artifacts are intentionally not downloaded here.

## Tests

Fixture tests cover:

- registry resolution for both architecture strings;
- nested 3B and 8B text dimensions and all relevant YaRN/query-scale fields;
- YaRN cache values against an independently computed reference;
- query-scale positions 0, 16383, 16384, and 32768, including floor/log edges;
- wrapper/tower loader name mapping and deliberate vision/projector skips;
- loud image, GGUF, and FP8 refusal messages.

Existing Mistral, Qwen3 dense, registry, and rotary embedding tests remain
regression gates.  RED-first evidence is captured in the implementation
handoff; no token-parity test is fabricated without a real oracle.

## Gates and evidence

Before push: configure and build a CPU-only Release tree, run focused
Ministral/config/rotary/registry tests, run existing Mistral and Qwen3-dense
tests, and run applicable scripts/checkers.  The final report records exact
commands and whether each gate is satisfied, pending, or blocked by the
environment.

## Stop conditions

Stop and report rather than silently broadening scope if:

1. the shared dense seam cannot support YaRN or row scaling without an
   untracked backend exception;
2. a checkpoint tensor cannot be classified as loaded or deliberately skipped;
3. a requested image, GGUF, or FP8 input would reach an accidental fallback;
4. a real-checkpoint or token-parity claim would require downloading a large
   artifact or access to unavailable GPU/oracle hardware.
