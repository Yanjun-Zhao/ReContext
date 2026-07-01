import argparse
import hashlib
import os
import numpy as np
import random
import json
import time
import yaml
from copy import deepcopy
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from data_utils import load_eval_data
from recontext.custom_mixin import RescaleConfig
from recontext.custom_modeling_llama import RescaleLlamaForCausalLM
from recontext.custom_modeling_qwen3 import RescaleQwen3ForCausalLM
from recontext.custom_modeling_qwen3_moe import RescaleQwen3MoeForCausalLM

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DAC_COMPRESSED_DATA_ROOT = os.path.join(_REPO_ROOT, "data_eval_compressed")
_DAC_SUPPORTED_COMPRESSION_RATES = (0.5, 0.25)

# Maps HF model_type -> (BaseClass, RescaleClass)
_MODEL_CLASSES = {
    "llama":     (LlamaForCausalLM,     RescaleLlamaForCausalLM),
    "qwen3_moe": (Qwen3MoeForCausalLM,  RescaleQwen3MoeForCausalLM),
    "qwen3":     (Qwen3ForCausalLM,     RescaleQwen3ForCausalLM),
}


def reset_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=23)

    # data
    parser.add_argument("--dataset", type=str, default="tom_tracking_0.5k")
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument(
        "--no_shuffle_test_subset",
        action="store_true",
        default=False,
        help="When --test_size is set, take the first N examples in dataset order instead of shuffling first.",
    )
    parser.add_argument("--no_chat_template", action="store_false", dest="use_chat_template", default=True)
    parser.add_argument("--stop_on_newline", action="store_true", default=False)

    # model
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--think", action="store_true", default=False)
    parser.add_argument("--enable_yarn", action="store_true", default=False,
                        help="Enable YaRN rope scaling (recommended for Qwen3 long context).")
    parser.add_argument("--yarn_factor", type=float, default=None,
                        help="YaRN rope scaling factor. If unset, inferred from max_model_len.")
    parser.add_argument("--yarn_original_max_position_embeddings", type=int, default=None,
                        help="Override original_max_position_embeddings in YaRN config.")
    parser.add_argument("--yarn_beta_fast", type=float, default=None,
                        help="Optional YaRN beta_fast override.")
    parser.add_argument("--yarn_beta_slow", type=float, default=None,
                        help="Optional YaRN beta_slow override.")

    # control
    parser.add_argument("--strip_thinking", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--disable_cache", dest="enable_cache", action="store_false", default=True)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--auto_skip", action="store_true", default=False)


    # decoding method
    parser.add_argument("--decoding_method", type=str, default="flash",
                        choices=["flash", "dysco", "ReContext", "attnsharp", "DAC"])
    # ReContext config (YAML + overrides)
    parser.add_argument("--recontext_cfgs_path", type=str, default=None, help="path to ReContext configs")
    parser.add_argument("--recontext_qrheads", type=str, default=None, help="qr heads")
    parser.add_argument("--recontext_top_k", type=int, default=None, help="top k")
    parser.add_argument("--recontext_top_p", type=float, default=None, help="top p")
    parser.add_argument("--recontext_strength", type=float, default=None, help="strength")
    parser.add_argument("--recontext_decay_factor", type=float, default=None, help="decay factor")
    parser.add_argument("--recontext_ctx_warmup", type=int, default=None, help="context warmup")
    parser.add_argument("--recontext_interv_warmup", type=str, default=None, help="intervention warmup")
    parser.add_argument("--recontext_replay_rounds", type=int, default=None, help="number of evidence replay rounds")
    parser.add_argument(
        "--recontext_selection_scope",
        type=str,
        choices=["context", "full_prompt"],
        default=None,
        help="Token selection scope for sentence replay.",
    )
    parser.add_argument(
        "--recontext_dedup_inserted_sentences",
        type=int,
        choices=[0, 1],
        default=None,
        help="Whether to remove sentences already inserted by earlier replay rounds.",
    )
    parser.add_argument("--recontext_rescale_template", action="store_true", default=False, help="rescale template")
    parser.add_argument("--recontext_static_rescaling", action="store_true", default=False, help="static rescaling")
    parser.add_argument(
        "--recontext_wrap_sentence_txt",
        type=int,
        choices=[0, 1],
        default=None,
        help="Whether to wrap replayed sentence text with the Extra Info template (1=yes, 0=no).",
    )
    parser.add_argument(
        "--recontext_replay_position",
        type=str,
        choices=["before_question", "after_question", "after_question_user_side"],
        default=None,
        help=(
            "Where to insert sentence-replay evidence relative to the original question segment. "
            "'before_question' preserves the current behavior; 'after_question' appends the Extra Info "
            "after the original question suffix as already tokenized by the chat template; "
            "'after_question_user_side' rebuilds the prompt so the Extra Info remains user-side context."
        ),
    )
    parser.add_argument(
        "--recontext_use_extension_inputs",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Whether to actually insert the replayed sentence text back into the prompt "
            "(1=yes, 0=no). Setting 0 reproduces the effect of forcing `extension_inputs = None`."
        ),
    )
    parser.add_argument(
        "--recontext_importance_dump_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for saving per-sample ReContext importance details before top-k selection. "
            "Each sample is saved as a torch .pt file."
        ),
    )

    # attnsharp
    parser.add_argument("--attention_logits_temperature", type=float, default=None)

    # DAC
    parser.add_argument(
        "--dac_compression_rate",
        type=float,
        default=None,
        help=(
            "Offline DAC compressed-data keep ratio. Supported values are 0.5 and 0.25, "
            "selecting data_eval_compressed/keep_0p5 or keep_0p25."
        ),
    )
    parser.add_argument(
        "--dac_compressed_data_root",
        type=str,
        default=_DEFAULT_DAC_COMPRESSED_DATA_ROOT,
        help="Root directory containing DAC compressed data subfolders keep_0p5 and keep_0p25.",
    )
    parser.add_argument(
        "--dac_compress_ratio",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option from online DAC compression. "
            "Use --dac_compression_rate for offline compressed-data evaluation."
        ),
    )

    args = parser.parse_args()
    if args.decoding_method == "DAC":
        _normalize_dac_args(args)
    args.model = args.model.rstrip("/")
    saving_name = args.model.replace("/", "-")
    args.output_dir = os.path.join(args.output_dir, args.decoding_method, saving_name)
    return args


def _is_close_to_supported_dac_rate(value):
    for rate in _DAC_SUPPORTED_COMPRESSION_RATES:
        if abs(float(value) - rate) < 1e-9:
            return rate
    return None


def _dac_rate_to_dir_name(rate):
    rate = _is_close_to_supported_dac_rate(rate)
    if rate is None:
        supported = ", ".join(str(x) for x in _DAC_SUPPORTED_COMPRESSION_RATES)
        raise ValueError(f"Unsupported DAC compression rate. Supported values: {supported}")
    tag = f"{rate:g}".replace(".", "p")
    return f"keep_{tag}"


def _resolve_dac_compression_rate(args):
    if args.dac_compression_rate is not None:
        rate = _is_close_to_supported_dac_rate(args.dac_compression_rate)
        if rate is None:
            supported = ", ".join(str(x) for x in _DAC_SUPPORTED_COMPRESSION_RATES)
            raise ValueError(f"--dac_compression_rate must be one of: {supported}")
        return rate

    if args.dac_compress_ratio is None:
        return 0.5

    direct_rate = _is_close_to_supported_dac_rate(args.dac_compress_ratio)
    if direct_rate is not None:
        print(
            "[DAC] --dac_compress_ratio is deprecated for offline evaluation; "
            f"interpreting it as --dac_compression_rate {direct_rate:g}."
        )
        return direct_rate

    legacy_keep_rate = _is_close_to_supported_dac_rate(1 - args.dac_compress_ratio)
    if legacy_keep_rate is not None:
        print(
            "[DAC] --dac_compress_ratio is deprecated for offline evaluation; "
            f"interpreting legacy deletion ratio {args.dac_compress_ratio:g} "
            f"as --dac_compression_rate {legacy_keep_rate:g}."
        )
        return legacy_keep_rate

    supported = ", ".join(str(x) for x in _DAC_SUPPORTED_COMPRESSION_RATES)
    raise ValueError(
        "--dac_compress_ratio is deprecated and could not be mapped to an offline "
        f"DAC compression rate. Use --dac_compression_rate with one of: {supported}"
    )


def _normalize_dac_args(args):
    args.dac_compression_rate = _resolve_dac_compression_rate(args)
    args.dac_compression_dir = _dac_rate_to_dir_name(args.dac_compression_rate)
    args.dac_data_path = os.path.join(args.dac_compressed_data_root, args.dac_compression_dir)


def get_eval_data_path(args):
    if args.decoding_method != "DAC":
        return "data_eval"
    if not os.path.isdir(args.dac_data_path):
        raise FileNotFoundError(
            "DAC compressed data path not found: "
            f"{args.dac_data_path}. Expected subfolders like "
            f"{os.path.join(args.dac_compressed_data_root, 'keep_0p5')} and "
            f"{os.path.join(args.dac_compressed_data_root, 'keep_0p25')}."
        )
    return args.dac_data_path


def hash8(s):
    return hashlib.md5(str(s).encode()).hexdigest()[:8]


def get_output_path(args, rescale_config=None):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.decoding_method == "flash":
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    elif args.decoding_method == "attnsharp":
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_attnsharp{args.attention_logits_temperature}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    elif args.decoding_method == "DAC":
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_dacoffline{args.dac_compression_dir}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    elif args.decoding_method in ["dysco", "ReContext"]:
        cfg = rescale_config
        head_hash = hash8(cfg.selected_heads)
        dynamic_tag = "dynamic" if cfg.dynamic_rescale else "static"
        replay_rounds = getattr(cfg, "replay_rounds", 1)
        effective_replay_position = (
            "before_question" if replay_rounds > 1 else getattr(cfg, "replay_position", "before_question")
        )
        replay_tag = ""
        if effective_replay_position != "before_question":
            replay_tag = f"_replay{effective_replay_position}"
        selection_scope = getattr(cfg, "selection_scope", "full_prompt")
        dedup_inserted_sentences = getattr(cfg, "dedup_inserted_sentences", True)
        rounds_tag = ""
        if replay_rounds != 1 or selection_scope != "full_prompt" or not dedup_inserted_sentences:
            rounds_tag = (
                f"_rounds{replay_rounds}"
                f"_scope{selection_scope}"
                f"_dedup{int(bool(dedup_inserted_sentences))}"
            )
        output_filename = (
            f"{args.dataset}_modlen{args.max_model_len}_max{args.max_tokens}"
            f"t{args.temperature}p{args.top_p}"
            f"_{dynamic_tag}rescalehead{head_hash}"
            f"k{cfg.top_k}p{cfg.top_p}s{cfg.strength}df{cfg.decay_factor}"
            f"_ctxwarm{cfg.context_warmup_steps}intwarm{cfg.intervention_warmup_steps}"
            f"_wrapsent{int(bool(cfg.wrap_sentence_txt))}"
            f"_useext{int(bool(cfg.use_extension_inputs))}"
            f"{replay_tag}"
            f"{rounds_tag}"
            f"scaletemp{args.recontext_rescale_template}"
            f"_think{args.think}_{args.seed}and{args.generation_seed}"
            f"_testsz{args.test_size}.json"
        )
    else:
        raise ValueError(f"Unknown decoding method: {args.decoding_method}")

    return os.path.join(args.output_dir, output_filename)


def detect_model_type(model_path, config=None):
    if config is None:
        config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type.lower()
    # Match against known types (order matters: qwen3_moe before qwen3)
    for key in ["llama", "qwen3_moe", "qwen3"]:
        if key in model_type:
            return key
    raise ValueError(f"Unsupported model type: {model_type}")


def maybe_enable_yarn(args, model_type, config):
    if "qwen3" not in model_type:
        return config

    base_max_len = int(getattr(config, "max_position_embeddings", 0) or 0)
    need_long_context = base_max_len > 0 and args.max_model_len > base_max_len
    if not args.enable_yarn and not need_long_context:
        return config

    cfg = config
    if args.yarn_factor is not None:
        yarn_factor = float(args.yarn_factor)
    elif base_max_len > 0:
        yarn_factor = float(max(args.max_model_len / base_max_len, 1.0))
    else:
        yarn_factor = 4.0

    rope_scaling = {
        "type": "yarn",
        "factor": yarn_factor,
    }

    if args.yarn_original_max_position_embeddings is not None:
        rope_scaling["original_max_position_embeddings"] = int(args.yarn_original_max_position_embeddings)
    elif base_max_len > 0:
        rope_scaling["original_max_position_embeddings"] = base_max_len

    if args.yarn_beta_fast is not None:
        rope_scaling["beta_fast"] = float(args.yarn_beta_fast)
    if args.yarn_beta_slow is not None:
        rope_scaling["beta_slow"] = float(args.yarn_beta_slow)

    cfg.rope_scaling = rope_scaling
    if args.max_model_len > 0:
        cfg.max_position_embeddings = max(args.max_model_len, base_max_len)

    print("Enabled YaRN rope scaling:", cfg.rope_scaling)
    print("Configured max_position_embeddings:", cfg.max_position_embeddings)
    return cfg


def build_rescale_config(args, tokenizer, model_type):
    """Load YAML config + CLI overrides -> RescaleConfig."""
    # Load base config from YAML
    if args.recontext_cfgs_path:
        with open(args.recontext_cfgs_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # CLI overrides (non-None values take precedence)
    if args.recontext_qrheads is not None:
        cfg["selected_heads"] = args.recontext_qrheads
    if args.recontext_top_k is not None:
        if args.recontext_top_k > 0:
            cfg["top_k"] = args.recontext_top_k
        else:
            cfg.pop("top_k", None)  # disable top_k (use top_p only)
    if args.recontext_top_p is not None:
        if args.recontext_top_p > 0:
            cfg["top_p"] = args.recontext_top_p
        else:
            cfg.pop("top_p", None)  # disable top_p (use top_k only)
    if args.recontext_strength is not None:
        cfg["strength"] = args.recontext_strength
    if args.recontext_decay_factor is not None:
        cfg["decay_factor"] = args.recontext_decay_factor
    if args.recontext_ctx_warmup is not None:
        cfg["context_warmup_steps"] = args.recontext_ctx_warmup
    if args.recontext_interv_warmup is not None:
        cfg["intervention_warmup"] = args.recontext_interv_warmup
    if args.recontext_replay_rounds is not None:
        cfg["replay_rounds"] = args.recontext_replay_rounds
    if args.recontext_selection_scope is not None:
        cfg["selection_scope"] = args.recontext_selection_scope
    if args.recontext_dedup_inserted_sentences is not None:
        cfg["dedup_inserted_sentences"] = bool(args.recontext_dedup_inserted_sentences)
    if args.recontext_rescale_template:
        cfg["scale_template_tokens"] = True
    if args.recontext_static_rescaling:
        cfg["dynamic_rescale"] = False
    if args.recontext_wrap_sentence_txt is not None:
        cfg["wrap_sentence_txt"] = bool(args.recontext_wrap_sentence_txt)
    if args.recontext_replay_position is not None:
        cfg["replay_position"] = args.recontext_replay_position
    if args.recontext_use_extension_inputs is not None:
        cfg["use_extension_inputs"] = bool(args.recontext_use_extension_inputs)

    # Parse selected_heads string -> list of tuples
    selected_heads = eval(cfg["selected_heads"])

    # Compute intervention_warmup_steps
    intervention_warmup = cfg.get("intervention_warmup", "auto")
    if intervention_warmup == "auto":
        if args.use_chat_template:
            if "qwen3" in model_type:
                dummy = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt",
                    enable_thinking=args.think)
                dummy_no_gen = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=False, return_tensors="pt",
                    enable_thinking=args.think)
            else:
                dummy = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt")
                dummy_no_gen = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=True, add_generation_prompt=False, return_tensors="pt")
            intervention_warmup_steps = dummy.shape[1] - dummy_no_gen.shape[1]
        else:
            intervention_warmup_steps = 2
    elif str(intervention_warmup).isdigit():
        intervention_warmup_steps = int(intervention_warmup)
    else:
        raise ValueError(f"Unknown intervention warmup setting: {intervention_warmup}")

    # Template sequences (None when rescale_template is enabled — no masking)
    template_sequences = None
    if not cfg.get("scale_template_tokens", False):
        raw_templates = cfg.get("template_sequences", [])
        if raw_templates:
            template_sequences = [torch.LongTensor(seq) for seq in raw_templates]

    rescale_config = RescaleConfig(
        selected_heads=selected_heads,
        top_k=cfg.get("top_k"),
        top_p=cfg.get("top_p"),
        strength=cfg["strength"],
        decay_factor=cfg["decay_factor"],
        context_warmup_steps=cfg.get("context_warmup_steps", 0),
        intervention_warmup_steps=intervention_warmup_steps,
        dynamic_rescale=cfg.get("dynamic_rescale", True),
        wrap_sentence_txt=cfg.get("wrap_sentence_txt", True),
        replay_position=cfg.get("replay_position", "before_question"),
        use_extension_inputs=cfg.get("use_extension_inputs", True),
        replay_rounds=cfg.get("replay_rounds", 1),
        selection_scope=cfg.get("selection_scope", "full_prompt"),
        dedup_inserted_sentences=cfg.get("dedup_inserted_sentences", True),
        template_sequences=template_sequences,
    )

    return rescale_config


def _find_chat_prefix_token_length_before_last_user_content(conversation, tokenizer, template_kwargs):
    last_user_idx = None
    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx].get("role") == "user":
            last_user_idx = idx
            break
    if last_user_idx is None:
        return None, None

    conversation_upto_last_user = conversation[: last_user_idx + 1]
    if not conversation_upto_last_user:
        return None, None

    prompt_upto_last_user = tokenizer.apply_chat_template(
        conversation_upto_last_user,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    prompt_with_empty_last_user = list(conversation_upto_last_user)
    prompt_with_empty_last_user[-1] = dict(prompt_with_empty_last_user[-1])
    prompt_with_empty_last_user[-1]["content"] = ""
    prompt_with_empty_last_user = tokenizer.apply_chat_template(
        prompt_with_empty_last_user,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )

    # The common prefix pins the first character of the final user content
    # in the exact rendered prompt, regardless of chat-template quirks.
    boundary_char_pos = 0
    for full_char, empty_char in zip(prompt_upto_last_user, prompt_with_empty_last_user):
        if full_char != empty_char:
            break
        boundary_char_pos += 1

    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not full_text.startswith(prompt_upto_last_user):
        return None, None

    context_len = _find_token_prefix_length_in_full_text(
        full_text,
        tokenizer,
        boundary_char_pos,
        add_special_tokens=False,
    )
    return context_len, last_user_idx


def _find_chat_last_user_content_token_bounds(conversation, tokenizer, template_kwargs):
    last_user_idx = None
    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx].get("role") == "user":
            last_user_idx = idx
            break
    if last_user_idx is None:
        return None, None, None

    conversation_upto_last_user = conversation[: last_user_idx + 1]
    if not conversation_upto_last_user:
        return None, None, None

    prompt_upto_last_user = tokenizer.apply_chat_template(
        conversation_upto_last_user,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    prompt_with_empty_last_user = list(conversation_upto_last_user)
    prompt_with_empty_last_user[-1] = dict(prompt_with_empty_last_user[-1])
    prompt_with_empty_last_user[-1]["content"] = ""
    prompt_with_empty_last_user = tokenizer.apply_chat_template(
        prompt_with_empty_last_user,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )

    content_start_char_pos = 0
    for full_char, empty_char in zip(prompt_upto_last_user, prompt_with_empty_last_user):
        if full_char != empty_char:
            break
        content_start_char_pos += 1

    suffix_match_len = 0
    max_suffix_len = min(
        len(prompt_upto_last_user) - content_start_char_pos,
        len(prompt_with_empty_last_user) - content_start_char_pos,
    )
    while suffix_match_len < max_suffix_len:
        if prompt_upto_last_user[-1 - suffix_match_len] != prompt_with_empty_last_user[-1 - suffix_match_len]:
            break
        suffix_match_len += 1
    content_end_char_pos = len(prompt_upto_last_user) - suffix_match_len

    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not full_text.startswith(prompt_upto_last_user):
        return None, None, None

    content_start_token_len = _find_token_prefix_length_in_full_text(
        full_text,
        tokenizer,
        content_start_char_pos,
        add_special_tokens=False,
    )
    content_end_token_len = _find_token_prefix_length_in_full_text(
        full_text,
        tokenizer,
        content_end_char_pos,
        add_special_tokens=False,
    )
    return content_start_token_len, content_end_token_len, last_user_idx


def _find_token_prefix_length_in_full_text(full_text, tokenizer, prefix_char_length, add_special_tokens):
    if prefix_char_length < 0 or prefix_char_length > len(full_text):
        return None

    full_encoding = tokenizer(
        full_text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    offsets = full_encoding.get("offset_mapping")
    if offsets is None:
        return None

    prefix_token_length = len(offsets)
    for idx, (_, end) in enumerate(offsets):
        if end >= prefix_char_length:
            prefix_token_length = idx + 1
            break
    return prefix_token_length


def _find_marker_boundary_token_length(full_text, tokenizer, boundary_pattern, closing_marker, add_special_tokens):
    boundary_pos = full_text.rfind(boundary_pattern)
    if boundary_pos < 0:
        return None
    prefix_char_length = boundary_pos + len(closing_marker)
    return _find_token_prefix_length_in_full_text(
        full_text,
        tokenizer,
        prefix_char_length,
        add_special_tokens=add_special_tokens,
    )


def _find_user_content_anchor_token_length(full_content, prefix_char_length, tokenizer, use_chat_template, template_kwargs):
    if use_chat_template:
        rendered_full = tokenizer.apply_chat_template(
            [{"role": "user", "content": full_content}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        rendered_empty = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        content_start_char_pos = 0
        for full_char, empty_char in zip(rendered_full, rendered_empty):
            if full_char != empty_char:
                break
            content_start_char_pos += 1
        anchor_char_pos = content_start_char_pos + prefix_char_length
        return _find_token_prefix_length_in_full_text(
            rendered_full,
            tokenizer,
            anchor_char_pos,
            add_special_tokens=False,
        )

    return _find_token_prefix_length_in_full_text(
        full_content,
        tokenizer,
        prefix_char_length,
        add_special_tokens=True,
    )


_HELMET_REPLAY_PREFIXES = (
    "kilt_nq",
    "nq",
    "kilt_triviaqa",
    "triviaqa",
    "kilt_hotpotqa",
    "hotpot",
    "kilt_popqa",
    "popqa",
    "narrativeqa",
    "infbench_qa_eng",
    "infbench_choice_eng",
    "infqa",
    "infmc",
)


def _is_helmet_question_boundary_dataset(dataset_name):
    dataset_key = dataset_name.lower()
    return any(dataset_key.startswith(prefix) for prefix in _HELMET_REPLAY_PREFIXES)


def _find_last_question_marker_char_pos(input_prompt):
    marker_pos = -1
    for marker in ("\n\nQuestion:", "\nQuestion:"):
        marker_pos = max(marker_pos, input_prompt.rfind(marker))
    return None if marker_pos < 0 else marker_pos


def _build_user_side_replay_context(ex, tokenizer, model_type, use_chat_template, think, dataset_name):
    template_kwargs = {}
    if "qwen3" in model_type:
        template_kwargs["enable_thinking"] = think

    input_prompt = ex["input_prompt"]

    if dataset_name.startswith("mrcr") and isinstance(input_prompt, list):
        _, content_end_token_len, last_user_idx = _find_chat_last_user_content_token_bounds(
            input_prompt,
            tokenizer,
            template_kwargs,
        )
        if content_end_token_len is None or last_user_idx is None:
            return None
        return {
            "user_side_replay_kind": "mrcr_last_user_append",
            "user_side_anchor_token_length": content_end_token_len,
            "user_side_conversation": deepcopy(input_prompt),
            "user_side_last_user_idx": last_user_idx,
            "user_side_template_kwargs": template_kwargs,
        }

    if dataset_name.startswith("longbenchv2") and isinstance(input_prompt, str):
        suffix_markers = [
            '\n\nFormat your response as follows: "The correct answer is (insert choice here)".',
            '\n\nLet\'s think step by step. After thinking, choose a single, most likely answer. Output your final answer follows: "The correct answer is (insert choice here)".',
            "\n\nLet's think step by step first and then choose a single answer. Format your response following the example below.\n\nExample Response Format:\nREASONING:\n(reasoning process here)\n\nANSWER:\nThe correct answer is (insert choice here).",
        ]
        suffix_start = None
        for marker in suffix_markers:
            marker_pos = input_prompt.find(marker)
            if marker_pos >= 0:
                suffix_start = marker_pos
                break
        if suffix_start is None:
            return None
        prefix_text = input_prompt[:suffix_start]
        suffix_text = input_prompt[suffix_start:]
        anchor_token_length = _find_user_content_anchor_token_length(
            input_prompt,
            len(prefix_text),
            tokenizer,
            use_chat_template,
            template_kwargs,
        )
        return {
            "user_side_replay_kind": "single_user_string_insert",
            "user_side_anchor_token_length": anchor_token_length,
            "user_side_prefix_text": prefix_text,
            "user_side_suffix_text": suffix_text,
            "user_side_use_chat_template": use_chat_template,
            "user_side_template_kwargs": template_kwargs,
        }

    if dataset_name.startswith("clipper") and isinstance(input_prompt, str):
        suffix_marker = (
            "\n\nFirst provide an explanation of your decision-making process, and then provide your final answer. "
            "Use the following format:\n\n<explanation>YOUR EXPLANATION</explanation>\n<answer>YOUR ANSWER</answer>"
        )
        suffix_start = input_prompt.find(suffix_marker)
        if suffix_start < 0:
            return None
        prefix_text = input_prompt[:suffix_start]
        suffix_text = input_prompt[suffix_start:]
        anchor_token_length = _find_user_content_anchor_token_length(
            input_prompt,
            len(prefix_text),
            tokenizer,
            use_chat_template,
            template_kwargs,
        )
        return {
            "user_side_replay_kind": "single_user_string_insert",
            "user_side_anchor_token_length": anchor_token_length,
            "user_side_prefix_text": prefix_text,
            "user_side_suffix_text": suffix_text,
            "user_side_use_chat_template": use_chat_template,
            "user_side_template_kwargs": template_kwargs,
        }

    if _is_helmet_question_boundary_dataset(dataset_name) and isinstance(input_prompt, str):
        question_start = _find_last_question_marker_char_pos(input_prompt)
        if question_start is None:
            return None
        prefix_text = input_prompt[:question_start]
        suffix_text = input_prompt[question_start:]
        anchor_token_length = _find_user_content_anchor_token_length(
            input_prompt,
            len(prefix_text),
            tokenizer,
            use_chat_template,
            template_kwargs,
        )
        return {
            "user_side_replay_kind": "single_user_string_insert",
            "user_side_anchor_token_length": anchor_token_length,
            "user_side_prefix_text": prefix_text,
            "user_side_suffix_text": suffix_text,
            "user_side_use_chat_template": use_chat_template,
            "user_side_template_kwargs": template_kwargs,
        }

    return None


def build_prompt_segment_context(ex, tokenizer, model_type, use_chat_template, think, dataset_name, replay_position="before_question"):
    template_kwargs = {}
    if "qwen3" in model_type:
        template_kwargs["enable_thinking"] = think

    input_prompt = ex["input_prompt"]

    if dataset_name.startswith("mrcr") and isinstance(input_prompt, list):
        conversation = input_prompt
        context_len, last_user_idx = _find_chat_prefix_token_length_before_last_user_content(
            conversation,
            tokenizer,
            template_kwargs,
        )
        if last_user_idx is None:
            return None

        full_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs,
        )
        question_len = full_ids.shape[1] - context_len
        if context_len <= 0 or question_len <= 0:
            return None
        result = {
            "prompt_context_token_length": context_len,
            "prompt_question_token_length": question_len,
        }
        if replay_position == "after_question_user_side":
            user_side_context = _build_user_side_replay_context(
                ex, tokenizer, model_type, use_chat_template, think, dataset_name,
            )
            if user_side_context is None:
                raise ValueError("Failed to build user-side replay context for MRCR prompt.")
            result.update(user_side_context)
        return result

    if dataset_name.startswith("longbenchv2") and isinstance(input_prompt, str):
        closing_marker = "</text>"
        boundary_pattern = "</text>\n\nWhat is the correct answer to this question:"

        if use_chat_template:
            conversation = [{"role": "user", "content": input_prompt}]
            full_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            context_len = _find_marker_boundary_token_length(
                full_text,
                tokenizer,
                boundary_pattern,
                closing_marker,
                add_special_tokens=False,
            )
            full_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
        else:
            context_len = _find_marker_boundary_token_length(
                input_prompt,
                tokenizer,
                boundary_pattern,
                closing_marker,
                add_special_tokens=True,
            )
            full_ids = tokenizer.encode(input_prompt, return_tensors="pt")

        if context_len is None:
            return None
        question_len = full_ids.shape[1] - context_len
        if context_len <= 0 or question_len <= 0:
            return None
        result = {
            "prompt_context_token_length": context_len,
            "prompt_question_token_length": question_len,
        }
        if replay_position == "after_question_user_side":
            user_side_context = _build_user_side_replay_context(
                ex, tokenizer, model_type, use_chat_template, think, dataset_name,
            )
            if user_side_context is None:
                raise ValueError("Failed to build user-side replay context for LongBench prompt.")
            result.update(user_side_context)
        return result

    if dataset_name.startswith("clipper") and isinstance(input_prompt, str):
        closing_marker = "</context>"
        boundary_pattern = "</context>\n\n\n<statement>"

        if use_chat_template:
            conversation = [{"role": "user", "content": input_prompt}]
            full_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            context_len = _find_marker_boundary_token_length(
                full_text,
                tokenizer,
                boundary_pattern,
                closing_marker,
                add_special_tokens=False,
            )
            full_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
        else:
            context_len = _find_marker_boundary_token_length(
                input_prompt,
                tokenizer,
                boundary_pattern,
                closing_marker,
                add_special_tokens=True,
            )
            full_ids = tokenizer.encode(input_prompt, return_tensors="pt")

        if context_len is None:
            return None
        question_len = full_ids.shape[1] - context_len
        if context_len <= 0 or question_len <= 0:
            return None
        result = {
            "prompt_context_token_length": context_len,
            "prompt_question_token_length": question_len,
        }
        if replay_position == "after_question_user_side":
            user_side_context = _build_user_side_replay_context(
                ex, tokenizer, model_type, use_chat_template, think, dataset_name,
            )
            if user_side_context is None:
                raise ValueError("Failed to build user-side replay context for CLIPPER prompt.")
            result.update(user_side_context)
        return result

    if _is_helmet_question_boundary_dataset(dataset_name) and isinstance(input_prompt, str):
        question_start = _find_last_question_marker_char_pos(input_prompt)
        if question_start is None:
            return None

        context_len = _find_user_content_anchor_token_length(
            input_prompt,
            question_start,
            tokenizer,
            use_chat_template,
            template_kwargs,
        )
        if use_chat_template:
            conversation = [{"role": "user", "content": input_prompt}]
            full_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
        else:
            full_ids = tokenizer.encode(input_prompt, return_tensors="pt")

        if context_len is None:
            return None
        question_len = full_ids.shape[1] - context_len
        if context_len <= 0 or question_len <= 0:
            return None
        result = {
            "prompt_context_token_length": context_len,
            "prompt_question_token_length": question_len,
        }
        if replay_position == "after_question_user_side":
            user_side_context = _build_user_side_replay_context(
                ex, tokenizer, model_type, use_chat_template, think, dataset_name,
            )
            if user_side_context is None:
                raise ValueError("Failed to build user-side replay context for HELMET-style prompt.")
            result.update(user_side_context)
        return result

    return None


def duplicate_question_in_prompt(ex, dataset_name):
    input_prompt = ex["input_prompt"]

    if dataset_name.startswith("mrcr") and isinstance(input_prompt, list):
        last_user_message = None
        for message in reversed(input_prompt):
            if message.get("role") == "user":
                last_user_message = deepcopy(message)
                break
        if last_user_message is None:
            return ex

        updated_ex = dict(ex)
        updated_ex["input_prompt"] = list(input_prompt) + [last_user_message]
        return updated_ex

    if dataset_name.startswith("longbenchv2") and isinstance(input_prompt, str):
        marker = "</text>"
        marker_pos = input_prompt.find(marker)
        if marker_pos < 0:
            return ex

        question_part = input_prompt[marker_pos + len(marker):]
        if not question_part:
            return ex

        updated_ex = dict(ex)
        updated_ex["input_prompt"] = input_prompt + question_part
        return updated_ex

    if dataset_name.startswith("clipper") and isinstance(input_prompt, str):
        marker = "</context>"
        marker_pos = input_prompt.find(marker)
        if marker_pos < 0:
            return ex

        question_part = input_prompt[marker_pos + len(marker):]
        if not question_part:
            return ex

        updated_ex = dict(ex)
        updated_ex["input_prompt"] = input_prompt + question_part
        return updated_ex

    return ex


def augment_dataset_prompts(dataset, dataset_name):
    return [duplicate_question_in_prompt(ex, dataset_name) for ex in dataset]


def prepare_input_ids(ex, tokenizer, model_type, use_chat_template, think):
    if isinstance(ex["input_prompt"], list):
        input_prompt = ex["input_prompt"]
        if "qwen3" in model_type:
            input_ids = tokenizer.apply_chat_template(
                input_prompt, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=think)
        else:
            input_ids = tokenizer.apply_chat_template(
                input_prompt, tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
    else:
        if use_chat_template:
            input_prompt = [{"role": "user", "content": ex["input_prompt"]}]
            if "qwen3" in model_type:
                input_ids = tokenizer.apply_chat_template(
                    input_prompt, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", enable_thinking=think)
            else:
                input_ids = tokenizer.apply_chat_template(
                    input_prompt, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt")
        else:
            input_ids = tokenizer.encode(ex["input_prompt"], return_tensors="pt")
    return input_ids


def setup_stop_token_ids(model, tokenizer, model_type):
    stop_token_ids = model.generation_config.eos_token_id
    stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
    stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
    stop_token_ids = list(set(
        [tokenizer.convert_tokens_to_ids(s) for s in stop] + stop_token_ids
    ))
    if "llama" in model_type:
        stop_token_ids.remove(tokenizer.unk_token_id)
    stop_token_ids = [x for x in stop_token_ids if x is not None]
    return stop_token_ids


def get_decoding_kwargs(args, model_type):
    if "qwen3" in model_type and args.think:
        return {
            "max_new_tokens": args.max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
        }
    else:
        return {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.temperature > 0,
        }


def print_inserted_sentence_txt(generation_logging):
    sentence_txt = generation_logging.get("sentence_txt", "")
    round_inserted_sentences = generation_logging.get("round_inserted_sentences", [])
    print("<FINAL INSERTED SENTENCES BY REPLAY ROUND>")
    if round_inserted_sentences:
        for round_idx, round_sentence_txt in enumerate(round_inserted_sentences, start=1):
            print(f"[round {round_idx}]")
            if round_sentence_txt.strip():
                print(round_sentence_txt)
            else:
                print("[EMPTY]")
        return
    if sentence_txt.strip():
        print(sentence_txt)
    else:
        print("[EMPTY]")


def extract_question_text(ex, dataset_name, generation_logging=None):
    input_prompt = ex.get("input_prompt", "")
    if isinstance(input_prompt, list):
        for message in reversed(input_prompt):
            if message.get("role") == "user":
                return message.get("content", "").strip()
        return str(input_prompt)

    if isinstance(input_prompt, str):
        question_start = None
        if dataset_name.startswith("clipper"):
            marker_pos = input_prompt.find("</context>")
            if marker_pos >= 0:
                question_start = marker_pos + len("</context>")
        elif dataset_name.startswith("longbenchv2"):
            marker_pos = input_prompt.find("</text>")
            if marker_pos >= 0:
                question_start = marker_pos + len("</text>")
        else:
            question_start = _find_last_question_marker_char_pos(input_prompt)

        if question_start is not None:
            return input_prompt[question_start:].strip()

    if generation_logging is not None:
        replay_question = generation_logging.get("debug_replay_question_text", "")
        if replay_question.strip():
            return replay_question.strip()

    return str(input_prompt).strip()


def strip_thinking_for_display(content, args):
    if args.strip_thinking and args.think and "</think>" in content:
        return content.rsplit("</think>", 1)[1]
    return content


def print_generation_debug_sample(
    ex,
    args,
    content,
    generation_logging=None,
    print_full_prompt=False,
    use_sentence_replay=False,
):
    print("-" * 50)
    if print_full_prompt:
        print("<INPUT PROMPT>")
        print(ex["input_prompt"])
    print("<QUESTION>")
    question_text = extract_question_text(ex, args.dataset, generation_logging)
    print(question_text if question_text else "[EMPTY]")
    if use_sentence_replay and generation_logging is not None:
        print_inserted_sentence_txt(generation_logging)
    print("<REFERENCE OUTPUT>")
    print(ex["reference_output"])
    print("<LLM OUTPUT>")
    print(strip_thinking_for_display(content, args))


def print_replay_prompt_segments(generation_logging):
    print("<REPLAY PROMPT SEGMENTS>")
    print("<REPLAY CONTEXT>")
    replay_context = generation_logging.get("debug_replay_context_text", "")
    print(replay_context if replay_context else "[EMPTY]")
    print("<REPLAY INSERTED SENTENCE_TXT>")
    replay_inserted = generation_logging.get("debug_replay_inserted_text", "")
    print(replay_inserted if replay_inserted else "[EMPTY]")
    print("<REPLAY QUESTION>")
    replay_question = generation_logging.get("debug_replay_question_text", "")
    print(replay_question if replay_question else "[EMPTY]")


def save_importance_details_dump(
    args,
    sample_idx,
    input_ids,
    prompt_segment_context,
    generation_logging,
    importance_details,
    rescale_config,
    tokenizer,
):
    if not args.recontext_importance_dump_dir:
        return

    dump_dir = os.path.abspath(args.recontext_importance_dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    input_token_ids = input_ids[0].detach().cpu()
    input_token_id_list = input_token_ids.tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_token_id_list)
    context_token_length = int(prompt_segment_context.get("prompt_context_token_length", 0))

    cpu_importance_details = []
    for detail in importance_details:
        cpu_detail = {}
        for key, value in detail.items():
            if torch.is_tensor(value):
                cpu_detail[key] = value.detach().cpu()
            else:
                cpu_detail[key] = value
        cpu_importance_details.append(cpu_detail)

    filename = (
        f"{args.dataset}_sample{sample_idx:03d}"
        f"_topk{rescale_config.top_k}"
        f"_rounds{getattr(rescale_config, 'replay_rounds', 1)}"
        f"_ctxwarm{rescale_config.context_warmup_steps}.pt"
    )
    save_path = os.path.join(dump_dir, filename)
    torch.save(
        {
            "sample_idx": sample_idx,
            "dataset": args.dataset,
            "model": args.model,
            "decoding_method": args.decoding_method,
            "input_token_ids": input_token_ids,
            "input_tokens": input_tokens,
            "context_token_length": context_token_length,
            "question_token_length": int(prompt_segment_context.get("prompt_question_token_length", 0)),
            "context_token_ids": input_token_ids[:context_token_length],
            "context_tokens": input_tokens[:context_token_length],
            "rescale_config": dict(rescale_config.__dict__),
            "generation_logging": generation_logging,
            "importance_details": cpu_importance_details,
        },
        save_path,
    )
    generation_logging["importance_dump_path"] = save_path
    print(f"<IMPORTANCE DUMP> {save_path}")


def should_run_boundary_sanity_example(args):
    return (
        args.decoding_method == "ReContext"
        and args.dataset == "mrcr_2needle_64k"
        and args.model.lower().rstrip("/") == "qwen/qwen3-8b"
    )


def run_flash_generation(args, model, tokenizer, dataset, model_type):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")
        # import pdb
        # pdb.set_trace()
        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        content = model.generate(input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken
        content = tokenizer.decode(content[0][input_ids.shape[1]:], skip_special_tokens=True)
        if i < 2:
            print("-" * 50)
            print("<INPUT PROMPT>")
            print(ex["input_prompt"])
            print("<REFERENCE OUTPUT>")
            print(ex["reference_output"])
            print("<OUTPUT>")
            print(content)
        else:
            print("-" * 50)
            print("<REFERENCE OUTPUT>")
            print(ex["reference_output"])
            print("<OUTPUT>")
            print(content)
        outputs.append({"prompt": ex["input_prompt"], "output": content, "success": True, "time_taken": time_taken})
        # torch.cuda.empty_cache()
    return outputs


def run_rescale_generation(
    args,
    model,
    tokenizer,
    dataset,
    model_type,
    rescale_config,
    use_sentence_replay=False,
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)
    print("Rescale config", rescale_config.__dict__)

    run_boundary_sanity_example = use_sentence_replay and should_run_boundary_sanity_example(args)
    boundary_sanity_example_done = False
    if run_boundary_sanity_example:
        print("Boundary sanity example is enabled for this run (one sample only).")

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")

        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        print("INPUT LENGTH", input_ids.shape[1])

        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        decoding_kwargs["rescale_config"] = rescale_config
        dump_importance_details = bool(args.recontext_importance_dump_dir and use_sentence_replay)
        if dump_importance_details:
            decoding_kwargs["return_importance_details"] = True
        prompt_segment_context = None
        if use_sentence_replay:
            decoding_kwargs["tokenizer"] = tokenizer
            effective_replay_position = (
                "before_question"
                if getattr(rescale_config, "replay_rounds", 1) > 1
                else rescale_config.replay_position
            )
            prompt_segment_context = build_prompt_segment_context(
                ex,
                tokenizer,
                model_type,
                args.use_chat_template,
                args.think,
                args.dataset,
                replay_position=effective_replay_position,
            )
            if prompt_segment_context is None:
                raise ValueError(
                    "decoding_method=ReContext requires a prompt/context boundary for sentence replay, "
                    f"but none was found for dataset={args.dataset!r} at sample index {i}."
                )
            if (
                run_boundary_sanity_example
                and not boundary_sanity_example_done
            ):
                prompt_segment_context = dict(prompt_segment_context)
                prompt_segment_context["debug_boundary_sanity_check"] = True
                boundary_sanity_example_done = True
            decoding_kwargs["prompt_segment_context"] = prompt_segment_context
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        generation_result = model.rescale_generate(input_ids, **decoding_kwargs)
        importance_details = None
        if dump_importance_details:
            content, generation_logging, importance_details = generation_result
        else:
            content, generation_logging = generation_result
        time_taken = time.time() - time_taken
        if dump_importance_details:
            save_importance_details_dump(
                args,
                i,
                input_ids,
                prompt_segment_context,
                generation_logging,
                importance_details,
                rescale_config,
                tokenizer,
            )
        replay_token_count = 0
        if use_sentence_replay:
            replay_token_count = generation_logging.get(
                "prompt_token_delta",
                generation_logging.get("forced_append_tokens", 0),
            )
        content = tokenizer.decode(
            content[0][input_ids.shape[1] + replay_token_count:],
            skip_special_tokens=True,
        )
        print_generation_debug_sample(
            ex,
            args,
            content,
            generation_logging=generation_logging,
            print_full_prompt=i < 2,
            use_sentence_replay=use_sentence_replay,
        )
        # print_replay_prompt_segments(generation_logging)
        outputs.append({
            "prompt": ex["input_prompt"], "output": content, "success": True,
            "time_taken": time_taken, "generation_logging": generation_logging,
        })
        # torch.cuda.empty_cache()
    return outputs


def run_attnsharp_generation(args, model, tokenizer, dataset, model_type):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")

        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        decoding_kwargs["use_attnsharp"] = True
        decoding_kwargs["attention_logits_temperature"] = args.attention_logits_temperature
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        content, generation_logging = model.rescale_generate(input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken
        content = tokenizer.decode(content[0][input_ids.shape[1]:], skip_special_tokens=True)
        if i < 2:
            print("-" * 50)
            print("<INPUT PROMPT>")
            print(ex["input_prompt"])
            print("<REFERENCE OUTPUT>")
            print(ex["reference_output"])
            print("<OUTPUT>")
            print(content)
        else:
            print("-" * 50)
            print("<REFERENCE OUTPUT>")
            print(ex["reference_output"])
            print("<OUTPUT>")
            print(content)
        outputs.append({
            "prompt": ex["input_prompt"], "output": content, "success": True,
            "time_taken": time_taken, "generation_logging": generation_logging,
        })
        # torch.cuda.empty_cache()
    return outputs


def run_dac_generation(args, model, tokenizer, dataset, model_type):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = None
    if args.stop_on_newline:
        stop_token_ids = setup_stop_token_ids(model, tokenizer, model_type)
    print("Use chat template", args.use_chat_template, "Stop on newline", args.stop_on_newline)
    print(
        "DAC config",
        {
            "mode": "offline_compressed_data",
            "compression_rate": args.dac_compression_rate,
            "data_path": args.dac_data_path,
        },
    )

    outputs = []
    for i, ex in tqdm(enumerate(dataset), desc="Running DAC generation", total=len(dataset)):
        if isinstance(ex["input_prompt"], list) and not args.use_chat_template:
            raise ValueError("Input prompt is already a list, chat template should be applied.")

        input_ids = prepare_input_ids(ex, tokenizer, model_type, args.use_chat_template, args.think)
        input_ids = input_ids.to(model.device)

        decoding_kwargs = get_decoding_kwargs(args, model_type)
        if args.stop_on_newline:
            decoding_kwargs["eos_token_id"] = stop_token_ids

        dac_logging = {
            "dac_mode": "offline_compressed_data",
            "dac_compression_rate": args.dac_compression_rate,
            "dac_data_path": args.dac_data_path,
            "prompt_tokens": input_ids.shape[1],
        }

        reset_all_seeds(args.generation_seed)
        time_taken = time.time()
        content = model.generate(input_ids, **decoding_kwargs)
        time_taken = time.time() - time_taken
        content = tokenizer.decode(content[0][input_ids.shape[1]:], skip_special_tokens=True)
        if i < 2:
            print("-" * 50)
            print("<INPUT PROMPT>")
            print(ex["input_prompt"])
            print("<DAC LOGGING>")
            print(dac_logging)
            print("<REFERENCE OUTPUT>")
            print(ex["reference_output"])
            print("<OUTPUT>")
            print(content)
        else:
            print("-" * 50)
            print("<DAC LOGGING>")
            print(dac_logging)
            print("<REFERENCE OUTPUT>")
            print(ex["reference_output"])
            print("<OUTPUT>")
            print(content)
        outputs.append({
            "prompt": ex["input_prompt"], "output": content, "success": True,
            "time_taken": time_taken, "generation_logging": dac_logging,
        })
    return outputs


def main():
    args = _parse_args()

    # Validate
    if args.think and "qwen3" not in args.model.lower():
        raise ValueError("Thinking mode is only supported for Qwen3 models.")
    if args.decoding_method == "attnsharp":
        assert args.attention_logits_temperature is not None and args.attention_logits_temperature > 0
    if args.decoding_method == "DAC":
        get_eval_data_path(args)

    # Detect model type and prepare config
    model_config = AutoConfig.from_pretrained(args.model)
    model_type = detect_model_type(args.model, config=model_config)
    print(f"Detected model type: {model_type}")
    model_config = maybe_enable_yarn(args, model_type, model_config)
    BaseModelClass, RescaleModelClass = _MODEL_CLASSES[model_type]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build rescale config if needed
    rescale_config = None

    if args.decoding_method in ["dysco", "ReContext"]:
        rescale_config = build_rescale_config(args, tokenizer, model_type)

    # Get output path
    output_path = get_output_path(args, rescale_config)
    print("OUTPUT PATH", output_path)

    # Auto skip
    if args.auto_skip:
        score_path = output_path.replace(".json", "scores.json")
        if os.path.exists(output_path) and os.path.exists(score_path):
            print(f"Auto-skipping: both {output_path} and {score_path} exist")
            return


    # Load data
    data_path = get_eval_data_path(args)
    print("DATA PATH", data_path)
    dataset, eval_func = load_eval_data(args.dataset, data_path=data_path)
    # dataset = augment_dataset_prompts(dataset, args.dataset)
    print(f"Dataset size: {len(dataset)}")
    if args.test_size > 0:
        random.seed(args.seed)
        if not args.no_shuffle_test_subset and not args.dataset.startswith("clipper"):
            random.shuffle(dataset)
        dataset = dataset[:args.test_size]

    # Load model
    if args.decoding_method == "flash":
        model = BaseModelClass.from_pretrained(
            args.model, device_map="auto", attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, config=model_config)
    elif args.decoding_method in ["dysco", "ReContext", "attnsharp"]:
        model = RescaleModelClass.from_pretrained(
            args.model, attn_implementation="flash_attention_2", device_map="auto",
            torch_dtype=torch.bfloat16, config=model_config)
        print("Using model class:", RescaleModelClass.__name__)
    elif args.decoding_method == "DAC":
        model = BaseModelClass.from_pretrained(
            args.model, device_map="auto", attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, config=model_config)
        print("Using model class:", BaseModelClass.__name__)

    # Run generation
    if args.decoding_method == "flash":
        outputs = run_flash_generation(args, model, tokenizer, dataset, model_type)
    elif args.decoding_method == "dysco":
        outputs = run_rescale_generation(
            args, model, tokenizer, dataset, model_type,
            rescale_config, use_sentence_replay=False)
    elif args.decoding_method == "ReContext":
        outputs = run_rescale_generation(
            args, model, tokenizer, dataset, model_type,
            rescale_config, use_sentence_replay=True)
    elif args.decoding_method == "attnsharp":
        outputs = run_attnsharp_generation(args, model, tokenizer, dataset, model_type)
    elif args.decoding_method == "DAC":
        outputs = run_dac_generation(args, model, tokenizer, dataset, model_type)

    # Evaluate
    all_metrics = []
    saving_info = []
    for ex, output in zip(dataset, outputs):
        thinking_part = None
        if args.strip_thinking:
            if args.think and "</think>" in output["output"]:
                thinking_part = output["output"].rsplit("</think>", 1)[0].replace("<think>", "")
                output_part = output["output"].rsplit("</think>", 1)[1]
                output["output"] = output_part

        ex_saving = {
            "data": ex,
            "output": {
                "prompt": output["prompt"], "output": output["output"],
                "success": output["success"], "thinking_part": thinking_part,
                "generation_logging": output.get("generation_logging", None),
                "time_taken": output["time_taken"],
            },
        }
        if args.skip_eval:
            ex_saving["metric"] = None
        else:
            mets, _ = eval_func(output["output"], ex)
            all_metrics.append(mets)
            ex_saving["metric"] = mets
        saving_info.append(ex_saving)

    # Save
    if args.skip_eval:
        output_content = {
            "args": args.__dict__,
            "saving_info": saving_info,
            "test_size": len(dataset),
        }
        with open(output_path, "w") as f:
            json.dump(output_content, f, indent=2)
        return

    # CLIPPER paired evaluation
    if "clipper" in args.dataset:
        assert len(all_metrics) % 2 == 0
        num_pairs = len(all_metrics) // 2
        paired_correct = sum(
            1 for i in range(num_pairs)
            if all_metrics[2*i]["accuracy"] == 1 and all_metrics[2*i+1]["accuracy"] == 1
        )
        avg_metrics = {"accuracy": paired_correct / num_pairs}
        for k in all_metrics[0].keys():
            if k != "accuracy":
                avg_metrics[k] = np.mean([x[k] for x in all_metrics])
    else:
        avg_metrics = {k: np.mean([x[k] for x in all_metrics]) for k in all_metrics[0].keys()}

    print([f"{k}: {v*100:.1f}" for k, v in avg_metrics.items()])

    output_content = {
        "args": args.__dict__,
        "saving_info": saving_info,
        "avg_metrics": avg_metrics,
        "test_size": len(dataset),
    }
    with open(output_path, "w") as f:
        json.dump(output_content, f, indent=2)
    with open(output_path.replace(".json", "scores.json"), "w") as f:
        json.dump(avg_metrics, f, indent=2)
    

if __name__ == "__main__":
    main()
