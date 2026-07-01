# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, List, Tuple, Dict

import torch
import torch.distributed as dist
from torch import nn
from transformers.cache_utils import DynamicCache

from transformers.dynamic_module_utils import (
    check_python_requirements,
    get_cached_module_file,
    get_class_in_module,
    resolve_trust_remote_code,
)
from transformers.generation.utils import GenerationMixin
from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GENERATION_MODES_MAPPING,
)
from transformers.utils import (
    is_accelerate_available,
    logging,
)
try:
    from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize
except ImportError:
    PunktSentenceTokenizer = None
    sent_tokenize = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)
_TOKEN_PIECE_CACHE = {}
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?=$|\s)(?:[^\S\n]*)|\n+")
_INLINE_SPECIAL_MARKER_RE = re.compile(r"\s*<\|[^|>]*\|>\s*")
_TRAILING_SPECIAL_MARKER_RE = re.compile(
    r"(?:\s*(?:<end>|</s>|<\|endoftext\|>|<\|im_end\|>|<\|eot_id\|>|<\|[^|>]+\|>))+\s*$"
)
_MULTISPACE_RE = re.compile(r"[^\S\n]{2,}")
_BOUNDARY_SANITY_EXAMPLE_DONE = False

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module


# RescaleConfig as a dataclass
@dataclass
class RescaleConfig:
    selected_heads: str = None
    top_k: int = None
    top_p: float = None
    selection_method: str = None
    strength: float = None
    decay_factor: float = None
    context_warmup_steps: int = 0
    intervention_warmup_steps: int = 0
    dynamic_rescale: bool = True
    wrap_sentence_txt: bool = True
    replay_position: str = "before_question"
    use_extension_inputs: bool = True
    replay_rounds: int = 1
    selection_scope: str = "full_prompt"
    dedup_inserted_sentences: bool = True
    template_sequences: list = None  # list of torch.LongTensor for template token sequences

    def __post_init__(self):
        assert self.selected_heads is not None
        assert self.top_k is not None or self.top_p is not None
        if self.top_k is not None and self.top_p is not None:
            self.selection_method = "hybrid"
        elif self.top_k is not None:
            self.selection_method = "top_k"
        elif self.top_p is not None:
            self.selection_method = "top_percentile"
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        assert self.strength is not None
        assert self.decay_factor is not None
        if self.replay_position not in {"before_question", "after_question", "after_question_user_side"}:
            raise ValueError(f"Unknown replay_position: {self.replay_position}")
        self.replay_rounds = int(self.replay_rounds)
        if self.replay_rounds < 1:
            raise ValueError(f"replay_rounds must be >= 1, got {self.replay_rounds}")
        if self.selection_scope not in {"context", "full_prompt"}:
            raise ValueError(f"Unknown selection_scope: {self.selection_scope}")
        self.dedup_inserted_sentences = bool(self.dedup_inserted_sentences)



def _nucleus_mask(attn: torch.Tensor, p: float):
    """
    attn: [B, L] attention weights (assumed >= 0)
    p: float in (0, 1]
    returns: mask [B, L] (True = kept)
    """
    # normalize if needed
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # sort descending
    sorted_vals, sorted_idx = torch.sort(attn, dim=-1, descending=True)

    # cumulative probability
    cumvals = sorted_vals.cumsum(dim=-1)

    # keep tokens until cumulative prob >= p
    keep_sorted = cumvals <= p

    # always keep at least one token
    keep_sorted[..., 0] = True

    # scatter back to original order
    mask = torch.zeros_like(attn, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)

    return mask


def _build_intervention_vector(selected_mask, reference_tensor, strength, non_template_mask, dtype=None):
    """Build attention logits intervention vector from selected mask and strength.

    Args:
        selected_mask: bool tensor marking important tokens
        reference_tensor: tensor whose shape/device to match for the output
        strength: rescaling multiplier (>=99.0 means mask non-top to -inf)
        non_template_mask: inverted template mask (True = non-template token), or None
        dtype: explicit dtype for the output tensor; when None inherits from reference_tensor
    """
    if non_template_mask is not None and 0 < strength < 99.0:
        if selected_mask.shape[1] < non_template_mask.shape[1]:
            selected_mask = selected_mask & non_template_mask[:, :selected_mask.shape[1]]
        else:
            selected_mask[:, :non_template_mask.shape[1]] = (
                selected_mask[:, :non_template_mask.shape[1]] & non_template_mask
            )

    ones_kwargs = {"dtype": dtype} if dtype is not None else {}
    if 0 < strength < 99.0:
        vec = torch.ones_like(reference_tensor, **ones_kwargs)
        vec[selected_mask] = strength
        vec[:, -1] = strength
        vec = torch.log(vec)
    elif strength >= 99.0:
        vec = torch.full_like(reference_tensor, float('-inf'), **ones_kwargs)
        vec[selected_mask] = 0.0
        vec[:, -1] = 0.0
    else:
        raise ValueError(f"Invalid strength: {strength}")
    return vec


def _mask_selected_tokens(selected_mask, non_template_mask):
    if non_template_mask is None:
        return selected_mask
    if selected_mask.shape[1] < non_template_mask.shape[1]:
        return selected_mask & non_template_mask[:, :selected_mask.shape[1]]

    masked = selected_mask.clone()
    masked[:, :non_template_mask.shape[1]] = (
        masked[:, :non_template_mask.shape[1]] & non_template_mask
    )
    return masked


def _select_important_tokens(importance, generation_logging, selection_method, top_tokens=None, top_percentile=None):
    """Select important context tokens via top-k, nucleus (top-p), or hybrid masking.

    Updates generation_logging in place.
    """
    if selection_method == "top_k":
        selected_k = min(top_tokens, importance.shape[1])
        top_vals, top_indices = torch.topk(importance, k=selected_k, dim=1)
        selected_mask = torch.zeros_like(importance, dtype=torch.bool)
        selected_mask[:, top_indices] = True
        generation_logging["avg_num_token"] += selected_k
        generation_logging["avg_nucleus_mass"] += torch.sum(top_vals).item()
        generation_logging["scale_by_token"] += 1.0
    elif selection_method == "top_percentile":
        selected_mask = _nucleus_mask(importance, top_percentile)
        generation_logging["avg_num_token"] += selected_mask.sum(dim=1)[0].item()
        generation_logging["avg_nucleus_mass"] += top_percentile
        generation_logging["scale_by_nucleus"] += 1.0
    elif selection_method == "hybrid":
        selected_mask = _nucleus_mask(importance, top_percentile)
        num_nuclues = selected_mask.sum(dim=1)[0].item()
        if num_nuclues > top_tokens:
            selected_k = min(top_tokens, importance.shape[1])
            top_vals, top_indices = torch.topk(importance, k=selected_k, dim=1)
            selected_mask = torch.zeros_like(importance, dtype=torch.bool)
            selected_mask[:, top_indices] = True
            generation_logging["avg_num_token"] += selected_k
            generation_logging["avg_nucleus_mass"] += torch.sum(top_vals).item()
            generation_logging["scale_by_token"] += 1.0
        else:
            generation_logging["avg_num_token"] += num_nuclues
            generation_logging["avg_nucleus_mass"] += top_percentile
            generation_logging["scale_by_nucleus"] += 1.0
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")
    return selected_mask


def _select_topk_rank_range(importance, candidate_mask, start_rank: int, end_rank: int, max_count: int = None):
    """Select candidate tokens ranked in [start_rank, end_rank) by importance."""
    selected_mask = torch.zeros_like(importance, dtype=torch.bool)
    if start_rank < 0 or end_rank <= start_rank:
        return selected_mask

    rank_limit = min(end_rank, importance.shape[1])
    if rank_limit <= start_rank:
        return selected_mask

    top_vals, top_indices = torch.topk(importance, k=rank_limit, dim=1)
    range_vals = top_vals[:, start_rank:rank_limit]
    range_indices = top_indices[:, start_rank:rank_limit]
    range_valid = range_vals > 0
    for batch_idx in range(importance.shape[0]):
        valid_positions = range_indices[batch_idx][range_valid[batch_idx]]
        if max_count is not None:
            valid_positions = valid_positions[:max_count]
        if valid_positions.numel() > 0:
            selected_mask[batch_idx, valid_positions] = True
    return selected_mask & candidate_mask


def _apply_importance_decay(cur_importance, past_importance, decay_factor):
    """Blend current attention importance with past via decay, then normalize."""
    cur_importance[:, :-1] += past_importance * decay_factor
    cur_importance = cur_importance / torch.sum(cur_importance, dim=1)
    return cur_importance


def _aggregate_head_attention(attention_outputs, selected_heads):
    """Extract and average attention weights across selected (layer, head) pairs."""
    per_head = []
    for layer, head in selected_heads:
        per_head.append(attention_outputs[layer][:, head,])
    return torch.stack(per_head, dim=0).mean(dim=0).squeeze(1)


def obtain_template_sequence_mask(input_ids, template_sequences):
    """Build a boolean mask marking template token positions in input_ids."""
    batch_size, max_length = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for template in template_sequences:
        template = template.to(input_ids.device)
        template_len = len(template)
        for i in range(max_length - template_len + 1):
            matches = torch.all(input_ids[:, i:i+template_len] == template.unsqueeze(0), dim=1)
            for b in range(batch_size):
                if matches[b]:
                    mask[b, i:i+template_len] = True
    return mask


def _collect_sentence_char_spans(text: str):
    if not text:
        return []

    if sent_tokenize is not None:
        try:
            sentences = sent_tokenize(text)
            spans = []
            cursor = 0
            for sentence in sentences:
                if not sentence:
                    continue
                start = text.find(sentence, cursor)
                if start < 0:
                    continue
                end = start + len(sentence)
                if not text[start:end].isspace():
                    spans.append((start, end))
                cursor = end
            if spans:
                return spans
        except LookupError:
            pass

    if PunktSentenceTokenizer is not None:
        spans = [
            (start, end)
            for start, end in PunktSentenceTokenizer().span_tokenize(text)
            if end > start and not text[start:end].isspace()
        ]
        if spans:
            return spans

    spans = []
    sent_start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        sent_end = match.end()
        if sent_end > sent_start and not text[sent_start:sent_end].isspace():
            spans.append((sent_start, sent_end))
        sent_start = sent_end
    if sent_start < len(text) and not text[sent_start:].isspace():
        spans.append((sent_start, len(text)))
    return spans


def _strip_trailing_special_markers(text: str) -> str:
    if not text:
        return text
    text = _INLINE_SPECIAL_MARKER_RE.sub(" ", text)
    text = _TRAILING_SPECIAL_MARKER_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _get_cached_token_pieces(tokenizer, token_ids):
    tokenizer_key = (
        id(tokenizer),
        getattr(tokenizer, "name_or_path", None),
        getattr(tokenizer, "vocab_size", None),
    )
    cache = _TOKEN_PIECE_CACHE.setdefault(tokenizer_key, {})

    missing_ids = []
    seen_missing = set()
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id not in cache and token_id not in seen_missing:
            missing_ids.append(token_id)
            seen_missing.add(token_id)

    if missing_ids:
        decoded_missing = tokenizer.batch_decode(
            [[token_id] for token_id in missing_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cache.update(zip(missing_ids, decoded_missing))

    return cache


def _decode_token_ids_for_debug(tokenizer, token_ids) -> str:
    if token_ids is None:
        return ""
    if isinstance(token_ids, torch.Tensor):
        if token_ids.numel() == 0:
            return ""
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        token_ids = token_ids.detach().cpu().tolist()
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _selected_tokens_to_sentence_info(selected_mask, reference_input_ids, tokenizer):
    assert reference_input_ids.shape[0] == 1
    selected_positions = torch.nonzero(selected_mask[0], as_tuple=False).flatten().tolist()
    if not selected_positions:
        return {
            "sentence_text": "",
            "chosen_sentence_ids": [],
            "selected_positions": [],
            "token_sentence_ids": [],
            "sentence_text_by_id": {},
        }

    token_ids = reference_input_ids[0].detach().cpu().tolist()
    piece_cache = _get_cached_token_pieces(tokenizer, token_ids)

    pieces = []
    selected_token_spans = []
    selected_pos_idx = 0
    next_selected_position = selected_positions[0]
    cursor = 0
    for idx, token_id in enumerate(token_ids):
        piece = piece_cache[int(token_id)]
        pieces.append(piece)
        next_cursor = cursor + len(piece)
        if selected_pos_idx < len(selected_positions) and idx == next_selected_position:
            selected_token_spans.append((cursor, next_cursor))
            selected_pos_idx += 1
            if selected_pos_idx < len(selected_positions):
                next_selected_position = selected_positions[selected_pos_idx]
        cursor = next_cursor

    decoded_text = "".join(pieces)
    sentence_spans = _collect_sentence_char_spans(decoded_text)
    if not sentence_spans:
        return {
            "sentence_text": "",
            "chosen_sentence_ids": [],
            "selected_positions": selected_positions,
            "token_sentence_ids": [[] for _ in selected_positions],
            "sentence_text_by_id": {},
        }

    chosen_sentence_ids = []
    seen_sentence_ids = set()
    sentence_idx = 0
    token_sentence_ids = []
    for token_start, token_end in selected_token_spans:
        while sentence_idx < len(sentence_spans) and sentence_spans[sentence_idx][1] <= token_start:
            sentence_idx += 1

        candidate_idx = sentence_idx
        current_token_sentence_ids = []
        while candidate_idx < len(sentence_spans) and sentence_spans[candidate_idx][0] < token_end:
            current_token_sentence_ids.append(candidate_idx)
            if candidate_idx not in seen_sentence_ids:
                seen_sentence_ids.add(candidate_idx)
                chosen_sentence_ids.append(candidate_idx)
            if sentence_spans[candidate_idx][1] >= token_end:
                break
            candidate_idx += 1
        token_sentence_ids.append(current_token_sentence_ids)

    cleaned_sentences = []
    sentence_text_by_id = {}
    for sentence_idx in chosen_sentence_ids:
        span_start, span_end = sentence_spans[sentence_idx]
        cleaned_sentence = _strip_trailing_special_markers(decoded_text[span_start:span_end])
        if cleaned_sentence.strip():
            sentence_text_by_id[sentence_idx] = cleaned_sentence
            cleaned_sentences.append(cleaned_sentence)

    return {
        "sentence_text": "\n".join(cleaned_sentences),
        "chosen_sentence_ids": chosen_sentence_ids,
        "selected_positions": selected_positions,
        "token_sentence_ids": token_sentence_ids,
        "sentence_text_by_id": sentence_text_by_id,
    }


def _selected_tokens_to_sentence_text(selected_mask, reference_input_ids, tokenizer):
    sentence_info = _selected_tokens_to_sentence_info(selected_mask, reference_input_ids, tokenizer)
    return (
        sentence_info["sentence_text"],
        sentence_info["chosen_sentence_ids"],
        sentence_info["selected_positions"],
    )


def _snapshot_past_key_values_to_cpu(past_key_values):
    if past_key_values is None:
        return None

    if isinstance(past_key_values, tuple):
        legacy_cache = past_key_values
    elif hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
    else:
        raise TypeError(f"Unsupported cache type for snapshot: {type(past_key_values)!r}")

    return tuple(
        (
            key_states.detach().cpu().clone(),
            value_states.detach().cpu().clone(),
        )
        for key_states, value_states in legacy_cache
    )


def _restore_past_key_values_from_cpu(cache_snapshot, config, device):
    if cache_snapshot is None:
        return None

    restored_cache = DynamicCache(config=config)
    for layer_idx, (key_states, value_states) in enumerate(cache_snapshot):
        restored_cache.update(
            key_states.to(device=device),
            value_states.to(device=device),
            layer_idx,
        )
    return restored_cache


def _trim_last_token_from_past_key_values(past_key_values, config, device):
    if past_key_values is None:
        return None

    if isinstance(past_key_values, tuple):
        legacy_cache = past_key_values
        return tuple(
            (
                key_states[:, :, :-1, :].contiguous(),
                value_states[:, :, :-1, :].contiguous(),
            )
            for key_states, value_states in legacy_cache
        )

    if not hasattr(past_key_values, "to_legacy_cache"):
        raise TypeError(f"Unsupported cache type for trim: {type(past_key_values)!r}")

    legacy_cache = past_key_values.to_legacy_cache()
    trimmed_cache = DynamicCache(config=config)
    for layer_idx, (key_states, value_states) in enumerate(legacy_cache):
        if key_states.shape[2] <= 0:
            raise ValueError("Cannot trim an empty KV cache.")
        trimmed_cache.update(
            key_states[:, :, :-1, :].to(device=device).contiguous(),
            value_states[:, :, :-1, :].to(device=device).contiguous(),
            layer_idx,
        )
    return trimmed_cache


def _extend_max_length_stopping_criteria(stopping_criteria, added_tokens: int):
    if added_tokens <= 0:
        return
    for criterion in stopping_criteria:
        if hasattr(criterion, "max_length") and criterion.max_length is not None:
            criterion.max_length += added_tokens


def _normalize_user_side_inserted_text(inserted_text: str) -> str:
    if not inserted_text:
        return inserted_text

    # A plain leading space keeps the user-content prefix tokenization stable.
    # Leading newlines can retokenize the last original token before the anchor.
    stripped = inserted_text.lstrip()
    if not stripped:
        return " "
    return " " + stripped


def _split_sentence_text(sentence_txt: str) -> List[str]:
    if not sentence_txt:
        return []
    return [sentence.strip() for sentence in sentence_txt.splitlines() if sentence.strip()]


def _sentence_dedup_key(sentence: str) -> str:
    return _MULTISPACE_RE.sub(" ", sentence.strip())


def _filter_sentence_info_for_replay(sentence_info, seen_inserted_sentence_keys):
    sentence_text_by_id = sentence_info.get("sentence_text_by_id", {})
    inserted_sentences = []
    inserted_sentence_keys = []
    accepted_sentence_ids = set()
    round_seen_sentence_keys = set()

    for sentence_id in sentence_info.get("chosen_sentence_ids", []):
        sentence = sentence_text_by_id.get(sentence_id, "")
        sentence_key = _sentence_dedup_key(sentence)
        if (
            not sentence_key
            or sentence_key in seen_inserted_sentence_keys
            or sentence_key in round_seen_sentence_keys
        ):
            continue
        inserted_sentences.append(sentence)
        inserted_sentence_keys.append(sentence_key)
        accepted_sentence_ids.add(sentence_id)
        round_seen_sentence_keys.add(sentence_key)

    covered_sentence_ids = set()
    wasted_selected_positions = []
    for position, token_sentence_ids in zip(
        sentence_info.get("selected_positions", []),
        sentence_info.get("token_sentence_ids", []),
    ):
        accepted_token_sentence_ids = [
            sentence_id
            for sentence_id in token_sentence_ids
            if sentence_id in accepted_sentence_ids
        ]
        uncovered_sentence_ids = [
            sentence_id
            for sentence_id in accepted_token_sentence_ids
            if sentence_id not in covered_sentence_ids
        ]
        if uncovered_sentence_ids:
            covered_sentence_ids.add(uncovered_sentence_ids[0])
        else:
            wasted_selected_positions.append(position)

    return inserted_sentences, inserted_sentence_keys, wasted_selected_positions


def _format_multi_round_inserted_text(round_sentence_texts: List[str], wrap_sentence_txt: bool) -> str:
    sentence_block = "\n".join(
        sentence_text.strip()
        for sentence_text in round_sentence_texts
        if sentence_text and sentence_text.strip()
    )
    if not sentence_block:
        return ""
    if wrap_sentence_txt:
        return (
            "\n below are possible supporting evidence from the context: \n"
            f"<Extra Info>\n{sentence_block}\n</Extra Info>\n"
        )
    return sentence_block


class CustomGenerationMixin(GenerationMixin):
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes.
    Inheriting from this class causes the model to have special generation-related behavior, such as loading a
    `GenerationConfig` at initialization time or ensuring `generate`-related tests are run in `transformers` CI.

    A model class should inherit from `GenerationMixin` to enable calling methods like `generate`, or when it
    has defined a custom `generate` method that relies on `GenerationMixin`, directly or indirectly, which
    approximately shares the same interface to public methods like `generate`. Three examples:
        - `LlamaForCausalLM` should inherit from `GenerationMixin` to enable calling `generate` and other public
            methods in the mixin;
        - `BlipForQuestionAnswering` has a custom `generate` method that approximately shares the same interface as
           `GenerationMixin.generate` (it has a few extra arguments, and the same output). That function also calls
           `GenerationMixin.generate` indirectly, through an inner model. As such, `BlipForQuestionAnswering` should
           inherit from `GenerationMixin` to benefit from all generation-related automation in our codebase;
        - `BarkModel` has a custom `generate` method and one of its inner models calls `GenerationMixin.generate`.
            However, its `generate` does not share the same interface as `GenerationMixin.generate`. In this case,
            `BarkModel` should NOT inherit from `GenerationMixin`, as it breaks the `generate` interface.

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def load_custom_generate(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ) -> Callable:
        """
        Loads and returns a custom generate function, given a model repo.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                 Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
            trust_remote_code (`bool`, *optional*):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            **kwargs:
                Additional keyword arguments for remote code loading.

        Raises:
            OSError: If `pretrained_model_name_or_path` does not contain a `custom_generate` subdirectory.

        Returns:
            A callable that can be used to generate text.
        """
        # Fetches the generate.py file from the model repo. If it doesn't exist, a file in `.no_exist` cache directory
        # is created (preventing future hub requests), and an OSError is raised.
        try:
            module = get_cached_module_file(
                pretrained_model_name_or_path, module_file="custom_generate/generate.py", **kwargs
            )
        except OSError:
            raise OSError(
                f"`{pretrained_model_name_or_path}` does not contain a `custom_generate` subdirectory with a "
                "`generate.py` file, can't load the custom generate function."
            )

        # Handle opt-in `trust_remote_code` and related exceptions
        is_local_code = os.path.exists(pretrained_model_name_or_path)
        error_message = (
            f"The repository `{pretrained_model_name_or_path}` contains custom generation code that will override "
            "the default `generate` method."
        )
        resolve_trust_remote_code(
            trust_remote_code,
            pretrained_model_name_or_path,
            has_local_code=is_local_code,
            has_remote_code=not is_local_code,
            error_message=error_message,
        )

        # Load the custom generate function
        check_python_requirements(
            pretrained_model_name_or_path, requirements_file="custom_generate/requirements.txt", **kwargs
        )
        custom_generate_function = get_class_in_module("generate", module)
        return custom_generate_function

    @torch.no_grad()
    def rescale_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[Union[str, Callable]] = None,
        rescale_config: Optional[RescaleConfig] = None,
        return_importance_details: bool = False,
        # for baselines
        use_attnsharp: bool = False,
        attention_logits_temperature: Optional[float] = None,
        tokenizer=None,
        prompt_segment_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor, Dict[str, Any]]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://huggingface.co/papers/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            use_model_defaults (`bool`, *optional*):
                When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
                generation configuration (`model.generation_config`), as opposed to the global defaults
                (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
                `True`.
            custom_generate (`str` or `Callable`, *optional*):
                One of the following:
                - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
                  `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
                  repository fully replaces the generation logic, and the return type may differ.
                - `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
                - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
                  run the decoding loop.
                For more information, see [the docs](../../generation_strategies#custom-generation-methods).
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
        trust_remote_code = kwargs.pop("trust_remote_code", None)

        if custom_generate is not None and isinstance(custom_generate, str):
            # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
            # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
            # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
            global_keys_to_exclude = {
                "self",
                "kwargs",
                "global_keys_to_exclude",
                "trust_remote_code",
                "custom_generate",
            }
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
        generation_mode_kwargs = self._extract_generation_mode_kwargs(
            custom_generate,
            kwargs,
            synced_gpus,
            assistant_model,
            streamer,
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)
        if isinstance(custom_generate, Callable):
            decoding_method = custom_generate
        else:
            # type() required to access the unbound class-level method
            decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])

        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        # Some generation modes (e.g. assisted) need `inputs_tensor` to rerun encoder.forward()
        if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
            generation_mode_kwargs["inputs_tensor"] = inputs_tensor
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # Expand inputs depending on the generation mode
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        assert GENERATION_MODES_MAPPING[generation_mode] == "_sample"
        # 9. Call generation mode
        if rescale_config is not None:
            result = self._rescale_sample(
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                rescale_config=rescale_config,
                return_importance_details=return_importance_details,
                tokenizer=tokenizer,
                prompt_segment_context=prompt_segment_context,
                **generation_mode_kwargs,
                **model_kwargs,
            )
        elif use_attnsharp and attention_logits_temperature is not None:
            result = self._sharp_sample(
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                attention_logits_temperature=attention_logits_temperature,
                **generation_mode_kwargs,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _get_attn_weights(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = repeat_kv(key_states, num_key_value_groups)
    
        # Scale before multiplication to prevent overflow
        scale = 1.0 / math.sqrt(head_dim)
        scaled_queries = query_states * scale
        attn_weights = torch.matmul(scaled_queries, key_states.transpose(2,3))

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")
        
        # make causal mask and add it to attention weights.
        causal_mask = self._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True) # Log-sum-exp of attention weights for numerical stability in softmax.
        attn_weights = torch.exp(attn_weights - attn_lses) # softmax
        return attn_weights

    def _prepare_replay_model_kwargs(self, model_kwargs, full_input_ids, past_key_values_override=Ellipsis):
        replay_kwargs = model_kwargs.copy()
        replay_kwargs.pop("cache_position", None)
        replay_kwargs.pop("position_ids", None)
        replay_kwargs.pop("decoder_position_ids", None)
        if past_key_values_override is Ellipsis:
            pass
        elif past_key_values_override is None:
            replay_kwargs.pop("past_key_values", None)
        else:
            replay_kwargs["past_key_values"] = past_key_values_override

        if self.config.is_encoder_decoder:
            replay_kwargs["decoder_attention_mask"] = torch.ones_like(full_input_ids, dtype=torch.long)
        else:
            replay_kwargs["attention_mask"] = torch.ones_like(full_input_ids, dtype=torch.long)
        return replay_kwargs

    def _run_replay_forward(
        self,
        full_input_ids: torch.LongTensor,
        model_kwargs: Dict[str, Any],
        model_forward,
        compute_logits: bool = False,
        compute_last_logits_only: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        replay_kwargs = self._prepare_replay_model_kwargs(model_kwargs, full_input_ids)
        replay_kwargs = self._get_initial_cache_position(full_input_ids.shape[1], full_input_ids.device, replay_kwargs)
        cache_position = replay_kwargs.get("cache_position")
        if (
            replay_kwargs.get("past_key_values") is not None
            and cache_position is not None
            and cache_position.numel() == 0
        ):
            # HF generation currently errors when cache length exactly matches
            # input_ids length. Roll back one cached token and recompute it.
            replay_kwargs["past_key_values"] = _trim_last_token_from_past_key_values(
                replay_kwargs["past_key_values"],
                self.config,
                full_input_ids.device,
            )
            replay_kwargs["cache_position"] = torch.tensor(
                [full_input_ids.shape[1] - 1],
                dtype=torch.long,
                device=full_input_ids.device,
            )
        model_inputs = self.prepare_inputs_for_generation(full_input_ids, **replay_kwargs)
        outputs = model_forward(
            **model_inputs,
            compute_logits=compute_logits,
            compute_last_logits_only=compute_last_logits_only,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        next_kwargs = self._prepare_replay_model_kwargs(
            replay_kwargs,
            full_input_ids,
            past_key_values_override=outputs.past_key_values,
        )
        return outputs, next_kwargs

    def _build_replay_state_from_snapshot(self, model_kwargs, cache_snapshot, prefix_length, device):
        replay_kwargs = model_kwargs.copy()
        replay_kwargs.pop("past_key_values", None)
        replay_kwargs.pop("cache_position", None)
        replay_kwargs.pop("position_ids", None)
        replay_kwargs.pop("decoder_position_ids", None)
        replay_kwargs["past_key_values"] = _restore_past_key_values_from_cpu(
            cache_snapshot, self.config, device,
        )
        if self.config.is_encoder_decoder:
            replay_kwargs["decoder_attention_mask"] = torch.ones(
                (1, prefix_length), dtype=torch.long, device=device,
            )
        else:
            replay_kwargs["attention_mask"] = torch.ones(
                (1, prefix_length), dtype=torch.long, device=device,
            )
        return replay_kwargs

    def _build_context_replay_state(self, model_kwargs, cache_snapshot, context_length, device):
        return self._build_replay_state_from_snapshot(
            model_kwargs=model_kwargs,
            cache_snapshot=cache_snapshot,
            prefix_length=context_length,
            device=device,
        )

    def _run_prompt_boundary_sanity_example(
        self,
        tokenizer,
        original_input_ids,
        context_token_length,
        question_input_ids,
        extension_inputs,
        replay_input_ids,
        generation_state,
    ):
        global _BOUNDARY_SANITY_EXAMPLE_DONE
        if _BOUNDARY_SANITY_EXAMPLE_DONE:
            return

        context_input_ids = original_input_ids[:, :context_token_length]
        expected_replay = context_input_ids
        if extension_inputs is not None:
            expected_replay = torch.cat([expected_replay, extension_inputs], dim=1)
        expected_replay = torch.cat([expected_replay, question_input_ids], dim=1)

        boundary_ok = torch.equal(question_input_ids, original_input_ids[:, context_token_length:])
        replay_structure_ok = torch.equal(replay_input_ids, expected_replay)

        restored_cache = generation_state.get("past_key_values")
        cache_seq_len = int(restored_cache.get_seq_length()) if restored_cache is not None else -1
        cache_boundary_ok = cache_seq_len == context_token_length

        verify_kwargs = self._prepare_replay_model_kwargs(generation_state, replay_input_ids)
        verify_kwargs = self._get_initial_cache_position(replay_input_ids.shape[1], replay_input_ids.device, verify_kwargs)
        verify_inputs = self.prepare_inputs_for_generation(replay_input_ids, **verify_kwargs)
        prepared_input_ids = verify_inputs.get("input_ids")
        expected_resume_ids = replay_input_ids[:, context_token_length:]
        resume_suffix_ok = (
            prepared_input_ids is not None
            and prepared_input_ids.shape[1] == expected_resume_ids.shape[1]
            and torch.equal(prepared_input_ids, expected_resume_ids)
        )
        cache_position = verify_inputs.get("cache_position")
        resume_start_pos = None
        if cache_position is not None and cache_position.numel() > 0:
            resume_start_pos = int(cache_position.reshape(-1)[0].item())
        resume_position_ok = resume_start_pos == context_token_length

        print("<PROMPT BOUNDARY SANITY EXAMPLE>")
        print(
            f"[1] boundary_detected={boundary_ok} "
            f"context_tokens={context_token_length} question_tokens={question_input_ids.shape[1]}"
        )
        print(f"[2] replay_prompt_is_context_plus_extension_plus_question={replay_structure_ok}")
        print(f"[3] kv_cache_captured_at_context_end={cache_boundary_ok} cache_seq_len={cache_seq_len}")
        print(
            f"[4] resumed_from_context_end={resume_suffix_ok and resume_position_ok} "
            f"resume_input_tokens={(prepared_input_ids.shape[1] if prepared_input_ids is not None else -1)} "
            f"expected_tokens={expected_resume_ids.shape[1]} resume_start_pos={resume_start_pos}"
        )

        all_checks_ok = boundary_ok and replay_structure_ok and cache_boundary_ok and resume_suffix_ok and resume_position_ok
        if not all_checks_ok:
            raise AssertionError("Prompt boundary sanity example failed. Check [1]-[4] logs above.")

        _BOUNDARY_SANITY_EXAMPLE_DONE = True

    def _run_append_replay_sanity_example(
        self,
        original_input_ids,
        extension_inputs,
        replay_input_ids,
        generation_state,
    ):
        global _BOUNDARY_SANITY_EXAMPLE_DONE
        if _BOUNDARY_SANITY_EXAMPLE_DONE:
            return

        expected_replay = original_input_ids
        if extension_inputs is not None:
            expected_replay = torch.cat([expected_replay, extension_inputs], dim=1)

        replay_structure_ok = torch.equal(replay_input_ids, expected_replay)

        restored_cache = generation_state.get("past_key_values")
        cache_seq_len = int(restored_cache.get_seq_length()) if restored_cache is not None else -1
        prefix_length = original_input_ids.shape[1]
        cache_boundary_ok = cache_seq_len == prefix_length

        verify_kwargs = self._prepare_replay_model_kwargs(generation_state, replay_input_ids)
        verify_kwargs = self._get_initial_cache_position(
            replay_input_ids.shape[1],
            replay_input_ids.device,
            verify_kwargs,
        )
        verify_inputs = self.prepare_inputs_for_generation(replay_input_ids, **verify_kwargs)
        prepared_input_ids = verify_inputs.get("input_ids")
        expected_resume_ids = replay_input_ids[:, prefix_length:]
        if expected_resume_ids.shape[1] == 0:
            resume_suffix_ok = prepared_input_ids is None or prepared_input_ids.shape[1] == 0
        else:
            resume_suffix_ok = (
                prepared_input_ids is not None
                and prepared_input_ids.shape[1] == expected_resume_ids.shape[1]
                and torch.equal(prepared_input_ids, expected_resume_ids)
            )

        cache_position = verify_inputs.get("cache_position")
        resume_start_pos = None
        if cache_position is not None and cache_position.numel() > 0:
            resume_start_pos = int(cache_position.reshape(-1)[0].item())
        resume_position_ok = resume_start_pos == prefix_length

        print("<APPEND REPLAY SANITY EXAMPLE>")
        print(f"[1] replay_prompt_is_original_prompt_plus_extension={replay_structure_ok}")
        print(f"[2] kv_cache_captured_at_original_prompt_end={cache_boundary_ok} cache_seq_len={cache_seq_len}")
        print(
            f"[3] resumed_from_original_prompt_end={resume_suffix_ok and resume_position_ok} "
            f"resume_input_tokens={(prepared_input_ids.shape[1] if prepared_input_ids is not None else -1)} "
            f"expected_tokens={expected_resume_ids.shape[1]} resume_start_pos={resume_start_pos}"
        )

        all_checks_ok = replay_structure_ok and cache_boundary_ok and resume_suffix_ok and resume_position_ok
        if not all_checks_ok:
            raise AssertionError("Append replay sanity example failed. Check [1]-[3] logs above.")

        _BOUNDARY_SANITY_EXAMPLE_DONE = True

    def _run_user_side_replay_sanity_example(
        self,
        original_input_ids,
        anchor_token_length,
        replay_input_ids,
        generation_state,
    ):
        global _BOUNDARY_SANITY_EXAMPLE_DONE
        if _BOUNDARY_SANITY_EXAMPLE_DONE:
            return

        prefix_ok = torch.equal(
            replay_input_ids[:, :anchor_token_length],
            original_input_ids[:, :anchor_token_length],
        )

        restored_cache = generation_state.get("past_key_values")
        cache_seq_len = int(restored_cache.get_seq_length()) if restored_cache is not None else -1
        cache_boundary_ok = cache_seq_len == anchor_token_length

        verify_kwargs = self._prepare_replay_model_kwargs(generation_state, replay_input_ids)
        verify_kwargs = self._get_initial_cache_position(
            replay_input_ids.shape[1],
            replay_input_ids.device,
            verify_kwargs,
        )
        verify_inputs = self.prepare_inputs_for_generation(replay_input_ids, **verify_kwargs)
        prepared_input_ids = verify_inputs.get("input_ids")
        expected_resume_ids = replay_input_ids[:, anchor_token_length:]
        resume_suffix_ok = (
            prepared_input_ids is not None
            and prepared_input_ids.shape[1] == expected_resume_ids.shape[1]
            and torch.equal(prepared_input_ids, expected_resume_ids)
        )

        cache_position = verify_inputs.get("cache_position")
        resume_start_pos = None
        if cache_position is not None and cache_position.numel() > 0:
            resume_start_pos = int(cache_position.reshape(-1)[0].item())
        resume_position_ok = resume_start_pos == anchor_token_length

        print("<USER-SIDE REPLAY SANITY EXAMPLE>")
        print(f"[1] replay_prefix_matches_original_prefix={prefix_ok} anchor_tokens={anchor_token_length}")
        print(f"[2] kv_cache_captured_at_user_side_anchor={cache_boundary_ok} cache_seq_len={cache_seq_len}")
        print(
            f"[3] resumed_from_user_side_anchor={resume_suffix_ok and resume_position_ok} "
            f"resume_input_tokens={(prepared_input_ids.shape[1] if prepared_input_ids is not None else -1)} "
            f"expected_tokens={expected_resume_ids.shape[1]} resume_start_pos={resume_start_pos}"
        )

        all_checks_ok = prefix_ok and cache_boundary_ok and resume_suffix_ok and resume_position_ok
        if not all_checks_ok:
            raise AssertionError("User-side replay sanity example failed. Check [1]-[3] logs above.")

        _BOUNDARY_SANITY_EXAMPLE_DONE = True

    def _build_user_side_replay_input_ids(
        self,
        tokenizer,
        prompt_segment_context,
        inserted_text: str,
        device,
    ):
        replay_kind = prompt_segment_context.get("user_side_replay_kind")
        anchor_token_length = int(prompt_segment_context.get("user_side_anchor_token_length", -1))
        if anchor_token_length <= 0:
            raise ValueError(f"Invalid user-side replay anchor token length: {anchor_token_length}")

        template_kwargs = prompt_segment_context.get("user_side_template_kwargs", {})
        if replay_kind == "mrcr_last_user_append":
            conversation = copy.deepcopy(prompt_segment_context["user_side_conversation"])
            last_user_idx = int(prompt_segment_context["user_side_last_user_idx"])
            conversation[last_user_idx] = dict(conversation[last_user_idx])
            conversation[last_user_idx]["content"] = conversation[last_user_idx]["content"] + inserted_text
            replay_input_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
        elif replay_kind == "single_user_string_insert":
            prefix_text = prompt_segment_context["user_side_prefix_text"]
            suffix_text = prompt_segment_context["user_side_suffix_text"]
            rebuilt_prompt = prefix_text + inserted_text + suffix_text
            if prompt_segment_context.get("user_side_use_chat_template", True):
                replay_input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": rebuilt_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    **template_kwargs,
                )
            else:
                replay_input_ids = tokenizer(
                    rebuilt_prompt,
                    return_tensors="pt",
                ).input_ids
        else:
            raise ValueError(f"Unknown user-side replay kind: {replay_kind}")

        replay_input_ids = replay_input_ids.to(device)
        if anchor_token_length >= replay_input_ids.shape[1]:
            raise ValueError(
                f"User-side replay anchor {anchor_token_length} must be smaller than replay prompt length "
                f"{replay_input_ids.shape[1]}"
            )
        return replay_input_ids, anchor_token_length

    def _compute_sentence_replay_round_importance(
        self,
        replay_input_ids: torch.LongTensor,
        model_kwargs: Dict[str, Any],
        model_forward,
        context_cache_snapshot,
        context_token_length: int,
        selected_heads,
        context_warmup_steps: int,
        decay_factor: float,
        return_importance_details: bool = False,
        round_idx: Optional[int] = None,
    ):
        total_len_prompt = replay_input_ids.shape[1]
        look_back_len = min(max(context_warmup_steps, 0), total_len_prompt)
        if look_back_len <= 0:
            raise ValueError("Sentence replay mode requires context_warmup_steps > 0.")

        look_back_inputs = replay_input_ids[:, -look_back_len:]
        prefix_input_ids = replay_input_ids[:, :-look_back_len]
        current_input_ids = prefix_input_ids

        if context_token_length <= prefix_input_ids.shape[1]:
            current_state = self._build_context_replay_state(
                model_kwargs,
                context_cache_snapshot,
                context_token_length,
                replay_input_ids.device,
            )
            if prefix_input_ids.shape[1] > context_token_length:
                prefix_outputs, current_state = self._run_replay_forward(
                    prefix_input_ids,
                    current_state,
                    model_forward,
                    compute_logits=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                del prefix_outputs
        else:
            current_state = self._prepare_replay_model_kwargs(
                model_kwargs,
                prefix_input_ids,
                past_key_values_override=None,
            )
            if prefix_input_ids.shape[1] > 0:
                prefix_outputs, current_state = self._run_replay_forward(
                    prefix_input_ids,
                    current_state,
                    model_forward,
                    compute_logits=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                del prefix_outputs

        past_token_importance = torch.zeros_like(prefix_input_ids, dtype=torch.float32)
        importance_details = []

        for look_back_idx in range(look_back_inputs.shape[1]):
            current_input_ids = torch.cat(
                [current_input_ids, look_back_inputs[:, look_back_idx:look_back_idx + 1]],
                dim=-1,
            )
            outputs, current_state = self._run_replay_forward(
                current_input_ids,
                current_state,
                model_forward,
                compute_logits=False,
                output_attentions=True,
                output_hidden_states=False,
            )
            cur_token_importance = _aggregate_head_attention(outputs.attentions, selected_heads).to(torch.float32)
            if return_importance_details:
                per_token_importance = {
                    "round_idx": round_idx,
                    "position_idx": current_input_ids.shape[1] - 1,
                    "attention_weights": cur_token_importance.cpu(),
                }
            past_token_importance = _apply_importance_decay(
                cur_token_importance,
                past_token_importance,
                decay_factor,
            )
            if return_importance_details:
                per_token_importance["context_scores"] = past_token_importance.cpu()
                importance_details.append(per_token_importance)
            del outputs

        return past_token_importance, importance_details

    def _multi_round_sentence_replay_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        rescale_config: Optional[RescaleConfig] = None,
        return_importance_details: bool = False,
        tokenizer=None,
        prompt_segment_context: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor, Dict[str, Any]]:
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        assert rescale_config is not None
        assert tokenizer is not None
        if self.config.is_encoder_decoder:
            raise ValueError("Multi-round sentence replay mode currently supports decoder-only models only.")

        selected_heads = rescale_config.selected_heads
        context_warmup_steps = int(rescale_config.context_warmup_steps)
        decay_factor = rescale_config.decay_factor
        top_tokens = rescale_config.top_k
        top_percentile = rescale_config.top_p
        selection_method = rescale_config.selection_method
        wrap_sentence_txt = rescale_config.wrap_sentence_txt
        replay_position = rescale_config.replay_position
        use_extension_inputs = rescale_config.use_extension_inputs
        replay_rounds = int(rescale_config.replay_rounds)
        selection_scope = rescale_config.selection_scope
        dedup_inserted_sentences = bool(rescale_config.dedup_inserted_sentences)

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Sentence replay mode currently only supports batch size 1."

        context_token_length = int(prompt_segment_context.get("prompt_context_token_length", 0))
        if context_token_length <= 0 or context_token_length >= input_ids.shape[1]:
            raise ValueError(f"Invalid prompt context length for sentence replay: {context_token_length}")

        question_input_ids = input_ids[:, context_token_length:]
        original_input_ids = input_ids
        device = original_input_ids.device

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option "
                        "`CompileConfig(fullgraph=True)` as FA2 introduces graph breaks. We overrode the option "
                        "with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        context_input_ids = original_input_ids[:, :context_token_length]
        context_state = self._prepare_replay_model_kwargs(
            model_kwargs,
            context_input_ids,
            past_key_values_override=None,
        )
        context_outputs, context_state = self._run_replay_forward(
            context_input_ids,
            context_state,
            model_forward,
            compute_logits=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        context_cache_snapshot = _snapshot_past_key_values_to_cpu(context_state.get("past_key_values"))
        if context_cache_snapshot is None:
            raise ValueError("Failed to capture context KV cache for multi-round sentence replay mode.")
        del context_outputs, context_state

        generation_logging = {
            "avg_num_token": 0.0,
            "avg_nucleus_mass": 0.0,
            "scale_by_token": 0.0,
            "scale_by_nucleus": 0.0,
            "num_generations": 0,
            "forced_append_tokens": 0,
            "requested_replay_position": replay_position,
            "effective_replay_position": "before_question",
            "use_extension_inputs": bool(use_extension_inputs),
            "replay_rounds": replay_rounds,
            "selection_scope": selection_scope,
            "dedup_inserted_sentences": bool(dedup_inserted_sentences),
            "chosen_sentence_ids": [],
            "selected_positions": [],
            "sentence_txt": "",
            "debug_replay_context_text": "",
            "debug_replay_inserted_text": "",
            "debug_replay_question_text": "",
            "round_selected_sentences": [],
            "round_inserted_sentences": [],
            "round_selected_positions": [],
            "round_chosen_sentence_ids": [],
            "round_backup_selected_sentences": [],
            "round_backup_inserted_sentences": [],
            "round_backup_selected_positions": [],
            "round_backup_chosen_sentence_ids": [],
            "round_backup_rank_ranges": [],
        }

        current_replay_input_ids = original_input_ids
        extension_inputs = None
        inserted_round_texts = []
        seen_inserted_sentence_keys = set()
        importance_details = []

        for round_idx in range(1, replay_rounds + 1):
            prompt_importance, round_importance_details = self._compute_sentence_replay_round_importance(
                replay_input_ids=current_replay_input_ids,
                model_kwargs=model_kwargs,
                model_forward=model_forward,
                context_cache_snapshot=context_cache_snapshot,
                context_token_length=context_token_length,
                selected_heads=selected_heads,
                context_warmup_steps=context_warmup_steps,
                decay_factor=decay_factor,
                return_importance_details=return_importance_details,
                round_idx=round_idx,
            )
            if return_importance_details:
                importance_details.extend(round_importance_details)

            template_sequence_mask = None
            if rescale_config.template_sequences:
                template_sequence_mask = obtain_template_sequence_mask(
                    current_replay_input_ids,
                    rescale_config.template_sequences,
                )
            candidate_mask = torch.ones_like(prompt_importance, dtype=torch.bool)
            if selection_scope == "context":
                candidate_mask[:, context_token_length:] = False
            if template_sequence_mask is not None:
                candidate_mask = candidate_mask & ~template_sequence_mask

            scoped_importance = prompt_importance.masked_fill(~candidate_mask, 0.0)
            if return_importance_details:
                importance_details.append({
                    "kind": "selection_input",
                    "round_idx": round_idx,
                    "raw_scores": prompt_importance.cpu(),
                    "selection_scores": scoped_importance.cpu(),
                    "candidate_mask": candidate_mask.cpu(),
                })
            selected_mask = torch.zeros_like(scoped_importance, dtype=torch.bool)
            if scoped_importance.sum().item() > 0:
                generation_logging["num_generations"] += 1
                selected_mask = _select_important_tokens(
                    scoped_importance,
                    generation_logging,
                    selection_method,
                    top_tokens,
                    top_percentile,
                )
                selected_mask = selected_mask & candidate_mask

            selected_sentence_info = _selected_tokens_to_sentence_info(
                selected_mask,
                current_replay_input_ids,
                tokenizer,
            )
            sentence_txt = selected_sentence_info["sentence_text"]
            chosen_sentence_ids = selected_sentence_info["chosen_sentence_ids"]
            selected_positions = selected_sentence_info["selected_positions"]

            backup_sentence_txt = ""
            backup_chosen_sentence_ids = []
            backup_selected_positions = []
            backup_inserted_sentences = []
            backup_inserted_sentence_keys = []
            backup_rank_range = None
            backup_rank_ranges = []
            if dedup_inserted_sentences:
                (
                    initial_inserted_sentences,
                    initial_inserted_sentence_keys,
                    wasted_selected_positions,
                ) = _filter_sentence_info_for_replay(
                    selected_sentence_info,
                    seen_inserted_sentence_keys,
                )

                backup_selected_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
                if (
                    use_extension_inputs
                    and wasted_selected_positions
                    and selection_method == "top_k"
                    and top_tokens is not None
                    and top_tokens > 0
                ):
                    needed_backup_token_count = len(wasted_selected_positions)
                    max_backup_rank = int(top_tokens) * max(2, round_idx)
                    backup_start_rank = int(top_tokens)
                    while backup_start_rank < max_backup_rank and needed_backup_token_count > 0:
                        backup_end_rank = backup_start_rank + int(top_tokens)
                        backup_rank_range = [backup_start_rank, backup_end_rank]
                        backup_rank_ranges.append(backup_rank_range)
                        current_backup_selected_mask = _select_topk_rank_range(
                            scoped_importance,
                            candidate_mask,
                            backup_start_rank,
                            backup_end_rank,
                            max_count=needed_backup_token_count,
                        )
                        current_backup_selected_positions = torch.nonzero(
                            current_backup_selected_mask[0],
                            as_tuple=False,
                        ).flatten().tolist()
                        if current_backup_selected_positions:
                            backup_selected_mask = backup_selected_mask | current_backup_selected_mask
                        needed_backup_token_count -= len(current_backup_selected_positions)
                        backup_start_rank = backup_end_rank

                    backup_sentence_info = _selected_tokens_to_sentence_info(
                        backup_selected_mask,
                        current_replay_input_ids,
                        tokenizer,
                    )
                    backup_sentence_txt = backup_sentence_info["sentence_text"]
                    backup_chosen_sentence_ids = backup_sentence_info["chosen_sentence_ids"]
                    backup_selected_positions = backup_sentence_info["selected_positions"]

                combined_selected_mask = selected_mask | backup_selected_mask
                combined_sentence_info = _selected_tokens_to_sentence_info(
                    combined_selected_mask,
                    current_replay_input_ids,
                    tokenizer,
                )
                inserted_sentences, inserted_sentence_keys, _ = _filter_sentence_info_for_replay(
                    combined_sentence_info,
                    seen_inserted_sentence_keys,
                )
                initial_inserted_sentence_key_set = set(initial_inserted_sentence_keys)
                backup_inserted_sentences = [
                    sentence
                    for sentence, sentence_key in zip(inserted_sentences, inserted_sentence_keys)
                    if sentence_key not in initial_inserted_sentence_key_set
                ]
                backup_inserted_sentence_keys = [
                    sentence_key
                    for sentence_key in inserted_sentence_keys
                    if sentence_key not in initial_inserted_sentence_key_set
                ]
            else:
                inserted_sentences = _split_sentence_text(sentence_txt)
                inserted_sentence_keys = []

            if not use_extension_inputs:
                inserted_sentences = []
                inserted_sentence_keys = []
            else:
                seen_inserted_sentence_keys.update(inserted_sentence_keys)

            selected_sentence_txt = sentence_txt
            inserted_sentence_txt = "\n".join(inserted_sentences)
            print(f"[ReContext replay round {round_idx}] selected sentences before dedup:")
            print(selected_sentence_txt if selected_sentence_txt.strip() else "[EMPTY]")
            if backup_rank_range is not None:
                backup_rank_range_txt = ", ".join(
                    f"top {rank_range[0] + 1} to top {rank_range[1]}"
                    for rank_range in backup_rank_ranges
                )
                print(
                    f"[ReContext replay round {round_idx}] backup selected sentences "
                    f"from {backup_rank_range_txt} before dedup:"
                )
                print(backup_sentence_txt if backup_sentence_txt.strip() else "[EMPTY]")
                print(f"[ReContext replay round {round_idx}] backup inserted sentences after dedup:")
                backup_inserted_sentence_txt = "\n".join(backup_inserted_sentences)
                print(backup_inserted_sentence_txt if backup_inserted_sentence_txt.strip() else "[EMPTY]")
            print(f"[ReContext replay round {round_idx}] inserted sentences after dedup:")
            print(inserted_sentence_txt if inserted_sentence_txt.strip() else "[EMPTY]")

            generation_logging["round_selected_sentences"].append(selected_sentence_txt)
            generation_logging["round_inserted_sentences"].append(inserted_sentence_txt)
            generation_logging["round_selected_positions"].append(
                selected_positions + backup_selected_positions
            )
            generation_logging["round_chosen_sentence_ids"].append(
                chosen_sentence_ids + backup_chosen_sentence_ids
            )
            generation_logging["round_backup_selected_sentences"].append(backup_sentence_txt)
            generation_logging["round_backup_inserted_sentences"].append(
                "\n".join(backup_inserted_sentences)
            )
            generation_logging["round_backup_selected_positions"].append(backup_selected_positions)
            generation_logging["round_backup_chosen_sentence_ids"].append(backup_chosen_sentence_ids)
            generation_logging["round_backup_rank_ranges"].append(backup_rank_ranges)
            generation_logging["chosen_sentence_ids"] = chosen_sentence_ids + backup_chosen_sentence_ids
            generation_logging["selected_positions"] = selected_positions + backup_selected_positions

            if inserted_sentence_txt.strip():
                inserted_round_texts.append(inserted_sentence_txt)

            inserted_text = _format_multi_round_inserted_text(inserted_round_texts, wrap_sentence_txt)
            if inserted_text.strip():
                extension_inputs = tokenizer(
                    inserted_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)
                if extension_inputs.shape[1] == 0:
                    extension_inputs = None
            else:
                extension_inputs = None

            current_replay_input_ids = original_input_ids[:, :context_token_length]
            if extension_inputs is not None:
                current_replay_input_ids = torch.cat([current_replay_input_ids, extension_inputs], dim=1)
            current_replay_input_ids = torch.cat([current_replay_input_ids, question_input_ids], dim=1)

        generation_logging["sentence_txt"] = "\n".join(inserted_round_texts)
        if extension_inputs is not None:
            generation_logging["forced_append_tokens"] = extension_inputs.shape[1]

        prompt_token_delta = current_replay_input_ids.shape[1] - original_input_ids.shape[1]
        generation_logging["prompt_token_delta"] = int(prompt_token_delta)
        if prompt_token_delta > 0:
            _extend_max_length_stopping_criteria(stopping_criteria, prompt_token_delta)
            if generation_config.max_length is not None:
                generation_config.max_length += prompt_token_delta

        generation_logging["debug_replay_context_text"] = _decode_token_ids_for_debug(
            tokenizer,
            original_input_ids[:, :context_token_length],
        )
        generation_logging["debug_replay_inserted_text"] = _decode_token_ids_for_debug(
            tokenizer,
            extension_inputs,
        )
        generation_logging["debug_replay_question_text"] = _decode_token_ids_for_debug(
            tokenizer,
            question_input_ids,
        )

        generation_state = self._build_context_replay_state(
            model_kwargs,
            context_cache_snapshot,
            context_token_length,
            device,
        )
        del context_cache_snapshot

        run_boundary_sanity_example = bool(prompt_segment_context.get("debug_boundary_sanity_check", False))
        if run_boundary_sanity_example:
            self._run_prompt_boundary_sanity_example(
                tokenizer=tokenizer,
                original_input_ids=original_input_ids,
                context_token_length=context_token_length,
                question_input_ids=question_input_ids,
                extension_inputs=extension_inputs,
                replay_input_ids=current_replay_input_ids,
                generation_state=generation_state,
            )

        replay_input_ids = current_replay_input_ids
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=device):
            outputs, generation_state = self._run_replay_forward(
                replay_input_ids,
                generation_state,
                model_forward,
                compute_logits=True,
                compute_last_logits_only=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)
            next_token_scores = logits_processor(replay_input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            replay_input_ids = torch.cat([replay_input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(replay_input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            del outputs

        if streamer is not None:
            streamer.end()

        if generation_logging["num_generations"] > 0:
            generation_logging["avg_num_token"] /= generation_logging["num_generations"]
            generation_logging["avg_nucleus_mass"] /= generation_logging["num_generations"]
            generation_logging["scale_by_token"] /= generation_logging["num_generations"]
            generation_logging["scale_by_nucleus"] /= generation_logging["num_generations"]

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=replay_input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=generation_state.get("past_key_values"),
            )

        if return_importance_details:
            return replay_input_ids, generation_logging, importance_details
        return replay_input_ids, generation_logging

    def _sentence_replay_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        rescale_config: Optional[RescaleConfig] = None,
        return_importance_details: bool = False,
        tokenizer=None,
        prompt_segment_context: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor, Dict[str, Any]]:
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        assert rescale_config is not None
        assert tokenizer is not None
        if self.config.is_encoder_decoder:
            raise ValueError("Sentence replay mode currently supports decoder-only models only.")
        if int(getattr(rescale_config, "replay_rounds", 1)) > 1:
            return self._multi_round_sentence_replay_sample(
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                rescale_config=rescale_config,
                return_importance_details=return_importance_details,
                tokenizer=tokenizer,
                prompt_segment_context=prompt_segment_context,
                **model_kwargs,
            )

        selected_heads = rescale_config.selected_heads
        context_warmup_steps = int(rescale_config.context_warmup_steps)
        decay_factor = rescale_config.decay_factor
        top_tokens = rescale_config.top_k
        top_percentile = rescale_config.top_p
        selection_method = rescale_config.selection_method
        wrap_sentence_txt = rescale_config.wrap_sentence_txt
        replay_position = rescale_config.replay_position
        use_extension_inputs = rescale_config.use_extension_inputs
        selection_scope = getattr(rescale_config, "selection_scope", "full_prompt")
        dedup_inserted_sentences = getattr(rescale_config, "dedup_inserted_sentences", True)

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Sentence replay mode currently only supports batch size 1."

        context_token_length = int(prompt_segment_context.get("prompt_context_token_length", 0))
        if context_token_length <= 0 or context_token_length >= input_ids.shape[1]:
            raise ValueError(f"Invalid prompt context length for sentence replay: {context_token_length}")

        template_sequence_mask = None
        if rescale_config.template_sequences:
            template_sequence_mask = obtain_template_sequence_mask(input_ids, rescale_config.template_sequences)
        non_template_mask = ~template_sequence_mask if template_sequence_mask is not None else None

        total_len_prompt = input_ids.shape[1]
        look_back_len = min(max(context_warmup_steps, 0), total_len_prompt)
        if look_back_len <= 0:
            raise ValueError("Sentence replay mode requires context_warmup_steps > 0.")

        look_back_inputs = input_ids[:, -look_back_len:]
        prefix_input_ids = input_ids[:, :-look_back_len]
        question_input_ids = input_ids[:, context_token_length:]
        original_input_ids = input_ids

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        context_cache_snapshot = None
        full_prompt_cache_snapshot = None
        current_state = model_kwargs.copy()
        current_input_ids = prefix_input_ids

        if context_token_length <= prefix_input_ids.shape[1]:
            context_input_ids = original_input_ids[:, :context_token_length]
            context_outputs, current_state = self._run_replay_forward(
                context_input_ids,
                current_state,
                model_forward,
                compute_logits=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            context_cache_snapshot = _snapshot_past_key_values_to_cpu(current_state.get("past_key_values"))
            del context_outputs
            if prefix_input_ids.shape[1] > context_token_length:
                prefix_outputs, current_state = self._run_replay_forward(
                    prefix_input_ids,
                    current_state,
                    model_forward,
                    compute_logits=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                del prefix_outputs
        elif prefix_input_ids.shape[1] > 0:
            prefix_outputs, current_state = self._run_replay_forward(
                prefix_input_ids,
                current_state,
                model_forward,
                compute_logits=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            del prefix_outputs

        past_token_importance = torch.zeros_like(prefix_input_ids, dtype=torch.float32)
        importance_details = []

        for look_back_idx in range(look_back_inputs.shape[1]):
            current_input_ids = torch.cat([current_input_ids, look_back_inputs[:, look_back_idx:look_back_idx + 1]], dim=-1)
            outputs, current_state = self._run_replay_forward(
                current_input_ids,
                current_state,
                model_forward,
                compute_logits=False,
                output_attentions=True,
                output_hidden_states=False,
            )
            cur_token_importance = _aggregate_head_attention(outputs.attentions, selected_heads).to(torch.float32)
            if return_importance_details:
                per_token_importance = {
                    "position_idx": current_input_ids.shape[1] - 1,
                    "attention_weights": cur_token_importance.cpu(),
                }
            past_token_importance = _apply_importance_decay(cur_token_importance, past_token_importance, decay_factor)
            if return_importance_details:
                per_token_importance["context_scores"] = past_token_importance.cpu()
                importance_details.append(per_token_importance)

            if context_cache_snapshot is None and current_input_ids.shape[1] == context_token_length:
                context_cache_snapshot = _snapshot_past_key_values_to_cpu(current_state.get("past_key_values"))
            del outputs

        if context_cache_snapshot is None:
            raise ValueError("Failed to capture context KV cache for sentence replay mode.")
        full_prompt_cache_snapshot = _snapshot_past_key_values_to_cpu(current_state.get("past_key_values"))
        if full_prompt_cache_snapshot is None:
            raise ValueError("Failed to capture full prompt KV cache for sentence replay mode.")
        del current_state

        # By default, select over the full prompt to preserve the original behavior.
        # The context scope keeps candidates inside the original long context only.
        prompt_importance = past_token_importance.clone()
        prompt_candidate_mask = torch.ones_like(prompt_importance, dtype=torch.bool)
        if selection_scope == "context":
            prompt_candidate_mask[:, context_token_length:] = False
        if non_template_mask is not None:
            prompt_candidate_mask = prompt_candidate_mask & non_template_mask
        prompt_importance = prompt_importance.masked_fill(~prompt_candidate_mask, 0.0)
        if return_importance_details:
            importance_details.append({
                "kind": "selection_input",
                "round_idx": 1,
                "raw_scores": past_token_importance.cpu(),
                "selection_scores": prompt_importance.cpu(),
                "candidate_mask": prompt_candidate_mask.cpu(),
            })

        generation_logging = {
            "avg_num_token": 0.0,
            "avg_nucleus_mass": 0.0,
            "scale_by_token": 0.0,
            "scale_by_nucleus": 0.0,
            "num_generations": 0,
            "forced_append_tokens": 0,
            "requested_replay_position": replay_position,
            "effective_replay_position": "before_question",
            "use_extension_inputs": bool(use_extension_inputs),
            "replay_rounds": 1,
            "selection_scope": selection_scope,
            "dedup_inserted_sentences": bool(dedup_inserted_sentences),
            "chosen_sentence_ids": [],
            "selected_positions": [],
            "sentence_txt": "",
            "debug_replay_context_text": "",
            "debug_replay_inserted_text": "",
            "debug_replay_question_text": "",
        }

        selected_mask = torch.zeros_like(prompt_importance, dtype=torch.bool)
        if prompt_importance.sum().item() > 0:
            generation_logging["num_generations"] = 1
            selected_mask = _select_important_tokens(
                prompt_importance,
                generation_logging,
                selection_method,
                top_tokens,
                top_percentile,
            )
            selected_mask = selected_mask & prompt_candidate_mask

        sentence_txt, chosen_sentence_ids, selected_positions = _selected_tokens_to_sentence_text(
            selected_mask,
            original_input_ids,
            tokenizer,
        )
        generation_logging["chosen_sentence_ids"] = chosen_sentence_ids
        generation_logging["selected_positions"] = selected_positions
        generation_logging["sentence_txt"] = sentence_txt

        inserted_text = None
        extension_inputs = None
        if sentence_txt.strip():
            if wrap_sentence_txt:
                sentence_txt = (
                    "\n below are possible supporting evidence from the context: \n"
                    f"<Extra Info>\n{sentence_txt}\n</Extra Info>\n"
                )
            inserted_text = sentence_txt
            if replay_position == "after_question_user_side":
                inserted_text = _normalize_user_side_inserted_text(inserted_text)
            extension_inputs = tokenizer(
                inserted_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(original_input_ids.device)
            if extension_inputs.shape[1] == 0:
                inserted_text = None
                extension_inputs = None

        #pdb_test
        if not use_extension_inputs:
            extension_inputs = None

        use_after_question_replay = replay_position == "after_question" and extension_inputs is not None
        use_after_question_user_side_replay = (
            replay_position == "after_question_user_side" and extension_inputs is not None
        )
        generation_logging["effective_replay_position"] = (
            "after_question_user_side"
            if use_after_question_user_side_replay
            else ("after_question" if use_after_question_replay else "before_question")
        )

        if extension_inputs is not None:
            generation_logging["forced_append_tokens"] = extension_inputs.shape[1]

        user_side_anchor_token_length = None
        if use_after_question_user_side_replay:
            replay_input_ids, user_side_anchor_token_length = self._build_user_side_replay_input_ids(
                tokenizer=tokenizer,
                prompt_segment_context=prompt_segment_context,
                inserted_text=inserted_text,
                device=original_input_ids.device,
            )
        elif use_after_question_replay:
            replay_input_ids = original_input_ids
            replay_input_ids = torch.cat([replay_input_ids, extension_inputs], dim=1)
        else:
            replay_input_ids = original_input_ids[:, :context_token_length]
            if extension_inputs is not None:
                replay_input_ids = torch.cat([replay_input_ids, extension_inputs], dim=1)
            replay_input_ids = torch.cat([replay_input_ids, question_input_ids], dim=1)

        prompt_token_delta = replay_input_ids.shape[1] - original_input_ids.shape[1]
        generation_logging["prompt_token_delta"] = int(prompt_token_delta)
        if prompt_token_delta > 0:
            _extend_max_length_stopping_criteria(stopping_criteria, prompt_token_delta)
            if generation_config.max_length is not None:
                generation_config.max_length += prompt_token_delta
        generation_logging["debug_replay_context_text"] = _decode_token_ids_for_debug(
            tokenizer,
            original_input_ids[:, :context_token_length],
        )
        generation_logging["debug_replay_inserted_text"] = _decode_token_ids_for_debug(
            tokenizer,
            extension_inputs,
        )
        generation_logging["debug_replay_question_text"] = _decode_token_ids_for_debug(
            tokenizer,
            question_input_ids,
        )

        if use_after_question_user_side_replay:
            anchor_prefix_input_ids = original_input_ids[:, :user_side_anchor_token_length]
            # `model_kwargs` may hold a mutable cache object initialized by HF generate().
            # Earlier replay passes can fill that cache in-place, so for the user-side
            # branch we must explicitly restart from a cache-free state at the anchor.
            anchor_state = self._prepare_replay_model_kwargs(
                model_kwargs,
                anchor_prefix_input_ids,
                past_key_values_override=None,
            )
            anchor_outputs, anchor_state = self._run_replay_forward(
                anchor_prefix_input_ids,
                anchor_state,
                model_forward,
                compute_logits=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            user_side_cache_snapshot = _snapshot_past_key_values_to_cpu(anchor_state.get("past_key_values"))
            if user_side_cache_snapshot is None:
                raise ValueError("Failed to capture user-side replay KV cache snapshot.")
            del anchor_outputs, anchor_state
            generation_state = self._build_replay_state_from_snapshot(
                model_kwargs=model_kwargs,
                cache_snapshot=user_side_cache_snapshot,
                prefix_length=user_side_anchor_token_length,
                device=original_input_ids.device,
            )
            del user_side_cache_snapshot
        elif use_after_question_replay:
            generation_state = self._build_replay_state_from_snapshot(
                model_kwargs=model_kwargs,
                cache_snapshot=full_prompt_cache_snapshot,
                prefix_length=original_input_ids.shape[1],
                device=original_input_ids.device,
            )
            del full_prompt_cache_snapshot
        else:
            generation_state = self._build_context_replay_state(
                model_kwargs,
                context_cache_snapshot,
                context_token_length,
                original_input_ids.device,
            )
            del context_cache_snapshot

        run_boundary_sanity_example = bool(prompt_segment_context.get("debug_boundary_sanity_check", False))
        if run_boundary_sanity_example:
            if use_after_question_user_side_replay:
                self._run_user_side_replay_sanity_example(
                    original_input_ids=original_input_ids,
                    anchor_token_length=user_side_anchor_token_length,
                    replay_input_ids=replay_input_ids,
                    generation_state=generation_state,
                )
            elif use_after_question_replay:
                self._run_append_replay_sanity_example(
                    original_input_ids=original_input_ids,
                    extension_inputs=extension_inputs,
                    replay_input_ids=replay_input_ids,
                    generation_state=generation_state,
                )
            else:
                self._run_prompt_boundary_sanity_example(
                    tokenizer=tokenizer,
                    original_input_ids=original_input_ids,
                    context_token_length=context_token_length,
                    question_input_ids=question_input_ids,
                    extension_inputs=extension_inputs,
                    replay_input_ids=replay_input_ids,
                    generation_state=generation_state,
                )

        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=original_input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=original_input_ids.device):
            outputs, generation_state = self._run_replay_forward(
                replay_input_ids,
                generation_state,
                model_forward,
                compute_logits=True,
                compute_last_logits_only=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=original_input_ids.device)
            next_token_scores = logits_processor(replay_input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            replay_input_ids = torch.cat([replay_input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(replay_input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            del outputs

        if streamer is not None:
            streamer.end()

        if generation_logging["num_generations"] > 0:
            generation_logging["avg_num_token"] /= generation_logging["num_generations"]
            generation_logging["avg_nucleus_mass"] /= generation_logging["num_generations"]
            generation_logging["scale_by_token"] /= generation_logging["num_generations"]
            generation_logging["scale_by_nucleus"] /= generation_logging["num_generations"]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=replay_input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=generation_state.get("past_key_values"),
                )
            return GenerateDecoderOnlyOutput(
                sequences=replay_input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=generation_state.get("past_key_values"),
            )

        if return_importance_details:
            return replay_input_ids, generation_logging, importance_details
        return replay_input_ids, generation_logging

    def _rescale_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        rescale_config: Optional[RescaleConfig] = None,
        return_importance_details: bool = False,
        tokenizer=None,
        prompt_segment_context: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor, Dict[str, Any]]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        if prompt_segment_context is not None and tokenizer is not None:
            return self._sentence_replay_sample(
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                rescale_config=rescale_config,
                return_importance_details=return_importance_details,
                tokenizer=tokenizer,
                prompt_segment_context=prompt_segment_context,
                **model_kwargs,
            )

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample


        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        assert rescale_config is not None
        selected_heads = rescale_config.selected_heads
        context_warmup_steps = rescale_config.context_warmup_steps
        intervention_warmup_steps = rescale_config.intervention_warmup_steps
        decay_factor = rescale_config.decay_factor
        strength = rescale_config.strength
        top_tokens = rescale_config.top_k
        top_percentile = rescale_config.top_p
        selection_method = rescale_config.selection_method
        max_selected_layer = max(selected_heads, key=lambda x: x[0])[0]

        # compute template_sequence_mask from rescale_config
        template_sequence_mask = None
        if rescale_config.template_sequences:
            template_sequence_mask = obtain_template_sequence_mask(input_ids, rescale_config.template_sequences)

        # segment the inputs
        tf_ending_len = input_ids.shape[-1]
        total_truncate_length = intervention_warmup_steps + context_warmup_steps
        look_back_inputs = input_ids[:, -total_truncate_length:]
        input_ids = input_ids[:, :-total_truncate_length]
        look_back_start_len = tf_ending_len - total_truncate_length
        intervention_start_len = tf_ending_len - intervention_warmup_steps
        print(tf_ending_len, total_truncate_length, look_back_start_len, intervention_start_len)
        dynamic_rescale = rescale_config.dynamic_rescale

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        assert batch_size == 1 # currently only support batch size 1

        importance_details = []
        static_mask_initialized = False
        past_token_importance = None
        if template_sequence_mask is not None:
            # template_sequence_mask = template_sequence_mask[:,:intervention_start_len] # cannot mask out tokens after intervention starts
            template_sequence_mask = ~template_sequence_mask # invert the mask to keep non-template tokens

        generation_logging = {"avg_num_token": 0.0, "avg_nucleus_mass": 0.0, "scale_by_token": 0.0, "scale_by_nucleus": 0.0, "num_generations": 0}
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            position_idx = cur_len - 1
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                # import pdb
                # pdb.set_trace()
                # print("Prefill stage, just run the model forward to fill the cache, no importance calculation yet.")

                outputs = self(**model_inputs, compute_logits=False, return_dict=True)
                is_prefill = False
                past_token_importance = torch.zeros_like(input_ids)
            # in context warmup stage, only get the attention weights, still reading the context
            elif cur_len < intervention_start_len:
                outputs = model_forward(
                        **model_inputs,
                        compute_logits=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                # import pdb
                # pdb.set_trace()
                # print(f"Context warmup stage, cur_len: {cur_len}, getting attention weights and calculating importance scores, but not intervening yet.")
                attention_peak = outputs.attentions
                cur_token_importance = _aggregate_head_attention(attention_peak, selected_heads)

                if return_importance_details:
                    per_token_importance = {}
                    per_token_importance["position_idx"] = position_idx
                    per_token_importance["attention_weights"] = cur_token_importance.cpu()
                past_token_importance = _apply_importance_decay(cur_token_importance, past_token_importance, decay_factor)

                if return_importance_details:
                    per_token_importance["context_scores"] = past_token_importance.cpu()
                    importance_details.append(per_token_importance)
            else:
                if dynamic_rescale:
                    # SPECULATIVE PASS; skip updating the past key value
                    outputs = model_forward(
                        **model_inputs,
                        attention_logits_intervention_vector=None,
                        compute_logits=False,
                        output_attentions=True,
                        skip_update_past_key_value=True,
                        layer_early_stopping=max_selected_layer,
                        return_dict=True,
                    )
                    # import pdb
                    # pdb.set_trace()
                    # print(f"Intervention stage, cur_len: {cur_len}, calculating importance scores, and applying rescaling intervention.")
                    attention_peak = outputs.attentions
                    cur_token_importance = _aggregate_head_attention(attention_peak, selected_heads)
                    
                    if return_importance_details:
                        per_token_importance = {}
                        per_token_importance["position_idx"] = position_idx
                        per_token_importance["attention_weights"] = cur_token_importance.cpu()

                    past_token_importance = _apply_importance_decay(cur_token_importance, past_token_importance, decay_factor)

                    generation_logging["num_generations"] += 1
                    selected_mask = _select_important_tokens(past_token_importance, generation_logging, selection_method, top_tokens, top_percentile)
                    attention_logits_intervention_vector = _build_intervention_vector(selected_mask, past_token_importance, strength, template_sequence_mask)
                # statically rescale
                else:
                    # import pdb
                    # pdb.set_trace()
                    # print(f"Intervention stage, cur_len: {cur_len}, applying rescaling intervention based on importance scores calculated in the context warmup stage, without further calculating importance scores.")
                    # selected_mask,
                    if not static_mask_initialized:
                        static_mask_initialized = True
                        
                        generation_logging["num_generations"] += 1
                        selected_mask = _select_important_tokens(past_token_importance, generation_logging, selection_method, top_tokens, top_percentile)
                    # extend selected mask to the length of the input ids
                    selected_mask = torch.cat([selected_mask, torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)], dim=1)
                    attention_logits_intervention_vector = _build_intervention_vector(selected_mask, input_ids, strength, template_sequence_mask, dtype=past_token_importance.dtype)

                # second pass, get logits;
                # REAL PASS
                outputs = model_forward(
                    **model_inputs,
                    attention_logits_intervention_vector=attention_logits_intervention_vector,
                    output_attentions=False,
                    return_dict=True,
                )
                # import pdb
                # pdb.set_trace()
                # print(f"After applying intervention, getting the output logits and continuing generation as normal.")
                if return_importance_details:
                    per_token_importance["context_scores"] = past_token_importance.cpu()
                    importance_details.append(per_token_importance)
                # clear the logits if still doing teacher forcing
                if look_back_inputs is not None:
                    outputs.logits = None

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue


            if outputs.logits is not None:
                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits and outputs.logits is not None:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            teacher_forcing = False
            if look_back_inputs is not None:
                next_tokens = look_back_inputs[:, 0]
                look_back_inputs = look_back_inputs[:, 1:]
                teacher_forcing = True
                if look_back_inputs.shape[1] == 0:
                    look_back_inputs = None
                    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            else:
                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if not teacher_forcing and has_eos_stopping_criteria:
                # if there are still look_back_inputs, we need to use them
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None and not teacher_forcing:
                streamer.put(next_tokens.cpu())

            if teacher_forcing:
                # The look-back suffix is part of the prompt, so prompt-side EOS/newline
                # tokens must not stop generation before the model produces anything.
                this_peer_finished = False
            else:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        generation_logging["avg_num_token"] /= generation_logging["num_generations"]
        generation_logging["avg_nucleus_mass"] /= generation_logging["num_generations"]
        generation_logging["scale_by_token"] /= generation_logging["num_generations"]
        generation_logging["scale_by_nucleus"] /= generation_logging["num_generations"]
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            if return_importance_details:
                return input_ids, generation_logging, importance_details
            return input_ids, generation_logging

    def _sharp_sample( # for baselines, sharpenning attention or sharpenning system prompt
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True

        attention_logits_temperature = model_kwargs.pop("attention_logits_temperature", None)
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, compute_last_logits_only=True, attention_logits_temperature=attention_logits_temperature, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, attention_logits_temperature=attention_logits_temperature, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids, None
