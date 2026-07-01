#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
LOG_DIR="${LOG_DIR:-logs/reproduce/recontext_qwen3_4b}"
MODEL="Qwen/Qwen3-4B"
CFG="recontext_cfgs/qwen3_4b.yaml"

mkdir -p "${LOG_DIR}"

run_recontext() {
  local dataset="$1"
  local max_tokens="$2"
  local stop_on_newline="$3"
  local top_k="$4"
  local replay_rounds="$5"
  local selection_scope="$6"
  local replay_position="$7"
  local wrap_sentence_txt="$8"

  local log_path="${LOG_DIR}/qwen3_4b_ReContext_${dataset}_128k_max${max_tokens}_topK${top_k}_rounds${replay_rounds}_scope${selection_scope}_replay${replay_position}_wrap${wrap_sentence_txt}.log"
  local args=(
    --seed 42
    --generation_seed 23
    --dataset "${dataset}"
    --test_size -1
    --model "${MODEL}"
    --max_tokens "${max_tokens}"
    --temperature 0.0
    --top_p 1.0
    --max_model_len 131072
  )

  args+=(
    --enable_yarn
    --output_dir results
    --decoding_method "ReContext"
    --recontext_cfgs_path "${CFG}"
    --recontext_top_p 0
    --recontext_top_k "${top_k}"
    --recontext_strength 2.0
    --recontext_decay_factor 0.75
    --recontext_ctx_warmup 8
    --recontext_interv_warmup auto
    --recontext_replay_rounds "${replay_rounds}"
    --recontext_selection_scope "${selection_scope}"
    --recontext_dedup_inserted_sentences 1
    --recontext_wrap_sentence_txt "${wrap_sentence_txt}"
    --recontext_replay_position "${replay_position}"
    --recontext_use_extension_inputs 1
  )

  if [[ "${stop_on_newline}" == "1" ]]; then
    args+=(--stop_on_newline)
  fi

  printf '[ReContext][qwen3_4b] dataset=%s top_k=%s rounds=%s scope=%s log=%s\n' \
    "${dataset}" "${top_k}" "${replay_rounds}" "${selection_scope}" "${log_path}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q run_eval.py' "${GPU}" "${PYTHON_BIN}"
    printf ' %q' "${args[@]}"
    printf ' > %q 2>&1\n' "${log_path}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" run_eval.py "${args[@]}" > "${log_path}" 2>&1
}


run_recontext "kilt_nq" 20 1 32 1 "context" "before_question" 0
run_recontext "kilt_triviaqa" 20 1 8 3 "context" "before_question" 0
run_recontext "kilt_hotpotqa" 20 1 8 3 "context" "before_question" 0
run_recontext "kilt_popqa_3" 20 1 32 1 "context" "before_question" 0
run_recontext "narrativeqa_130772" 100 0 8 2 "full_prompt" "before_question" 0
run_recontext "infbench_qa_eng_130862" 10 0 8 2 "full_prompt" "before_question" 0
run_recontext "infbench_choice_eng_130862" 4096 0 16 2 "context" "before_question" 0
run_recontext "clipper" 512 0 8 2 "context" "before_question" 0
