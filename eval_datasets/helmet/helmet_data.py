import os
import json
import random
import hashlib
import math
import numpy as np

from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk as hf_load_from_disk

from .helmet_utils import parse_rankings, calculate_retrieval_metrics, drop_duplicates, calculate_metrics, parse_output
from functools import partial

# use top version for better instruction following
def load_msmarco_rerank(dataset_name: str, path: str, seed=42):
    #### TODO: HARD CODED config for now
    assert dataset_name in ["msmarco_8k", "msmarco_16k", "msmarco_32k", "msmarco_64k", "msmarco_128k"]
    level_to_k_mapping = {
        "msmarco_8k": 50,
        "msmarco_16k": 130,
        "msmarco_32k": 285,
        "msmarco_64k": 600,
        "msmarco_128k": 1000,
    }
    data_path = os.path.join(path, f"msmarco/test_reranking_data_k{level_to_k_mapping[dataset_name]}_dep3.jsonl")
    demo_path = os.path.join(path, "msmarco/test_reranking_data_k10_dep3.jsonl")
    shots = 2
    k_values = [10]
    ####

    random.seed(seed)
    user_template = "You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}"
    system_template = "Ranking:"
    prompt_template = user_template + "\n" + system_template

    data = hf_load_dataset("json", data_files=data_path)["train"]
    demos = hf_load_dataset("json", data_files=demo_path)["train"]

    # could also do this question by question, but not necessary if we are sampling
    demo_filtered = False
    if len(demos) > 2*len(data):
        qids = set(data["qid"])
        demos = demos.filter(lambda x: x["qid"] not in qids)
        demo_filtered = True

    def update(sample, demos):
        passage_text = ""

        passage_template = "[ID: {id}] Document (Title: {title}): {text}"  if "title" in sample["ctxs"][0] else "[ID: {id}] Document: {text}"
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])
        gold_ranking = " > ".join([x['id'] for x in sorted(sample["ctxs"], key=lambda x: x["label"], reverse=True)])
        demo_text = ""

        if shots > 0:
            # need to make sure we don't pick the same question as the demos
            if not demo_filtered:
                demos = demos.filter(lambda x: x["qid"] != sample["qid"])
            # hashlib is deterministic while hash() is not in Python>=3.3, the seed has to be a positive integer
            h = abs(int(hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest(), 16) % 2**31)
            demo = demos.shuffle(seed=h)
            demo = drop_duplicates(demo, 'qid').select(range(shots))

            demo_ids = set()
            for d in demo:
                if d["qid"] in demo_ids or len(demo_ids) >= shots:
                    continue
                demo_ids.add(d["qid"])
                # sort ids by label
                ids = sorted(d["ctxs"], key=lambda x: x["label"], reverse=True)
                ranking = " > ".join([x['id'] for x in ids])
                demo_text += "\n\n".join([passage_template.format(**c) for c in d['ctxs']]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"

        qrel = [[c['id'], str(c['label'])] for c in sample["ctxs"]]
        return {"context": passage_text, "question": sample["query"], "demos": demo_text, "answer": gold_ranking, "qrel": qrel}

    data = data.map(lambda x: update(x, demos), remove_columns=["query", "ctxs"])

    def post_process(output, example):
        example = example["item"]
        parsed_pred = parse_rankings(output)
        o = {"parsed_output": parsed_pred}
        qrels = {example["qid"]: {c[0]: int(c[1]) for c in example["qrel"]}}
        mets = calculate_retrieval_metrics(results={example['qid']: parsed_pred}, qrels=qrels, k_values=k_values)
        mets = {**mets, "num_preds": len(parsed_pred)}
        return mets, o

    data_purged = []
    for d in data:
        input_prompt = prompt_template.format(**d)
        reference_output = d["answer"]
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": d})

    return data_purged, post_process




def load_msmarcotop_rerank(dataset_name: str, path: str, seed=42):
    #### TODO: HARD CODED config for now
    assert dataset_name in ["msmarcotop_8k", "msmarcotop_16k", "msmarcotop_32k", "msmarcotop_64k", "msmarcotop_128k"]
    level_to_k_mapping = {
        "msmarcotop_8k": 50,
        "msmarcotop_16k": 130,
        "msmarcotop_32k": 285,
        "msmarcotop_64k": 600,
        "msmarcotop_128k": 1000,
    }
    data_path = os.path.join(path, f"msmarco/test_reranking_data_k{level_to_k_mapping[dataset_name]}_dep3.jsonl")
    demo_path = os.path.join(path, "msmarco/test_reranking_data_k10_dep3.jsonl")
    shots = 2
    top_k = 10
    k_values = [10]
    ####

    random.seed(seed)
    user_template = "You are provided with a list of documents, each indicated by their ID. Find the top 10 most relavant documents, and rank them based on their relevance to the question in descending order from most relelvant to least relevant texts. Include the top 10 documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID0 > ID1 > ID2 ...\n\n{demos}{context}\n\nQuery: {question}"
    system_template = "Ranking:"
    prompt_template = user_template + "\n" + system_template

    data = hf_load_dataset("json", data_files=data_path)["train"]
    demos = hf_load_dataset("json", data_files=demo_path)["train"]

    # could also do this question by question, but not necessary if we are sampling
    demo_filtered = False
    if len(demos) > 2*len(data):
        qids = set(data["qid"])
        demos = demos.filter(lambda x: x["qid"] not in qids)
        demo_filtered = True

    def update(sample, demos):
        passage_text = ""

        passage_template = "[ID: {id}] Document (Title: {title}): {text}"  if "title" in sample["ctxs"][0] else "[ID: {id}] Document: {text}"
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])
        gold_ranking = " > ".join([x['id'] for x in sorted(sample["ctxs"], key=lambda x: x["label"], reverse=True)][:top_k])
        demo_text = ""

        if shots > 0:
            # need to make sure we don't pick the same question as the demos
            if not demo_filtered:
                demos = demos.filter(lambda x: x["qid"] != sample["qid"])
            # hashlib is deterministic while hash() is not in Python>=3.3, the seed has to be a positive integer
            h = abs(int(hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest(), 16) % 2**31)
            demo = demos.shuffle(seed=h)
            demo = drop_duplicates(demo, 'qid').select(range(shots))

            demo_ids = set()
            for d in demo:
                if d["qid"] in demo_ids or len(demo_ids) >= shots:
                    continue
                demo_ids.add(d["qid"])
                # sort ids by label
                ids = sorted(d["ctxs"], key=lambda x: x["label"], reverse=True)
                ranking = " > ".join([x['id'] for x in ids])
                demo_text += "\n\n".join([passage_template.format(**c) for c in d['ctxs']]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"

        qrel = [[c['id'], str(c['label'])] for c in sample["ctxs"]]
        return {"context": passage_text, "question": sample["query"], "demos": demo_text, "answer": gold_ranking, "qrel": qrel}

    data = data.map(lambda x: update(x, demos), remove_columns=["query", "ctxs"])

    def post_process(output, example):
        example = example["item"]
        parsed_pred = parse_rankings(output)
        o = {"parsed_output": parsed_pred}
        qrels = {example["qid"]: {c[0]: int(c[1]) for c in example["qrel"]}}
        mets = calculate_retrieval_metrics(results={example['qid']: parsed_pred}, qrels=qrels, k_values=k_values)
        mets = {**mets, "num_preds": len(parsed_pred)}
        return mets, o

    data_purged = []
    for d in data:
        input_prompt = prompt_template.format(**d)
        reference_output = d["answer"]
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": d})

    return data_purged, post_process


def qa_post_process(output, example, compute_rouge=False):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    example = example["item"]
    prediction = output
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer, compute_rouge=compute_rouge)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer, compute_rouge=compute_rouge)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    return mets, {"parsed_output": parsed_pred}


_DEFAULT_LONG_CONTEXT_LENGTHS = {
    "narrativeqa": 130772,
    "infbench_qa_eng": 130862,
    "infbench_choice_eng": 130862,
}

_LEGACY_RAG_LENGTH_TO_K = {
    "8k": 50,
    "16k": 105,
    "32k": 220,
    "64k": 440,
    "128k": 1000,
}

_LEGACY_TRUNCATION_LENGTHS = {
    "narrativeqa": {
        "8k": 7892,
        "16k": 16084,
        "32k": 32468,
        "64k": 65236,
        "128k": 130772,
    },
    "infbench_qa_eng": {
        "8k": 7982,
        "16k": 16174,
        "32k": 32558,
        "64k": 65326,
        "128k": 130862,
    },
    "infbench_choice_eng": {
        "8k": 7982,
        "16k": 16174,
        "32k": 32558,
        "64k": 65326,
        "128k": 130862,
    },
}

_KILT_DATASET_CONFIGS = {
    "kilt_nq": {
        "data_file": "kilt/nq-dev-multikilt_1000_k1000_dep6.jsonl",
        "demo_file": "kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
    },
    "kilt_triviaqa": {
        "data_file": "kilt/triviaqa-dev-multikilt_1000_k1000_dep6.jsonl",
        "demo_file": "kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
    },
    "kilt_hotpotqa": {
        "data_file": "kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl",
        "demo_file": "kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
    },
    "kilt_popqa_3": {
        "data_file": "kilt/popqa_test_1000_k1000_dep6.jsonl",
        "demo_file": "kilt/popqa_test_1000_k3_dep6.jsonl",
        "popularity_threshold": 3.0,
    },
}


def _resolve_helmet_data_root(path: str) -> str:
    if os.path.isdir(os.path.join(path, "kilt")):
        return path

    parent = os.path.dirname(path.rstrip(os.sep))
    if parent and os.path.isdir(os.path.join(parent, "kilt")):
        return parent

    return path


def _canonicalize_helmet_dataset_name(dataset_name: str) -> str:
    dataset_name = dataset_name.lower().replace("_nonchat_", "_")
    legacy_suffix = dataset_name.rsplit("_", 1)[-1]

    if legacy_suffix in _LEGACY_RAG_LENGTH_TO_K:
        if (
            dataset_name.startswith(("hotpot_nocot_", "hotpotqa_", "kilt_hotpotqa_"))
            or (dataset_name.startswith("hotpot_") and not dataset_name.startswith("hotpot_cot_"))
        ):
            return f"hotpot_nocot_{legacy_suffix}"
        if dataset_name.startswith(("nq_", "kilt_nq_")):
            return f"nq_{legacy_suffix}"
        if dataset_name.startswith(("triviaqa_", "kilt_triviaqa_")):
            return f"triviaqa_{legacy_suffix}"
        if dataset_name.startswith(("popqa_", "popqa_3_", "kilt_popqa_", "kilt_popqa_3_")):
            return f"kilt_popqa_3_{legacy_suffix}"

    if dataset_name.startswith("hotpot_nocot_") and dataset_name.rsplit("_", 1)[-1] in _LEGACY_RAG_LENGTH_TO_K:
        return dataset_name
    if dataset_name in {"hotpot", "hotpotqa", "hotpot_nocot"}:
        return "kilt_hotpotqa"
    if dataset_name.startswith("nq_") and dataset_name.rsplit("_", 1)[-1] in _LEGACY_RAG_LENGTH_TO_K:
        return dataset_name
    if dataset_name in {"nq"}:
        return "kilt_nq"
    if dataset_name.startswith("triviaqa_") and dataset_name.rsplit("_", 1)[-1] in _LEGACY_RAG_LENGTH_TO_K:
        return dataset_name
    if dataset_name in {"triviaqa"}:
        return "kilt_triviaqa"
    if dataset_name in {"popqa", "popqa_3", "kilt_popqa", "kilt_popqa_3"}:
        return "kilt_popqa_3"

    if dataset_name in {"narrativeqa"}:
        return f"narrativeqa_{_DEFAULT_LONG_CONTEXT_LENGTHS['narrativeqa']}"
    if dataset_name.startswith("narrativeqa_"):
        suffix = dataset_name.rsplit("_", 1)[-1]
        if suffix in _LEGACY_TRUNCATION_LENGTHS["narrativeqa"]:
            return f"narrativeqa_{_LEGACY_TRUNCATION_LENGTHS['narrativeqa'][suffix]}"
        return dataset_name

    if dataset_name in {"infqa", "infbench_qa", "infbench_qa_eng"}:
        return f"infbench_qa_eng_{_DEFAULT_LONG_CONTEXT_LENGTHS['infbench_qa_eng']}"
    if dataset_name.startswith("infbench_qa_eng_"):
        suffix = dataset_name.rsplit("_", 1)[-1]
        if suffix in _LEGACY_TRUNCATION_LENGTHS["infbench_qa_eng"]:
            return f"infbench_qa_eng_{_LEGACY_TRUNCATION_LENGTHS['infbench_qa_eng'][suffix]}"
        return dataset_name
    if dataset_name.startswith(("infqa_", "infbench_qa_")):
        suffix = dataset_name.rsplit("_", 1)[-1]
        if suffix in _LEGACY_TRUNCATION_LENGTHS["infbench_qa_eng"]:
            return f"infbench_qa_eng_{_LEGACY_TRUNCATION_LENGTHS['infbench_qa_eng'][suffix]}"
        return f"infbench_qa_eng_{suffix}"

    if dataset_name in {"infmc", "infbench_mc", "infbench_choice", "infbench_choice_eng"}:
        return f"infbench_choice_eng_{_DEFAULT_LONG_CONTEXT_LENGTHS['infbench_choice_eng']}"
    if dataset_name.startswith("infbench_choice_eng_"):
        suffix = dataset_name.rsplit("_", 1)[-1]
        if suffix in _LEGACY_TRUNCATION_LENGTHS["infbench_choice_eng"]:
            return f"infbench_choice_eng_{_LEGACY_TRUNCATION_LENGTHS['infbench_choice_eng'][suffix]}"
        return dataset_name
    if dataset_name.startswith(("infmc_", "infbench_mc_", "infbench_choice_")):
        suffix = dataset_name.rsplit("_", 1)[-1]
        if suffix in _LEGACY_TRUNCATION_LENGTHS["infbench_choice_eng"]:
            return f"infbench_choice_eng_{_LEGACY_TRUNCATION_LENGTHS['infbench_choice_eng'][suffix]}"
        return f"infbench_choice_eng_{suffix}"

    return dataset_name


def _extract_dataset_truncation_length(dataset_name: str, prefix: str) -> int:
    if dataset_name == prefix:
        return _DEFAULT_LONG_CONTEXT_LENGTHS[prefix]

    if not dataset_name.startswith(prefix + "_"):
        raise ValueError(f"Dataset {dataset_name} does not match prefix {prefix}")

    suffix = dataset_name.rsplit("_", 1)[-1]
    if suffix.endswith("k"):
        legacy_lengths = _LEGACY_TRUNCATION_LENGTHS.get(prefix, {})
        if suffix in legacy_lengths:
            return legacy_lengths[suffix]
    return int(suffix)


def _load_jsonl_dataset(path: str):
    return hf_load_dataset("json", data_files=path)["train"]


def _load_optional_jsonl_dataset(path: str):
    if not os.path.exists(path):
        print(f"Demo file {path} not found. Falling back to zero-shot prompts.")
        # logger.info(f"Demo file {path} is not available locally. Falling back to zero-shot prompts.")
        return None
    return _load_jsonl_dataset(path)


def _selected_100_data_file(data_file: str) -> str:
    selected_file = data_file.replace("_1000_", "_100_")
    if selected_file.endswith(".jsonl"):
        selected_file = selected_file[:-6] + ".json"
    return selected_file


def load_kilt_qa(dataset_name: str, path: str, seed=42):
    dataset_name = _canonicalize_helmet_dataset_name(dataset_name)
    data_root = _resolve_helmet_data_root(path)

    cfg = None
    ctx_limit = None
    if dataset_name in _KILT_DATASET_CONFIGS:
        cfg = dict(_KILT_DATASET_CONFIGS[dataset_name])
    else:
        dataset_base = None
        legacy_suffix = dataset_name.rsplit("_", 1)[-1]
        if dataset_name.startswith("hotpot_nocot_"):
            dataset_base = "kilt_hotpotqa"
        elif dataset_name.startswith("nq_"):
            dataset_base = "kilt_nq"
        elif dataset_name.startswith("triviaqa_"):
            dataset_base = "kilt_triviaqa"
        elif dataset_name.startswith("kilt_popqa_3_"):
            dataset_base = "kilt_popqa_3"

        if dataset_base is None or legacy_suffix not in _LEGACY_RAG_LENGTH_TO_K:
            raise ValueError(f"Unknown KILT dataset: {dataset_name}")

        cfg = dict(_KILT_DATASET_CONFIGS[dataset_base])
        legacy_k = _LEGACY_RAG_LENGTH_TO_K[legacy_suffix]
        ctx_limit = legacy_k
        cfg["data_file"] = cfg["data_file"].replace("k1000", f"k{legacy_k}")
        demo_file = cfg.get("demo_file")
        if demo_file is not None:
            cfg["demo_file"] = demo_file

    data_file = cfg["data_file"]
    selected_data_file = _selected_100_data_file(data_file)
    if os.path.exists(os.path.join(data_root, selected_data_file)):
        data_file = selected_data_file

    data_path = os.path.join(data_root, data_file)
    demo_path = os.path.join(data_root, cfg["demo_file"])
    if not os.path.exists(data_path) and "k1000" not in data_file:
        fallback_cfg = _KILT_DATASET_CONFIGS[
            "kilt_hotpotqa"
            if "hotpot" in dataset_name
            else "kilt_triviaqa"
            if "triviaqa" in dataset_name
            else "kilt_popqa_3"
            if "popqa" in dataset_name
            else "kilt_nq"
        ]
        print(f"Legacy KILT file {data_path} not found. Falling back to {fallback_cfg['data_file']} and truncating ctxs to k{ctx_limit}.")
        data_path = os.path.join(data_root, fallback_cfg["data_file"])
        demo_path = os.path.join(data_root, fallback_cfg["demo_file"])

    data = _load_jsonl_dataset(data_path)
    demo_data = _load_optional_jsonl_dataset(demo_path)

    popularity_threshold = cfg.get("popularity_threshold")
    if popularity_threshold is not None:
        data = data.filter(lambda x: math.log10(x["s_pop"]) < popularity_threshold)
        if demo_data is not None:
            demo_data = demo_data.filter(lambda x: math.log10(x["s_pop"]) < popularity_threshold)

    shots = 2 if demo_data is not None else 0

    user_template = (
        "Use the given documents to write a concise and short answer to the question. "
        "Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
    )
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    key = "id" if "id" in data.column_names else "question"
    demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
    passage_template = "Document (Title: {title}): {text}"

    def update(sample):
        demos = demo_data
        demo_text = ""
        if shots > 0 and demos is not None:
            if "popqa" in dataset_name:
                demos = demos.filter(lambda x: x[key] != sample[key])

            hashed_key = str(sample[key])
            local_seed = int(hashlib.sha256(hashed_key.encode("utf-8")).hexdigest(), 16) % 2**31
            demos = demos.shuffle(seed=local_seed)
            demos = drop_duplicates(demos, key).select(range(min(shots, len(demos))))
            demo_text = "\n\n".join(
                [
                    demo_template.format(
                        **d,
                        documents="\n\n".join([passage_template.format(**c) for c in d["ctxs"]]),
                        answer=d["answers"][0],
                    )
                    for d in demos
                ]
            )
            if demo_text:
                demo_text += "\n\n"

        ctxs = sample["ctxs"]
        if ctx_limit is not None:
            ctxs = ctxs[:ctx_limit]

        passage_text = ""
        if len(ctxs) > 0:
            passage_text = "\n\n".join([passage_template.format(**c) for c in ctxs])
        return {"demos": demo_text, "context": passage_text, "answer": sample["answers"]}

    data = data.map(update)

    data_purged = []
    for d in data:
        input_prompt = prompt_template.format(**d)
        reference_output = " ".join(d["answer"])
        item = {"question": d["question"], "answer": d["answer"]}
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": item})

    return data_purged, partial(qa_post_process, compute_rouge=True)



def load_ruler(dataset_name: str, path: str, seed=42):
    sub_task_to_datafile = {
        "niah_s_1": "niah_single_1",
        "niah_s_2": "niah_single_2",
        "niah_s_3": "niah_single_3",
        "niah_mk_1": "niah_multikey_1",
        "niah_mk_2": "niah_multikey_2",
        "niah_mk_3": "niah_multikey_3",
        "niah_mq": "niah_multiquery",
        "niah_mv": "niah_multivalue",
        "cwe": "cwe",
        "fwe": "fwe",
        "vt": "vt",
        "qa_1": "qa_1",
        "qa_2": "qa_2",
    }
    assert dataset_name.startswith("ruler_")
    dataset_name = dataset_name.split("_", 1)[1]
    dataset,length_name = dataset_name.rsplit("_", 1) # length like 8k, 16k, 32k, 64k, 128k
    length_name = str(int(length_name[:-1]) * 1024)
    data_path = os.path.join(path, f"ruler/{sub_task_to_datafile[dataset]}/validation_{length_name}.jsonl")
    data = hf_load_dataset("json", data_files=data_path)["train"]
    user_template = "{context}\n\n{question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    # https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
    if "niah_mv" in dataset or "niah_mq" in dataset:
        user_template = "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system_template = "The special magic {type_needle_v} for {query} mentioned in the provided text are"
    elif "niah" in dataset:
        user_template = "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system_template = "The special magic {type_needle_v} for {query} mentioned in the provided text is"
    elif "vt" in dataset:
        user_template = "{example}Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above."
        system_template = "Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are:"
    elif "cwe" in dataset:
        user_template = "{example}Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?"
        system_template = "Answer: The top 10 words that appear most often in the list are:"
    elif "fwe" in dataset:
        user_template = "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words.\n{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?"
        system_template = "Answer: According to the coded text above, the three most frequently appeared words are:"
    elif "qa" in dataset:
        # note that for qa, instead of calculating the recall, we simply check for substring exact match
        user_template = "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {question}"
        system_template = "Answer:"
    else:
        raise NotImplementedError(f"Unknown ruler dataset {dataset}")
    prompt_template = user_template + "\n" + system_template

    def process_example(example):
        return {
            "question": example["query"] if "query" in example else example["question"] if "question" in example else "",
            "example": example["example"] + "\n\n" if "example" in example and example["example"] != "" else "",
            "answer": example["answer"] if "answer" in example else example['outputs'],
        }
    data = data.map(process_example)

    def post_process(output: str, example: dict):
        # we don't do any parsing since we are only checking for substring exact match
        prediction = output
        answer = example["item"]["answer"]
        recall = sum([a.lower() in prediction.lower() for a in answer]) / len(answer)
        mets = {"ruler_recall": recall}
        return mets, {"parsed_output": prediction}

    data_purged = []
    for d in data:
        input_prompt = prompt_template.format(**d)
        reference_output = " ".join(d["answer"])
        d = {"question": d["question"], "answer": d["answer"]}
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": d})

    return data_purged, post_process




def load_hotpot(dataset_name, path, seed=42):
    return load_kilt_qa(dataset_name, path, seed=seed)


def load_hotpot_cot(dataset_name, path, seed=42):
    #### TODO: HARD CODED config for now
    assert dataset_name in ["hotpot_cot_8k", "hotpot_cot_16k", "hotpot_cot_32k", "hotpot_cot_64k", "hotpot_cot_128k"]

    level_to_k_mapping = {
       "8k": 50,
        "16k": 105,
        "32k": 220,
        "64k": 440,
        "128k": 1000,
    }
    path = _resolve_helmet_data_root(path)
    data_path = os.path.join(path, f"kilt/hotpotqa-dev-multikilt_1000_k{level_to_k_mapping[dataset_name.split('_',)[-1]]}_dep3.jsonl")
    max_test_samples = 100
    ####

    user_template = "You will be given a list of documents and a question, please answer the question based on the documents.\n\nDOCUMENTS:\n{context}\n\n\nNow answer the following question based on the documents above.\nQuestion: {question}"

    data = hf_load_dataset("json", data_files=data_path)["train"]

    key = "id" if "id" in data.column_names else "question"

    # some datasets do not have id (e.g., nq), so we assume unique questions
    random.seed(seed)
    keys = set(data[key])
    keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
    data = data.filter(lambda x: x[key] in keys)

    passage_template = "Document (Title: {title}): {text}"
    def update(sample):
        passage_text = ""
        if len(sample['ctxs']) > 0:
            passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])
        return {"context": passage_text, "answer": sample["answers"]}
    data = data.map(update)

    data_purged = []
    for d in data:
        input_prompt = user_template.format(**d)
        reference_output = " ".join(d["answer"])
        del d["context"]
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": d})
    
    return data_purged, qa_post_process


def load_icl(dataset_name, path, flag_balance=True, seed=42):
    max_test_sample = 500 # inherented from config
    #input_max_length: 8192,16384,32768,65536,8192,16384,32768,65536,8192,16384,32768,65536,8192,16384,32768,65536
    # datasets: icl_banking77_360shot_balance,icl_banking77_720shot_balance,icl_banking77_1450shot_balance,icl_banking77_2900shot_balance,icl_clinic150_440shot_balance,icl_clinic150_880shot_balance,icl_clinic150_1750shot_balance,icl_clinic150_3525shot_balance,icl_nlu_510shot_balance,icl_nlu_1020shot_balance,icl_nlu_2040shot_balance,icl_nlu_4080shot_balance
    length_to_shot_mapping = {}
    trec_corse_length_to_shot_mapping = {
        "8k": 400,
        "16k": 800,
        "32k": 1600,
        "64k": 3300,
        "128k": 6600,
    }
    trec_fine_length_to_shot_mapping = {
        "8k": 400,
        "16k": 800,
        "32k": 1600,
        "64k": 3200,
        "128k": 6400,
    }

    if "trec_fine" in dataset_name.lower():
        print(os.path.join(path, "trec"))
        shot = trec_fine_length_to_shot_mapping[dataset_name.split("_")[-1]]
        # train_data = hf_load_dataset("CogComp/trec", trust_remote_code=True)["train"]
        # test_data = hf_load_dataset("CogComp/trec", trust_remote_code=True)["test"]
        trec_data = hf_load_from_disk(os.path.join(path, "icl/trec"))
        train_data = trec_data["train"]
        test_data = trec_data["test"]
        id2label = train_data.features['fine_label'].names
        text_field = "text"
        label_field = "fine_label"
        num_labels = 50
    elif "trec_coarse" in dataset_name.lower():
        shot = trec_corse_length_to_shot_mapping[dataset_name.split("_")[-1]]
        trec_data = hf_load_from_disk(os.path.join(path, "icl/trec"))
        train_data = trec_data["train"]
        test_data = trec_data["test"]
        id2label = train_data.features['coarse_label'].names
        text_field = "text"
        label_field = "coarse_label"
        num_labels = 6
    # elif "banking77" in dataset_name.lower():
    #     train_data = load_dataset("PolyAI/banking77", trust_remote_code=True)["train"]
    #     test_data = load_dataset("PolyAI/banking77", trust_remote_code=True)["test"]
    #     id2label = train_data.features["label"].names
    #     id2label = {i: id2label[i] for i in range(len(id2label))}
    #     text_field = "text"
    #     label_field = "label"
    #     num_labels = 77
    # elif "clinic150" in dataset_name.lower():
    #     train_data = load_dataset("clinc_oos", "plus")["train"]
    #     test_data = load_dataset("clinc_oos", "plus")["validation"]
    #     id2label = train_data.features["intent"].names
    #     text_field = "text"
    #     label_field = "intent"
    #     num_labels = 151
    # elif "nlu" in dataset_name.lower():
    #     data = load_dataset("xingkunliuxtracta/nlu_evaluation_data", trust_remote_code=True)["train"]
    #     id2label = data.features["label"].names
    #     data = data.train_test_split(test_size=0.1, seed=seed)
    #     train_data = data["train"]
    #     test_data = data["test"]
    #     text_field = "text"
    #     label_field = "label"
    #     num_labels = 68
    else:
        raise NotImplementedError(f"Unknown ICL dataset")
   
    def balance_labels(data, shots, seed):
        # for each data point, we are going to sample a random set of demos with balanced labels
        # there are two places where randomness is involved: the selection of the demos and the final shuffle
        rand = random.Random(seed)

        label_mapping = {x[label_field]: [] for x in data}
        for x in data:
            label_mapping[x[label_field]].append(x)

        # rearrange the data such that every label has the same number of samples
        # they are also in consecutive sets with random order in each set
        num_rounds = math.ceil(shots / len(label_mapping))
        new_data = [[] for _ in range(num_rounds)]
        for _, samples in label_mapping.items():
            indices = rand.sample(range(len(samples)), num_rounds % len(samples))
            while len(indices) < num_rounds:
                # sample with replacement if necessary, shouldn't happen unless we have very many shots
                indices += rand.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))

            for i, idx in enumerate(indices):
                new_data[i].append(samples[idx])

        for i in range(len(new_data)):
            # this shuffles the order of the labels within each set
            rand.shuffle(new_data[i])
        new_data = [item for sublist in new_data for item in sublist][:shots]
        return new_data

    if max_test_sample is not None and len(test_data) > max_test_sample:
        test_data = test_data.shuffle(seed=seed)
        # we also balance the output labels
        test_data = balance_labels(test_data, max_test_sample, seed)
        test_data = datasets.Dataset.from_list(test_data)

    item_template = "{text}\nlabel: {label}"
    user_template = "Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}"
    system_template = "label:"
    prompt_template = user_template + "\n" + system_template

    def preprocess(sample):
        # use a different seed for every sample, but is also deterministic and affected by the set seed
        local_seed = (int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16) + seed) % 2**31
        np.random.seed(local_seed)
        if flag_balance:
            demos = balance_labels(train_data, shot, local_seed)
        else:
            demos = []
            while len(demos) < shot:
                demos += list(np.random.choice(train_data, min(len(train_data), shot - len(demos)), replace=False))

        if "natural_label" in dataset_name:
            label_mapping = [id2label[i] for i in range(num_labels)]
        else:
            # we map the labels to a random integer
            label_mapping = list(range(num_labels))
            random.seed(local_seed)
            random.shuffle(label_mapping)

        context = "\n\n".join([
            item_template.format(text=selected_item[text_field], label=str(label_mapping[int(selected_item[label_field])]))
            for selected_item in demos]
        )
        return {"context": context, "question": sample[text_field], "answer": str(label_mapping[int(sample[label_field])])}

    final_data = test_data.map(preprocess, num_proc=40)

    def post_process(output, example):
        prediction = output
        example = example["item"]
        answer = example["answer"]
        prediction = parse_output(prediction, system_template)
        mets = calculate_metrics(prediction, answer)
        return mets, {"parsed_output": prediction}

    data_purged = []
    for d in final_data:
        input_prompt = prompt_template.format(**d)
        reference_output = str(d["answer"])
        del d["context"]
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": d})

    return data_purged, post_process


def truncate_llama2(max_length: int, data, postfix_text=" ... [the rest of the text is omitted]"):
    # use the llama 2 tokenizer to truncate to max_length, which only applies to the main document (context) and exclude the instructions and the demos
    # this is to make sure that every model see the same amount of information
    from transformers import AutoTokenizer
    try:
        # use local first
        tokenizer = AutoTokenizer.from_pretrained("models/Llama-2-7b-hf")
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    separator_length = len(tokenizer(postfix_text)["input_ids"])

    def truncate(sample):
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > max_length:
            # truncate
            sample["context"] = sample["context"][:tokens["offset_mapping"][max_length-separator_length][1]] + postfix_text
        return sample
    return data.map(truncate, num_proc=16)


def load_narrativeqa(dataset_name, path, seed=42):
    dataset_name = _canonicalize_helmet_dataset_name(dataset_name)
    data_root = _resolve_helmet_data_root(path)
    truncate_length = _extract_dataset_truncation_length(dataset_name, "narrativeqa")
    shots = 2

    user_template = (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible.\n\n"
        "{demo}{context}\n\nQuestion: {question}"
    )
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    selected_path = os.path.join(data_root, "narrativeqa", f"{dataset_name}_100.json")
    if os.path.exists(selected_path):
        data = _load_jsonl_dataset(selected_path)
    else:
        all_data = hf_load_dataset("narrativeqa")
        data = all_data["test"].shuffle(seed=seed)

        demo_examples = []
        if shots > 0:
            demo_examples = list(all_data["train"].shuffle(seed=seed).select(range(shots)))

        data = data.map(
            lambda example: {
                "context": example["document"]["text"],
                "question": example["question"]["text"],
                "answer": [ans["text"] for ans in example["answers"]],
                "demo": (
                    ""
                    if shots == 0
                    else "For example:\n\n"
                    + "\n\n".join(
                        [
                            f"Question: {ex['question']['text']}\nAnswer: {ex['answers'][0]['text']}"
                            for ex in demo_examples
                        ]
                    )
                    + "\n\nNow, use the following story to answer the question:\n\n"
                ),
            },
            remove_columns=["document", "answers"],
        )

        data = truncate_llama2(truncate_length, data)

    data_purged = []
    for d in data:
        input_prompt = prompt_template.format(**d)
        reference_output = " ".join(d["answer"])
        item = {"question": d["question"], "answer": d["answer"]}
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": item})

    return data_purged, partial(qa_post_process, compute_rouge=True)


def load_infbench(dataset_name, path, seed=42):
    dataset_name = _canonicalize_helmet_dataset_name(dataset_name)
    data_root = _resolve_helmet_data_root(path)
    shots = 2

    truncate_prefix = "infbench_qa_eng" if "qa_eng" in dataset_name else "infbench_choice_eng"
    truncate_length = _extract_dataset_truncation_length(dataset_name, truncate_prefix)

    if "qa_eng" in dataset_name:
        user_template = (
            "You are given a story and a question. Answer the question as concisely as you can, "
            "using a single phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
        )
        system_template = "Answer:"
        split_name = "longbook_qa_eng"
        post_process = partial(qa_post_process, compute_rouge=True)
    elif "choice_eng" in dataset_name:
        user_template = (
            "You are given a story and a question with multiple choices. Choose the best answer "
            "from the options provided. Only one of the following options is correct, output the "
            "answer using one single letter (A, B, C, or D). Don't say anything else.\n\n"
            "{demo}{context}\n\nQuestion: {question}\nOptions:\n{options}"
        )
        system_template = "Answer:"
        split_name = "longbook_choice_eng"

        def choice_post_process(output, example):
            prediction = output
            example = example["item"]
            answer = example["answer"]
            mets = calculate_metrics(prediction, answer, compute_rouge=True)
            mets.pop("substring_exact_match")

            parsed_pred = parse_output(prediction)
            if parsed_pred is not None:
                new_mets = calculate_metrics(parsed_pred, answer, compute_rouge=True)
                new_mets.pop("substring_exact_match")
                mets = {k: max(v, new_mets[k]) for k, v in mets.items()}

            mets["substring_exact_match"] = False
            if answer[1].lower() in prediction.lower():
                mets["substring_exact_match"] = True
                mets["exact_match"] = True
            return mets, {"parsed_output": parsed_pred}

        post_process = choice_post_process
    else:
        raise NotImplementedError(f"Unsupported InfiniteBench dataset: {dataset_name}")

    prompt_template = user_template + "\n\n" + system_template
    selected_path = os.path.join(data_root, "infbench", f"{dataset_name}_100.json")

    def process_example(example):
        update = {"question": example["input"], "demo": ""}
        if "choice" in dataset_name:
            options = "A. {}\nB. {}\nC. {}\nD. {}".format(*example["options"])
            answer = example["options"].index(example["answer"][0])
            answer = chr(ord("A") + answer)
            update["options"] = options
            update["answer"] = [answer, f"{answer}. {example['answer'][0]}"]
        return update

    def add_demos(example):
        demos = data.filter(lambda x: x["id"] != example["id"]).shuffle(seed=seed)
        demos = demos.select(range(min(shots, len(demos))))
        if "qa_eng" in dataset_name:
            temp = "[story text]\nQuestion: {question}\nAnswer: {answer[0]}"
            demo = "\n\n".join([temp.format(**x) for x in demos])
        else:
            temp = "[story text]\nQuestion: {question}\nOptions:\n{options}\nAnswer: {answer[0]}"
            demo = "\n\n".join([temp.format(**x) for x in demos])
        return {"demo": f"For example:\n\n{demo}\n\nNow, read the following story:\n\n"}

    if os.path.exists(selected_path):
        data = _load_jsonl_dataset(selected_path)
    else:
        from datasets import Value, Sequence, Features

        ft = Features(
            {
                "id": Value("int64"),
                "context": Value("string"),
                "input": Value("string"),
                "answer": Sequence(Value("string")),
                "options": Sequence(Value("string")),
            }
        )
        data = hf_load_dataset("xinrongzhang2022/infinitebench", features=ft)
        data = data[split_name]
        data = data.map(process_example)

        if shots > 0:
            data = data.map(add_demos)

        data = truncate_llama2(truncate_length, data)

    data_purged = []
    for d in data:
        input_prompt = prompt_template.format(**d)
        reference_output = str(d["answer"])
        item = {"question": d["question"], "answer": d["answer"]}
        if "options" in d:
            item["options"] = d["options"]
        data_purged.append({"input_prompt": input_prompt, "reference_output": reference_output, "item": item})

    return data_purged, post_process


def load_helmet_data(dataset_name: str, path: str = "data/helmet"):
    """
    return data as a list and a eval function
    """

    dataset_name = _canonicalize_helmet_dataset_name(dataset_name)
    path = _resolve_helmet_data_root(path)

    if "msmarco_" in dataset_name:
        return load_msmarco_rerank(dataset_name, path)
    elif "msmarcotop_" in dataset_name:
        return load_msmarcotop_rerank(dataset_name, path)
    elif (
        dataset_name in _KILT_DATASET_CONFIGS
        or dataset_name.startswith("hotpot_nocot_")
        or dataset_name.startswith("nq_")
        or dataset_name.startswith("triviaqa_")
        or dataset_name.startswith("kilt_popqa_3_")
    ):
        return load_kilt_qa(dataset_name, path)
    elif "hotpot_cot_" in dataset_name:
        return load_hotpot_cot(dataset_name, path)
    elif dataset_name.startswith("narrativeqa"):
        return load_narrativeqa(dataset_name, path)
    elif "ruler_" in dataset_name:
        return load_ruler(dataset_name, path)
    elif "icl_" in dataset_name:
        return load_icl(dataset_name, path)
    elif dataset_name.startswith("infbench_qa_eng") or dataset_name.startswith("infbench_choice_eng"):
        return load_infbench(dataset_name, path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
