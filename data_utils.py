from eval_datasets.longproc.longproc_data import load_longproc_data
from eval_datasets.longbenchv2.longbenchv2_data import load_longbenchv2_data
from eval_datasets.helmet.helmet_data import load_helmet_data
from eval_datasets.mrcr.mrcr_data import load_mrcr_data
from eval_datasets.clipper.clipper_data import load_clipper_data

from eval_datasets.longbenchv2.longbenchv2_data import load_retrieved_longbenchv2_data
from eval_datasets.clipper.clipper_data import load_retrieved_clipper_data


_HELMET_DATASETS = [
    "kilt_",
    "nq",
    "triviaqa",
    "hotpot",
    "popqa",
    "narrativeqa",
    "infbench",
    "infqa",
    "infmc",
]

_LONGPROC_EVAL_DATASETS = [
    "path_walking",
]

_MRCR_EVAL_DATASETS = [
    "mrcr_2needle",
    "mrcr_4needle",
    "mrcr_8needle",
]

_LONGBENCHV2_EVAL_DATASETS = [
    "longbenchv2_"
]

_CLIPPER_DATASETS = [
    "clipper"
]



def load_retrieved_eval_data(dataset_name: str, retriever: str, k: int):
    """
    return list of datapoints ({"input_prompt", "reference_output", "item" }) and a eval func;
    only support clipper and longbenchv2
    """

    dataset_key = dataset_name.lower()

    if any([dataset_key.startswith(prefix) for prefix in _LONGBENCHV2_EVAL_DATASETS]):
        return load_retrieved_longbenchv2_data(dataset_name, data_path="data_eval/lbv2_qr", retrieval_path="results_retriever/lbv2", retriever=retriever, k=k)
    elif any([dataset_key.startswith(prefix) for prefix in _CLIPPER_DATASETS]):
        return load_retrieved_clipper_data(dataset_name, data_path="data_eval/clipper", retrieval_path="results_retriever/clipper", retriever=retriever, k=k)
    else:
        raise ValueError(f"Unknown retriever eval dataset: {dataset_name}")



def load_eval_data(dataset_name: str, data_path: str = "data_eval"):
    """
    return list of datapoints ({"input_prompt", "reference_output", "item" }) and a eval func;
    """

    dataset_key = dataset_name.lower()

    if any([dataset_key.startswith(prefix) for prefix in _LONGPROC_EVAL_DATASETS]):
        return load_longproc_data(dataset_name, path=f"{data_path}/longproc",)
    elif any([dataset_key.startswith(prefix) for prefix in _LONGBENCHV2_EVAL_DATASETS]):
        return load_longbenchv2_data(dataset_name, path=f"{data_path}/lbv2_qr")
    elif any([dataset_key.startswith(prefix) for prefix in _HELMET_DATASETS]):
        return load_helmet_data(dataset_name, path=data_path)
    elif any([dataset_key.startswith(prefix) for prefix in _MRCR_EVAL_DATASETS]):
        return load_mrcr_data(dataset_name, path=f"{data_path}/mrcr")
    elif any([dataset_key.startswith(prefix) for prefix in _CLIPPER_DATASETS]):
        return load_clipper_data(dataset_name, path=f"{data_path}/clipper")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def test_load_eval_data():
    dataset_name = "path_walking_16k"

    data, eval_func = load_eval_data(dataset_name)
    print(len(data))
    print(f"Loaded {len(data)} datapoints for {dataset_name}")
    print("*" * 50)
    print("PROMPT:")
    print(data[0]["input_prompt"] + "|||")
    print("*" * 50)
    print("REFERENCE OUTPUT:")
    print(data[0]["reference_output"])
    print("*" * 50)
    print("EVAL:")
    print(eval_func("<Route>" + data[0]["reference_output"] + "</Route>", data[0]))
    print("*" * 50)

if __name__ == "__main__":
    test_load_eval_data()
