from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# 强制 Hugging Face/Transformers 只使用本地文件，避免评估时联网下载。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ccr.data import CCRDataCollator, FraudTextDataset
from ccr.model import CCRClassifier


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 JSON 对象：{path}")
    return data


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_local_model_source(model_name_or_path: str) -> Path:
    """
    解析基础编码器的本地目录。

    1. 如果传入的是本地目录，直接使用；
    2. 如果传入的是 Hugging Face 仓库名，只在本地缓存中查找；
    3. 不允许联网下载。
    """
    direct_path = Path(model_name_or_path).expanduser()
    if direct_path.exists():
        return direct_path.resolve()

    try:
        cached_path = snapshot_download(
            repo_id=model_name_or_path,
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            "没有在本机 Hugging Face 缓存中找到基础模型："
            f"{model_name_or_path}\n"
            "当前脚本不会联网下载。请确认仍在原训练服务器上运行，"
            "或者通过 --model-name 传入基础模型的本地目录。"
        ) from exc

    return Path(cached_path).resolve()


def load_state_dict(checkpoint: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """兼容直接保存 state_dict 或外层带 state_dict/model_state_dict 的检查点。"""
    try:
        obj = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        state = obj["model_state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"无法识别检查点格式：{checkpoint}")

    # 兼容 DataParallel 保存出的 module. 前缀。
    if state and all(str(key).startswith("module.") for key in state.keys()):
        state = {str(key)[7:]: value for key, value in state.items()}

    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评估已训练的 CCR 模型")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="训练好的权重，例如 outputs/ccr_run/ccr_best.pt",
    )
    parser.add_argument("--test-file", default="data/processed/test.csv")
    parser.add_argument(
        "--run-config",
        default=None,
        help="训练配置；默认读取 checkpoint 同目录下的 config.json",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "基础编码器的本地目录，或训练时的仓库名。"
            "仓库名只会从本地 Hugging Face 缓存读取，不会下载。"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="本地 tokenizer 目录；默认使用 checkpoint 同目录下的 tokenizer",
    )
    parser.add_argument("--feature-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到检查点：{checkpoint}")

    test_file = Path(args.test_file).expanduser().resolve()
    if not test_file.exists():
        raise FileNotFoundError(f"找不到测试文件：{test_file}")

    run_dir = checkpoint.parent
    config_path = (
        Path(args.run_config).expanduser().resolve()
        if args.run_config
        else run_dir / "config.json"
    )
    train_config = load_json(config_path)

    model_name = first_not_none(args.model_name, train_config.get("model_name"))
    feature_size = first_not_none(args.feature_size, train_config.get("feature_size"), 128)
    max_length = first_not_none(args.max_length, train_config.get("max_length"), 384)
    batch_size = first_not_none(
        args.batch_size,
        train_config.get("eval_batch_size"),
        train_config.get("batch_size"),
        16,
    )

    if not model_name:
        raise ValueError(
            "无法确定基础模型。请保留 checkpoint 同目录下的 config.json，"
            "或通过 --model-name 指定本地模型目录。"
        )

    local_model_dir = resolve_local_model_source(str(model_name))

    if args.tokenizer_dir:
        tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    elif (run_dir / "tokenizer").exists():
        tokenizer_dir = (run_dir / "tokenizer").resolve()
    else:
        tokenizer_dir = local_model_dir

    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"找不到本地 tokenizer：{tokenizer_dir}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"检查点：{checkpoint}")
    print(f"测试集：{test_file}")
    print(f"本地基础模型：{local_model_dir}")
    print(f"本地 tokenizer：{tokenizer_dir}")
    print(f"设备：{device}")
    print("离线模式：已启用，不会联网下载")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        use_fast=True,
        local_files_only=True,
    )

    dataset = FraudTextDataset(str(test_file), tokenizer, int(max_length))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=CCRDataCollator(tokenizer),
    )

    # 这里只利用本地基础模型目录创建网络结构，随后完整加载 ccr_best.pt。
    model = CCRClassifier(str(local_model_dir), 2, int(feature_size)).to(device)
    state_dict = load_state_dict(checkpoint, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    sample_ids: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            current_ids = batch.pop("sample_ids")
            current_labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}

            logits, _ = model(**inputs)
            fraud_prob = F.softmax(logits, dim=-1)[:, 1].cpu()
            pred = logits.argmax(dim=-1).cpu()

            labels.extend(current_labels.tolist())
            predictions.extend(pred.tolist())
            probabilities.extend(fraud_prob.tolist())
            sample_ids.extend(current_ids.tolist())

    accuracy = float(accuracy_score(labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "ccr": accuracy,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_samples": len(labels),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["非诈骗", "诈骗"],
            zero_division=0,
            output_dict=True,
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "prediction": predictions,
            "fraud_probability": probabilities,
        }
    ).to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    evaluation_config = {
        "checkpoint": str(checkpoint),
        "test_file": str(test_file),
        "run_config": str(config_path),
        "model_name_in_training_config": model_name,
        "local_model_dir": str(local_model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "feature_size": int(feature_size),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device": str(device),
        "offline": True,
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"评估结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
