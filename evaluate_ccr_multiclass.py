from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ccr_multiclass import (
    ID2LABEL,
    NUM_LABELS,
    FraudMulticlassDataset,
    MulticlassCCRClassifier,
    MulticlassCCRDataCollator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估训练好的八分类 CCR 模型"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/ccr_multiclass_eval",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="默认优先使用 checkpoint 同目录下的 tokenizer/",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default=None,
        help='例如 "cuda"、"cuda:0" 或 "cpu"',
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def safe_torch_load(path: str | Path, map_location="cpu") -> dict:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    label_ids = list(range(NUM_LABELS))
    target_names = [ID2LABEL[i] for i in label_ids]

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=label_ids,
            average="weighted",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=label_ids,
    )

    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    binary_labels = (labels != 0).astype(int)
    binary_predictions = (predictions != 0).astype(int)
    binary_precision, binary_recall, binary_f1, _ = (
        precision_recall_fscore_support(
            binary_labels,
            binary_predictions,
            average="binary",
            zero_division=0,
        )
    )

    fraud_as_normal = int(
        sum(matrix[true_label, 0] for true_label in range(1, NUM_LABELS))
    )

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "binary_normal_vs_fraud": {
            "accuracy": float(
                accuracy_score(
                    binary_labels,
                    binary_predictions,
                )
            ),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "fraud_as_normal_count": fraud_as_normal,
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


@torch.no_grad()
def run_evaluation(
    model: MulticlassCCRClassifier,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    labels_all: list[int] = []
    predictions_all: list[int] = []
    sample_ids_all: list[int] = []
    probabilities_all: list[np.ndarray] = []

    for batch in tqdm(loader, desc="评估"):
        labels = batch["labels"].to(device)
        sample_ids = batch["sample_ids"].to(device)
        model_inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in {"labels", "sample_ids"}
        }

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, _ = model(**model_inputs)
            loss = F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            )

        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

        total_loss += float(loss.item())
        total_examples += labels.size(0)

        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        sample_ids_all.extend(sample_ids.cpu().tolist())
        probabilities_all.extend(
            probabilities.cpu().numpy()
        )

    labels_np = np.asarray(labels_all, dtype=np.int64)
    predictions_np = np.asarray(
        predictions_all, dtype=np.int64
    )
    probabilities_np = np.asarray(
        probabilities_all, dtype=np.float32
    )

    metrics = compute_metrics(labels_np, predictions_np)
    metrics["loss"] = total_loss / max(total_examples, 1)

    return {
        "metrics": metrics,
        "labels": labels_np,
        "predictions": predictions_np,
        "sample_ids": np.asarray(sample_ids_all),
        "probabilities": probabilities_np,
    }


def save_results(
    output_dir: Path,
    dataset: FraudMulticlassDataset,
    result: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_df = pd.DataFrame(
        metrics["classification_report"]
    ).T
    report_df.to_csv(
        output_dir / "classification_report.csv",
        encoding="utf-8-sig",
    )

    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"真实_{ID2LABEL[i]}" for i in range(NUM_LABELS)],
        columns=[f"预测_{ID2LABEL[i]}" for i in range(NUM_LABELS)],
    )
    matrix_df.to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    predictions_df = dataset.df.reset_index(drop=True).copy()
    predictions_df["true_label"] = result["labels"]
    predictions_df["true_label_name"] = [
        ID2LABEL[int(label)] for label in result["labels"]
    ]
    predictions_df["prediction"] = result["predictions"]
    predictions_df["prediction_name"] = [
        ID2LABEL[int(label)]
        for label in result["predictions"]
    ]
    predictions_df["correct"] = (
        result["labels"] == result["predictions"]
    )
    predictions_df["confidence"] = (
        result["probabilities"].max(axis=1)
    )

    for label_id in range(NUM_LABELS):
        predictions_df[
            f"prob_{label_id}_{ID2LABEL[label_id]}"
        ] = result["probabilities"][:, label_id]

    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    test_file: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
    device_name: str | None = None,
    fp16: bool = True,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    test_file = Path(test_file)
    output_dir = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"没有找到模型权重：{checkpoint_path}"
        )
    if not test_file.exists():
        raise FileNotFoundError(
            f"没有找到测试集：{test_file}"
        )

    checkpoint = safe_torch_load(checkpoint_path)
    required = {
        "model_state_dict",
        "model_name",
        "feature_size",
        "num_labels",
        "max_length",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"checkpoint 缺少必要字段：{sorted(missing)}"
        )

    if int(checkpoint["num_labels"]) != NUM_LABELS:
        raise ValueError(
            f"checkpoint 的 num_labels="
            f"{checkpoint['num_labels']}，不是八分类模型。"
        )

    device = resolve_device(device_name)
    amp_enabled = bool(fp16 and device.type == "cuda")

    if tokenizer_path is None:
        local_tokenizer = checkpoint_path.parent / "tokenizer"
        tokenizer_source = (
            local_tokenizer
            if local_tokenizer.exists()
            else checkpoint["model_name"]
        )
    else:
        tokenizer_source = Path(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
    )

    dataset = FraudMulticlassDataset(
        test_file,
        tokenizer=tokenizer,
        max_length=int(checkpoint["max_length"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MulticlassCCRDataCollator(tokenizer),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MulticlassCCRClassifier(
        model_name=checkpoint["model_name"],
        num_labels=int(checkpoint["num_labels"]),
        feature_size=int(checkpoint["feature_size"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    result = run_evaluation(
        model,
        loader,
        device,
        amp_enabled,
    )
    save_results(output_dir, dataset, result)

    metrics = result["metrics"]
    print("\n" + "=" * 80)
    print(f"测试集：{test_file}")
    print(
        f"Accuracy={metrics['accuracy']:.4%} | "
        f"Macro-F1={metrics['macro_f1']:.4%} | "
        f"Weighted-F1={metrics['weighted_f1']:.4%} | "
        f"诈骗误判为正常="
        f"{metrics['binary_normal_vs_fraud']['fraud_as_normal_count']}"
    )
    print(f"结果目录：{output_dir}")

    return metrics


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_file=args.test_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        fp16=args.fp16,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
