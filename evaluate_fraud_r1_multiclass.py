from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ccr_multiclass import ID2LABEL, NUM_LABELS
from evaluate_ccr_multiclass import evaluate_checkpoint


STRATEGIES = {
    "original": "原始测试集",
    "credibility": "建立可信度",
    "urgency": "制造紧迫感",
    "emotion": "情感操纵",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评估八分类 CCR 在原始与 Fraud-R1 三种攻击集上的表现"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--original-test-file",
        default="data/processed_multiclass/test.csv",
    )
    parser.add_argument(
        "--attack-dir",
        default="data/fraud_r1_compact_multiclass",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fraud_r1_multiclass_eval",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def per_class_recall(
    metrics: dict[str, Any],
    label_name: str,
) -> float:
    report = metrics["classification_report"]
    return float(report.get(label_name, {}).get("recall", 0.0))


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    attack_dir = Path(args.attack_dir)
    datasets = {
        "original": Path(args.original_test_file),
        "credibility": attack_dir / "test_fraud_r1_credibility.csv",
        "urgency": attack_dir / "test_fraud_r1_urgency.csv",
        "emotion": attack_dir / "test_fraud_r1_emotion.csv",
    }

    all_metrics: dict[str, dict[str, Any]] = {}

    for key, file_path in datasets.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"没有找到 {STRATEGIES[key]}：{file_path}"
            )

        print("\n" + "=" * 90)
        print(f"开始评估：{STRATEGIES[key]}")
        print("=" * 90)

        metrics = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            test_file=file_path,
            output_dir=output_root / key,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            fp16=args.fp16,
            tokenizer_path=args.tokenizer_path,
        )
        all_metrics[key] = metrics

    original = all_metrics["original"]
    summary_rows = []

    for key, metrics in all_metrics.items():
        binary = metrics["binary_normal_vs_fraud"]

        summary_rows.append(
            {
                "dataset": key,
                "dataset_cn": STRATEGIES[key],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "binary_fraud_recall": binary["recall"],
                "fraud_as_normal_count": binary[
                    "fraud_as_normal_count"
                ],
                "accuracy_drop_vs_original": (
                    original["accuracy"] - metrics["accuracy"]
                ),
                "macro_f1_drop_vs_original": (
                    original["macro_f1"] - metrics["macro_f1"]
                ),
                "weighted_f1_drop_vs_original": (
                    original["weighted_f1"]
                    - metrics["weighted_f1"]
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_root / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 各类别 Recall 对比，便于观察哪种诈骗最容易被攻击影响。
    per_class_rows = []
    for label_id in range(NUM_LABELS):
        label_name = ID2LABEL[label_id]
        row = {
            "label_id": label_id,
            "label_name": label_name,
        }
        for key, metrics in all_metrics.items():
            row[f"{key}_recall"] = per_class_recall(
                metrics,
                label_name,
            )
            row[f"{key}_recall_drop_vs_original"] = (
                per_class_recall(original, label_name)
                - per_class_recall(metrics, label_name)
            )
        per_class_rows.append(row)

    pd.DataFrame(per_class_rows).to_csv(
        output_root / "per_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (output_root / "all_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display = summary_df.copy()
    percent_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "binary_fraud_recall",
        "accuracy_drop_vs_original",
        "macro_f1_drop_vs_original",
        "weighted_f1_drop_vs_original",
    ]
    for column in percent_columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4%}"
        )

    print("\n" + "=" * 100)
    print("八分类 Fraud-R1 鲁棒性评估完成")
    print("=" * 100)
    print(
        display[
            [
                "dataset_cn",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "binary_fraud_recall",
                "fraud_as_normal_count",
                "accuracy_drop_vs_original",
                "macro_f1_drop_vs_original",
            ]
        ].to_string(index=False)
    )

    print(f"\n总指标：{output_root / 'all_metrics.json'}")
    print(f"汇总表：{output_root / 'summary.csv'}")
    print(
        f"各类别 Recall："
        f"{output_root / 'per_class_recall.csv'}"
    )


if __name__ == "__main__":
    main()
