from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


STRATEGIES = ["credibility", "urgency", "emotion"]

STRATEGY_NAMES = {
    "credibility": "建立可信度",
    "urgency": "制造紧迫感",
    "emotion": "情感操纵",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评估 Fraud-R1 三种攻击测试集"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="训练好的 CCR 权重，例如 outputs/ccr_run/ccr_best.pt",
    )
    parser.add_argument(
        "--attack-dir",
        default="data/fraud_r1_rule_based",
        help="三个攻击测试集所在目录",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fraud_r1_rule_based_eval",
        help="评估结果保存目录",
    )
    parser.add_argument(
        "--model-name",
        default="hfl/chinese-roberta-wwm-ext",
        help="必须与训练 CCR 时使用的基础模型一致",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=128,
        help="必须与训练 CCR 时的 feature_size 一致",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="必须与训练和原始测试时的最大长度一致",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="评估批大小；显存不足可改为 8 或 4",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='可选，例如 "cuda"、"cuda:0" 或 "cpu"',
    )
    parser.add_argument(
        "--evaluate-script",
        default="evaluate_ccr.py",
        help="单个测试集评估脚本的位置",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    attack_dir = Path(args.attack_dir)
    output_root = Path(args.output_dir)
    evaluate_script = Path(args.evaluate_script)

    if not checkpoint.exists():
        raise FileNotFoundError(f"没有找到 CCR 模型权重：{checkpoint}")

    if not attack_dir.exists():
        raise FileNotFoundError(f"没有找到攻击测试集目录：{attack_dir}")

    if not evaluate_script.exists():
        raise FileNotFoundError(
            f"没有找到 {evaluate_script}。请把本脚本放到和 evaluate_ccr.py 同一级目录。"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for strategy in STRATEGIES:
        test_file = attack_dir / f"test_fraud_r1_{strategy}.csv"

        if not test_file.exists():
            print(f"\n[跳过] 没有找到测试集：{test_file}")
            continue

        strategy_output = output_root / strategy
        strategy_output.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(evaluate_script),
            "--checkpoint",
            str(checkpoint),
            "--test-file",
            str(test_file),
            "--model-name",
            args.model_name,
            "--feature-size",
            str(args.feature_size),
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(strategy_output),
        ]

        if args.device:
            command.extend(["--device", args.device])

        print("\n" + "=" * 80)
        print(f"开始评估：{STRATEGY_NAMES[strategy]}")
        print(f"测试集：{test_file}")
        print("=" * 80)

        subprocess.run(command, check=True)

        metrics_file = strategy_output / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(
                f"评估已经结束，但没有找到指标文件：{metrics_file}"
            )

        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        all_metrics[strategy] = metrics

        confusion_matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]

        summary_rows.append(
            {
                "strategy": strategy,
                "strategy_cn": STRATEGY_NAMES[strategy],
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    if not all_metrics:
        raise RuntimeError(
            "没有评估任何测试集。请检查 attack-dir 和三个 CSV 文件名是否正确。"
        )

    all_metrics_file = output_root / "all_metrics.json"
    all_metrics_file.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_root / "summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("三种攻击测试集评估完成")
    print("=" * 80)

    display_df = summary_df.copy()
    for column in ["accuracy", "precision", "recall", "f1"]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4%}" if pd.notna(value) else ""
        )

    print(
        display_df[
            [
                "strategy_cn",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ].to_string(index=False)
    )

    print(f"\n完整指标：{all_metrics_file}")
    print(f"汇总表格：{summary_file}")


if __name__ == "__main__":
    main()
