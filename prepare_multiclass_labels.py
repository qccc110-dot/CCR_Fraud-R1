from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABEL2ID = {
    "正常通话": 0,
    "客服诈骗": 1,
    "银行诈骗": 2,
    "投资诈骗": 3,
    "钓鱼诈骗": 4,
    "彩票诈骗": 5,
    "绑架诈骗": 6,
    "身份盗窃": 7,
}


def normalize_binary_label(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    raise ValueError(f"无法识别二分类标签：{value!r}")


def convert_csv(input_path: Path, output_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        required = {"sample_id", "text", "label", "fraud_type"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{input_path.name} 缺少字段：{sorted(missing)}")

        output_fields = []
        for field in fields:
            output_fields.append(field)
            if field == "label":
                output_fields += ["label_name", "binary_label"]

        counts = Counter()
        total = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                binary_label = normalize_binary_label(row["label"])
                label_name = (
                    "正常通话"
                    if binary_label == 0
                    else str(row.get("fraud_type", "")).strip()
                )

                if label_name not in LABEL2ID:
                    raise ValueError(
                        f"未知或缺失类别：{label_name!r}, "
                        f"sample_id={row.get('sample_id')}"
                    )

                row["binary_label"] = binary_label
                row["label_name"] = label_name
                row["label"] = LABEL2ID[label_name]
                writer.writerow(row)

                counts[label_name] += 1
                total += 1

    return {
        "source": str(input_path),
        "output": str(output_path),
        "total_rows": total,
        "class_counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = convert_csv(Path(args.input), Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
