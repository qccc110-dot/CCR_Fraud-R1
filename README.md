# CCR_Fraud-R1

基于 **CCR（Causal Correlation Robustness）** 与 **Fraud-R1 式社会工程增强策略** 的中文虚假通话检测与鲁棒性分析项目。

本项目同时实现：

- 中文虚假通话二分类：正常通话 / 诈骗通话；
- 中文虚假通话八分类：正常通话 + 7 类诈骗类型；
- CCR 两阶段训练；
- 类别无关的 Fraud-R1 式三阶段攻击测试集构造；
- 原始测试集与攻击测试集上的二分类、八分类评估；
- Accuracy、Precision、Recall、F1、Macro-F1、混淆矩阵及逐类别 Recall 分析。

仓库地址：

```text
https://github.com/qccc110-dot/CCR_Fraud-R1
```

---

## 1. 项目简介

传统诈骗文本分类模型可能依赖“转账”“验证码”“银行卡”等表面关键词。当诈骗文本加入更正式的业务措辞、紧迫性表达或情感诱导后，模型预测可能发生变化。

本项目采用中文 RoBERTa 作为文本编码器，并参考 CCR 方法进行两阶段训练：

1. **Stage 1：基础分类与特征解耦**
   - 交叉熵分类损失；
   - 特征协方差非对角项正则；
   - 训练编码器、特征层与分类头。

2. **Stage 2：因果校准**
   - 冻结编码器和特征层；
   - 根据“真实类别 × Stage 1 是否预测正确”估计 IPW；
   - 使用逐特征反事实遮蔽构造因果约束；
   - 仅更新最终分类层。

完成训练后，使用类别无关的 Fraud-R1 式增强测试集评估模型在以下三种社会工程策略下的鲁棒性：

- 建立可信度；
- 制造紧迫感；
- 情感操纵。

---

## 2. 任务定义

### 2.1 二分类任务

| 标签 | 含义 |
|---:|---|
| 0 | 正常通话 |
| 1 | 诈骗通话 |

### 2.2 八分类任务

| 标签 | 类别 |
|---:|---|
| 0 | 正常通话 |
| 1 | 客服诈骗 |
| 2 | 银行诈骗 |
| 3 | 投资诈骗 |
| 4 | 钓鱼诈骗 |
| 5 | 彩票诈骗 |
| 6 | 绑架诈骗 |
| 7 | 身份盗窃 |

---

## 3. 项目目录结构

完成数据准备、二分类训练、八分类训练和鲁棒性评估后，推荐目录结构如下：

```text
CCR_Fraud-R1/
├── ccr/                                      # 二分类 CCR 模型包
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── ccr_multiclass/                           # 八分类 CCR 模型包
│   ├── __init__.py
│   ├── data.py
│   ├── labels.py
│   └── model.py
├── data/
│   ├── original/                             # 原始训练集和测试集
│   ├── processed/                            # 二分类 train/val/test
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── data_report.json
│   ├── processed_multiclass/                 # 八分类 train/val/test
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── label_mapping.json
│   │   └── data_report.json
│   ├── fraud_r1_class_agnostic_binary/       # 二分类攻击集
│   │   ├── test_fraud_r1_credibility.csv
│   │   ├── test_fraud_r1_urgency.csv
│   │   └── test_fraud_r1_emotion.csv
│   └── fraud_r1_class_agnostic_multiclass/   # 八分类攻击集
│       ├── test_fraud_r1_credibility.csv
│       ├── test_fraud_r1_urgency.csv
│       ├── test_fraud_r1_emotion.csv
│       ├── label_mapping.json
│       ├── verification_report.json
│       └── examples_before_after.txt
├── outputs/
│   ├── ccr_run/                              # 二分类训练结果
│   ├── ccr_multiclass_run/                   # 八分类训练结果
│   ├── ccr_multiclass_eval/                  # 八分类原始测试集评估
│   ├── fraud_r1_class_agnostic_binary_eval/  # 二分类攻击评估结果
│   └── fraud_r1_class_agnostic_multiclass_eval/
├── prepare_data.py
├── prepare_multiclass_labels.py
├── generate_class_agnostic_attacks.py
├── train_ccr.py
├── train_ccr_multiclass.py
├── evaluate_ccr.py
├── evaluate_ccr_multiclass.py
├── evaluate_fraud_r1_testsets.py
├── evaluate_fraud_r1_multiclass.py
├── check_multiclass_text_and_predictions.py
├── check_truncated_difference.py
├── requirements.txt
├── .gitignore
└── README.md
```

> 训练好的模型权重文件通常较大，本仓库默认不上传 `*.pt`、`*.pth`、`*.ckpt` 和 `*.safetensors`。克隆仓库后需要重新训练得到权重。

---

## 4. 环境要求

推荐环境：

- Linux；
- Python 3.10；
- NVIDIA GPU；
- CUDA 12.4；
- PyTorch 2.6.0；
- Transformers 4.51.3。

本项目曾使用：

```text
NVIDIA Driver: 550.78
CUDA: 12.4
Python: 3.10
PyTorch: 2.6.0+cu124
Transformers: 4.51.3
```

没有 GPU 时也可使用 CPU，但训练速度会明显降低。

---

## 5. 克隆项目

```bash
git clone https://github.com/qccc110-dot/CCR_Fraud-R1.git
cd CCR_Fraud-R1
```

---

## 6. 创建环境

### 6.1 使用 Conda

```bash
conda create -n ccr python=3.10 -y
conda activate ccr
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

如需手动安装 CUDA 12.4 对应 PyTorch：

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

安装其他依赖：

```bash
pip install \
  transformers==4.51.3 \
  pandas==2.2.3 \
  scikit-learn==1.6.1 \
  numpy==2.1.3 \
  tqdm==4.67.1 \
  safetensors==0.5.3 \
  huggingface-hub==0.30.2 \
  tokenizers==0.21.1
```

### 6.3 验证环境

```bash
python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 7. Hugging Face 模型下载

项目默认使用：

```text
hfl/chinese-roberta-wwm-ext
```

网络环境无法直接访问 Hugging Face 时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

永久配置：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## 8. 数据说明

### 8.1 `data/original/`

保存原始训练集和测试集，用于数据清洗和复现实验。

### 8.2 `data/processed/`

二分类标准数据：

```text
train.csv
val.csv
test.csv
```

至少包含：

```text
sample_id
text
label
```

标签定义：

```text
0 = 正常通话
1 = 诈骗通话
```

### 8.3 `data/processed_multiclass/`

八分类标准数据，常见字段：

```text
sample_id
text
label
label_name
binary_label
fraud_type
```

### 8.4 类别无关攻击数据

正式鲁棒性实验使用：

```text
data/fraud_r1_class_agnostic_binary/
data/fraud_r1_class_agnostic_multiclass/
```

三种攻击集：

```text
test_fraud_r1_credibility.csv
test_fraud_r1_urgency.csv
test_fraud_r1_emotion.csv
```

构造原则：

1. 只修改诈骗样本；
2. 正常样本保持不变；
3. 生成攻击文本时不读取 `fraud_type`；
4. 三种策略使用统一、类别无关的话术；
5. 保留原始诈骗意图与真实标签；
6. 避免加入直接提示诈骗类型的信息。

---

## 9. 数据预处理

### 9.1 生成二分类数据

```bash
python prepare_data.py
```

输出：

```text
data/processed/
```

查看脚本参数：

```bash
python prepare_data.py --help
```

### 9.2 生成八分类数据

```bash
python prepare_multiclass_labels.py
```

输出：

```text
data/processed_multiclass/
```

检查类别分布：

```bash
python - <<'PY'
import pandas as pd

for split in ["train", "val", "test"]:
    path = f"data/processed_multiclass/{split}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    print("\n", split)
    print(df["label_name"].value_counts())
PY
```

---

## 10. 训练二分类 CCR

```bash
python train_ccr.py \
  --train-file "data/processed/train.csv" \
  --val-file "data/processed/val.csv" \
  --test-file "data/processed/test.csv" \
  --model-name "hfl/chinese-roberta-wwm-ext" \
  --output-dir "outputs/ccr_run"
```

训练完成后主要生成：

```text
outputs/ccr_run/
├── stage1_best.pt
├── ccr_best.pt
├── tokenizer/
├── config.json
└── test_metrics.json
```

最终二分类模型：

```text
outputs/ccr_run/ccr_best.pt
```

---

## 11. 评估二分类模型

### 11.1 原始测试集

```bash
python evaluate_ccr.py \
  --checkpoint "outputs/ccr_run/ccr_best.pt" \
  --test-file "data/processed/test.csv" \
  --output-dir "outputs/ccr_eval"
```

### 11.2 三种攻击测试集

```bash
python evaluate_fraud_r1_testsets.py \
  --checkpoint "outputs/ccr_run/ccr_best.pt" \
  --attack-dir "data/fraud_r1_class_agnostic_binary" \
  --output-dir "outputs/fraud_r1_class_agnostic_binary_eval"
```

重点查看：

```text
outputs/fraud_r1_class_agnostic_binary_eval/summary.csv
outputs/fraud_r1_class_agnostic_binary_eval/all_metrics.json
```

---

## 12. 训练八分类 CCR

```bash
python train_ccr_multiclass.py \
  --train-file "data/processed_multiclass/train.csv" \
  --val-file "data/processed_multiclass/val.csv" \
  --test-file "data/processed_multiclass/test.csv" \
  --model-name "hfl/chinese-roberta-wwm-ext" \
  --output-dir "outputs/ccr_multiclass_run" \
  --max-length 384 \
  --feature-size 128 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --stage1-epochs 3 \
  --stage2-epochs 10
```

主要输出：

```text
outputs/ccr_multiclass_run/
├── stage1_best.pt
├── ccr_best.pt
├── tokenizer/
├── config.json
├── stage1_history.csv
├── stage2_history.csv
├── stage2_ipw_train.csv
├── stage1_test_predictions.csv
├── ccr_test_predictions.csv
├── stage1_test_confusion_matrix.csv
├── ccr_test_confusion_matrix.csv
└── test_metrics.json
```

最终八分类模型：

```text
outputs/ccr_multiclass_run/ccr_best.pt
```

显存不足时：

```bash
python train_ccr_multiclass.py \
  --train-file "data/processed_multiclass/train.csv" \
  --val-file "data/processed_multiclass/val.csv" \
  --test-file "data/processed_multiclass/test.csv" \
  --output-dir "outputs/ccr_multiclass_run" \
  --batch-size 4 \
  --eval-batch-size 8
```

---

## 13. 评估八分类模型

### 13.1 原始测试集

```bash
python evaluate_ccr_multiclass.py \
  --checkpoint "outputs/ccr_multiclass_run/ccr_best.pt" \
  --test-file "data/processed_multiclass/test.csv" \
  --output-dir "outputs/ccr_multiclass_eval"
```

输出：

```text
outputs/ccr_multiclass_eval/
├── metrics.json
├── predictions.csv
├── classification_report.csv
└── confusion_matrix.csv
```

### 13.2 三种攻击测试集

```bash
python evaluate_fraud_r1_multiclass.py \
  --checkpoint "outputs/ccr_multiclass_run/ccr_best.pt" \
  --original-test-file "data/processed_multiclass/test.csv" \
  --attack-dir "data/fraud_r1_class_agnostic_multiclass" \
  --output-dir "outputs/fraud_r1_class_agnostic_multiclass_eval"
```

输出：

```text
outputs/fraud_r1_class_agnostic_multiclass_eval/
├── original/
├── credibility/
├── urgency/
├── emotion/
├── summary.csv
├── per_class_recall.csv
└── all_metrics.json
```

---

## 14. 重新生成类别无关攻击数据

```bash
python generate_class_agnostic_attacks.py
```

建议输出到：

```text
data/fraud_r1_class_agnostic_multiclass/
```

从八分类攻击集生成二分类副本：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

source_dir = Path("data/fraud_r1_class_agnostic_multiclass")
output_dir = Path("data/fraud_r1_class_agnostic_binary")
output_dir.mkdir(parents=True, exist_ok=True)

for strategy in ["credibility", "urgency", "emotion"]:
    filename = f"test_fraud_r1_{strategy}.csv"
    df = pd.read_csv(source_dir / filename, encoding="utf-8-sig")

    df["multiclass_label"] = df["label"]

    if "binary_label" in df.columns:
        df["label"] = df["binary_label"].astype(int)
    else:
        df["label"] = (df["label"].astype(int) != 0).astype(int)

    df.to_csv(
        output_dir / filename,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        strategy,
        "总数:", len(df),
        "正常:", int((df["label"] == 0).sum()),
        "诈骗:", int((df["label"] == 1).sum()),
    )
PY
```

---

## 15. 检查攻击数据

```bash
python check_multiclass_text_and_predictions.py \
  --project-root "." \
  --eval-dir "outputs/fraud_r1_class_agnostic_multiclass_eval"
```

正常情况下：

- 正常样本在三种攻击集中保持一致；
- 诈骗样本在三种策略中不同；
- 八分类转换前后文本完全一致；
- 三种攻击的预测不应因文件重复而完全相同。

---

## 16. 主要实验结果

### 16.1 二分类结果

| 测试集 | Accuracy | Precision | Recall | F1 | 诈骗漏检数 |
|---|---:|---:|---:|---:|---:|
| 原始测试集 | 99.92% | 99.86% | 100.00% | 99.93% | 0 |
| 建立可信度 | 99.61% | 99.86% | 99.42% | 99.64% | 8 |
| 制造紧迫感 | 98.70% | 99.85% | 97.76% | 98.80% | 31 |
| 情感操纵 | 97.92% | 99.85% | 96.32% | 98.05% | 51 |

### 16.2 八分类结果

| 测试集 | Accuracy | Macro Recall | Macro F1 | Weighted F1 | 诈骗判为正常 |
|---|---:|---:|---:|---:|---:|
| 原始测试集 | 84.49% | 81.25% | 77.94% | 84.84% | 0 |
| 建立可信度 | 80.45% | 75.44% | 70.44% | 77.32% | 2 |
| 制造紧迫感 | 80.33% | 74.70% | 69.84% | 77.10% | 3 |
| 情感操纵 | 80.06% | 72.78% | 69.16% | 76.88% | 3 |

结论：

1. 攻击强度逐级增强时，二分类和八分类性能均下降；
2. 情感操纵影响最大，制造紧迫感次之；
3. 二分类仍能较稳定识别“是否诈骗”；
4. 八分类对具体诈骗类型的判断更容易受到攻击影响；
5. Macro-F1 下降幅度大于 Accuracy，说明少数类别和类别边界受影响更明显。

---

## 17. 常见问题

### 17.1 模型下载失败

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 17.2 CUDA 显存不足

```bash
--batch-size 4 --eval-batch-size 8
```

仍不足时：

```bash
--batch-size 2 --eval-batch-size 4
```

### 17.3 找不到模型权重

仓库不上传训练好的 `*.pt` 权重，需要先运行：

```bash
python train_ccr.py
```

或：

```bash
python train_ccr_multiclass.py
```

### 17.4 `ModuleNotFoundError: No module named 'ccr'`

确认当前目录为项目根目录：

```bash
pwd
ls
```

### 17.5 三种攻击结果完全相同

检查三个 CSV 的 `text` 是否相同：

```bash
python check_multiclass_text_and_predictions.py
```

---

## 18. `.gitignore` 推荐配置

```gitignore
# Python
__pycache__/
*.py[cod]

# 编辑器
.vscode/
.idea/

# 模型权重
*.pt
*.pth
*.ckpt
*.safetensors

# 日志和临时文件
*.log
*.tmp

# 系统文件
.DS_Store
Thumbs.db
```

代码、数据、CSV、JSON 和评估结果均可上传，仅忽略模型权重和临时文件。

---

## 19. 推荐复现顺序

```text
1. 克隆仓库
2. 创建并激活 ccr 环境
3. 安装 requirements.txt
4. 检查 data/processed 和 data/processed_multiclass
5. 训练二分类 CCR
6. 评估二分类原始测试集
7. 评估二分类攻击测试集
8. 训练八分类 CCR
9. 评估八分类原始测试集
10. 评估八分类攻击测试集
11. 查看 summary.csv、all_metrics.json 和 per_class_recall.csv
```

完整命令：

```bash
conda create -n ccr python=3.10 -y
conda activate ccr

pip install -r requirements.txt

python train_ccr.py \
  --train-file data/processed/train.csv \
  --val-file data/processed/val.csv \
  --test-file data/processed/test.csv \
  --output-dir outputs/ccr_run

python evaluate_fraud_r1_testsets.py \
  --checkpoint outputs/ccr_run/ccr_best.pt \
  --attack-dir data/fraud_r1_class_agnostic_binary \
  --output-dir outputs/fraud_r1_class_agnostic_binary_eval

python train_ccr_multiclass.py \
  --train-file data/processed_multiclass/train.csv \
  --val-file data/processed_multiclass/val.csv \
  --test-file data/processed_multiclass/test.csv \
  --output-dir outputs/ccr_multiclass_run

python evaluate_fraud_r1_multiclass.py \
  --checkpoint outputs/ccr_multiclass_run/ccr_best.pt \
  --original-test-file data/processed_multiclass/test.csv \
  --attack-dir data/fraud_r1_class_agnostic_multiclass \
  --output-dir outputs/fraud_r1_class_agnostic_multiclass_eval
```

---

## 20. 参考工作

1. Zhou Y, Zhu Z. *Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective*. NAACL 2025.
2. Yang S, Zhu S, Wu Z, et al. *Fraud-R1: A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements*. Findings of ACL 2025.
3. Cui Y, Che W, Liu T, et al. *Revisiting Pre-Trained Models for Chinese Natural Language Processing*. Findings of EMNLP 2020.
4. Devlin J, Chang M W, Lee K, et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019.

---

## 21. 免责声明

本项目仅用于课程学习、科研实验与模型鲁棒性分析。

仓库中的诈骗文本与社会工程策略用于安全研究和防御评估，不应被用于实际欺诈、钓鱼、诱导或其他违法用途。使用者应遵守当地法律法规及数据集授权要求。

---

## 22. License

当前仓库未明确指定开源许可证。公开发布前，建议根据数据和代码来源选择合适的许可证，例如 MIT License 或 Apache License 2.0。

若原始数据存在课程授权、隐私或再分发限制，应优先遵守原数据使用协议。
