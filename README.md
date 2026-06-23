# 中文虚假通话 CCR 二分类项目

本项目把论文 **Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective** 中的 CCR 思路迁移到中文虚假通话二分类任务。

## 1. 已实现内容

- 原始 CSV 清洗与字段转换；
- 标签统一：`False -> 0（非诈骗）`，`True -> 1（诈骗）`；
- 删除无文本、无有效标签和重复样本；
- 从课程训练集划分训练集与验证集；
- Stage 1：中文 Transformer + 交叉熵 + 协方差解耦正则；
- Stage 2：冻结特征提取器，仅训练最后分类层；
- 根据 Stage 1 在训练集上的正确/错误情况估计 IPW；
- 加入逐特征遮蔽的反事实因果约束；
- 输出 Accuracy、Precision、Recall、F1、预测概率和模型权重。

该实现是针对你数据格式重写的紧凑、可运行版本，遵循论文和作者官方代码中的核心两阶段训练逻辑，但没有复制作者为 WILDS、MultiNLI 等数据集编写的大量专用代码。

## 2. 项目结构

```text
CCR_中文诈骗分类完整代码/
├── ccr/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── data/processed/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── data_report.json
├── prepare_data.py
├── train_ccr.py
├── evaluate_ccr.py
├── requirements.txt
├── run_all.bat
└── run_all.sh
```

## 3. 安装环境

推荐 Python 3.10 或 3.11。

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 如果需要特定 CUDA 版本，建议先按 PyTorch 官网命令安装，再运行 `pip install -r requirements.txt`。

## 4. 数据准备

把原始文件放到项目根目录：

```text
训练集结果.csv
测试集结果.csv
```

运行：

```bash
python prepare_data.py \
  --train-input "训练集结果.csv" \
  --test-input "测试集结果.csv" \
  --output-dir data/processed
```

当前已替你生成清洗后的 `train.csv`、`val.csv`、`test.csv`，因此也可以直接训练。

清洗后统一字段：

| 字段 | 含义 |
|---|---|
| sample_id | 当前 split 内样本编号 |
| text | 输入对话文本 |
| label | 0=非诈骗，1=诈骗 |
| interaction_strategy | 原始交互策略元数据 |
| call_type | 原始通话类型元数据 |
| fraud_type | 原始诈骗类型元数据 |
| source_split | 原始来源 |

元数据列不会作为输入特征，只用于后续分组分析。

## 5. 训练完整 CCR

```bash
python train_ccr.py \
  --train-file data/processed/train.csv \
  --val-file data/processed/val.csv \
  --test-file data/processed/test.csv \
  --model-name hfl/chinese-roberta-wwm-ext \
  --output-dir outputs/ccr_run \
  --max-length 384 \
  --feature-size 128 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --stage1-epochs 3 \
  --stage2-epochs 10 \
  --lambda-cov 0.5 \
  --lambda-causal 0.1
```

显存不足时先改：

```bash
--batch-size 4 --eval-batch-size 8 --max-length 256
```

模型下载不稳定时，可以使用 Hugging Face 镜像：

Windows CMD：

```bat
set HF_ENDPOINT=https://hf-mirror.com
```

Linux：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

也可将 `--model-name` 改成 `bert-base-chinese`。

## 6. 输出文件

训练完成后，`outputs/ccr_run/` 中包括：

- `stage1_best.pt`：第一阶段模型；
- `ccr_best.pt`：最终 CCR 模型；
- `stage1_history.csv`、`stage2_history.csv`：训练曲线数据；
- `stage2_ipw_train.csv`：训练样本的 Stage 1 正误与 IPW；
- `stage1_test_predictions.csv`：第一阶段测试预测；
- `ccr_test_predictions.csv`：最终 CCR 测试预测；
- `test_metrics.json`：两阶段测试指标；
- `config.json`：运行参数。

## 7. 独立测试

```bash
python evaluate_ccr.py \
  --checkpoint outputs/ccr_run/ccr_best.pt \
  --test-file data/processed/test.csv \
  --model-name hfl/chinese-roberta-wwm-ext \
  --feature-size 128 \
  --output-dir outputs/eval
```

`model-name` 和 `feature-size` 必须与训练时一致。

## 8. CCR 两阶段含义

### Stage 1

优化：

```text
交叉熵 + lambda_cov × 协方差矩阵非对角项 Frobenius 范数
```

其目的不是直接找出某个具体关键词，而是尽量让最终特征表示彼此解耦。

### Stage 2

固定 Transformer 和特征层，只训练分类器。Stage 1 中每个类别的正确样本和错误样本被视为两个潜在组，并以逆倾向权重平衡。随后逐个遮蔽 128 维特征，比较原始预测与反事实预测，形成因果约束。

## 9. 重要说明

1. 当前阶段没有进行 Fraud-R1 测试集改写。
2. 当前提供的课程测试集只用于评估，不参与训练。
3. `interaction_strategy`、`call_type`、`fraud_type` 不输入模型，避免标签泄漏。
4. 训练集和测试集已检查，没有发现完全相同的对话文本。
5. 作者论文使用英文 `bert-base-uncased`；本项目替换为中文 RoBERTa，这是任务迁移所必需的调整。
