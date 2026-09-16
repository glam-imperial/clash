# CLASH - Hearing It or Reading It?


Official code layout for controlled lexical-prosodic separation experiments in sarcasm detection.

本仓库给出统一、与贡献者身份无关的论文实现。代码覆盖 WORLD 音频构造、O/L/P/F 客观验证、音频大模型首决策位置打分、全量 context probe、七模型横向比较、AUROC 分解、cluster bootstrap、mixed-effects、多重比较校正和论文图生成。仓库只保存代码、配置与本 README；不包含语音、元数据、标签、预测结果、模型权重或论文结果表。

## Repository structure

```text
CLASH/
├── analysis/
│   ├── decomposition.py
│   ├── statistics.py
│   └── visualization.py
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── prompt.yaml
├── experiments/
│   ├── context_probe.py
│   └── inference.py
├── preprocessing/
│   ├── neutralization.py
│   ├── validation.py
│   └── world_processing.py
├── .gitignore
└── README.md
```

截图中的 `._gitignore` 是 macOS AppleDouble 元数据，不是 GitHub 项目文件。本仓库通过 `.gitignore` 中的 `._*` 排除它。

### File inventory

| File | Responsibility |
|---|---|
| `analysis/decomposition.py` | 计算 O/L/P/F AUROC、主效应、F-based 辅助效应、两种 logit 口径差异与 H4 可识别性 |
| `analysis/statistics.py` | conversation/speaker cluster bootstrap、经验 p 值、BH-FDR、七模型 mixed-effects |
| `analysis/visualization.py` | 生成 condition AUROC、主效应 CI 和分数口径敏感性图 |
| `configs/dataset.yaml` | 数据规模、路径、条件、采样率、WORLD、bootstrap 与 gate 参数 |
| `configs/model.yaml` | 模型地址/类、设备、dtype、模型清单与主/敏感性打分口径 |
| `configs/prompt.yaml` | target-only、audio+context 提示词和 Yes/No 词元变体 |
| `experiments/context_probe.py` | 全量 Qwen3 context probe 与七模型分析的统一入口 |
| `experiments/inference.py` | 音频模型加载、首个回答位置 logits、token-set logsumexp 和 single-token 对照 |
| `preprocessing/neutralization.py` | 基于 WORLD 参数生成成对 O/L/P/F 条件及 manifest |
| `preprocessing/validation.py` | 输入 schema/唯一键/行数审计，以及 O-P 的逐样本 F0/能量验证 |
| `preprocessing/world_processing.py` | 音频 I/O、WORLD 分析/合成、帧功率匹配和固定频谱载体估计 |

## Experimental conditions

每条语句保留同一个 `(dataset, id)`，只改变音频条件：

| Condition | Meaning | Implementation |
|---|---|---|
| `O` | Original | 重采样后的原始单声道音频 |
| `L` | Lexical-only | 保留时变频谱包络，压缩 F0 变化并中和逐帧能量 |
| `P` | Prosody-only | 保留 F0、能量包络与非周期成分，用数据集级固定频谱载体替换词汇频谱 |
| `F` | Fully neutralised | 固定频谱载体，同时中和 F0、能量和时变非周期成分 |
| `L_TTS` | Lexical-only TTS | 可选敏感性臂；由外部 TTS 生成，不由本仓库伪造为 WORLD-L |

`neutralization.py` 先在每个数据集内、无标签地估计固定 carrier，再为每条语句生成四个配对条件。CMMA 的 L 默认保留很小的 F0 spread 以避免完全抹除字调；MUStARD 默认将句内 voiced F0 压平。所有值均在 `configs/dataset.yaml` 中显式配置。

这套代码是公开仓库的统一规范实现。若需要复现已经归档的历史 waveform，必须同时保留当时的音频、依赖版本和配置；不同 WORLD/音高追踪版本不能假定逐采样点相同。

## Methodological policy

### Scores

主分数在模型的第一个回答位置计算：

```text
score = logsumexp(logits[Yes-token set])
      - logsumexp(logits[No-token set])
```

`single_token_margin` 使用配置中首个 Yes token 与 No token 的原始 logit 差，仅作实现敏感性分析。代码只接纳可独立编码为一个 token 的答案变体；若配置与 tokenizer 不兼容会立即报错，不会悄悄改用其他位置。

### Primary metric and decomposition

主指标是 AUROC。对 `c ∈ {O,L,P,F}`：

```text
AUC_c = AUROC(label, score_c)
```

正文优先使用不依赖 F 基线的两个量：

```text
L_minus_P = AUC_L - AUC_P
O_minus_L = AUC_O - AUC_L
```

下列量仍会输出，但只能在同时报告 `AUC_F` 及其 CI 后解释：

```text
L_minus_F = AUC_L - AUC_F
P_minus_F = AUC_P - AUC_F
interaction = AUC_O - AUC_L - AUC_P + AUC_F
```

原因是 F 在部分 dataset/setting 下不是随机地板，并且对 logit 提取细节更敏感。

### Cluster inference

- CMMA：优先从 `C_<conversation>_U_...` 提取 conversation cluster，失败时回退到 speaker。
- MUStARD：speaker cluster；七模型留出集只有 3 个 speaker，因此推断必须标为 exploratory。
- bootstrap 始终按 cluster 重抽，并保持同一语句 O/L/P/F 配对。
- 95% CI 使用 2.5%/97.5% 分位数。
- 所有报告效应的双侧经验 p 值统一做 Benjamini-Hochberg FDR。

### Mixed-effects

不同模型的原始 margin 不可直接比较。代码先按真实标签校正方向，再在模型内标准化：

```text
adjusted_margin = margin * (+1 if label=1 else -1)
z_margin = zscore(adjusted_margin within model)
```

Omnibus 模型：

```text
z_margin ~ C(model) * C(condition, Treatment(reference='O')) + C(dataset)
```

随机截距为 `dataset::speaker`。分数据集模型移除 dataset 固定效应；固定效应 p 值按模型族做 BH-FDR。正确输入行数是：

```text
7 models * 944 utterances * 4 conditions = 26,432 rows
```

`validation.py` 先把 metadata 约束为每个 `(dataset,id)` 一行，再以 `many_to_one` 合并。这样可阻止 condition-level metadata 只按 `(dataset,id)` 合并造成四倍复制。

## H3 and H4

H3 预期“不同模型家族分别专门依赖韵律、词汇或交互”。当前证据未显示这种分工能跨数据集和模型稳定复现。因此横向表可以支持保守的模型观察，但 H3 本身应报告为 **not supported**。

H4 比较 target-only 与 audio+context 的韵律依赖。只有两个 setting 的 F 都与 0.5 随机地板兼容时，`P_minus_F` 才是同义、可比较的估计量。`context_probe.py` 使用主口径 token-set 分数和 `AUC_F` cluster-bootstrap CI 自动判定；任一 setting 的 CI 排除 0.5 时输出：

```text
H4 not identifiable under the current F-baseline design
```

这不等于“context 没有效应”，而是当前 F 基线设计不能把模型行为变化、context 信息冗余和 AUROC 尺度效应分开。代码仍报告不含 F 的 `L_minus_P` 变化，供描述性分析使用。

## Installation

建议 Python 3.10 或更高版本。先创建隔离环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell 激活命令：

```powershell
.\.venv\Scripts\Activate.ps1
```

分析依赖：

```bash
python -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib pyyaml
```

音频构造与验证另需：

```bash
python -m pip install librosa soundfile pyworld
```

模型推理另需与所选模型兼容的 PyTorch、Transformers、Accelerate 和模型官方依赖：

```bash
python -m pip install torch transformers accelerate
```

模型类和 checkpoint 不在代码里猜测。请按实际安装的 Transformers 版本，在 `configs/model.yaml` 填写 `path_or_id`、`processor_path_or_id` 和 `transformers_class`。

## Configuration

### `configs/dataset.yaml`

填写外部目录：

```yaml
paths:
  results_root: /path/to/RESULTS_ROOT
  token_set_root: /path/to/TOKEN_SET_ROOT
  output_dir: outputs/clash_analysis
```

论文规模预期为：

| Dataset | Full sample | Seven-model holdout | Holdout speakers |
|---|---:|---:|---:|
| CMMA | 3,935 | 781 | 22 |
| MUStARD | 690 | 163 | 3 |
| Total | 4,625 | 944 | 25 |

### `configs/model.yaml`

每个要运行的模型至少设置：

```yaml
models:
  qwen3_omni_30b:
    path_or_id: /path/or/huggingface-id
    processor_path_or_id: /path/or/huggingface-id
    transformers_class: InstalledTransformersModelClass
```

统一 runner 依赖模型 processor 支持多模态 chat template，并能在 forward 输出中提供 `.logits`。不满足接口的模型需要显式 adapter，不能通过改文件名伪装成已兼容。

### `configs/prompt.yaml`

提示词强制模型把 `Yes`/`No` 放在首个回答 token。任何 prompt 修改都会改变决策位置和 score，应与结果一起版本化。

## Input schemas

### Neutralisation manifest

最小字段：

```text
dataset,id,audio_path
```

可以附带 label、speaker、context 等字段；它们会原样写入生成后的 `conditions_manifest.csv`。

### Inference manifest

最小字段：

```text
dataset,id,condition,audio_path
```

`audio_plus_context` 还必须提供 `context_text`。建议同时携带 `label` 和 `speaker` 以方便后续审计。

### Analysis roots

`RESULTS_ROOT`：

```text
RESULTS_ROOT/
├── metadata/
│   ├── all.csv
│   └── test_all.csv
├── qwen3/
│   ├── qwen3_token_logits_full.csv
│   └── qwen3_audio_plus_context_token_logits_full.csv
└── analysis/audio_7models/
    └── predictions_7models.csv
```

`TOKEN_SET_ROOT`：

```text
TOKEN_SET_ROOT/
├── metadata/
│   └── clash_metadata_final.csv
└── predictions/
    ├── prediction_qwen3omni_target_logprob.csv
    └── prediction_qwen3omni_context_logprob.csv
```

Metadata 最低字段为 `dataset,id,label,speaker`。预测键为 `dataset,id,condition`；七模型表另需 `model`。loader 会记录实际采用的 score/margin 列并核对精确行数。

## Running the pipeline

### 1. Construct O/L/P/F audio

```bash
python preprocessing/neutralization.py \
  --manifest /path/to/original_audio_manifest.csv \
  --output-dir /path/to/neutralised_audio \
  --config configs/dataset.yaml
```

输出目录必须为空，防止不同配置的 waveform 混合。

### 2. Validate O-P preservation

```bash
python preprocessing/validation.py \
  --manifest /path/to/neutralised_audio/conditions_manifest.csv \
  --output-dir /path/to/validation_output \
  --config configs/dataset.yaml
```

输出逐样本 F0/能量相关和逐数据集汇总。汇总同时报告 mean、median、`>= threshold` 比例及负相关比例，不能只用均值给双峰分布下结论。WORLD 去词化后重新提取 F0 可能发生清浊/八度错误；因此低 F0 相关必须结合直接输入声码器的 F0 轨复核，不能自动等同于构造失败。

### 3. Run model scoring

```bash
python experiments/inference.py \
  --manifest /path/to/inference_manifest.csv \
  --model qwen3_omni_30b \
  --setting target_only \
  --output /path/to/qwen3_target.csv
```

Context 设置：

```bash
python experiments/inference.py \
  --manifest /path/to/inference_manifest_with_context.csv \
  --model qwen3_omni_30b \
  --setting audio_plus_context \
  --output /path/to/qwen3_context.csv
```

每行保留 token-set 主分数、single-token 对照、Yes/No logsumexp 与 error。存在任何 error 时进程返回非零状态。

### 4. Run full analysis

使用 YAML 中路径：

```bash
python experiments/context_probe.py --config configs/dataset.yaml
```

或从命令行覆盖：

```bash
python experiments/context_probe.py \
  --results-root /path/to/RESULTS_ROOT \
  --token-set-root /path/to/TOKEN_SET_ROOT \
  --output-dir /path/to/analysis_output \
  --bootstrap-replicates 2000 \
  --seed 20260914
```

Smoke test 可以使用 100-200 次 bootstrap；论文结果建议至少 2,000 次。使用 `--no-figures` 可以只生成统计表。

## Generated analysis outputs

这些文件由运行产生，不提交到代码仓库：

| Output | Meaning |
|---|---|
| `score_method_comparison.csv` | token-set 与 single-token 的 Spearman、AUROC 差、分数标准差 |
| `full_single_token_auc_effects.csv` | 全量 single-token Qwen3 结果 |
| `full_token_set_auc_effects.csv` | 全量主口径 Qwen3 结果 |
| `seven_model_auc_effects.csv` | 七模型共同留出集及 cluster CI/FDR |
| `h4_f_baseline_assessment.csv` | 各 dataset/setting 的 F 地板诊断 |
| `h4_decision.csv` | H4 是否可识别的逐数据集判定 |
| `mixed_effects_omnibus.csv` | 七模型 omnibus mixed-effects |
| `mixed_effects_by_dataset.csv` | 分数据集 mixed-effects；小 cluster 数自动标 exploratory |
| `analysis_audit.json` | 输入、版本、行数、seed、统计口径、模型错误与输出索引 |
| `*.png` | 三张可重复生成的论文分析图 |

预期审计计数：

```text
full_single_token_rows = 37,000
full_token_set_rows    = 46,250
seven_model_rows       = 26,432
seven_models           = 7
score_comparison_cells = 16
```

## Reproducibility checklist

- [ ] 从干净 commit、固定配置和空输出目录开始。
- [ ] 保存 Python、WORLD、Transformers、模型 revision、dtype、seed 和完整命令。
- [ ] 核对每个 prediction key 唯一、condition 完整、label 与 metadata 一致。
- [ ] 明确结果来自全量 4,625 还是七模型留出集 944。
- [ ] 明确 Qwen3 主口径是 token-set，single-token 只作敏感性分析。
- [ ] 解释 F-based 量前先报告 `AUC_F` 和 cluster CI。
- [ ] MUStARD speaker-level 推断标为 exploratory。
- [ ] 检查 mixed-effects 的 `**MODEL_ERROR**`、convergence 和 warnings。
- [ ] 检查 `analysis_audit.json` 的精确行数和 H4 判定。
- [ ] 将代码 commit、配置、外部输入 checksum 和输出 checksum 一起归档。

## Limitations

- 数据、模型、权重、原始预测和正式人工可懂度/MOS 结果不随代码发布。
- 通用 `inference.py` 只支持遵循 Transformers 多模态 chat-template 和 `.logits` 接口的模型；其他家族需要可审计 adapter。
- F 只是 approximately neutralised control，不能预设为理论上的无信息条件。
- P 的能量保真应看完整分布；F0 从去词化音频重提取可能产生测量伪影。
- 训练过的 probes 必须只在 speaker-independent holdout 上评价；零样本 LLM 可以报告全量结果。
- 推理噪声、模型 revision 和 tokenization 差异必须通过版本固定或重复运行量化。

## Citation and license

公开 GitHub 前，请补充最终论文引用、仓库 URL 和经作者/机构确认的许可证。当前代码不替作者假定 DOI 或软件许可。

