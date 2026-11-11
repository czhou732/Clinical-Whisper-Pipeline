# ClinicalWhisper

[English README](README.md)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20559786.svg)](https://doi.org/10.5281/zenodo.20559786)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.08.728970-b31b1b)](https://doi.org/10.64898/2026.06.08.728970)

**面向临床访谈转录与研究分析、注重隐私的本地处理流程。**

ClinicalWhisper v5.3 在本机处理临床音频。处理期间，音频、转录文本和分析结果不会发送到外部推理服务。

---

## 当前版本：v5.3

- **MOSS-Transcribe-Diarize 0.9B**：在一次处理中完成转录与说话人区分，取代早期版本的 Whisper + Pyannote 主流程。
- **OpenMED 标识符遮蔽**：在本地识别并遮蔽姓名、日期、地址、电话号码、病历号等信息，之后再运行基于规则的安全检查。自动遮蔽可能漏检，分享数据前仍需人工复核。
- **评分器 v2**：本地 Llama-3-8B 根据受访者原话生成四项 0–3 的实验性研究评分。Apple Silicon 使用 MLX，其他平台使用 Hugging Face Transformers。
- **桌面应用和批处理**：macOS 桌面应用用于交互式处理；命令行批处理可将整个目录的录音汇总为 CSV。

早期版本曾使用 RoBERTa 情感分析和 Ollama。这些组件不属于当前主流程。

---

## 架构概览

| 类别 | 当前实现 |
|---|---|
| 转录与说话人区分 | MOSS-Transcribe-Diarize 0.9B；自动选择 Apple Silicon、CUDA 或 CPU 路径 |
| 标识符遮蔽 | OpenMED `OpenMed-PII-SuperClinical-Small-44M-v1`，加规则安全检查 |
| 声学特征 | OpenSMILE eGeMAPSv02：音高、响度、抖动、闪烁和 VTA |
| LLM 研究评分 | Llama-3-8B；四项 0–3 评分，每项必须有受访者原话作为证据 |
| 输出 | 每个录音一个完整 JSON；批处理或桌面导出一个汇总 CSV |

## 主流程

```text
音频（WAV / MP3 / M4A / MP4 / OGG / Opus / FLAC / AAC）
    │
    ▼  PyAV 分块解码为 16 kHz 单声道音频
┌───────────────────────────────────────────────────────────────┐
│                    ClinicalWhisper v5.3                       │
│                                                               │
│  1. MOSS 转录与说话人区分（归一化副本）                       │
│  2. OpenMED 标识符遮蔽及规则检查                              │
│  3. OpenSMILE 声学分析（保留原始增益的副本）                  │
│  4. 访谈者 / 受访者角色识别                                   │
│  5. 本地 LLM 研究评分（长访谈按窗口处理）                     │
│                         │                                     │
│                         ▼                                     │
│                 JSON 分析报告 + 汇总 CSV                      │
└───────────────────────────────────────────────────────────────┘
```

转录使用电平归一化副本，以减少安静录音产生空转录的情况。响度和 VTA 会受增益影响，因此声学特征使用保留原始增益的音频副本。

---

## 指标状态与限制

### 标准声学特征

`pitch_mean_st`、`pitch_cv`、`loudness_cv`、`jitter` 和 `shimmer` 来自 OpenSMILE 的 eGeMAPSv02 标准参数集。

### 实验性 LLM 评分

评分器 v2 输出以下四项 0–3 研究评分：

- `anhedonia_content`：受访者对兴趣和愉悦的描述；
- `depressed_mood_content`：受访者对低落情绪的描述；
- `affect_flatness`：受访者语言中情感表达的丰富程度；
- `engagement_level`：受访者参与对话和主动展开回答的程度。

每项评分都必须附有受访者原话；没有足够文本证据时，输出为空值，而不是用默认数字代替。这些评分尚未与 PHQ-9、SHAPS、HAM-D、SCID、临床人员评分或其他临床参考标准进行外部效度验证，不能用于诊断或治疗决策。

仓库中的 `evals/reports/reliability_v2.md` 报告了测试—重测可靠性：13 段短测试文本（12 段合成情境和 1 段合成双人访谈）各运行 5 次。

| 评分 | 5 次运行均有证据的文本数 | ICC(1,1) | 95% CI | 5 次运行均值的可靠性 |
|---|---:|---:|---:|---:|
| `anhedonia_content` | 8 / 13 | 0.956 | 0.89–0.99 | 0.99 |
| `depressed_mood_content` | 5 / 13 | 0.826 | 0.55–0.98 | 0.96 |
| `affect_flatness` | 9 / 13 | 0.309 | 0.05–0.70 | 0.69 |
| `engagement_level` | 11 / 13 | 0.693 | 0.46–0.89 | 0.92 |

这些合成样例较短且刻意设置得较清晰，因此对真实访谈可能过于乐观。`affect_flatness` 即使平均 5 次运行仍然不可靠。可靠性表示重复运行的一致程度，不等于临床准确率或临床效度。`evals/` 中的临床情境测试是软件回归测试，不是临床验证研究。

### VTA

VTA 是项目自定义派生指标：`-ln(CV_F0 × CV_Energy)`。该指标尚无外部验证；2.4 和 4.6 的分段来自组成特征的内部范围，并非真实临床样本校准结果。

---

## 系统要求

| 项目 | 要求 |
|---|---|
| Python | 3.11 或更高版本 |
| 推荐硬件 | Apple Silicon Mac（MLX 路径）；Windows、Linux 和 Intel Mac 使用 Transformers，并在可用时使用 CUDA |
| ffmpeg | 无需单独安装；PyAV 自带解码所需的 FFmpeg 库 |
| 磁盘与网络 | 从源码首次运行需要下载模型；基础 DMG 包含转录和遮蔽模型。约 5 GB 的临床评分模型通过独立的 Scoring 附加组件安装 |

---

## 桌面应用（推荐）

在 macOS 上构建应用和 DMG：

```bash
./build_dmg.sh
```

挂载 `ClinicalWhisper.dmg`，将 **ClinicalWhisper** 拖到 Applications。由于应用尚未经过 Apple 公证，第一次启动时请右键点击应用并选择 **Open**。

将音频拖入窗口并点击 Process。窗口会显示各阶段进度和结果；**Save CSV…** 可将汇总表保存到指定位置。

基础应用可以转录并遮蔽英文访谈。实验性临床评分需要另外安装 **ClinicalWhisper Scoring** 附加组件；其他语言的标识符遮蔽、姓名静音、Praat 和 Kintsugi 功能也通过各自的可选附加组件提供。请在应用底部点击 **Install an add-on…** 并选择已挂载附加组件中的文件夹。

打包应用默认写入：

```text
~/ClinicalWhisper/
├── Output/       # <name>_analysis.json、汇总 CSV 和可选导出
└── config.yaml   # 可选的用户配置
```

桌面应用直接读取用户选择的源文件，不移动或复制源音频。该数据目录位于 Documents 之外，以降低常见 iCloud Documents 设置自动同步研究数据的风险。可通过 `CLINICALWHISPER_DATA_DIR` 修改根目录。自定义配置文件应放在 `~/ClinicalWhisper/config.yaml`。

从源码运行桌面应用：

```bash
uv run python main_app.py
```

---

## 从源码安装

```bash
git clone https://github.com/czhou732/Clinical-Whisper-Pipeline.git
cd Clinical-Whisper-Pipeline
uv venv
```

在 macOS 上激活环境：

```bash
source .venv/bin/activate
```

安装项目和 MOSS 可选依赖：

```bash
uv pip install -e ".[diarization]"
```

## 命令行批处理

```bash
uv run python batch_processor.py --input ./Input --output ./Output/summary.csv --config config.example.yaml
```

每个录音在 CSV 中占一行，包括研究标识符、质量标记、词数、时长、四项评分器 v2 输出、时间特征，以及整段录音和受访者级别的声学特征。完整转录、引用证据、来源信息、警告和分窗口结果保存在每个录音的 JSON 文件中。

### 关键配置

```yaml
moss:
  model: "OpenMOSS-Team/MOSS-Transcribe-Diarize"
  device: "auto"          # auto | mps | cuda | cpu
  dtype: "auto"
  hotwords: []

pii_scrubbing:
  enabled: true
  confidence_threshold: 0.5
  strict: true             # 遮蔽失败时终止，不输出未遮蔽转录

llm_scoring:
  enabled: true
  mlx_model: "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
  hf_model: "NousResearch/Meta-Llama-3-8B-Instruct"
  window_words: 2800
  transcript_scope: "dialogue"
  samples: 5

audio_retention: "archive"
keep_models_loaded: false
```

### 内存处理

批处理先转录文件并释放转录模型，再加载评分模型，因此两个大型模型不会同时常驻内存。音频按块解码到磁盘，避免将长录音完整加载到内存。

### 长访谈评分

超过一个上下文窗口的转录会在说话轮次边界分段，并逐窗口评分后聚合。JSON 中的 `_meta.coverage` 记录成功覆盖比例，`_meta.per_window_scores` 保留各窗口结果。失败或部分成功的运行不会被当作完整结果隐藏起来。

### 当前未接入主流程的模块

`longitudinal.py`、`question_detector.py`、`llm_embeddings.py` 和 `sentiment_analyzer.py` 可作为独立工具使用，但当前桌面应用和批处理主流程不会导入它们。

### 可复现性

每个分析 JSON 都包含 `provenance`：解析后的模型提交哈希、评分设置、软件包版本、平台信息，以及从 Git checkout 运行时的源代码提交。转录或严格模式下的标识符遮蔽失败会终止当前任务，而不是生成看似完整的结果。

---

## 支持与问题报告

使用问题或可复现的软件缺陷请提交到 [GitHub Issues](https://github.com/czhou732/Clinical-Whisper-Pipeline/issues)。请提供 ClinicalWhisper 版本、操作系统、准确命令或应用操作步骤，以及完整错误信息。

桌面应用底部的 **Save diagnostics for a bug report** 可生成经过处理的诊断压缩包，其中不含音频和转录文本，并会移除文件名。也可将该压缩包发送给 Peter Zhou：`czhou732@usc.edu`。

不要在公开 issue 中上传临床音频、原始转录、身份信息、凭据、模型权重或其他私密研究数据。疑似安全或隐私问题请通过电子邮件私下报告。

---

## 引用

如果在研究中使用 ClinicalWhisper，请引用相关的 [bioRxiv 预印本](https://doi.org/10.64898/2026.06.08.728970)：

```bibtex
@article{zhou2026clinicalwhisper,
  title={Cross-Modal Benchmarking of Acoustic Prosody and Ventral Striatal BOLD for Depression-Related Anhedonia Classification: A Pre-Registered Study with the ClinicalWhisper Pipeline},
  author={Zhou, Chengdong and Wu, Muqing and Xiang, Yi and Itti, Laurent},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.08.728970}
}
```

---

## 许可证

本项目采用 [MIT License](LICENSE)。第三方模型和组件的许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
