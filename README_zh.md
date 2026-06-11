[English](README.md) | 中文

# ClinicalWhisper

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ChengdongPeter/Clinical-Whisper)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**隐私优先的临床语音分析流水线 — 转录、说话人分离、情感分析与声学特征提取，100% 本地处理。**

ClinicalWhisper 在本地设备上完整处理临床音频录音，数据不会离开设备，适用于 HIPAA/GDPR 等合规环境。

---

## v3.0 更新内容

- 🧠 **基于 Transformer 的情感分析** — 使用 `cardiffnlp/twitter-roberta-base-sentiment-latest` 替代 TextBlob，提供更准确的临床情感评分
- 🧪 **完整测试套件** — 20+ 项 pytest 测试，覆盖情感分析、声学特征、流水线集成及 Zhou Index
- 📊 **批量处理模式** — 一次性处理整个目录的音频文件，生成汇总 CSV
- 📈 **纵向追踪** — 跨会话追踪患者指标，自动检测趋势
- 📦 **现代化打包** — 支持 `pyproject.toml` 和 `pip install -e .`
- 🔄 **GitHub Actions CI** — 每次推送和 Pull Request 自动运行测试
- 🔧 **NLTK 句子分词** — 相比正则分割，能更好地处理缩写和边界情况
- 📋 **CSV/SPSS 导出** — 将分析 JSON 展平为 R/SPSS 兼容的表格数据

---

## 功能特性

| 类别 | 说明 |
|---|---|
| **语音转录** | Apple Silicon 上使用 MLX Whisper（快 4–10 倍），其他平台回退至 OpenAI Whisper |
| **说话人分离** | pyannote.audio — 逐说话人识别与标注 |
| **情感分析** | 基于 Transformer 的评分（1–10），语调检测，关键时刻识别 |
| **声学特征** | OpenSMILE eGeMAPSv02 — 音高、响度、jitter、shimmer、Zhou Index（VTA） |
| **批量处理** | 目录 → CSV 流水线，逐文件错误处理 |
| **纵向追踪** | 基于线性回归的跨会话趋势检测 |
| **会议智能** | 基于 Ollama 的本地 LLM 摘要（可选） |
| **Plaud 集成** | 通过 USB 或导出文件夹自动导入 Plaud Note 设备录音 |
| **隐私保护** | 100% 本地处理 — 零网络请求，零云端依赖 |
| **输出格式** | Markdown（兼容 Obsidian）、JSON 分析结果、CSV 导出 |

---

## 系统架构

```
音频文件 (.m4a/.mp3/.wav/.mp4)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    ClinicalWhisper                       │
│                                                         │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────┐ │
│  │ 预处理    │──▶│ MLX Whisper │──▶│ 说话人分离       │ │
│  │ (FFmpeg)  │   │ / OpenAI    │   │ (pyannote.audio)  │ │
│  └──────────┘   └──────┬──────┘   └────────┬─────────┘ │
│                        │                   │           │
│                        ▼                   │           │
│              ┌─────────────────┐           │           │
│              │   片段构建器    │◀──────────┘           │
│              └────────┬────────┘                       │
│                       │                                │
│          ┌────────────┼────────────┐                   │
│          ▼            ▼            ▼                   │
│  ┌──────────────┐ ┌────────┐ ┌───────────┐            │
│  │ 情感分析     │ │ 统计   │ │ 声学特征  │            │
│  │ (RoBERTa)    │ │        │ │ (OpenSMILE│            │
│  │              │ │        │ │  eGeMAPS) │            │
│  └──────┬───────┘ └───┬────┘ └─────┬─────┘            │
│         └─────────────┼────────────┘                   │
│                       ▼                                │
│              ┌─────────────────┐                       │
│              │ JSON + Markdown │                       │
│              │ 输出            │                       │
│              └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────┐
│ 研究工具              │
│ • 批量处理            │
│ • CSV/SPSS 导出       │
│ • 纵向追踪            │
└───────────────────────┘
```

---

## 安装

### 前置要求

- Python 3.9+
- FFmpeg（`brew install ffmpeg`）
- 推荐使用 Apple Silicon Mac（以获得 MLX 加速）

### 安装步骤

```bash
git clone https://github.com/czhou732/ClinicalWhisper.git
cd ClinicalWhisper
python3 -m venv venv
source venv/bin/activate

# 使用 pyproject.toml 安装（可编辑模式）
pip install -e .

# Apple Silicon 用户（推荐 — 快 4–10 倍）：
pip install mlx-whisper

# 下载 NLTK 数据（仅需一次）
python -c "import nltk; nltk.download('punkt_tab')"
```

### 可选依赖

```bash
# 说话人分离（需要 HuggingFace token）
pip install pyannote.audio
export HF_TOKEN="your_token_here"

# 声学特征
pip install opensmile soundfile

# 会议智能
# 从 https://ollama.ai 安装 Ollama，然后：
ollama pull gpt-oss:20b
```

---

## 快速开始

### 单文件处理（监听模式）

```bash
source venv/bin/activate
python main.py
```

将音频文件（`.m4a`、`.mp3`、`.wav`、`.mp4`）拖入 `Input/` 文件夹，包含完整分析的转录文件将出现在配置的输出目录中。

### 异步 API 模式

```bash
# 启动 API 服务器
python api.py

# 提交任务
curl -X POST http://127.0.0.1:8000/jobs \
  -F "file=@recording.m4a"
```

### 作为守护进程运行（macOS）

```bash
bash scripts/start_pipeline.sh        # 启动流水线 + Plaud 连接器
bash scripts/start_pipeline.sh --stop  # 停止所有服务
```

---

## 批量处理

一次性处理整个目录的音频文件：

```bash
python batch_processor.py --input ./audio_dir --output results.csv
```

使用自定义配置：

```bash
python batch_processor.py \
  --input /path/to/recordings \
  --output /path/to/results.csv \
  --config config.yaml
```

输出 CSV 列说明：

| 列名 | 说明 |
|---|---|
| `filename` | 源音频文件名 |
| `word_count` | 转录文本总词数 |
| `duration_minutes` | 音频时长 |
| `sentiment_score` | 整体情感评分（1–10） |
| `sentiment_label` | Positive / Neutral / Negative |
| `vta` | Zhou Index（语音快感缺乏指标） |
| `pitch_mean_st` | 平均基频（半音） |
| `pitch_cv` | 音高变异系数 |
| `loudness_mean_db` | 平均响度（dB） |
| `loudness_cv` | 响度变异系数 |
| `jitter` | 音高微扰 |
| `shimmer` | 振幅微扰 |

---

## 研究工具

### CSV 导出（兼容 R/SPSS）

将分析 JSON 展平为单个 CSV 文件，用于统计分析：

```bash
python export_features.py --input ./Output --output features.csv
```

输出包含 21 列，涵盖人口统计、情感、声学及元数据 — 可直接导入 R、SPSS 或 pandas。

### 纵向追踪

追踪患者在多个会话中的指标变化：

```bash
python longitudinal.py --patient P001 --input ./Output
```

通过线性回归检测 VTA、音高变异性、响度变异性、情感评分和词数的趋势（改善 / 稳定 / 下降）。

---

## 配置

所有设置位于 `config.yaml`：

```yaml
model: "small.en"                        # Whisper 模型大小
mlx_model: "mlx-community/whisper-small.en-mlx"

sentiment:
  enabled: true
  sentences_per_segment: 5

diarization:
  enabled: true                          # 需要 pyannote.audio + HF token

acoustic_features:
  enabled: true                          # 需要 opensmile

audio_extensions: [".m4a", ".mp3", ".wav", ".mp4"]
```

完整配置选项（包括 Plaud 集成、会议智能和流水线设置）请参阅 [config.yaml](config.yaml)。

---

## 测试

```bash
# 运行完整测试套件
pytest

# 详细输出
pytest -v

# 运行特定测试模块
pytest tests/test_sentiment.py -v
```

测试套件覆盖：
- 情感分析（Transformer 模型加载、评分、边界情况）
- 声学特征提取与 VTA 计算
- 批量处理与 CSV 导出
- 纵向趋势检测
- 配置加载

---

## 输出格式

每个处理完成的文件会生成：

1. **Markdown 笔记** — 兼容 Obsidian，包含 YAML 前置元数据、完整转录、说话人标签、情感分析和声学指标
2. **JSON 分析文件**（`*_analysis.json`）— 机器可读的输出，包含所有提取的特征

JSON 结构：

```json
{
  "job_id": "batch_a1b2c3d4e5f6",
  "model": "small.en",
  "statistics": {
    "word_count": 1247,
    "duration_seconds": 423.5
  },
  "overall_sentiment": {
    "label": "Neutral",
    "score": 6,
    "polarity": 0.112
  },
  "overall_acoustics": {
    "pitch_mean_st": 42.3,
    "pitch_cv": 0.187,
    "loudness_mean_db": 65.2,
    "vta": 3.41
  },
  "segments": [...],
  "speaker_acoustics": {...}
}
```

---

## Zhou Index

ClinicalWhisper 计算自定义的**语音快感缺乏指标（VTA）** — Zhou Index：

```
V_anh = −log(CV_F0 × CV_Energy)
```

其中：
- **CV_F0** = 基频变异系数（音高变异性）
- **CV_Energy** = 响度/能量变异系数

更高的 VTA 值表示更平坦、更单调的语音 — 这是快感缺乏和抑郁状态的潜在生物标志物。该指标基于 OpenSMILE 的 eGeMAPSv02 特征集，其中归一化标准差（stddevNorm）即为变异系数。

---

## 作者

**周成栋（Chengdong Zhou）** — 南加州大学本科研究员
*研究方向：计算神经科学与临床 AI*

- GitHub: [@czhou732](https://github.com/czhou732)

---

## 引用

如果您在研究中使用了 ClinicalWhisper，请引用：

```bibtex
@software{zhou2025clinicalwhisper,
  author    = {Zhou, Chengdong},
  title     = {ClinicalWhisper: Privacy-First Clinical Speech Analysis Pipeline},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/czhou732/ClinicalWhisper}
}
```

---

## 许可证

[MIT](LICENSE)
