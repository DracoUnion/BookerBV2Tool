# 01 · 总体架构

## 1. 项目定位

`BookerBV2Tool` 的目标是把"一段视频/音频"加工成可训练的 Bert-VITS2 声音克隆
数据集，并完成训练与合成。它不是通用推理框架，而是**单机可跑的完整工具链**：

```
视频/音频 ──► 切片 ─► 重采样 ─► 转写 ─► 标注列表 ─► 预处理 ─► BERT特征 ─► 训练 ─► 合成
            slice    resample   mark     mklist      preproc    bert-gen   train   infer
```

模型侧直接复用上游 Bert-VITS2 的 **v2.3** 模型库（`modules.py` / `attentions.py` /
`transforms.py` / `monotonic_align/` / `mel_processing.py`），本仓库将其打包为包内
模块并改为相对导入（见 commit `39f5ce8`）。

## 2. 设计目标与约束

| 目标/约束 | 实现方式 |
| --- | --- |
| 单机训练，无分布式 | `train.py` 移除 DDP，rank 恒为 0，设备自动选择（有 CUDA 用 GPU，否则 CPU） |
| CPU 可训练 | bf16 混合精度仅在 GPU 启用；WavLM/SLM 损失缺失时自动退化（`_NoWavLM`） |
| 面向中文数据集为主，兼顾日/英 | 三语文本规范化 + G2P + 三路 BERT 特征 |
| 数据质量门槛低 | SenseVoice 自动转写 + 自动划分训练/验证集 |
| 断点续跑友好 | `bert-gen` 跳过已存在 `_bert.pt`；训练自动加载最新检查点；检查点自动清理 |
| 依赖最少化 | 数据侧脚本只依赖 `librosa/soundfile/funasr/ffmpeg`；模型侧依赖 `torch/transformers` |

## 3. 总体分层

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI 层   __main__.py                                               │
│  全局参数(-sv/-cb/-eb/-jb) + 子命令(slice/resample/mark/mklist/      │
│  preproc/bert-gen/train/infer)                                       │
└─────────────────────────────────────────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────────────────────────────┐
   │  数据流水线层                                                   │
   │  slice.py  resample.py  mark.py+sencevoice.py  mklist.py        │
   │  preproc.py  bert_gen.py                                       │
   └────────────────────────────────────────────────────────────────┘
        │  产出 train_list.json / val_list.json / config.json / *_bert.pt
   ┌────┴──────────────────────────────────────────────────────────┐
   │  文本处理层  text/                                            │
   │  symbols.py(词表) cleaner.py(入口) chinese/english/japanese.py  │
   │  tone_sandhi.py(变调) bert_utils.py(模型下载)                   │
   └────────────────────────────────────────────────────────────────┘
        │  音素序列 + 声调 + 语言 id + BERT 特征
   ┌────┴──────────────────────────────────────────────────────────┐
   │  模型层                                                        │
   │  models.py SynthesizerTrn(+判别器)                             │
   │  modules.py(耦合流/卷积/ResBlock) attentions.py(Transformer)    │
   │  transforms.py(有理二次样条) monotonic_align(MAS)              │
   │  mel_processing.py(频谱) losses.py(损失)                       │
   └────────────────────────────────────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────────────────────────────┐
   │  驱动层                                                        │
   │  train.py(训练循环)  infer.py(推理)  data_utils.py(数据集)      │
   │  utils.py(检查点/日志/绘图)  config.py(spec_cache 开关)          │
   └────────────────────────────────────────────────────────────────┘
```

## 4. 模块依赖图（主要 import 关系）

```
__main__.py ──► slice/resample/mark/mklist/preproc/bert_gen（数据侧）
            ──► train（惰性导入）──► data_utils ──► mel_processing / utils / text.cleaner
            ──► infer（惰性导入）──► models / utils / text.cleaner

models.py ──► modules.py ──► attentions.py / commons.py / transforms.py
          ──► attentions.py ──► commons.py
          ──► monotonic_align ──► numba.core
          ──► text.symbols（词表/声调/语言表）
          ──► commons.py
```

设计要点：`train` / `infer` 通过 `__main__.py` 中的 `_train_handle` / `_infer_handle`
**惰性导入**，避免启动其它子命令时被 tensorboard、模型库等重依赖拖垮。

## 5. CLI 命令汇总

两个等价入口：`BookerBV2Tool <子命令>` 与 `bv2-tool <子命令>`（见
`pyproject.toml` 的 `[project.scripts]`）。

| 子命令 | 作用 | 关键输出 |
| --- | --- | --- |
| `slice` | 按静音切分音频 | `{名字}_0.wav …` |
| `resample` | 统一采样率到目标 SR（默认 44100） | 重采样 WAV |
| `mark` | SenseVoice 语音转写 | 每段音频同名 `.txt` 旁注 |
| `mklist` | 汇总 WAV+转写为标注列表 | `speaker_list.json` |
| `preproc` | 文本清洗、划分训练/验证集、写配置 | `train_list.json` `val_list.json` `config.json` |
| `bert-gen` | 生成每段音频的 BERT 音素向量 | `<音频>_bert.pt` |
| `train` | 训练模型 | `models/G_*.pth` `D_*.pth` `WD_*.pth` `DUR_*.pth` |
| `infer` | 文本转语音 | `out.wav` |

全局参数（放在子命令之前）：`-v/--version`、`-sv/--sencevoice`、`-eb/--enlish-bert`、
`-cb/--chinese-bert`、`-jb/--japanese-bert`。其中 BERT 三参数被 `preproc` /
`bert-gen` / `infer` 共同使用。

## 6. 端到端数据流（含中间文件）

| 阶段 | 输入 | 处理 | 输出 |
| --- | --- | --- | --- |
| 1 slice | 长音频/视频 | RMS 静音检测切分 | 语音片段 WAV |
| 2 resample | 片段 WAV | librosa 重采样 | 目标采样率 WAV |
| 3 mark | WAV | ffmpeg→SenseVoice(VAD+ASR) | `*.wav.txt` |
| 4 mklist | 目录 | 扫描 WAV+转写 | `speaker_list.json` |
| 5 preproc | 标注列表 | 文本清洗、过滤、划分 | `train/val_list.json` + `config.json` |
| 6 bert-gen | 训练/验证列表 | 并行 BERT 特征 | `*_bert.pt` |
| 7 train | 列表+特征+音频 | 对抗+KL+时长+Mel+SLM 训练 | 检查点 `G/D/WD/DUR_*.pth` |
| 8 infer | 文本+检查点 | 音素化→BERT→合成 | `out.wav` |

其中 preproc 会把包内 `config.json` 模板复制出去并写入 `spk2id`、`n_speakers`、
`version=2.3`、`training_files`、`validation_files`，作为后续 `bert-gen`/`train`/
`infer` 的配置输入。

## 7. 版本与打包

- 版本号：`2026.3.25`（`__init__.py`），模型架构版本 `2.3`。
- License：SATA（见仓库根 `LICENSE`）。
- 安装：`pip install -r requirements.txt` 或 `pip install .`。
- 运行时依赖：`torch`、`librosa`、`soundfile`、`scipy`、`transformers`、
  `tensorboard`、`funasr`（mark 用）、`pypinyin/jieba/cn2an`（中文 G2P）、
  `g2p_en`（英文）、`pyopenjtalk/jaconv/num2words`（日文）。
