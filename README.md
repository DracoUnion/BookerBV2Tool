# BookerBV2Tool

iBooker/DracoUnion BV2 声音克隆辅助工具。把一段或多段视频/音频切成语音片段、做语音转写、整理标注数据，最终用于训练 Bert-VITS2（BV2）声音克隆模型。

```bash
pip install -r requirements.txt
```

命令行入口有两种，等价：

```bash
BookerBV2Tool <子命令> [参数...]
bv2-tool   <子命令> [参数...]
```

## 工作流程总览

典型的“从视频到训练数据”流水线：

```
视频/音频
  │  (1) slice    按静音切分音频
  ▼
语音片段(WAV)
  │  (2) resample 统一采样率
  ▼
WAV(目标采样率)
  │  (3) mark     用 SenseVoice 逐段语音转写 → 每段生成 .wav.txt
  ▼
WAV + 转写(.txt)
  │  (4) mklist   汇总为 speaker_list.json
  ▼
speaker_list.json
  │  (5) preproc  划分训练/验证集、清洗文本 → train/val_list.json + config.json
  ▼
训练列表
  │  (6) bert-gen 为每段生成 BERT 音素向量 *_bert.pt
  ▼
带 BERT 特征的数据
  │  (7) train    训练模型
  ▼
模型权重
```

---

## 全局参数

这些参数放在子命令之前，对所有子命令生效（仅 mark / preproc / bert-gen 需要用到）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-v, --version` | — | 打印版本号 |
| `-sv, --sencevoice <路径>` | 环境变量 `SENCEVOICE_MODEL_PATH` 或空 | SenseVoice 模型目录（mark 用） |
| `-eb, --enlish-bert <模型>` | `microsoft/deberta-v3-large` | 英文 BERT 模型（注意拼写是 `enlish`） |
| `-cb, --chinese-bert <模型>` | `hfl/chinese-roberta-wwm-ext-large` | 中文 BERT 模型 |
| `-jb, --japanese-bert <模型>` | `ku-nlp/deberta-v2-large-japanese-char-wwm` | 日文 BERT 模型 |

```bash
# 例子：指定中文 BERT 模型跑 bert-gen
BookerBV2Tool -cb hfl/chinese-roberta-wwm-ext-large bert-gen -c config.json
```

---

## 1. slice — 按静音切分音频

把一条较长的音频按静音切成多个语音片段。

```bash
BookerBV2Tool slice 音频文件 [-o 输出目录] [静音参数...]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `audio` | 必填 | 输入音频路径 |
| `-o, --out` | 输入文件所在目录 | 输出目录 |
| `-dt, --db_thresh` | `-40` | 判定为静音的 dB 阈值 |
| `-ml, --min_length` | `5000` | 每个切片的**最小**长度（毫秒） |
| `-mi, --min_interval` | `300` | 触发切分的静音**最小**时长（毫秒） |
| `-hs, --hop_size` | `10` | 帧长（毫秒） |
| `-mk, --max_sil_kept` | `500` | 切片两端保留的静音**最大**时长（毫秒） |

输出为 `原文件名_0.wav`、`原文件名_1.wav` … 的片段，放在 `-o` 目录。

```bash
BookerBV2Tool slice lecture.mp4 -o clips
```

> 约束：`min_length >= min_interval >= hop_size` 且 `max_sil_kept >= hop_size`，否则报错。

---

## 2. resample — 统一采样率

把 WAV 重采样到目标采样率（训练通常需要统一采样率，如 44100）。

```bash
BookerBV2Tool resample 音频或目录 [-o 输出] [--sr 采样率] [-t 线程数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `audio` | 必填 | 单个 `.wav` 文件，或一个目录（目录下所有 wav 都会处理） |
| `--sr` | `44100` | 目标采样率 |
| `-t, --threads` | `8` | 并行线程数 |
| `-o, --out` | `.` | 输出目录 |

只处理 `.wav` 文件，非 WAV 会被忽略并提示。

```bash
BookerBV2Tool resample clips -o resampled --sr 44100
```

---

## 3. mark — 语音转写（生成旁注 .txt）

对每个音频片段做语音转写（SenseVoice），为每个音频生成同名 `.txt` 旁注文件，
内容为转写文本（片段按出现顺序以空格连接）。

```bash
BookerBV2Tool -sv <SenseVoice模型路径> mark 音频或目录
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `audio` | 必填 | 单个音频，或一个目录（目录下所有文件逐个处理） |

**依赖**：
- `ffmpeg` 在 PATH 中（非 `.mp3` 输入会先用 ffmpeg 转成临时 mp3）。
- SenseVoice 模型目录，通过 `-sv` 或环境变量 `SENCEVOICE_MODEL_PATH` 指定；
  该目录下需含 `fsmn-vad` 子目录作为 VAD 模型。

```bash
BookerBV2Tool -sv D:/models/SenseVoiceSmall mark resampled
# 生成 resampled/xxx.wav.txt
```

---

## 4. mklist — 生成标注列表

扫描目录，把每个“WAV + 同名 .txt 旁注”汇总成 `speaker_list.json`。

```bash
BookerBV2Tool mklist 目录 [-r 角色] [-l 语言]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dir` | 必填 | 存有 WAV 文件的目录 |
| `-r, --role` | `wizard` | 说话人角色名 |
| `-l, --lang` | `ZH` | 语言，`EN` / `ZH` / `JP` 三选一 |

输出 `speaker_list.json`（写在该目录下），形如：

```json
[
  {"file": "a.wav", "role": "wizard", "lang": "ZH", "sub": "你好"},
  {"file": "b.wav", "role": "wizard", "lang": "ZH", "sub": "世界"}
]
```

规则：`.txt` 旁注不存在或内容为空的 WAV 会被跳过；目录里没有 WAV 或没有有效
标注时会提示并提前结束。

```bash
BookerBV2Tool mklist resampled -r my_role -l ZH
```

---

## 5. preproc — 预处理，划分训练/验证集

读入标注列表，清洗文本（中文转拼音音素、数字转汉字、标点归一化等），按角色划分
训练/验证集，并更新 `config.json`（写入 `spk2id`、`n_speakers`、训练/验证文件路径）。

```bash
BookerBV2Tool preproc speaker_list.json [-tp 训练列表] [-vp 验证列表] [-cp 配置] [-vl 每语言验证数] [-mv 验证上限]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `transcription_path` | 必填 | 标注列表（如 `speaker_list.json`） |
| `-tp, --train-path` | `train_list.json` | 训练列表输出路径 |
| `-vp, --val-path` | `val_list.json` | 验证列表输出路径 |
| `-cp, --config-path` | `config.json` | 输出的 BV2 配置文件路径 |
| `-vl, --val-per-lang` | `4` | 每种语言每角色抽多少条做验证集 |
| `-mv, --max-val-total` | `12` | 验证集总数上限（超出部分并入训练集） |

**行为细节**：
- 首次运行会生成 `speaker_list_cleaned.json`（清洗后的标注缓存），后续运行直接复用。
- 标注里 `file` 对应的音频文件必须存在，否则该条被丢弃并提示；
  同一音频匹配多条文本会被视为数据错误并丢弃。
- 生成的 `config.json` 会写入 `version: "2.3"`，并填入 `spk2id` / `n_speakers` /
  `training_files` / `validation_files`。

```bash
BookerBV2Tool preproc speaker_list.json -tp train_list.json -vp val_list.json -cp config.json
```

---

## 6. bert-gen — 生成 BERT 音素向量

为训练/验证列表里的每条音频生成 BERT 音素特征，保存为 `<音频名>_bert.pt`（与音频同目录）。

```bash
BookerBV2Tool -cb <中文模型> -eb <英文模型> -jb <日文模型> bert-gen [-c 配置] [--num_processes 进程数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-c, --config` | `config.json` | preproc 生成或手写的配置 |
| `--num_processes` | `8` | 并行进程数 |

- 从配置里的 `data.training_files` 和 `data.validation_files` 读取数据。
- 配置里 `data.add_blank` 决定是否插入空白音素。
- 已存在 `_bert.pt` 的条目会跳过（可断点续跑）。

```bash
BookerBV2Tool bert-gen -c config.json --num_processes 4
```

---

## 7. train — 训练模型

用前面生成的数据训练 BV2 模型。训练参数写在 `config.json`（`data.*`、`train.*`、
`model.*` 等字段）中，命令行只指定配置路径。

```bash
BookerBV2Tool train [-c 配置]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-c, --config` | `config.json` | 配置文件路径 |

```bash
BookerBV2Tool train -c config.json
```

---

## 完整示例

```bash
# 1) 切成语音片段
BookerBV2Tool slice lecture.mp4 -o clips

# 2) 统一采样率
BookerBV2Tool resample clips -o resampled --sr 44100

# 3) 语音转写（需要 SenseVoice 模型 + ffmpeg）
BookerBV2Tool -sv D:/models/SenseVoiceSmall mark resampled

# 4) 生成标注列表
BookerBV2Tool mklist resampled -r wizard -l ZH

# 5) 划分训练/验证集并更新配置
BookerBV2Tool preproc resampled/speaker_list.json -tp train_list.json -vp val_list.json -cp config.json

# 6) 生成 BERT 特征
BookerBV2Tool bert-gen -c config.json --num_processes 4

# 7) 训练
BookerBV2Tool train -c config.json
```

## 测试

```bash
python -m pytest tests/ -q
```

详见 [tests/README.md](tests/README.md)。
