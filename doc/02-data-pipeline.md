# 02 · 数据流水线

数据流水线把"一段原始音视频"加工成 Bert-VITS2 训练所需的
`(音频, 说话人, 语言, 文本, 音素, 声调, word2ph, BERT特征)` 样本。各步骤彼此
独立、由输入文件类型解耦，可随时断点续跑。

## 1. slice — 按静音切分音频

`BookerBV2Tool slice <音频> [-o 目录] [-dt dB] [-ml 毫秒] [-mi 毫秒] [-hs 毫秒] [-mk 毫秒]`

实现位于 `slice.py` 的 `Slicer` 类。

**算法**（沿用 OpenVPI 的 Slicer，MIT License）：

1. 多声道先取均值；总帧数过短（`samples/hop <= min_length`）则整段保留不切。
2. 计算 RMS 能量帧序列，帧长 `win_size = min(min_interval, 4*hop_size)` 采样点、
   帧移 `hop_size` 为将毫秒换算成采样点后的值。
3. 阈值换算：`threshold = 10^(db_thresh/20)`；逐帧判断静音区间 `silence_start..i`。
4. 触发切分条件（`slice.py:119`）：
   - 片头静音过长（`is_leading_silence`），或
   - 静音区间 >= `min_interval` 且已累积片段 >= `min_length`（`need_slice_middle`）。
5. 按 `max_sil_kept` 决定保留静音的边界（最多分三档：保留 <= max_sil_kept、
   在两倍内取静音最深处折中、或各自取边界的静音最低点），从而在语音边界处
   优先选择噪声最小的切割点。开头的静音统一切到 0。
6. 尾部残留静音若 >= `min_interval`，同样切除。
7. 依据 `sil_tags` 生成切片，写入 `<原名>_<i>.wav`。

**不变量**（`slice.py:77-80`）：`min_length >= min_interval >= hop_size` 且
`max_sil_kept >= hop_size`，否则直接报错。

## 2. resample — 统一采样率

`resample <音频或目录> [--sr 44100] [-t 线程] [-o 目录]`

- 只处理 `.wav` 文件（`resample.py:16`）。
- 单文件：`librosa.load(path, sr=args.sr)` 重采样后 `soundfile.write` 回写。
- 目录：`ThreadPoolExecutor(args.threads)` 并行；每个文件独立 `copy.deepcopy(args)`
  避免共享可变对象。

## 3. mark — SenseVoice 语音转写

`mark <音频或目录>`（需 `-sv` 指定 SenseVoice 模型目录，或环境变量
`SENCEVOICE_MODEL_PATH`）。

实现：`mark.py` + `sencevoice.py`（funasr AutoModel）：

1. 非 `.mp3` 输入先用 `ffmpeg` 转成临时 mp3（`sencevoice.py:24-30`）。
2. 用 `fsmn-vad` 子目录的 VAD 模型切出语音段的起止时间（`max_single_segment_time`
   上限 30 s）。
3. 逐段裁剪为临时 mp3，交给 SenseVoice ASR（`language="auto"` 自动语言、
   `use_itn=True` 数字/标点反向文本归一化、`merge_vad=True` 断句）。
4. `rich_transcription_postprocess` 得到规整文本。
5. 每段记为 `{"start","end","time","text"}`；音频级结果按出现顺序用空格连接写入
   `audio.xxx.txt`（`mark.py:14-17`）。
6. 目录输入则对每个文件依次处理，单文件失败不影响其它（`mark_file_safe`）。

## 4. mklist — 生成标注列表

`mklist 目录 [-r 角色] [-l 语言]`

扫描目录中所有 `.wav`；对每个 WAV 找同名 `.txt` 转写（`wav.txt`），旁注不存在或
为空的跳过；产出 `speaker_list.json`：

```json
[{"file": "a.wav", "role": "wizard", "lang": "ZH", "sub": "你好"}]
```

## 5. preproc — 文本清洗与训练/验证集划分

`preproc speaker_list.json [-tp train] [-vp val] [-cp config] [-vl 每语言验证数] [-mv 验证总数上限]`

**Step 1 文本清洗**：每条 `sub` 经 `clean_text(model_by_lang, sub, lang)`（见
`03-text-processing.md`）得到 `norm_sub / phones / tones / word2ph`，写入
`<name>_cleaned.json` 缓存。缓存存在则直接复用（可断点续跑）。

**Step 2 过滤数据错误**（`preproc.py:59-75`）：
- 同一音频匹配多条文本 → 丢弃（数据错误）；
- 音频文件不存在 → 丢弃；
- 按 `role` 聚合，并给角色分配连续 `sid`（`spk2id`）。

**Step 3 划分**（`preproc.py:80-88`）：
- 每个角色单独 `shuffle`，前 `val_per_lang`（默认 4）条入验证集，其余入训练集；
- 验证集总数若超过 `max_val_total`（默认 12），超出部分并入训练集；
- 最终再次 `shuffle` 验证集。

**Step 4 写配置**：读取包内 `config.json` 模板，写入 `version="2.3"`、
`data.spk2id`、`data.n_speakers`、`data.training_files`、`data.validation_files`，
输出到 `config.json`。`n_speakers=0` 时走单说话人/零样本路径（训练侧见 05 章）。

## 6. bert-gen — 生成 BERT 音素向量

`bert-gen [-c config.json] [--num_processes 8]`

- 合并 `data.training_files` + `data.validation_files` 的所有条目。
- 对每条：由 `cleaned_text_to_sequence` 还原成整数序列（并在 `add_blank=true`
  时插入空白音素 0、word2ph 翻倍、首字 +1），再用对应语言 BERT 计算 phone-level
  特征（见 03 章），保存为 `<wav>.lower()_bert.pt`（`bert_gen.py:33`）。
- 已存在 `_bert.pt` 的条目跳过，支持断点续跑。
- `ProcessPoolExecutor` 并行，每进程独立加载 BERT 模型（设备 GPU/CPU 自动）。

**命名约定**：`data_utils.py` / `infer.py` 均按 `wav 路径小写 + 去掉 .wav 后加
_bert.pt` 查找，必须与 `bert-gen` 的产出命名一致（`data_utils.py:150`）。

## 7. 数据缺失时模型的容错

模型训练侧（`data_utils.py`）在 `_bert.pt` 缺失/损坏时**不报错**，而是用该语言路
径的随机向量 `torch.randn(1024, len(phone))`（`data_utils.py:159-177`）替代，保证
数据不齐也能把循环跑起来。这会在文本语义建模上引入噪声，正式训练前应保证
`_bert.pt` 齐备。