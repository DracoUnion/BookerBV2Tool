# BookerBV2Tool 单元测试

本目录包含对 `BookerBV2Tool` 源码的单元测试，使用 [pytest](https://docs.pytest.org/)。

## 运行

```bash
# 在仓库根目录下运行全部测试
python -m pytest tests/ -q

# 只运行某个模块的测试
python -m pytest tests/test_commons.py tests/test_utils.py -q

# 显示详细输出 / 覆盖率
python -m pytest tests/ -q -v
python -m pytest tests/ --cov=BookerBV2Tool
```

测试依赖项目运行时需要的第三方库（`torch`、`librosa`、`scipy`、
`pypinyin`、`jieba`、`cn2an`、`g2p_en`、`pyopenjtalk`、`num2words`、
`transformers`、`funasr` 等），但**不需要网络**：凡是会触发模型下载或
外部服务的路径（如 `transformers` tokenizer 加载、`pyopenjtalk` 词典下载、
`funasr` 推理）在测试里都被 mock 掉了。

## 目录结构

| 文件 | 覆盖模块 |
| --- | --- |
| `conftest.py` | 共享 fixture：把仓库根目录与包目录加入 `sys.path`、生成测试 WAV |
| `_compat.py` | 把 `data_utils.py` 遗留的绝对导入（`config`/`tools`/`commons`/`text`）映射到真实子模块 |
| `test_commons.py` | `commons.py`（张量/注意力/timing signal、KL、mask、grad clip 等） |
| `test_mel_processing.py` | `mel_processing.py`（STFT / Mel 频谱、动态范围压缩） |
| `test_losses.py` | `losses.py`（feature/discriminator/generator/kl 损失、WavLMLoss 骨架） |
| `test_slice.py` | `slice.py`（Slicer 静音切分、get_rms） |
| `test_resample.py` | `resample.py`（重采样单文件/目录） |
| `test_utils.py` | `utils.py`（HParams、checkpoint 读写/清理、模型混合、路径工具） |
| `test_symbols.py` | `text/symbols.py`（音素表、声调偏移、语言 id） |
| `test_chinese.py` | `text/chinese.py`（中文标点/数字归一化、G2P） |
| `test_english.py` | `text/english.py`（英文数字/缩写展开、音素工具） |
| `test_japanese.py` | `text/japanese.py`（平假名/片假名转音素、数字转日语） |
| `test_tone_sandhi.py` | `text/tone_sandhi.py`（变调规则） |
| `test_cleaner.py` | `text/cleaner.py`（音素→id 序列、按语言分发） |
| `test_bert_utils.py` | `text/bert_utils.py`（BERT 权重下载） |
| `test_mklist.py` | `mklist.py`（生成 `speaker_list.json`） |
| `test_mark.py` | `mark.py`（写转写旁注文件） |
| `test_preproc.py` | `preproc.py`（训练/验证集划分、写配置） |
| `test_bert_gen.py` | `bert_gen.py`（生成 BERT 音素向量） |
| `test_sencevoice.py` | `sencevoice.py`（SenseVoice 转写流程，全 mock） |
| `test_data_utils.py` | `data_utils.py`（数据集、collate、分桶采样器） |

## 未覆盖的模块

- **`models.py` / `train.py` / `__main__.py`**：依赖上游 Bert-VITS2 的
  `modules` / `attentions` / `monotonic_align` 源码，这些文件**不在本仓库内**，
  无法离线实例化模型，因此跳过。`train.py` 的许多逻辑实际上已在
  `test_data_utils.py`、`test_utils.py` 中通过其底层组件间接覆盖。
- **`text/*.bert`**：需要下载预训练模型，未测试（`get_bert_feature` 已在
  `test_cleaner.py` / `test_bert_gen.py` 中 mock 覆盖调用流程）。

## 关于“特征化测试”（characterization tests）

源码里存在若干真实缺陷，部分测试用 `# Characterization:` 注释把它们固定下来
（当维护者修复后这些测试会需要更新）：

- `commons.rand_slice_segments` 默认 `x_lengths=None` 时，在 torch≥2.x 下把
  Python `int` 传给 `torch.clamp(..., min=...)` 会抛 `TypeError`。
- `losses.kl_loss` 对相同的 N(0,1) 输入返回 `-0.5`（缺少 `exp(2*logs_p)` 归一化项）。
- `mel_processing` 用**位置参数**调用 `librosa.filters.mel`，而 librosa≥0.10 改为
  关键字参数，会抛 `TypeError`（测试中用 stub 替换了该调用）。
- `utils.clean_checkpoints` 的 `name_key` 正则 `._(\d+)\.pth` 无法匹配 `WD_*.pth`，
  遇到 WD 文件会抛 `AttributeError`。
- `sencevoice.py` 转换失败时引用未定义的 `fname`，实际抛 `NameError` 而非
  `FileNotFoundError`。
- `chinese.g2p` 对含未剥离的拉丁字符/标点的原始文本会因 `len(word2ph) == len(text)`
  断言失败（调用方应先 `text_normalize`）。
