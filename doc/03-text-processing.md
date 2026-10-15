# 03 · 文本处理

文本处理把"自然语言字符串"变成模型输入所需的**音素序列 + 声调序列 + 语言 id
序列 + BERT 音素特征**。入口统一在 `text/cleaner.py`，按语言分发到
`chinese.py` / `english.py` / `japanese.py`。

## 1. 词表（`text/symbols.py`）

音素表是**三语联合**的：对中文（opencpop 拼音方案）、日文（Julius 兼容）、英文
（CMU ARPA 兼容）音素取并集后排序，再补上标点与特殊符号。

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 中文音素 `zh_symbols` | 65 | opencpop-strict 拼音到音素的映射 |
| 日文音素 `ja_symbols` | 42 | 平/片假名→罗马音素（含长音 `a:` 等） |
| 英文音素 `en_symbols` | 39 | ARPA 音素（`aa, ae, ah, …`）+ 特殊 `V` |
| 并集 `normal_symbols` | 102 | 去重排序后 |
| 标点 `punctuation` | 7 | `! ? … , . ' -` |
| 其它 `pu_symbols` | 9 | 标点 + `SP` + `UNK` |
| 词表 `symbols` | **112** | `pad + normal_symbols + pu_symbols` |

关键约定：

- `symbols[0] == "_"` 为 **pad/blank** 音素，`add_blank` 时插入该 id（0）。
- `sil_phonemes_ids = symbols.index(pu_symbols)` 为静音音素 id（标点、SP、UNK）。
- `num_tones = 12`：中文 6 声 + 日文 2 声 + 英文 4 声；各语言声调 id 需加语言偏移
  （`language_tone_start_map`：ZH=0、JP=6、EN=12），见 `cleaned_text_to_sequence`。
- `language_id_map = {"ZH":0, "JP":1, "EN":2}`，`num_languages = 3`。

## 2. 统一入口 `text/cleaner.py`

| 函数 | 作用 |
| --- | --- |
| `text_to_sequence(model_name, text, lang)` | 全流程：clean_text + 转 id 序列 |
| `clean_text(model_name, text, lang)` | 返回 `(norm_text, phones, tones, word2ph)` |
| `cleaned_text_to_sequence(phones, tones, lang)` | 音素→id；声调加偏移；语言→lang_id 序列 |
| `get_bert_feature(model_name, norm_text, word2ph, device, style_text, style_weight)` | 计算 phone-level BERT 特征 |
| `get_model_name_by_lang(lang, args)` | 按语言返回对应 BERT 模型名 |

`cleaned_text_to_sequence` 具体做三件事（`cleaner.py:76-88`）：
1. 每个音素查 `_symbol_to_id` 得 id；
2. 声调序列整体加该语言偏移 `language_tone_start_map[lang]`（如中文 0..5 →
   0..5，日文 0/1 → 6/7，英文 0..3 → 12..15）；
3. 每个音素位置都填该语言的 `language_id_map[lang]`。

## 3. 中文 `text/chinese.py`

**文本规范化 `text_normalize`**：
1. `cn2an.transform(x, "an2cn")`：阿拉伯数字→汉字；
2. `replace_punctuation`：全角/中文标点→英文等价（`，→,`、`。→.`、`…` 保留、
   `嗯→恩`、`呣→母` 等）；再过滤非汉字/标点字符。

**G2P `g2p(model_name, text)`**：
1. 按标点切句，逐句 `_g2p`；
2. 用 `jieba.posseg` 分词，`pypinyin` 取声母/韵母（`Style.INITIALS`、
   `Style.FINALS_TONE3`）；
3. `ToneSandhi`（`tone_sandhi.py`，PaddleSpeech 实现）做轻声/变调修正；
4. 韵母缩写映射（`uei→ui`、`iou→iu`、`uen→un`）、零声母音节改写（`i→yi` 等）；
5. 查 `opencpop-strict.txt` 的 `pinyin→音素串` 映射，一个音节可展开为多个音素
   （`word2ph.append(len(phone))`）；
6. 句首尾补空白音素 `_`（`phones = ["_"] + phones + ["_"]`），word2ph 相应 +1；
7. 返回 `phones / tones / word2ph`，满足 `sum(word2ph) == len(phones)` 且
   `len(word2ph) == len(text)`。

## 4. 英文 `text/english.py`

- 数字/序数展开（`num2words`）、缩写展开（`expend_abbreviation`）、`Mr.` 等缩写
  词典（`dict_abbreviations`）。
- `g2p_en.G2p()` 输出 ARPA 音素（如 `AA0`），映射到 `en_symbols` 子集并转小写。
- 保留辅助数据文件 `cmudict.rep` 与 `cmudict_cache.pickle`（CMU 词典缓存）。
- 英文声调固定 0..3（词重音等级）。

## 5. 日文 `text/japanese.py`

- `pyopenjtalk`（OpenJTalk 的 Python 绑定）做文本→音素；`jaconv` 处理平/片假名
  互转与长音。
- 假名→音素映射表（`hiragana_map` 等），兼容 Julius 音素（`N`、`q`、长音 `a:`）。
- 数字→日文读音（`num2words`），标点统一。
- 日文声调固定 0/1。

## 6. BERT 特征（`text/cleaner.py:get_bert_feature`）

- 默认三语 BERT 模型（`__main__.py`）：
  - 中文 `hfl/chinese-roberta-wwm-ext-large`
  - 英文 `microsoft/deberta-v3-large`
  - 日文 `ku-nlp/deberta-v2-large-japanese-char-wwm`
- 用对应 `AutoTokenizer` + `AutoModelForMaskedLM` 对 `norm_text` 做 masked-LM 前向，
  取倒数第 3 层 `hidden_states`（`res["hidden_states"][-3][0].cpu()`）。
- `word2ph` 把 **word 级**特征重复成 **phone 级**（一个汉字可能对多个音素），
  得到 `[1024, len(phones)]` 张量，`.T` 返回。
- 可选 `style_text`（风格文本）时，用风格文本倒数第 3~4 层的均值按
  `style_weight` 加权混合（推理端默认关闭）。
- 模型加载有全局缓存（`name_model_map` / `name_tok_map`），进程内只下载一次。

## 7. 数据侧的复现路径

`preproc`/`bert-gen` 生成的 `phones/tones/word2ph` 是 `cleaned_text_to_sequence`
的直接输入；`data_utils.py` 与 `infer.py` 都按同样的方式做 `add_blank`
插入（空白音素 0、word2ph 翻倍、首字 +1），保证训练与推理的特征对齐。