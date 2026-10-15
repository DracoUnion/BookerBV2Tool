# 06 · 推理

推理把一段文本合成为音频，模型版本锁定 **2.3**（`infer.py:26`）。入口：

```
BookerBV2Tool [-cb 中文BERT] [-eb 英文BERT] [-jb 日文BERT] \
  infer "<文本>" <G_*.pth> [-c config.json] [-s 说话人] [-l 语言] [-o out.wav] \
  [--sdp_ratio] [--noise_scale] [--noise_scale_w] [--length_scale]
```

## 1. 加载

- `get_net_g`（`infer.py:34`）：按配置重建 `SynthesizerTrn`（`len(symbols)=112`，
  `spec_channels=filter_length//2+1=1025`，`segment_frames=segment_size/hop=32`，
  `n_speakers`、`**hps.model`），`load_checkpoint(skip_optimizer=True)` 只载权重。
- 设备自动：GPU 优先，否则 CPU（`infer.py:29-31`）。
- 说话人未指定时取 `data.spk2id` 第一个（`infer.py:186`）。

## 2. 文本 → 模型输入（`get_text`，`infer.py:51`）

1. 由语言选 BERT 模型名（`get_model_name_by_lang`）。
2. `clean_text`：该语言 `text_normalize` + G2P → `phones/tones/word2ph`。
3. `cleaned_text_to_sequence`：音素→id、声调加语言偏移、语言 id。
4. `add_blank=true` 时 `intersperse(_, 0)` 插空白音素（word2ph 同步翻倍）。
5. `get_bert_feature(model_name, norm_text, word2ph, device)` 对**该文本的语言**
   计算 phone 级 BERT 特征；**其它两路语言传随机噪声** `randn(1024, len(phone))`
   （`infer.py:70-83`），与训练侧缺失 `_bert.pt` 的兜底一致。

## 3. 合成（`SynthesizerTrn.infer`，`models.py:1449`）

流程图与 04 章 §11 相同：

```
enc_p(x, tone, language, bert, ja_bert, en_bert, g)
  → 先验 m_p/logs_p + x_mask

logw = SDP(reverse, noise_scale_w)*sdp_ratio + DP*(1-sdp_ratio)
w    = exp(logw) * x_mask * length_scale
w_ceil = ceil(w)
y_mask 由 w_ceil 确定帧数
attn = generate_path(w_ceil, attn_mask)       # 单调铺开
m_p/logs_p 按 attn 复制到频谱帧

z_p  = m_p + randn*exp(logs_p)*noise_scale
z    = flow(z_p, y_mask, g, reverse=True)     # 逆转流
o    = dec(z * y_mask, g)                       # HiFi-GAN 上采样波形
```

写成公式，生成的波形是：

$$
w = \mathrm{ceil}\!\big(\exp\big(\lambda\, \mathrm{SDP_{rev}}(x;\sigma_w)
+(1-\lambda)\,\mathrm{DP}(x)\big)\cdot M_x\cdot L\big),\qquad
z_p = m_p + \varepsilon\,\mathrm{e}^{\log s_p}
$$

- $\lambda$ = `sdp_ratio`（0.2），$\sigma_w$ = `noise_scale_w`（0.8），
  $L$ = `length_scale`（语速，>1 慢），$\varepsilon\sim N(0,1)$ 乘 `noise_scale`（0.6）。

## 4. 生成参数语义

| 参数 | 默认 | 语义 |
| --- | --- | --- |
| `--sdp_ratio` | 0.2 | SDP(随机时长)与 DP(确定时长)混合比；0=纯 DP，1=纯 SDP |
| `--noise_scale` | 0.6 | 先验采样噪声；越大越随机/不稳 |
| `--noise_scale_w` | 0.8 | 时长预测采样噪声（仅 SDP 路径） |
| `--length_scale` | 1.0 | 时长缩放；>1 语速放慢、<1 变快 |

`infer_handle` 里 `net_g.infer(...)[0][0,0]` 取出首个 batch/声道，`.cpu().float()
.numpy()` 用 `soundfile.write` 以 `data.sampling_rate` 写出。

## 5. 与训练的对应关系

- 训练里时长损失基于 MAS 对齐得出的真实 `w`；推理没有参考音频，只能靠 SDP/DP
  预测时长 → `generate_path` 构造对齐。这正是 MST 端到端训练"对齐在推理时由时长
  预测器隐式恢复"的标准做法。
- 训练/推理必须使用相同的 `add_blank`、同样的三语 BERT 映射，否则 BERT 特征长度
  与音素序列不一致会触发断言（`infer.py:68,85-87`）。
- `skip_start/skip_end` 参数（`infer.py:121-134`）支持裁剪首尾静音音素（CLI 未
  暴露），供上层调用。