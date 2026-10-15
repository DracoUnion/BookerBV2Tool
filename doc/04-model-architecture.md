# 04 · 模型架构

本文档详解本项目（Bert-VITS2 v2.3）的**完整模型架构**：端到端拓扑、每个子模块
的前向/训练/推理细节、判别器集合、与 `config.json` 的对应关系，以及实测参数量。

> 全部实现位于：`models.py`（顶层模型与判别器）、`modules.py`（流式单元与卷积）、
> `attentions.py`（Transformer 编码器）、`transforms.py`（有理二次样条）、
> `monotonic_align/core.py`（MAS 对齐，numba）。

## 1. 端到端拓扑总览

`SynthesizerTrn`（`models.py:1171`）是一个 **VITS 式的条件 VAE + 声码器**模型，由
五个子网络组成：

```
                        ┌──────────────────── 训练期参考路径 ────────────────────┐
   音素序列(+声调+语言)   │                                                       │
      + 三语 BERT       │                                                       │
        │               │   ┌────────────────────────┐                          │
        ▼               │   │ PosteriorEncoder enc_q  │  线性谱 spec [B,1025,T'] │
 ┌─────────────────┐    │   │ (WaveNet 后验，重参数化) │ ◄───────────────────────┘
 │  TextEncoder    │    │   └───────────┬────────────┘
 │  enc_p          │    │               │ z [B,192,T']
 │  Transformer    │    │               │ (样本)
 │  + 三语嵌入      │    │               ▼
 └───────┬─────────┘    │   ┌────────────────────────┐
         │              │   │ Flow z_p = flow(z)      │  可逆耦合流(4层 Transformer 耦合
 x,m_p,logs_p,x_mask    │   │ (先验分布: N(m_p,logs_p))│  + Flip；训练带 logdet)
 [B,192,T]              │   └───────────┬────────────┘
         │              │               │ z_p
         │              │               ▼
         ├──────────────┼────────►  MAS 单调对齐搜索 (num.nolib)
         │              │               │ 输出 attn 对齐矩阵 + 时长 w
         │  说话人嵌入 g │               ▼
         ▼   (emb_g)    │   ┌────────────────────────┐
 ┌──────────────────┐   │   │ 根据对齐扩展 m_p/logs_p  │  生成器 (HiFi-GAN)
 │ 时长预测器        │   │   └───────────┬────────────┘
 │  SDP(随机,流)     │   │               │ 随机切片 segment_size
 │  DP(确定性,卷积)  │   │               ▼
 └────────┬─────────┘   │        ┌───────────────────────┐
       w/ logw          │        │  Generator dec (HiFi- │  ► y_hat 波形 [B,1,segment]
   (训练 MAS 派生)       │        │  GAN 反卷积 + ResBlock)│
                         └────────┴───────────────────────┘
```

**与 VITS1 相比 VITS2 的差异**（都体现在二值配置开关里，见 §10）：

1. **说话人条件编码器**（`use_spk_conditioned_encoder`）：说话人嵌入 `g` 从仅注入
   解码器/后验，改为在 TextEncoder 第 3 个块（`cond_layer_idx=2`）额外相加注入
   （`attentions.py:107-111`）。
2. **噪声缩放 MAS**（`use_noise_scaled_mas`）：训练中对 MAS 的负对数似然打分加
   高斯噪声 `ε = std(neg_cent) * N(0,1) * noise_scale`，噪声随步数线性衰减
   （`models.py:1392-1398`），做"随机对齐"以提升鲁棒性。
3. **时长判别器**（`use_duration_discriminator`）：用 LSTM 判别真实/预测时长，
   加对抗损失（§8.2）。
4. **Transformer 耦合流**（`use_transformer_flow`，默认 True）替代 VITS1 的
   WaveNet 残差耦合流。
5. **添加空白音素**（`add_blank`）：所有序列插入 `_`（id 0），下游语义特征对齐。

## 2. 输入规格

模型输入张量由 `data_utils.py` 的 `TextAudioSpeakerCollate` 打包（按 spec 长度
降序 + 右侧 padding），batch 每个样本含：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `x` (phones) | `[B, T]` LongTensor | 音素 id，`symbols` 词表（112），pad=0 |
| `x_lengths` | `[B]` | 文本有效长度 |
| `spec` | `[B, 1025, T']` | 线性谱（STFT, 2048 点 → 1025 bin，`center=False`） |
| `spec_lengths` | `[B]` | 频谱有效帧数 |
| `y` (wav) | `[B, 1, L]` float | 归一化音频（`audio/32768.0`） |
| `y_lengths` | `[B]` | 音频长度 |
| `tone` | `[B, T]` | 声调 id（含语言偏移） |
| `language` | `[B, T]` | 语言 id（0/1/2，逐音素重复） |
| `bert/ja_bert/en_bert` | `[B, 1024, T]` | 三语 BERT 音素特征 |
| `sid` | `[B]` | 说话人 id |

维度关系：`T' = L / hop_length(512)`，帧率 = 44100/512 ≈ 86.1 Hz。

**帧对齐约定**：本仓库不使用 mel 后验（`use_mel_posterior_encoder=false`），后验
编码器吃**线性谱**；Mel 仅在训练损失里即时从线性谱（`spec_to_mel_torch`）与生成
音频（`mel_spectrogram_torch`）计算，用于 `loss_mel` 与可视化。

## 3. TextEncoder（文本编码器，`models.py:471`）

输入音素、声调、语言、三语 BERT，输出音素级隐藏特征 `x` 与先验分布
`N(m_p, logs_p)`。

**嵌入层：**
- `emb`: `nn.Embedding(112, 192)` — 音素嵌入（normal init, std=192^-0.5）
- `tone_emb`: `nn.Embedding(12, 192)` — 声调嵌入
- `language_emb`: `nn.Embedding(3, 192)` — 语言嵌入
- `bert_proj/ja_bert_proj/en_bert_proj`: 三个 `Conv1d(1024→192, 1)` — 三语 BERT
  各自投影到隐藏维度

**组合**（`models.py:567-577`）：

```
x = (emb(x) + tone_emb(tone) + language_emb(language)
     + bert_proj(bert)ᵀ + ja_bert_proj(ja_bert)ᵀ + en_bert_proj(en_bert)ᵀ)
    * sqrt(hidden_channels)          # 缩放保持方差
```

注意：三语 BERT 的投影**全部相加**（而不是按语言选择），所以每次推理三条 BERT
路都要有输入；本仓库语义上是"非本语言的 BERT 输入传随机向量"（`infer.py:71-83`）。

**Transformer 编码器**：`attentions.Encoder`（`attentions.py:37`），`isflow=True`，
6 层。每层 = `MultiHeadAttention` + LayerNorm + `FFN`(conv1x1→768→conv1x1) +
LayerNorm，带残差。注意力为 **相对位置注意力**（`window_size=4`，可学习相对位置
偏置 `emb_rel_k/emb_rel_v`，跨头共享，`heads_share=True`）。

说话人条件注入点：当 `use_spk_conditioned_encoder=true` 时，第 3 块（`i==2`）前把
`g` 经 `spk_emb_linear` 加到 `x` 上（`attentions.py:107-111`）；否则 `gin_channels=0`。

**输出投影**：`proj = Conv1d(192 → 2*192, 1)`，拆成 `m_p, logs_p`（均为
`[B,192,T]`），乘 `x_mask`。这定义**先验分布** $q(z_p|x) = \mathcal N(m_p, \exp^2 logs_p)$。

**参数量：约 700 万**（不含三语 BERT 本体，BERT 是外部固定提取器）。

## 4. PosteriorEncoder（后验编码器，`models.py:652`）

把线性谱映射到隐变量分布，作为 VAE 的编码器：

1. `pre = Conv1d(1025→192, 1)`；
2. `enc = modules.WN(192, kernel_size=5, dilation=1, n_layers=16, gin=g)` — 16 层
   **WaveNet 卷积**（$\times 2$ 通道 gated-tanh 单元，`fused_add_tanh_sigmoid_multiply`，
   带说话人条件通道 `cond_layer`）；
3. `proj = Conv1d(192→384, 1)` → 拆成 `m_q, logs_q`；
4. 重参数化采样：`z = m_q + randn*exp(logs_q)`（`models.py:720`）。

输出 `(z, m_q, logs_q, y_mask)`，`z` 为后验样本。**参数量约 1050 万**（16 层 WN
较大）。

## 5. 先验对流（Flow）

训练时把后验样本 `z` 映射到可逆先验空间：

```
z_p = flow(z, y_mask, g)
```

推理时则逆向 `z = flow(z_p, y_mask, g, reverse=True)`，把从先验采样的 `z_p`
变换回声学隐变量。无论用哪种流，都采用 **affine-coupling + Flip 交替**结构，且本
仓库默认打开 **Transformer 耦合**（`use_transformer_flow=true`）。

### 5.1 ResidualCouplingBlock（VITS1 风格，可选）

`models.py:592`：`n_flows=4` 层 `(ResidualCouplingLayer + Flip)`。耦合层的条件子网
是 4 层 WaveNet（`WN`），输出 `m`（`mean_only=true`，方差恒 0），可逆变换
`x1 = m + x1`（正向）/ `x1 = x1 - m`（逆向）。`mean_only` 耦合的 logdet 为 0。

### 5.2 TransformerCouplingBlock（默认，`models.py:118`）

`models.py:118`：`n_flows=4` 层 `(TransformerCouplingLayer + Flip)`。耦合层条件子网
为 Transformer `Encoder`（`attentions.py`，`isflow=True`，`n_layers_trans_flow=4`
层），支持 `share_parameter` 跨层共享 FFT 权重以省显存（默认关）。

**TransformerCouplingLayer**（`modules.py:519`）：`mean_only=true` 仿射耦合，
正向 `x1 = m + x1`、逆向 `x1 = x1 - m`；`logs` 恒 0，故每层 logdet 为 0，
整条流的 logdet 贡献全来自 Log/Flip（本仓库 `flow` 用于变换后验样本，训练时不
需要其 logdet）。

### 5.3 Flip（`modules.py:374`）

对通道维做反转（`torch.flip(x,[1])`），logdet=0。作用是把"层间强相关"打散，
保证耦合层可逆变换的组合足够表达。

> 流全体参数量：**约 2656 万**（Transformer 耦合流的最大头）。

## 6. 时长预测器

训练时 `attn.sum(2)` 得到每音素真实时长 `w`；推理时用预测的时长生成对齐。模型
同时具备两条时长预测路径，推理时按 `sdp_ratio` 混合。

### 6.1 StochasticDurationPredictor（SDP，随机，`models.py:213`）

把真实时长 `w` 的分布用 **6 段标准化流**（ElementwiseAffine + 3× ConvFlow +
Flip）建模，可输出多样化的时长，提升自然度。

- 前置 `pre`、`proj`、`DDSConv`（3 层 dilated depthwise-separable 卷积），把
  文本隐藏特征 `x` 编码为条件。
- **训练（NLL）**：对真实 `w` 构造后验 `q(z|w,x)`：`post_flows` 把噪声卷成
  `(z_u, z1)`，`u = sigmoid(z_u)`，`z0 = w - u`；再用 `log_flow(Log)` + 6 段 `flows`
  计算标准高斯 NLL，返回 `nll + logq`。
- **推理（采样）**：从标准正态采样 2 通道噪声（乘 `noise_scale`），逆向 4 段流
  （`flows[:-2]+[flows[-1]]`，去掉某个冗余 vflow），split 取 `logw = z0`。

### 6.2 DurationPredictor（DP，确定性，`models.py:352`）

两层 `Conv1d(→256) + LayerNorm + ReLU`，输出 `logw`。确定性、更快、缺多样性。

训练时二者都用量 `logw_ = log(w+1e-6)` 做 MSE（`models.py:1416-1426`）；SDP 另有
NLL 项。推理时 `logw = sdp(...,reverse)*sdp_ratio + dp(...)*(1-sdp_ratio)`
（`models.py:1502-1504`）。

**参数量：SDP ≈ 142 万，DP ≈ 44 万。**

## 7. Generator（HiFi-GAN 声码器，`models.py:724`）

把声学隐变量 `z`（帧率 ~86 Hz）上采样回原始波形采样率（44.1 kHz，总倍率
`8×8×2×2×2 = 512`，正好对应 hop=512）。

**上采样**：5 级 `ConvTranspose1d`（scale factor 8/8/2/2/2），初始通道 512 逐级减半
→ 256/128/64/32，卷积核 `16/16/8/2/2`，均 weight_norm。每级前接 `LeakyReLU(0.1)`。

**多核残差块**：3 个核 `(3,7,11)`，每个核内部的 dilated conv 序列复用上游
`ResBlock1`——`ResBlock1`（`modules.py:221`）每个核含 `(dilation=1,3,5)` 两串
卷积。C 级 × K 个核的输出**求和取平均**（`x = xs / num_kernels`）后进入下一级上采样。

**输出**：`conv_post = Conv1d(ch, 1, 7, pad=3)`（无 bias）+ `tanh` → 波形。

**说话人条件**：`cond = Conv1d(512 → 512, 1)`，在 `conv_pre` 之后加到第一个上采样
输入上（`models.py:801-803`）。推理时 `remove_weight_norm` 可加速。

**参数量：约 1466 万。**

## 8. 判别器集合（对抗训练）

### 8.1 MultiPeriodDiscriminator（MPD，`models.py:981`）

兼容 HiFi-GAN 的多判别器 GAN：1 个 **尺度判别器** `DiscriminatorS` + 5 个 **周期
判别器** `DiscriminatorP`（period∈{2,3,5,7,11}），默认 weight_norm。

- `DiscriminatorP`：把波形按周期 reshape 成 2D `[B,1,T/period,period]` 后走 5 层
  `Conv2d`(32→128→512→1024→1024)+`Conv2d(1024→1)`，捕获基频等周期结构。
- `DiscriminatorS`：1D 卷积（16→64→256→1024→1024→1024→1），带 groups 分组，
  捕获多尺度时域结构。
- 每个判别器同时返回 `fmap`（中间特征图），用于**特征匹配损失**。

**参数量：约 4675 万**（判别器合计是参数最大头之一）。

### 8.2 DurationDiscriminator（VITS2 特性，`models.py:18`）

两层原地卷积 + `dur_proj` 把真实/预测 `logw` 卷成特征，拼接后进**双向 LSTM**，
最后 `Linear→Sigmoid` 输出"时长真实度"，用标准 adv loss 训练。训练中分别喂
`(logw_, logw)` 与 `(logw_, logw_sdp)` 两对（`train.py:573-588`），让 DP 和 SDP
各自被鉴别。输入 `hidden_x` 等全部 `detach`。**参数量约 121 万。**

### 8.3 WavLMDiscriminator（SLM 对抗，`models.py:1030`）

配合 WavLM 语义损失：把 WavLM **13 层隐藏状态堆叠**（`768*13=9984` 维）经
`pre=Conv1d(9984→64)` + 3 层 Conv → Logit，作为语义判别器。**参数量约 117 万。**

## 9. 训练前向（`SynthesizerTrn.forward`，`models.py:1320`）

一次训练 step 的完整计算（对应 `train_and_evaluate` 的生成器部分）：

1. **说话人嵌入**：`g = emb_g(sid)[:, :, None]`（`[B,512,1]`），广播注入各处。
2. **文本编码**：`(x, m_p, logs_p, x_mask) = enc_p(...)`。
3. **后验编码**：`(z, m_q, logs_q, y_mask) = enc_q(spec, spec_lengths, g)`；
   `z_p = flow(z, y_mask, g)`。
4. **MAS 对齐**（`torch.no_grad`）：由先验与 `z_p` 构造负交叉熵打分
   `neg_cent`，`monotonic_align.maximum_path`（numba 动态规划，见 §10）得到对齐
   `attn`；时长 `w = attn.sum(2)`。
5. **时长损失**：SDP 的 NLL（除以 mask 求和）+ DP/SDP 与 `logw_` 的 MSE。
6. **先验铺开**：`m_p = attn @ m_pᵀ` 等，把每个文本音素对应的隐含对齐复制到频谱帧，
   得到与 `y_mask` 对齐的先验 `m_p/logs_p`。
7. **随机切片**：对 `z` 按 `y_lengths` 随机切 `segment_size/hop = 32` 帧
   （`rand_slice_segments`），`o = dec(z_slice, g)` 生成 16384 采样点音频。

返回 `(o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q,
logs_q), (x, logw, logw_, logw_sdp), g)`，供损失与可视化用。

## 10. MAS（Monotonic Alignment Search）

`monotonic_align/core.py` 的 `maximum_path_jit` 用 **numba JIT（nopython+nogil）**
动态规划求最优单调对齐：内层 `value[y,x] += max(prev_cur, prev_up)` 传播，
反向回溯记录路径。严格单调（`x≥y` 限制在可行带内），保证 TTS 的对齐是单调的。
`models.py` 里在打分上先加 `attn_mask = x_mask ⊗ y_mask`，负分位置被
`-1e9` 屏蔽（`maximum_path` 内 `max_neg_val`）。

## 11. 推理前向（`SynthesizerTrn.infer`，`models.py:1449`）

1. `enc_p` 编码 → 先验 `m_p/logs_p`。
2. 用 SDP/DP 混合时长 → `w = exp(logw)*x_mask*length_scale`，`ceil` 得每个音素帧
   数，`y_lengths = clamp_min(sum(w_ceil), 1)`。
3. 由 `generate_path(w_ceil, attn_mask)`（`commons.py:126`，按累计时长打码的单调
   对齐）复制先验到频谱帧。
4. 从先验采样：`z_p = m_p + randn*exp(logs_p)*noise_scale`。
5. `z = flow(z_p, y_mask, g, reverse=True)` — **逆转流**回到声学隐空间。
6. `o = dec(z* y_mask, g)` → 波形（可 `max_len` 截断）。

## 12. 参数量实测（默认 v2.3 配置）

| 组件 | 文件 | 参数量 |
| --- | --- | --- |
| 说话人嵌入 `emb_g` | models.py:1316 | 435,200 |
| 文本编码器 `enc_p` | models.py:471 | 7,005,888 |
| 后验编码器 `enc_q` | models.py:652 | 10,494,336 |
| Transformer 耦合流 `flow` | models.py:118 | 26,562,432 |
| 随机时长预测 `sdp` | models.py:213 | 1,415,664 |
| 确定时长预测 `dp` | models.py:352 | 444,353 |
| HiFi-GAN 生成器 `dec` | models.py:724 | 14,662,144 |
| **SynthesizerTrn 合计** | | **61,020,017** |
| 多周期判别器 MPD | models.py:981 | 46,747,132 |
| 时长判别器 DUR | models.py:18 | 1,209,409 |
| WavLM 判别器 WD | models.py:1030 | 1,173,634 |
| **全集训练参数量** | | **110,150,192** |

> 不含固定的三语 BERT（每语言数亿级外部参数量）与 WavLM 本体——它们在训练时冻结、
> 只作特征/损失来源。

## 13. config → 模型映射速查

| config（`model.*`） | 用途 |
| --- | --- |
| `inter_channels=192` | 隐变量维度（`z`）；TextEncoder 输出维度 |
| `hidden_channels=192` | 各子网隐藏维度 |
| `filter_channels=768` | Transformer FFN 中间维度 |
| `n_heads=2 , n_layers=6` | 文本编码器 Transformer 结构 |
| `kernel_size=3 , p_dropout=0.1` | Transformer 卷积核/丢弃率 |
| `use_transformer_flow=true` | 流用 Transformer 耦合（否→ResidualCouplingBlock） |
| `n_flow_layer=4` | 流层数 |
| `n_layers_trans_flow=4` | 耦合子网 Transformer 层数 |
| `use_duration_discriminator` | 时长判别器开/关 |
| `use_spk_conditioned_encoder` | 说话人条件注入 TextEncoder |
| `use_noise_scaled_mas` | 噪声缩放的 MAS |
| `use_mel_posterior_encoder` | 是否用 mel 作为后验输入（本仓库 false） |
| `resblock=1, resblock_kernel_sizes=[3,7,11]` | HiFi-GAN 残差块 |
| `upsample_rates=[8,8,2,2,2]` 等 | 声码器上采样 |
| `gin_channels=512` | 说话人嵌入维度（`emb_g`） |
| `slm.*` | WavLM 判别器结构/模型路径 |
| `data.n_speakers` | 说话人数；`0` 时用 `ReferenceEncoder` 换 `emb_g` |

## 14. 单说话人 / 零样本路径

当 `n_speakers == 0`（`models.py:1315-1318`）时，`emb_g` 被替换为
`ReferenceEncoder`（`models.py:1097`）：把参考音频 mel 经 6 层 conv 降采样 + 双向
GRU 汇成说话人嵌入，实现零样本/音箱-clone 的潜力。本仓库的 `preproc` 默认按多说话
人写配置，因此默认走 `emb_g` 路径；`infer.py` 的 `get_net_g` 同样接受 `n_speakers`，
若配置为 0 则需传参考音频 `y`（`infer` 接口已预留，CLI 未暴露）。