# 05 · 训练

训练实现位于 `train.py`。**单机单进程**：不做 DDP/distributed，`rank` 恒为 0，
设备自动选择（有 CUDA 用 GPU，否则 CPU）。模型、优化器、数据采样均按
`config.json` 驱动，CLI 只传配置文件路径。

## 1. 入口与设备

```
BookerBV2Tool train -c config.json
```

- `_device()`：`cuda` if 可用，否则 `cpu`（`train.py:61-63`）。
- `amp_enabled = bf16_run and torch.cuda.is_available()`：**CPU 上强制关闭 bf16**
  （`train.py:98`），避免 CPU 下 AMP 走不通。
- 固定 `torch.manual_seed(seed)`；`num_workers = min(train_ms.num_workers, cpu-1)`。
- 若配置 `model.use_noise_scaled_mas=true`，走"噪声缩放 MAS"，并随 `global_step`
  线性衰减 `mas_noise_scale`（初值 0.01，步进 2e-6，最低 0），`models.py:1392-1398`。
- 若配置 `use_duration_discriminator=true`，构建 `DurationDiscriminator`。

## 2. 数据加载

- `TextAudioSpeakerLoader`（`data_utils.py:16`）：
  - 读 `load_filepaths_and_text`（文件行按 `|` 拆分，下同）；
  - 过滤 `min_text_len(1) <= len(phones) <= max_text_len(384)`；
  - 音频经 `scipy.io.wavfile` 读入，`audio/32768.0` 归一化；校验采样率必须等于
    `data.sampling_rate`，否则抛错；
  - 频谱**惰性计算**：优先加载 `<wav>.spec.pt` 缓存（`data.spec_cache` 为 true
    时写回），否则用 `spectrogram_torch`（2048 点 STFT，1025 bin，center=False）
    现场算；可用 `.mel.pt`（mel 后验，默认关）。
  - BERT 特征从 `<wav>_bert.pt` 读取；缺失时回退随机向量（§4 容错）。
- `TextAudioSpeakerCollate`（`data_utils.py:191`）：按 spec 长度降序，右侧补零到
  batch 内最大长度，产出 `(x, x_lengths, spec, spec_lengths, wav, wav_lengths, sid,
  tone, language, bert, ja_bert, en_bert)`。
- `DistributedBucketSampler`（`data_utils.py:283`）：按音频 spec 长度分桶
  `[32,300,400,500,600,700,800,900,1000]`，桶内 gather 样本凑整除，返回按桶随机
  的 batch 索引，保证同 batch 长度接近（`num_replicas=1`）。

## 3. 训练主循环（`train_and_evaluate`，`train.py:422`）

每步的次序：

1. **生成器前向（bf16 AMP）** → `o, l_length, attn, ids_slice, x_mask, y_mask,
   (z,z_p,m_p,logs_p,m_q,logs_q), (hidden_x, logw, logw_, logw_sdp), g`。
2. 切真实 mel/音频：`y_mel = slice(mel, ids_slice, 32f)`、`y = slice(wav, ids*hop,
   segment_size(16384))`；`y_hat_mel = mel_spectrogram_torch(y_hat)`。
3. **判别器训练**（梯度不回流到生成器）：
   - MPD：`loss_disc_r/g`（LSGAN）；
   - 时长判别器（若启用）：`(logw_, logw)` 与 `(logw_, logw_sdp)` 两对；
   - WavLM 判别器（SLM on）：`loss_slm = wl.discriminator(y, y_hat)`;
   - 各自 `optim_d/dur/wd` 反向 + 混合精度 scaler。
4. **生成器训练**（梯度允许经生成器回传）：
   - `loss_dur` = `sum(l_length)`（时长：SDP NLL + DP/SDP MSE）；
   - `loss_mel` = `L1(y_mel, y_hat_mel) * c_mel(45)`；
   - `loss_kl` = `kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl(1.0)`；
   - `loss_fm` = 特征匹配损失（MPD fmap）；
   - `loss_gen` = 生成器 LSGAN 对抗损失；
   - `loss_lm/loss_lm_gen` = WavLM 特征 L1 + WavLM 判别器生成损失；
   - 总生成损失：
     `loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_lm + loss_lm_gen`
     （+ 时长判别器的 generator loss）；
   - `optim_g` 反向 + scaler.update()。

## 4. 损失明细

| 损失 | 公式/来源 | 系数 |
| --- | --- | --- |
| `loss_mel` | `L1(真mel, 生成mel)` | `c_mel=45` |
| `loss_kl` | $KL$（后验流 `z_p` vs 先验 `m_p/logs_p`，`losses.py:45`） | `c_kl=1.0` |
| `loss_dur` | SDP NLL + DP/SDP 时长 MSE（`models.py:1412-1426`） | 1 |
| `loss_gen` | LSGAN `mean((1-d(g))^2)` | 1 |
| `loss_fm` | Σ|fmap_r - fmap_g|（`losses.py:6`，×2） | 1 |
| `loss_lm` | WavLM 全层隐藏态 L1（冻结 WavLM） | 1 |
| `loss_lm_gen` | `(1 - WavLMDisc(mel特征生成))^2` | 1 |
| `loss_slm`(D) | WavLM 判别器 LSGAN | 1 |

WavLM 侧：`WavLMLoss`（`losses.py:63`）加载本地 `slm.model`（默认
`./slm/wavlm-base-plus`）。若目录不存在，`_NoWavLM` 占位把相关损失置 0 并跳过其
disc 训练（`train.py:66-80,616-630`），保证无 WavLM 模型时仍可训练（尤其 CPU）。

## 5. 优化器与调度

- 生成器/MPD/时长判定器三个（或四个）独立 `AdamW(lr=2e-4, betas=(0.8,0.99),
  eps=1e-9)`。
- `ExponentialLR(gamma=0.99995)` 每个 epoch 乘衰减；`last_epoch` 从恢复的检查点
  epoch 对齐。
- 原生 `autocast(dtype=bfloat16)` + `GradScaler`（CPU 关闭）。
- `clip_grad_value_(grad, None)` 仅记录 norm；bf16 时生成器 clip 500、判别器 200。

## 6. 检查点与恢复

`utils.py:load_checkpoint/save_checkpoint`：

- 保存：`G_{step}.pth / D_{step}.pth / WD_{step}.pth / DUR_{step}.pth`，内容
  `{model, iteration, optimizer, learning_rate}`。
- 恢复：自动找 `model_dir` 下数字最大的 `G_*.pth`（`latest_checkpoint_path`），
  `skip_optimizer` 默认 true；兼容旧检查点缺 `ja_bert_proj` 时自动补零。
- 可选底模：`train_ms.base.use_base_model` 时 `utils.download_checkpoint` 从
  HF 拉 `DUR_0/D_0/G_0.pth` 做初始化。
- `keep_ckpts`（默认 8）：按时间删除多余检查点腾空间。
- WavLM 等重特征缓存：`data.spec_cache` 由 `config.py` 读取（模块级单例，从 CWD
  的 `config.json` 读取 `data.spec_cache`）。

## 7. 评估与日志

- `eval_interval` 步评估一次：验证集上 `infer`（`sdp_ratio=0` 与 `=1` 各一次，
  `max_len=1000`），写 mel 图/音频到 TensorBoard。
- TensorBoard 标量：`loss/g/*`（total/fm/mel/dur/kl/lm/lm_gen/dur_gen/各判别器个
  数）、`loss/d_*`、`loss/dur_disc*`、`loss/wd/total`、学习率、三路 grad_norm；
  图像：真/伪 mel、全谱 mel、对齐 `attn`。目录 `model_dir/eval`。
- 日志文件 `train.log`（`utils.get_logger`），并记录 `githash` 用于比对代码版本。

## 8. 注意事项（来自 README）

- 音频采样率必须与 `data.sampling_rate` 一致，否则 `TextAudioSpeakerLoader` 报
  SR 不匹配。
- 模型数千万参数，CPU 单步耗时数秒~数分钟；建议 GPU 训练。
- `add_blank` 影响音素序列长度（约 ×2），训练与推理必须一致，否则 bert 特征维度
  断言失败。