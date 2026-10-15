# 07 · 配置参考

`config.json` 是**唯一的训练/推理配置源**。`preproc` 会以包内 `config.json`
（`BookerBV2Tool/config.json`）为模板，写入 `spk2id`、`n_speakers`、`version`、
训练/验证路径后生成到工作目录。下方列出全部字段及其默认值。

## 1. `train.*` — 训练超参

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `log_interval` | 200 | 每 N 步记日志/TensorBoard |
| `eval_interval` | 1000 | 每 N 步在验证集评估并保存检查点 |
| `seed` | 42 | 随机种子 |
| `epochs` | 1000 | 训练轮数 |
| `learning_rate` | 0.0002 | AdamW 学习率 |
| `betas` | [0.8, 0.99] | AdamW beta |
| `eps` | 1e-9 | AdamW epsilon |
| `batch_size` | 16 | 每 batch 样本数 |
| `bf16_run` | false | 是否 bf16 AMP（仅 GPU 生效） |
| `lr_decay` | 0.99995 | ExponentialLR gamma（每 epoch） |
| `segment_size` | 16384 | 生成器随机切片采样点数（→32 频谱帧） |
| `init_lr_ratio` | 1 | 初始学习率比例（本仓库未用） |
| `warmup_epochs` | 0 | 预热轮数（本仓库未用） |
| `c_mel` | 45 | Mel 损失系数 |
| `c_kl` | 1.0 | KL 损失系数 |
| `c_commit` | 100 | 提交损失系数（本仓库未用） |
| `skip_optimizer` | true | 恢复检查点时是否跳过优化器状态 |
| `freeze_ZH/EN/JP_bert` | false | 冻结对应语言 BERT 投影层（`net_g.enc_p.*_bert_proj`） |
| `freeze_emo` | false | 情感冻结（本仓库未用） |

## 2. `data.*` — 数据/信号参数

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `training_files` | train_list.json | 训练列表 |
| `validation_files` | val_list.json | 验证列表 |
| `max_wav_value` | 32768.0 | WAV 归一化除数 |
| `sampling_rate` | 44100 | 音频采样率（须与数据一致） |
| `filter_length` | 2048 | STFT 窗/FFT 点数（→1025 bin） |
| `hop_length` | 512 | 帧移 |
| `win_length` | 2048 | 窗长 |
| `n_mel_channels` | 128 | Mel 通道数（损失用） |
| `mel_fmin` / `mel_fmax` | 0.0 / null | Mel 频率范围 |
| `add_blank` | true | 是否插空白音素 `_` |
| `n_speakers` | 850 | 说话人数；0 时走 ReferenceEncoder |
| `cleaned_text` | true | 数据已清洗 |
| `spk2id` | — | `{角色名: id}`，preproc 写入 |
| `spec_cache` | false | 是否缓存线性谱 `.spec.pt`（`config.py` 从 CWD 读） |

## 3. `model.*` — 架构开关与维度

见 04 章 §13 的映射表。补充字段：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `use_spk_conditioned_encoder` | true | 说话人条件注入 TextEncoder |
| `use_noise_scaled_mas` | true | 噪声缩放 MAS（初值 0.01，步进 2e-6） |
| `use_mel_posterior_encoder` | false | 用 mel 做后验输入（默认线性谱） |
| `use_duration_discriminator` | true | 时长判别器 |
| `inter_channels / hidden_channels` | 192 / 192 | 隐变量/隐藏维度 |
| `filter_channels` | 768 | FFN 中间维度 |
| `n_heads / n_layers` | 2 / 6 | 文本 Transformer |
| `kernel_size / p_dropout` | 3 / 0.1 | 卷积核/丢弃率 |
| `resblock` | "1" | HiFi-GAN 残差块类型（"1"/"2"） |
| `resblock_kernel_sizes` | [3,7,11] | 残差块卷积核 |
| `resblock_dilation_sizes` | [[1,3,5]×3] | 残差块空洞 |
| `upsample_rates` | [8,8,2,2,2] | 上采样率（总 ×512） |
| `upsample_initial_channel` | 512 | 上采样初始通道 |
| `upsample_kernel_sizes` | [16,16,8,2,2] | 上采样卷积核 |
| `n_layers_q` | 3 | （上游保留） |
| `use_spectral_norm` | false | 判别器用谱归一化替代 weight_norm |
| `gin_channels` | 512 | 说话人嵌入维度 |
| `slm.model` | ./slm/wavlm-base-plus | WavLM 本地目录；缺失自动退化 |
| `slm.sr` | 16000 | WavLM 输入采样率 |
| `slm.hidden / nlayers / initial_channel` | 768 / 13 / 64 | WavLM 判别器结构 |

## 4. `train_ms.*` — 训练环境/产物

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `env.*` | localhost:10086, WORLD=1 | 分布式环境变量（单机不使用，保留兼容） |
| `base.use_base_model` | false | 是否从 HF 下载底模初始化 |
| `base.repo_id` | Stardust_minus/Bert-VITS2 | 底模仓库 |
| `base.model_image` | Bert-VITS2_2.3底模 | 底模说明 |
| `model_dir` | models | 检查点输出目录 |
| `config_path` | configs/config.json | （上游保留） |
| `num_workers` | 16 | DataLoader 进程数（自动 clamp 到 cpu-1） |
| `spec_cache` | false | 见 `data.spec_cache` |
| `keep_ckpts` | 8 | 保留的检查点数量（按时间清理） |

## 5. 其它顶层

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `mirror` | "" | 模型下载镜像 |
| `openi_token` | "" | OpenI 平台令牌 |
| `version` | "2.3" | 模型版本（preproc 写入，infer 校验 `latest_version`） |

## 6. 配置文件生命周期

```
包内模板 config.json
   │  preproc（读取 + 写 spk2id/n_speakers/version/路径）
   ▼
工作目录 config.json ──► bert-gen / train / infer 读取
```

`train` / `infer` 直接用 `-c` 指定的 `config.json`；`bert-gen` 也读该文件决定
`add_blank` 与训练/验证列表。