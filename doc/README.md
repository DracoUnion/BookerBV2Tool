# BookerBV2Tool 设计文档

本项目（`BookerBV2Tool`）是一套 Bert-VITS2（BV2，v2.3）声音克隆的**数据准备 + 训练 + 合成**工具链：把视频/音频切成语音片段、语音转写、整理标注、生成 BERT 音素特征，最终训练并推理一个多语言多说话人 TTS 模型。

本目录为整个项目的详细设计文档，按主题拆分如下：

| 文档 | 内容 | 阅读建议 |
| --- | --- | --- |
| [01-architecture.md](01-architecture.md) | 总体架构：设计目标、分层、CLI、模块依赖、数据流总览 | 入门必读 |
| [02-data-pipeline.md](02-data-pipeline.md) | 数据流水线：slice / resample / mark / mklist / preproc / bert-gen 的算法与数据结构 | 数据侧 |
| [03-text-processing.md](03-text-processing.md) | 文本处理：音素词表、中/英/日 G2P、BERT 特征、序列化 | 文本侧 |
| [04-model-architecture.md](04-model-architecture.md) | **模型架构详解**：SynthesizerTrn 及各子网络、判别器、训练/推理前向、参数量 | **重点文档** |
| [05-training.md](05-training.md) | 训练流程：单机训练、损失函数、优化器、检查点、评估 | 训练侧 |
| [06-inference.md](06-inference.md) | 推理：文本→音频的完整合成流程与生成参数 | 推理侧 |
| [07-configuration.md](07-configuration.md) | `config.json` 全部字段参考 | 配置速查 |

代码注释与本文档均使用中文；涉及代码处给出 `文件:行号` 指向。模型版本为
**Bert-VITS2 2.3**，与 `infer.py` 中 `latest_version = "2.3"` 一致。

## 相关链接

- 上游项目：<https://github.com/fishaudio/Bert-VITS2>
- 本项目仓库：<https://github.com/DracoUnion/BookerBV2Tool>
- 用户手册（命令行用法）：[`README.md`](../README.md)
- 单元测试说明：[`tests/README.md`](../tests/README.md)
