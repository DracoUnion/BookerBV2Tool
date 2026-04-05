import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from commons import init_weights, get_padding
from text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    """
    时长判别器（VITS2特性）

    用于判别真实时长和预测时长，提升时长预测的准确性。
    通过LSTM处理时长信息，输出概率值表示时长的真实性。

    参数:
        in_channels: 输入通道数
        filter_channels: 卷积滤波器通道数
        kernel_size: 卷积核大小
        p_dropout: Dropout概率
        gin_channels: 条件（说话人）嵌入通道数，0表示无条件
    """

    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 双向LSTM用于建模时序依赖
        self.LSTM = nn.LSTM(
            2 * filter_channels, filter_channels, batch_first=True, bidirectional=True
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(
            nn.Linear(2 * filter_channels, 1), nn.Sigmoid()
        )

    def forward_probability(self, x, dur):
        """
        计算时长概率

        参数:
            x: 隐藏特征 [B, C, T]
            dur: 时长对数 [B, 1, T]
        返回:
            output_prob: 时长真实性概率 [B, T, 1]
        """
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = x.transpose(1, 2)
        x, _ = self.LSTM(x)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        """
        前向传播

        参数:
            x: 输入特征 [B, C, T]
            x_mask: 掩码 [B, 1, T]
            dur_r: 真实时长对数 [B, 1, T]
            dur_hat: 预测时长对数 [B, 1, T]
            g: 说话人嵌入 [B, C, 1]
        返回:
            output_probs: 真实和预测时长的概率列表
        """
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, dur)
            output_probs.append(output_prob)

        return output_probs


class TransformerCouplingBlock(nn.Module):
    """
    Transformer耦合块（用于流模型）

    使用Transformer架构的耦合层，用于学习可逆变换。
    支持参数共享以节省内存。

    参数:
        channels: 通道数
        hidden_channels: 隐藏层通道数
        filter_channels: FFN滤波器通道数
        n_heads: 注意力头数
        n_layers: Transformer层数
        kernel_size: 卷积核大小
        p_dropout: Dropout概率
        n_flows: 流层数
        gin_channels: 条件通道数
        share_parameter: 是否共享参数
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        # 如果启用参数共享，创建共享的FFT模块
        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        前向/逆向传播

        参数:
            x: 输入 [B, C, T]
            x_mask: 掩码 [B, 1, T]
            g: 条件嵌入
            reverse: 是否逆向（用于推理）
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class StochasticDurationPredictor(nn.Module):
    """
    随机时长预测器（SDP）

    使用标准化流（Normalizing Flow）建模时长的分布，
    可以生成多样化的时长预测，增加语音的自然度。

    参数:
        in_channels: 输入通道数
        filter_channels: 滤波器通道数
        kernel_size: 卷积核大小
        p_dropout: Dropout概率
        n_flows: 流层数
        gin_channels: 条件通道数
    """

    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # 需要从未来版本中移除
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        """
        前向传播

        参数:
            x: 输入特征 [B, C, T]
            x_mask: 掩码 [B, 1, T]
            w: 真实时长（训练时使用）
            g: 条件嵌入
            reverse: 是否逆向（推理时生成时长）
            noise_scale: 噪声缩放因子
        """
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            # 训练模式：计算负对数似然
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            # 推理模式：从噪声生成时长
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # 移除无用的vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    """
    确定性时长预测器（DP）

    使用简单的卷积网络预测时长，输出确定性的时长值。
    相比SDP更快但缺乏多样性。

    参数:
        in_channels: 输入通道数
        filter_channels: 滤波器通道数
        kernel_size: 卷积核大小
        p_dropout: Dropout概率
        gin_channels: 条件通道数
    """

    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        """
        前向传播

        参数:
            x: 输入特征 [B, C, T]
            x_mask: 掩码 [B, 1, T]
            g: 条件嵌入
        返回:
            预测的对数时长 [B, 1, T]
        """
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class Bottleneck(nn.Sequential):
    """
    瓶颈层（用于GPT风格的模块）

    将输入投影到两个不同的隐藏维度。
    """

    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        super().__init__(*[c_fc1, c_fc2])


class Block(nn.Module):
    """
    GPT风格的Transformer块

    包含LayerNorm和MLP，使用残差连接。
    """

    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    """
    多层感知机（使用SwiGLU激活函数）

    参数:
        in_dim: 输入维度
        hidden_dim: 隐藏层维度
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # SwiGLU激活: silu(x @ W1) * (x @ W2)
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class TextEncoder(nn.Module):
    """
    文本编码器

    将输入的文本符号、音调、语言和BERT特征编码为隐藏表示，
    用于后续的音素时长预测和声学特征生成。

    参数:
        n_vocab: 词汇表大小
        out_channels: 输出通道数
        hidden_channels: 隐藏层通道数
        filter_channels: FFN滤波器通道数
        n_heads: 注意力头数
        n_layers: Transformer层数
        kernel_size: 卷积核大小
        p_dropout: Dropout概率
        gin_channels: 条件通道数
    """

    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 符号嵌入
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # 音调嵌入
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)

        # 语言嵌入
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)

        # BERT投影层（支持多语言）
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)    # 中文
        self.ja_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 日文
        self.en_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 英文

        # Transformer编码器
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        # 投影到输出（均值和对数方差）
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, tone, language, bert, ja_bert, en_bert, g=None):
        """
        前向传播

        参数:
            x: 文本符号索引 [B, T]
            x_lengths: 文本长度 [B]
            tone: 音调索引 [B, T]
            language: 语言索引 [B, T]
            bert: 中文BERT特征 [B, 1024, T]
            ja_bert: 日文BERT特征 [B, 1024, T]
            en_bert: 英文BERT特征 [B, 1024, T]
            g: 条件嵌入
        返回:
            x: 编码特征 [B, C, T]
            m: 均值 [B, C, T]
            logs: 对数标准差 [B, C, T]
            x_mask: 掩码 [B, 1, T]
        """
        # 投影BERT特征
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        en_bert_emb = self.en_bert_proj(en_bert).transpose(1, 2)

        # 组合所有嵌入
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
            + en_bert_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        # Transformer编码
        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        # 分割为均值和对数方差
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    """
    残差耦合块（用于流模型）

    使用仿射耦合层的标准化流，用于学习从后验到先验的可逆映射。

    参数:
        channels: 通道数
        hidden_channels: 隐藏层通道数
        kernel_size: 卷积核大小
        dilation_rate: 空洞率
        n_layers: 每层卷积层数
        n_flows: 流层数
        gin_channels: 条件通道数
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    """
    后验编码器

    从声学特征（频谱图）编码得到隐变量z的后验分布，
    用于VAE的推断和训练。

    参数:
        in_channels: 输入通道数（频谱图通道数）
        out_channels: 输出通道数（隐变量维度）
        hidden_channels: 隐藏层通道数
        kernel_size: 卷积核大小
        dilation_rate: 空洞率
        n_layers: WaveNet层数
        gin_channels: 条件通道数
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        前向传播

        参数:
            x: 输入频谱图 [B, C, T]
            x_lengths: 长度 [B]
            g: 条件嵌入
        返回:
            z: 采样得到的隐变量
            m: 后验均值
            logs: 后验对数标准差
            x_mask: 掩码
        """
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 重参数化技巧采样
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    """
    HiFi-GAN风格的声码器生成器

    将隐变量上采样并转换为波形音频。
    使用多尺度上采样和残差块。

    参数:
        initial_channel: 输入通道数
        resblock: 残差块类型（"1"或"2"）
        resblock_kernel_sizes: 残差块卷积核大小列表
        resblock_dilation_sizes: 残差块空洞大小列表
        upsample_rates: 上采样率列表
        upsample_initial_channel: 初始上采样通道数
        upsample_kernel_sizes: 上采样卷积核大小列表
        gin_channels: 条件通道数
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        # 创建上采样层
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # 创建残差块
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        """
        前向传播

        参数:
            x: 输入隐变量 [B, C, T]
            g: 条件嵌入
        返回:
            生成的波形 [B, 1, T']
        """
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        # 上采样和残差块处理
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """移除权重归一化（用于推理加速）"""
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    """
    周期判别器（Period Discriminator）

    对输入音频按周期分块后使用2D卷积进行判别。
    用于捕获音频的周期性特征（如基频）。

    参数:
        period: 周期长度
        kernel_size: 卷积核大小
        stride: 步长
        use_spectral_norm: 是否使用谱归一化
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入波形 [B, 1, T]
        返回:
            x: 判别结果
            fmap: 特征图列表（用于特征匹配损失）
        """
        fmap = []

        # 1D转2D：按周期分块
        b, c, t = x.shape
        if t % self.period != 0:  # 如果需要，先填充
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    """
    尺度判别器（Scale Discriminator）

    使用1D卷积在多个尺度上对音频进行判别。
    用于捕获不同时间尺度的音频特征。

    参数:
        use_spectral_norm: 是否使用谱归一化
    """

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入波形 [B, 1, T]
        返回:
            x: 判别结果
            fmap: 特征图列表
        """
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    多周期判别器（MPD）

    组合多个不同周期的周期判别器和一个尺度判别器，
    用于全面评估生成音频的真实性。

    参数:
        use_spectral_norm: 是否使用谱归一化
    """

    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]  # 不同的周期值

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        """
        前向传播

        参数:
            y: 真实音频
            y_hat: 生成音频
        返回:
            y_d_rs: 真实音频的判别结果列表
            y_d_gs: 生成音频的判别结果列表
            fmap_rs: 真实音频的特征图列表
            fmap_gs: 生成音频的特征图列表
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """
    WavLM判别器（用于SLM损失）

    对WavLM提取的特征进行判别，确保生成音频的语义一致性。

    参数:
        slm_hidden: WavLM隐藏层维度
        slm_layers: WavLM层数
        initial_channel: 初始通道数
        use_spectral_norm: 是否使用谱归一化
    """

    def __init__(
        self, slm_hidden=768, slm_layers=13, initial_channel=64, use_spectral_norm=False
    ):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.pre = norm_f(
            Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        前向传播

        参数:
            x: WavLM特征 [B, slm_hidden*slm_layers, T]
        返回:
            判别结果
        """
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x


class ReferenceEncoder(nn.Module):
    """
    参考编码器（用于单说话人模式）

    从参考音频中提取说话人嵌入，实现零样本语音克隆。

    输入: [N, Ty/r, n_mels*r] 梅尔频谱图
    输出: [N, ref_enc_gru_size] 说话人嵌入

    参数:
        spec_channels: 频谱图通道数
        gin_channels: 输出嵌入维度
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs, mask=None):
        """
        前向传播

        参数:
            inputs: 梅尔频谱图 [N, T, n_mels]
        返回:
            说话人嵌入 [N, gin_channels]
        """
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        """计算经过卷积后的通道数"""
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class SynthesizerTrn(nn.Module):
    """
    VITS训练用合成器

    完整的端到端TTS模型，包含：
    - 文本编码器（TextEncoder）
    - 后验编码器（PosteriorEncoder）
    - 生成器/声码器（Generator）
    - 流模型（Flow）
    - 时长预测器（DurationPredictor和StochasticDurationPredictor）

    支持多说话人和条件输入（音调、语言、BERT特征）。
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=4,
        flow_share_parameter=False,
        use_transformer_flow=True,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels

        # 文本编码器
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )

        # 声码器（解码器）
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        # 后验编码器
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        # 流模型（先验编码器）
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )

        # 随机时长预测器（SDP）
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )
        # 确定性时长预测器（DP）
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        # 说话人嵌入或多说话人参考编码器
        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        en_bert,
    ):
        """
        训练用的前向传播

        参数:
            x: 文本符号 [B, T]
            x_lengths: 文本长度 [B]
            y: 频谱图 [B, C, T']
            y_lengths: 频谱图长度 [B]
            sid: 说话人ID [B]
            tone: 音调 [B, T]
            language: 语言 [B, T]
            bert: 中文BERT特征 [B, 1024, T]
            ja_bert: 日文BERT特征 [B, 1024, T]
            en_bert: 英文BERT特征 [B, 1024, T]
        返回:
            o: 生成音频
            l_length: 时长损失
            attn: 对齐矩阵
            ids_slice: 切片索引
            x_mask: 文本掩码
            y_mask: 频谱图掩码
            (z, z_p, m_p, logs_p, m_q, logs_q): 隐变量和统计参数
            (x, logw, logw_, logw_sdp): 隐藏状态和时长对数
            g: 说话人嵌入
        """
        # 获取说话人嵌入
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        # 文本编码
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, en_bert, g=g
        )

        # 后验编码
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        # MAS（单调对齐搜索）找最佳对齐
        with torch.no_grad():
            # 负交叉熵
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            # 添加噪声（如果启用噪声缩放的MAS）
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            # 计算对齐路径
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        # 对齐后的时长
        w = attn.sum(2)

        # SDP损失（负对数似然）
        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        # 对数时长
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        logw_sdp = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=1.0)

        # DP和SDP的时长损失
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )
        l_length_sdp += torch.sum((logw_sdp - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        l_length = l_length_dp + l_length_sdp

        # 根据对齐扩展先验
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # 随机切片用于训练
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_, logw_sdp),
            g,
        )

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        en_bert,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
    ):
        """
        推理用的前向传播

        参数:
            x: 文本符号
            x_lengths: 文本长度
            sid: 说话人ID
            tone: 音调
            language: 语言
            bert: 中文BERT特征
            ja_bert: 日文BERT特征
            en_bert: 英文BERT特征
            noise_scale: 噪声缩放（控制音质多样性）
            length_scale: 时长缩放（控制语速）
            noise_scale_w: 时长噪声缩放
            max_len: 最大长度
            sdp_ratio: SDP和DP的混合比例（0=只用DP，1=只用SDP）
            y: 参考音频（用于单说话人模式）
        返回:
            o: 生成音频
            attn: 对齐矩阵
            y_mask: 输出掩码
            (z, z_p, m_p, logs_p): 隐变量和统计参数
        """
        # 获取说话人嵌入
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        # 文本编码
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, en_bert, g=g
        )

        # 时长预测（SDP和DP混合）
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )

        # 生成对齐路径
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        # 根据对齐扩展先验
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 从先验采样隐变量
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
