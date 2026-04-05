# flake8: noqa: E402
# 导入必要的库
import json
import platform
import os
from os import path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import argparse
import datetime

# 设置numba的日志级别为WARNING，减少不必要的日志输出
logging.getLogger("numba").setLevel(logging.WARNING)
from . import commons
import utils
from .data_utils import (
    TextAudioSpeakerLoader,      # 文本-音频-说话人数据加载器
    TextAudioSpeakerCollate,     # 数据批处理整理函数
    DistributedBucketSampler,    # 分布式分桶采样器
)
from .models import (
    SynthesizerTrn,              # 生成器模型（合成器）
    MultiPeriodDiscriminator,    # 多周期判别器
    DurationDiscriminator,       # 时长判别器（VITS2）
    WavLMDiscriminator,          # WavLM判别器（用于SLM损失）
)
from .losses import (
    generator_loss,              # 生成器损失函数
    discriminator_loss,          # 判别器损失函数
    feature_loss,                # 特征匹配损失
    kl_loss,                     # KL散度损失
    WavLMLoss,                   # WavLM损失（SLM）
)
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .text.symbols import symbols

# 启用CUDA TF32加速（在Ampere及以上架构的GPU上可提升性能）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # 如果遇到训练问题，请尝试禁用TF32
)
# 设置浮点矩阵乘法的精度为medium，平衡速度和精度
torch.set_float32_matmul_precision("medium")
# 启用Flash Attention（如果PyTorch版本支持）
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # 如果PyTorch版本低于2.0，此功能不可用

# 全局步数计数器，用于记录训练进度
global_step = 0


def train_handle(args):
    """
    训练处理函数 - 分布式训练的主入口

    参数:
        args: 命令行参数，包含配置文件路径等
    """
    # 读取配置文件
    config = json.loads(
        open(args.config, encoding='utf8').read())
    # 环境变量解析 - 从配置中加载环境变量
    envs = config['train_ms']['env']
    for env_name, env_value in envs.items():
        if env_name not in os.environ.keys():
            print("加载config中的配置{}".format(str(env_value)))
            os.environ[env_name] = str(env_value)
    print(
        "加载环境变量 \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    # 根据操作系统选择分布式后端：Windows使用gloo，Linux使用nccl
    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # 如果是Windows系统，切换到gloo后端
    # 初始化进程组 - 使用torchrun替代mp.spawn
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )
    # 获取当前进程的rank和本地rank
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()  # 获取GPU总数


    # 创建模型保存目录
    model_dir = config['train_ms']['model_dir']
    if not path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    # 设置随机种子，确保可复现性
    torch.manual_seed(config['train']['seed'])
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU

    global global_step
    # 只有rank 0的进程才创建日志和TensorBoard写入器
    if rank == 0:
        logger = utils.get_logger(model_dir)
        logger.info(config)
        utils.check_git_hash(model_dir)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))

    # 创建训练数据集
    train_dataset = TextAudioSpeakerLoader(config['data']['training_files'], config['data'])
    # 创建分布式分桶采样器 - 根据音频长度将数据分桶，提高效率
    train_sampler = DistributedBucketSampler(
        train_dataset,
        config['train']['batch_size'],
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],  # 桶的边界（音频长度）
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # 创建数据整理函数
    collate_fn = TextAudioSpeakerCollate()
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        num_workers=min(config['train_ms']['num_workers'], os.cpu_count() - 1),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader配置可以调整

    # 只有rank 0的进程才创建评估数据集
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(config['data']['validation_files'], config['data'])
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    # 检查是否使用噪声缩放的MAS（Monotonic Alignment Search）- VITS2特性
    if (
        "use_noise_scaled_mas" in config['model'].keys()
        and config['model']['use_noise_scaled_mas'] is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    # 检查是否使用时长判别器 - VITS2特性
    if (
        "use_duration_discriminator" in config['model']['keys']()
        and config['model']['use_duration_discriminator'] is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            config['model']['hidden_channels'],
            config['model']['hidden_channels'],
            3,
            0.1,
            gin_channels=config['model']['gin_channels'] if config['data']['n_speakers'] != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None

    # 检查是否使用说话人条件编码器 - VITS2特性
    if (
        "use_spk_conditioned_encoder" in config['model'].keys()
        and config['model']['use_spk_conditioned_encoder'] is True
    ):
        if config['data']['n_speakers'] == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    # 创建生成器（合成器）模型
    net_g = SynthesizerTrn(
        len(symbols),
        config['data']['filter_length'] // 2 + 1,
        config['train']['segment_size'] // config['data']['hop_length'],
        n_speakers=config['data']['n_speakers'],
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **config['model'],
    ).cuda(local_rank)

    # 根据配置冻结BERT编码器（可选）
    if getattr(config['train'], "freeze_ZH_bert", False):
        print("Freezing ZH bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False

    if getattr(config['train'], "freeze_EN_bert", False):
        print("Freezing EN bert encoder !!!")
        for param in net_g.enc_p.en_bert_proj.parameters():
            param.requires_grad = False

    if getattr(config['train'], "freeze_JP_bert", False):
        print("Freezing JP bert encoder !!!")
        for param in net_g.enc_p.ja_bert_proj.parameters():
            param.requires_grad = False

    # 创建多周期判别器
    net_d = MultiPeriodDiscriminator(config['model']['use_spectral_norm']).cuda(local_rank)
    # 创建WavLM判别器（用于SLM损失）
    net_wd = WavLMDiscriminator(
        config['model']['slm']['hidden'], config['model']['slm']['nlayers'], config['model']['slm']['initial_channel']
    ).cuda(local_rank)

    # 创建优化器
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        config['train']['learning_rate'],
        betas=config['train']['betas'],
        eps=config['train']['eps'],
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config['train']['learning_rate'],
        betas=config['train']['betas'],
        eps=config['train']['eps'],
    )
    optim_wd = torch.optim.AdamW(
        net_wd.parameters(),
        config['train']['learning_rate'],
        betas=config['train']['betas'],
        eps=config['train']['eps'],
    )
    # 时长判别器的优化器（如果启用）
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            config['train']['learning_rate'],
            betas=config['train']['betas'],
            eps=config['train']['eps'],
        )
    else:
        optim_dur_disc = None

    # 包装模型为DistributedDataParallel（DDP）
    net_g = DDP(net_g, device_ids=[local_rank], bucket_cap_mb=512)
    net_d = DDP(net_d, device_ids=[local_rank], bucket_cap_mb=512)
    net_wd = DDP(net_wd, device_ids=[local_rank], bucket_cap_mb=512)
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            bucket_cap_mb=512,
        )

    # 下载底模（预训练模型）
    if config['train_ms']['base']["use_base_model"]:
        utils.download_checkpoint(
            model_dir,
            config['train_ms']['base'],
            token=config['openi_token'],
            mirror=config['mirror'],
        )

    # 初始化学习率
    dur_resume_lr = config['train']['learning_rate']
    wd_resume_lr = config['train']['learning_rate']
    # 尝试加载时长判别器的检查点
    if net_dur_disc is not None:
        try:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=(
                    config['train']['skip_optimizer'] if "skip_optimizer" in config['train'] else True
                ),
            )
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        except:
            print("Initialize dur_disc")

    # 尝试加载生成器和判别器的检查点
    try:
        _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "G_*.pth"),
            net_g,
            optim_g,
            skip_optimizer=(
                config['train']['skip_optimizer'] if "skip_optimizer" in config['train'] else True
            ),
        )
        _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "D_*.pth"),
            net_d,
            optim_d,
            skip_optimizer=(
                 config['train']['skip_optimizer'] if "skip_optimizer" in config['train'] else True
            ),
        )
        if not optim_g.param_groups[0].get("initial_lr"):
            optim_g.param_groups[0]["initial_lr"] = g_resume_lr
        if not optim_d.param_groups[0].get("initial_lr"):
            optim_d.param_groups[0]["initial_lr"] = d_resume_lr

        epoch_str = max(epoch_str, 1)
        # global_step = (epoch_str - 1) * len(train_loader)
        # 从检查点文件名中提取全局步数
        global_step = int(
            utils.get_steps(utils.latest_checkpoint_path(model_dir, "G_*.pth"))
        )
        print(
            f"******************检测到模型存在，epoch为 {epoch_str}，gloabl step为 {global_step}*********************"
        )
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    # 尝试加载WavLM判别器的检查点
    try:
        _, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "WD_*.pth"),
            net_wd,
            optim_wd,
            skip_optimizer=(
                config['train']['skip_optimizer'] if "skip_optimizer" in config['train'] else True
            ),
        )
        if not optim_wd.param_groups[0].get("initial_lr"):
            optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
    except Exception as e:
        print(e)

    # 创建学习率调度器（指数衰减）
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config['train']['lr_decay'], last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config['train']['lr_decay'], last_epoch=epoch_str - 2
    )
    scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
        optim_wd, gamma=config['train']['lr_decay'], last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=config['train']['lr_decay'], last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    # 创建梯度缩放器（用于混合精度训练）
    scaler = GradScaler(enabled=config['train']['bf16_run'])

    # 创建WavLM损失对象（SLM损失）
    wl = WavLMLoss(
        config['model']['slm']['model'],
        net_wd,
        config['data']['sampling_rate'],
        config['model']['slm']['sr'],
    ).to(local_rank)

    # 训练循环
    for epoch in range(epoch_str, config['train']['epochs'] + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                config,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                config,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        # 每个epoch结束后更新学习率
        scheduler_g.step()
        scheduler_d.step()
        scheduler_wd.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
):
    """
    训练和评估函数 - 执行单个epoch的训练

    参数:
        rank: 全局进程rank
        local_rank: 本地GPU rank
        epoch: 当前epoch
        config: 配置字典
        nets: 模型列表 [net_g, net_d, net_dur_disc, net_wd, wl]
        optims: 优化器列表 [optim_g, optim_d, optim_dur_disc, optim_wd]
        schedulers: 学习率调度器列表
        scaler: 梯度缩放器（用于混合精度训练）
        loaders: 数据加载器列表 [train_loader, eval_loader]
        logger: 日志记录器
        writers: TensorBoard写入器列表 [writer, writer_eval]
    """
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # 设置数据采样器的epoch（确保分布式训练中数据打乱的一致性）
    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # 设置模型为训练模式
    net_g.train()
    net_d.train()
    net_wd.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    # 遍历训练数据
    for batch_idx, (
        x,              # 文本索引
        x_lengths,      # 文本长度
        spec,           # 频谱图
        spec_lengths,   # 频谱图长度
        y,              # 原始音频
        y_lengths,      # 音频长度
        speakers,       # 说话人ID
        tone,           # 音调
        language,       # 语言
        bert,           # BERT特征（中文）
        ja_bert,        # BERT特征（日文）
        en_bert,        # BERT特征（英文）
    ) in enumerate(tqdm(train_loader)):

        # 更新噪声缩放的MAS（如果启用）
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        # 将数据移动到GPU
        x, x_lengths = x.cuda(local_rank, non_blocking=True), x_lengths.cuda(
            local_rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(
            local_rank, non_blocking=True
        ), spec_lengths.cuda(local_rank, non_blocking=True)
        y, y_lengths = y.cuda(local_rank, non_blocking=True), y_lengths.cuda(
            local_rank, non_blocking=True
        )
        speakers = speakers.cuda(local_rank, non_blocking=True)
        tone = tone.cuda(local_rank, non_blocking=True)
        language = language.cuda(local_rank, non_blocking=True)
        bert = bert.cuda(local_rank, non_blocking=True)
        ja_bert = ja_bert.cuda(local_rank, non_blocking=True)
        en_bert = en_bert.cuda(local_rank, non_blocking=True)

        # 使用自动混合精度（AMP）进行前向传播
        with autocast(enabled=config['train']['bf16_run'], dtype=torch.bfloat16):
            # 生成器前向传播
            (
                y_hat,          # 生成的音频
                l_length,       # 时长损失
                attn,           # 注意力对齐
                ids_slice,      # 切片索引
                x_mask,         # 文本掩码
                z_mask,         # 隐变量掩码
                (z, z_p, m_p, logs_p, m_q, logs_q),  # 隐变量和统计参数
                (hidden_x, logw, logw_, logw_sdp),   # 隐藏状态和时长对数
                g,              # 说话人嵌入
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
            )

            # 将频谱图转换为梅尔频谱图
            mel = spec_to_mel_torch(
                spec,
                config['data']['filter_length'],
                config['data']['n_mel_channels'],
                config['data']['sampling_rate'],
                config['data']['mel_fmin'],
                config['data']['mel_fmax'],
            )
            # 对梅尔频谱图进行切片
            y_mel = commons.slice_segments(
                mel, ids_slice, config['train']['segment_size'] // config['data']['hop_length']
            )
            # 从生成的音频计算梅尔频谱图
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                config['data']['filter_length'],
                config['data']['n_mel_channels'],
                config['data']['sampling_rate'],
                config['data']['hop_length'],
                config['data']['win_length'],
                config['data']['mel_fmin'],
                config['data']['mel_fmax'],
            )

            # 对真实音频进行切片
            y = commons.slice_segments(
                y, ids_slice * config['data']['hop_length'], config['train']['segment_size']
            )  # 切片

            # ==================== 判别器训练 ====================
            # 多周期判别器前向传播（使用detach阻止梯度流向生成器）
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=config['train']['bf16_run'], dtype=torch.bfloat16):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            # 时长判别器前向传播（如果启用）
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
                y_dur_hat_r_sdp, y_dur_hat_g_sdp = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw_sdp.detach(),
                    g.detach(),
                )
                y_dur_hat_r = y_dur_hat_r + y_dur_hat_r_sdp
                y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp
                with autocast(enabled=config['train']['bf16_run'], dtype=torch.bfloat16):
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc

                # 时长判别器反向传播和优化
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

        # 多周期判别器反向传播和优化
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        if getattr(config['train'], "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        # WavLM判别器（SLM判别器）训练
        with autocast(enabled=config['train']['bf16_run'], dtype=torch.bfloat16):
            loss_slm = wl.discriminator(
                y.detach().squeeze(), y_hat.detach().squeeze()
            ).mean()

        optim_wd.zero_grad()
        scaler.scale(loss_slm).backward()
        scaler.unscale_(optim_wd)
        grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
        scaler.step(optim_wd)

        # ==================== 生成器训练 ====================
        with autocast(enabled=config['train']['bf16_run'], dtype=torch.bfloat16):
            # 多周期判别器前向传播（不detach，允许梯度流向生成器）
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
                _, y_dur_hat_g_sdp = net_dur_disc(hidden_x, x_mask, logw_, logw_sdp, g)
                y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp

            with autocast(enabled=config['train']['bf16_run'], dtype=torch.bfloat16):
                # 计算各项损失
                loss_dur = torch.sum(l_length.float())  # 时长损失
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * config['train']['c_mel']  # 梅尔频谱图损失
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config['train']['c_kl']  # KL散度损失

                loss_fm = feature_loss(fmap_r, fmap_g)  # 特征匹配损失
                loss_gen, losses_gen = generator_loss(y_d_hat_g)  # 生成器对抗损失

                loss_lm = wl(y.detach().squeeze(), y_hat.squeeze()).mean()  # WavLM损失
                loss_lm_gen = wl.generator(y_hat.squeeze())  # WavLM生成器损失

                # 总生成器损失
                loss_gen_all = (
                    loss_gen
                    + loss_fm
                    + loss_mel
                    + loss_dur
                    + loss_kl
                    + loss_lm
                    + loss_lm_gen
                )
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        # 生成器反向传播和优化
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        if getattr(config['train'], "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()  # 更新梯度缩放器

        # ==================== 日志记录和评估 ====================
        if rank == 0:
            # 定期记录训练日志
            if global_step % config['train']['log_interval'] == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                # 准备TensorBoard的标量字典
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "loss/wd/total": loss_slm,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "grad_norm_dur": grad_norm_dur,
                    "grad_norm_wd": grad_norm_wd,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                        "loss/g/lm": loss_lm,
                        "loss/g/lm_gen": loss_lm_gen,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                if net_dur_disc is not None:
                    scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})

                    scalar_dict.update(
                        {
                            "loss/dur_disc_g/{}".format(i): v
                            for i, v in enumerate(losses_dur_disc_g)
                        }
                    )
                    scalar_dict.update(
                        {
                            "loss/dur_disc_r/{}".format(i): v
                            for i, v in enumerate(losses_dur_disc_r)
                        }
                    )

                    scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
                    scalar_dict.update(
                        {
                            "loss/g/dur_gen_{}".format(i): v
                            for i, v in enumerate(losses_dur_gen)
                        }
                    )

                # 准备TensorBoard的图像字典
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            # 定期进行评估和保存检查点
            if global_step % config['train']['eval_interval'] == 0:
                evaluate(config, net_g, eval_loader, writer_eval)
                # 保存生成器检查点
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    config['train']['learning_rate'],
                    epoch,
                    os.path.join(config['train_ms']['model_dir'], "G_{}.pth".format(global_step)),
                )
                # 保存多周期判别器检查点
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    config['train']['learning_rate'],
                    epoch,
                    os.path.join(config['train_ms']['model_dir'], "D_{}.pth".format(global_step)),
                )
                # 保存WavLM判别器检查点
                utils.save_checkpoint(
                    net_wd,
                    optim_wd,
                    config['train']['learning_rate'],
                    epoch,
                    os.path.join(config['train_ms']['model_dir'], "WD_{}.pth".format(global_step)),
                )
                # 保存时长判别器检查点（如果启用）
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        config['train']['learning_rate'],
                        epoch,
                        os.path.join(config['train_ms']['model_dir'], "DUR_{}.pth".format(global_step)),
                    )
                # 清理旧检查点，只保留最新的N个
                keep_ckpts = config['train_ms']['keep_ckpts']
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=config['train_ms']['model_dir'],
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(config, generator, eval_loader, writer_eval):
    """
    评估函数 - 在验证集上评估模型性能

    参数:
        config: 配置字典
        generator: 生成器模型
        eval_loader: 评估数据加载器
        writer_eval: 评估用的TensorBoard写入器
    """
    generator.eval()  # 设置模型为评估模式
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():  # 不计算梯度
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
            en_bert,
        ) in enumerate(eval_loader):
            # 将数据移动到GPU
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            en_bert = en_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()

            # 使用SDP（随机时长预测器）和不使用SDP分别进行推理
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    en_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * config['data']['hop_length']

                # 计算梅尔频谱图
                mel = spec_to_mel_torch(
                    spec,
                    config['data']['filter_length'],
                    config['data']['n_mel_channels'],
                    config['data']['sampling_rate'],
                    config['data']['mel_fmin'],
                    config['data']['mel_fmax'],
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    config['data']['filter_length'],
                    config['data']['n_mel_channels'],
                    config['data']['sampling_rate'],
                    config['data']['hop_length'],
                    config['data']['win_length'],
                    config['data']['mel_fmin'],
                    config['data']['mel_fmax'],
                )

                # 保存生成的梅尔频谱图
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                # 保存生成的音频
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                # 保存真实的梅尔频谱图
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                # 保存真实的音频
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    # 将评估结果写入TensorBoard
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=config['data']['sampling_rate'],
    )
    generator.train()  # 恢复训练模式


