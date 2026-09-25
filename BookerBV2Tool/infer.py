#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""文本转语音推理（当前版本 2.3）。

移植自上游 ``Bert-VITS2/infer.py``，适配本仓库的包内相对导入，并移除
``oldVersion`` 旧版本兼容（本仓库仅训练/使用 2.3 版本模型）。
"""

import json

import torch

from . import commons
from . import utils
from .models import SynthesizerTrn
from .text.cleaner import (
    clean_text,
    cleaned_text_to_sequence,
    get_bert_feature,
    get_model_name_by_lang,
)
from .text.symbols import symbols

# 当前版本信息
latest_version = "2.3"


def _device():
    """选择推理设备：有 CUDA 用 GPU，否则 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_net_g(model_path: str, version: str, device: torch.device, hps):
    if version != latest_version:
        raise ValueError(
            f"本仓库仅支持当前版本 {latest_version} 的模型，收到版本号 {version}"
        )
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g


def get_text(text, language_str, hps, device, args, style_text=None, style_weight=0.7):
    """文本转音素序列 + 对应语言 BERT 特征。"""
    model_name = get_model_name_by_lang(language_str, args)
    norm_text, phone, tone, word2ph = clean_text(model_name, text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert_feature(
        model_name, norm_text, word2ph, device, style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "JP":
        bert = torch.randn(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    args,
    skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        args,
        style_text=style_text,
        style_weight=style_weight,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
            ja_bert,
            en_bert,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio


def infer_handle(args):
    """infer 子命令入口：加载模型，文本转语音并写出 WAV。"""
    device = _device()
    config = json.loads(open(args.config, encoding="utf-8").read())
    hps = utils.HParams(**config)

    # 未指定说话人时取配置中的第一个说话人
    sid = args.sid if args.sid is not None else list(hps.data.spk2id.keys())[0]

    net_g = get_net_g(args.model, latest_version, device, hps)
    audio = infer(
        args.text,
        args.sdp_ratio,
        args.noise_scale,
        args.noise_scale_w,
        args.length_scale,
        sid,
        args.language,
        hps,
        net_g,
        device,
        args,
    )
    import soundfile as sf

    sf.write(args.out, audio, hps.data.sampling_rate)
    print(f"Saved synthesized audio to {args.out}")