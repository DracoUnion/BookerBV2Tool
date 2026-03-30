import json
import torch
from multiprocessing import Pool
import commons
from os import path
import utils
from tqdm import tqdm
from text import cleaned_text_to_sequence, get_bert_feature, get_model_name_by_lang
import argparse
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .text.symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids

name_model_map = {}
name_tok_map = {}

def get_bert_feature(
    model_name,
    text,
    word2ph,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    style_text=None,
    style_weight=0.7,
):
    if model_name not in name_model_map:
        model = AutoModelForMaskedLM \
            .from_pretrained(model_name, trust_remote_code=True).to(device)
        name_model_map[model_name] = model
    else:
        model = name_model_map[model_name]
    if model_name not in name_tok_map:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        name_tok_map[model_name] = tokenizer
    else:
        tokenizer = name_tok_map[model_name]
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

def get_model_name_by_lang(lang, args):
    lang_map = {
        'ZH': args.chinese_bert,
        'EN': args.enlish_bert,
        'JP': args.japanese_bert,
    }
    return lang_map[lang] if lang in lang_map else ''


def process_line(x):
    line, add_blank = x
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_vec_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")
    if not path.isfile(bert_vec_path):
        bert = get_bert_feature(
            get_model_name_by_lang(language_str),
            text, word2ph, language_str, device,
        )
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_vec_path)



class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams_from_file(config_path):
    # print("config_path: ", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def bert_gen_handle(args):
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        num_processes = args.num_processes
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_line, zip(lines, add_blank)),
                total=len(lines),
            ):
                # 这里是缩进的代码块，表示循环体
                pass  # 使用pass语句作为占位符

    print(f"bert生成完毕!, 共有{len(lines)}个bert.pt生成!")


