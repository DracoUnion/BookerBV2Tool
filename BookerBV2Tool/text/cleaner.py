from . import chinese, japanese, english, cleaned_text_to_sequence
from .symbols import *
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

_symbol_to_id = {s: i for i, s in enumerate(symbols)}




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

def clean_text_bert(model_name, text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(model_name, norm_text)
    bert = get_bert_feature(model_name, norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(model_name, text, language):
    norm_text, phones, tones, word2ph = clean_text(model_name, text, language)
    return cleaned_text_to_sequence(phones, tones, language)


def clean_text(model_name, text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(model_name, norm_text)
    return norm_text, phones, tones, word2ph

if __name__ == "__main__":
    pass
