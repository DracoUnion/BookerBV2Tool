import json
from os import path
import os
from collections import defaultdict
from random import shuffle
from .text.cleaner import clean_text

DIR = path.dirname(__file__)
RAW_CONFIG_PATH = path.join(DIR, 'config.json')

def preprocess(args):
    transcription_path = args.transcription_path
    train_path = args.train_path
    val_path = args.val_path
    config_path = args.config_path
    val_per_lang = args.val_per_lang
    max_val_total = args.max_val_total

    cleaned_path = transcription_path + ".cleaned"

    if not path.isfile(cleaned_path):
        with open(transcription_path, "r", encoding="utf-8") as trans_file:
            lines = trans_file.readlines()
        if lines == 0:
            print(f'{transcription_path} 为空')
            return
        cleaned = []
        for line in lines:
            utt, spk, language, text = line.strip().split("|")
            norm_text, phones, tones, word2ph = clean_text(
                text, language
            )
            cleaned.append("{}|{}|{}|{}|{}|{}|{}"
                .format(
                    utt,
                    spk,
                    language,
                    norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                ))
        with open(cleaned_path, "w", encoding="utf-8") as of:
            of.write('\n'.join(cleaned))
    else:
        with open(cleaned_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # utt, spk, language, text, phones, tones, word2ph
        cleaned = [line.strip().split("|") for l in lines]

    transcription_path = cleaned_path
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0
    audioPaths = set()
    countSame = 0
    countNotFound = 0
    for utt, spk, language, text, phones, tones, word2ph in cleaned:
        if utt in audioPaths:
            # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
            print(f"重复音频文本：{line}")
            countSame += 1
            continue
        if not path.isfile(utt):
            # 过滤数据集错误：不存在对应音频
            print(f"没有找到对应的音频：{utt}")
            countNotFound += 1
            continue
        audioPaths.add(utt)
        spk_utt_map[language].append(line)
        if spk not in spk_id_map.keys():
            spk_id_map[spk] = current_sid
            current_sid += 1
    print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_lang]
        train_list += utts[val_per_lang:]

    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    json_config = json.load(open(RAW_CONFIG_PATH, encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    # 新增写入：写入训练版本、数据集路径
    json_config["version"] = '2.3'
    json_config["data"]["training_files"] = path.normpath(train_path).replace(
        "\\", "/"
    )
    json_config["data"]["validation_files"] = path.normpath(val_path).replace(
        "\\", "/"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    print("训练集和验证集生成完成！")