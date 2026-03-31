import json
from os import path
import os
from collections import defaultdict
from random import shuffle
from .text.cleaner import clean_text, get_model_name_by_lang
from .text import chinese, japanese, english

language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

DIR = path.dirname(__file__)
RAW_CONFIG_PATH = path.join(DIR, 'config.json')




def preprocess_handle(args):
    transcription_path = args.transcription_path
    train_path = args.train_path
    val_path = args.val_path
    config_path = args.config_path
    val_per_lang = args.val_per_lang
    max_val_total = args.max_val_total

    cleaned_path = transcription_path + ".cleaned"

    if not path.isfile(cleaned_path):
        with open(transcription_path, "r", encoding="utf-8") as trans_file:
            lines = json.loads(trans_file.read())
        if len(lines) == 0:
            print(f'{transcription_path} 为空')
            return
        cleaned = []
        for line in lines:
            norm_sub, phones, tones, word2ph = clean_text(
                get_model_name_by_lang(line['lang'], args),
                line['sub'], line['lang'],
            )
            cleaned.append(line | {
                'norm_sub': norm_sub,
                "phones": phones,
                "tones": tones,
                "word2ph": word2ph,
            })
        with open(cleaned_path, "w", encoding="utf-8") as of:
            of.write(json.dumps(cleaned, ensure_ascii=False))
    else:
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned = json.loads(f.read())

    transcription_path = cleaned_path
    role_file_map = defaultdict(list)
    role_id_map = {}
    current_sid = 0
    audioPaths = set()
    countSame = 0
    countNotFound = 0
    print(cleaned)
    for it in cleaned:
        if it['file'] in audioPaths:
            # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
            print(f"重复音频文本：{it['file']}")
            countSame += 1
            continue
        if not path.isfile(it['file']):
            # 过滤数据集错误：不存在对应音频
            print(f"没有找到对应的音频：{it['file']}")
            countNotFound += 1
            continue
        audioPaths.add(it['file'])
        role_file_map[it['role']].append(it['file'])
        if role not in role_id_map.keys():
            role_id_map[it['role']] = current_sid
            current_sid += 1
    print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    train_list = []
    val_list = []

    for role, files in role_file_map.items():
        shuffle(files)
        val_list += files[:val_per_lang]
        train_list += files[val_per_lang:]

    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_list, ensure_ascii=False))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(val_list, ensure_ascii=False))

    json_config = json.load(open(RAW_CONFIG_PATH, encoding="utf-8"))
    json_config["data"]["spk2id"] = role_id_map
    json_config["data"]["n_speakers"] = len(role_id_map)
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