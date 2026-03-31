import traceback
import json
import torch
from concurrent.futures import ProcessPoolExecutor
from . import commons
from os import path
from . import utils
from tqdm import tqdm
from .text.cleaner import cleaned_text_to_sequence, get_bert_feature, get_model_name_by_lang
import argparse
import torch.multiprocessing as mp

def process_line_safe(line, add_blank, args):
    try:
        process_line(line, add_blank, args)
    except:
        traceback.print_exc()

def process_line(line, add_blank, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    phone, tone, lang_ids = cleaned_text_to_sequence(line['phones'], line['tones'], line['lang'])
    word2ph = line['word2ph']
    file = line['file']
    sub = line['norm_sub']
    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        lang_ids = commons.intersperse(lang_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_vec_path = file.lower().replace(".wav", "_bert.pt")
    if not path.isfile(bert_vec_path):
        bert = get_bert_feature(
            get_model_name_by_lang(line['lang'], args),
            sub, word2ph, device,
        )
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_vec_path)


def bert_gen_handle(args):
    config_path = args.config
    config = json.loads(open(config_path, encoding='utf8').read())
    lines = []
    with open(config['data']['training_files'], encoding="utf-8") as f:
        lines.extend(json.loads(f.read()))

    with open(config['data']['validation_files'], encoding="utf-8") as f:
        lines.extend(json.loads(f.read()))
    add_blank = config['data']['add_blank']
    if len(lines) == 0:
        print(f'未找到训练或测试文件')
        return

    pool = ProcessPoolExecutor(args.num_processes)
    hdls = []
    for line in lines:
        h = pool.submit(process_line_safe, line, add_blank, args)
        hdls.append(h)
        if len(hdls) > args.num_processes:
            for h in hdls: h.result()
            hdls = []
    for h in hdls: 
        h.result()

    print(f"bert生成完毕!, 共有{len(lines)}个bert.pt生成!")


