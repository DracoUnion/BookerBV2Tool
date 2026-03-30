import torch
from multiprocessing import Pool
import commons
from os import path
import utils
from tqdm import tqdm
from text import check_bert_models, cleaned_text_to_sequence, get_bert_feature, get_model_name_by_lang
import argparse
import torch.multiprocessing as mp


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

def bert_gen_handle(args):
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    check_bert_models()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args, _ = parser.parse_known_args()
    