import argparse
import sys
import os
from . import __version__
from .slice import slice_handle
from .resample import resample_handle
from .mark import mark_handle
from .mklist import mklist_handle
from .preproc import preprocess_handle
from .bert_gen import bert_gen_handle

def main():
    parser = argparse.ArgumentParser(prog="BookerBV2Tool", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.add_argument('-sv', '--sencevoice', type=str,default=os.environ.get('SENCEVOICE_MODEL_PATH', '') , help='SenceVoice model path')
    parser.add_argument('-eb', '--enlish-bert', type=str,default='microsoft/deberta-v3-large' , help='SenceVoice model path')
    parser.add_argument('-cb', '--chinese-bert', type=str,default='hfl/chinese-roberta-wwm-ext-large' , help='SenceVoice model path')
    parser.add_argument('-jb', '--japanese-bert', type=str,default='ku-nlp/deberta-v2-large-japanese-char-wwm' , help='SenceVoice model path')
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()
    
    slice_parser = subparsers.add_parser("slice", help="slice audio")
    slice_parser.add_argument('audio', type=str, help='The audio to be sliced')
    slice_parser.add_argument('-o', '--out', type=str, help='Output directory of the sliced audio clips')
    slice_parser.add_argument('-dt', '--db_thresh', type=float, required=False, default=-40,
                        help='The dB threshold for silence detection')
    slice_parser.add_argument('-ml', '--min_length', type=int, required=False, default=5000,
                        help='The minimum milliseconds required for each sliced audio clip')
    slice_parser.add_argument('-mi', '--min_interval', type=int, required=False, default=300,
                        help='The minimum milliseconds for a silence part to be sliced')
    slice_parser.add_argument('-hs', '--hop_size', type=int, required=False, default=10,
                        help='Frame length in milliseconds')
    slice_parser.add_argument('-mk', '--max_sil_kept', type=int, required=False, default=500,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    slice_parser.set_defaults(func=slice_handle)

    resample_parser = subparsers.add_parser("resample", help="resample audio")
    resample_parser.add_argument('audio', type=str, help='The audio to be sliced')
    resample_parser.add_argument('--sr', type=int, default=44100, help="sampling rate")
    resample_parser.add_argument('-t', '--threads', type=int, default=8, help="num of threads")
    resample_parser.add_argument('-o', '--out', type=str, default='.', help="output dir")
    resample_parser.set_defaults(func=resample_handle)

    mark_parser = subparsers.add_parser("mark", help="mark audio")
    mark_parser.add_argument('audio', type=str, help='The audio to be sliced')
    mark_parser.set_defaults(func=mark_handle)

    mklist_parser = subparsers.add_parser("mklist", help="mark audio")
    mklist_parser.add_argument('dir', type=str, help='The audio dir')
    mklist_parser.add_argument('-r', "--role", default='wizard', type=str, help='role name')
    mklist_parser.add_argument('-l', '--lang', type=str,default='ZH', choices=['EN', 'ZH', 'JP'] , help='language')
    mklist_parser.set_defaults(func=mklist_handle)

    preproc_parser = subparsers.add_parser("preproc", help="preproccess speaker list")
    preproc_parser.add_argument('transcription_path', type=str, help='transcription path')
    preproc_parser.add_argument('-tp', "--train-path", default='train_list.json', type=str, help='train path')
    preproc_parser.add_argument('-vp', "--val-path", default='val_list.json', type=str, help='val path')
    preproc_parser.add_argument('-cp', "--config-path", default='config.json', type=str, help='config path')
    preproc_parser.add_argument('-vl', '--val-per-lang', type=int,default=4, help='val per lang')
    preproc_parser.add_argument('-mv', '--max-val-total', type=int,default=12, help='max val total')
    preproc_parser.set_defaults(func=preprocess_handle)

    bert_gen_parser = subparsers.add_parser("bert-gen", help="gen bert vectors")
    bert_gen_parser.add_argument(
        "-c", "--config", type=str, default='config.json'
    )
    bert_gen_parser.add_argument(
        "--num_processes", type=int, default=8
    )
    bert_gen_parser.set_defaults(func=bert_gen_handle)

    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()