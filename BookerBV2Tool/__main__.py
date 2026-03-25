import argparse
import sys
import os
from . import __version__

def main():
    parser = argparse.ArgumentParser(prog="BookerGptTool", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()
    
    '''
    trans_parser = subparsers.add_parser("trans-yaml", help="translate YAML files")
    trans_parser.add_argument("fname", help="yaml file name of dir")
    trans_parser.add_argument("-p", "--prompt", default=DFT_TRANS_PROMPT, help="prompt for trans")
    trans_parser.add_argument("-l", "--limit", type=int, default=3000, help="max token limit")
    trans_parser.add_argument("-t", "--threads", type=int, default=8, help="thread num")
    trans_parser.set_defaults(func=trans_yaml_handle)
    '''

    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()