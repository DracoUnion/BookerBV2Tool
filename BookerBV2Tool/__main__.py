import argparse
import sys
import os
from . import __version__
from .slice import slice

def main():
    parser = argparse.ArgumentParser(prog="BookerGptTool", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
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
    slice_parser.add_argument('-ms', '--max_sil_kept', type=int, required=False, default=500,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    slice_parser.set_defaults(func=slice)


    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()