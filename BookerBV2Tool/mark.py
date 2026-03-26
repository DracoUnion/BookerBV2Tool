import os
from os import path
import traceback
from .sencevoice import sencevoice

def mark_file_safe(args):
    try:
        mark_file(args)
    except:
        traceback.print_exc()

def mark_file(args):
    res = sencevoice(args)
    txt = ' '.join([r['text'] for r in res])
    ofname = args.audio + '.txt'
    print(ofname)
    open(ofname, 'w', encoding='utf8').write(txt)

def mark_dir(args):
    dir = args.audio
    fnames = os.listdir(dir)
    for f in fnames:
        args.audio = path.join(dir, f)
        mark_file_safe(args)

def mark_handle(args):
    if path.isfile(args.audio):
        mark_file(args)
    else:
        mark_dir(args)