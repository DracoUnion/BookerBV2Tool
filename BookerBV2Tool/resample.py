import librosa
import soundfile
import os
from os import path
from concurrent.futures import ThreadPoolExecutor
import copy
import traceback

def resample_file_safe(args):
    try:
        resample_file(args)
    except:
        traceback.print_exc()

def resample_file(args):
    if not args.fname.lower().endswith('.wav'):
        print('请提供 WAV 文件')
        return
    os.makedirs(args.out, exist_ok=True)
    ofname = path.join(args.out, path.basename(args.fname))
    wav, sr = librosa.load(args.fname, sr=args.sr)
    soundfile.write(ofname, wav, sr)

def resample_dir(args):
    pool = ThreadPoolExecutor(args.threads)
    hdls = []

    dir = args.fname
    fnames = os.listdir(dir)
    for f in fnames:
        args = copy.deepcopy(args)
        args.fname = path.join(dir, f)
        h = pool.submit(resample_file_safe, args)
        hdls.append(h)
        if len(hdls) > args.threads:
            for h in hdls: h.result()
            hdls = []

    for h in hdls: 
        h.result()

def resample_handle(args):
    if path.isfile(args.fname):
        resample_file(args)
    else:
        resample_dir(args)