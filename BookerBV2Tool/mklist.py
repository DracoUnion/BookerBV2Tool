from os import path
import os

def mklist_handle(args):
    if not path.isdir(args.dir):
        print('请提供存有 WAV 文件的目录')
        return
    audios = [f for f in os.listdir(args.dir) if f.endswith('.wav')]
    if not audios:
        print('请提供存有 WAV 文件的目录')
        return
    res = []
    for f in audios:
        print(f)
        sub_fname = path.join(args.dir, f + '.txt')
        if not path.isfile(sub_fname):
            print(f'{sub_fname} 不存在')
            continue
        sub = open(sub_fname, encoding='utf8').read().strip()
        if not sub:
            print(f'{sub_fname} 为空')
            continue
        res.append(f'{f}|{args.role}|{args.lang}|{sub}')

    if not res:
        print('未找到任何标注数据')
        return
    res = '\n'.join(res)
    ofname = path.join(args.dir, 'speaker.list')
    print(ofname)
    open(ofname, 'w', encoding='utf8').write(res)