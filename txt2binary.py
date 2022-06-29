from gensim.models import KeyedVectors
from tqdm import tqdm


def is_num(x):
    try:
        float(x)
    except:
        return False
    return True


def refine_embedding(inpath='tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt',
                     outpath='refine.txt'):
    with open(outpath, 'w', encoding='utf8') as fout:
        with open(inpath, 'r', encoding='utf8') as fin:
            fout.write('1956951\t100\n')  # 先wc -l统计过滤之后的行数
            for index, line in tqdm(enumerate(fin)):
                if index == 0 or is_num(line.split(' ')[0]):
                    continue
                fout.write(line)


if __name__ == '__main__':
    wv_from_text = KeyedVectors.load_word2vec_format("tencent-ailab-embedding-zh-d100-v0.2.0.txt", binary=False,
                                                     encoding='utf8')
    wv_from_text.save('TC.bin')
