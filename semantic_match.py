from gensim.models import KeyedVectors
import jieba
import numpy as np


def get_dict(dict, target_str, model):
    try:
        vec = model[target_str]
        dict[target_str] = vec
    except:
        leg = jieba.lcut_for_search(target_str)
        vec = model.get_mean_vector(leg)
        dict[target_str] = vec


def calculate_similarity(vec1, vec2):
    num = float(np.dot(vec1, vec2))
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


if __name__ == '__main__':
    model = KeyedVectors.load('test.bin')
    file = open('standardMeta.txt', 'r', encoding='utf-8')
    myDict = {}
    for line in file:
        get_dict(myDict, line.strip(), model)
    target_str=input().strip()
    try:
        match_vec = model[target_str]
    except:
        leg = jieba.lcut_for_search(target_str)
        match_vec = model.get_mean_vector(target_str)
    finally:
        array = []
        for k, v in myDict.items():
            array.append((k, calculate_similarity(v.tolist(), match_vec.tolist())))
        array = sorted(array, key=lambda x: x[1], reverse=True)
        print(jieba.lcut_for_search('QQ号码'))
    # print(model.n_similarity(leg1, leg2))
