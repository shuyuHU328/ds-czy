from gensim.models import KeyedVectors
import jieba
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

def get_dict(dict, str, model):
    try:
        vec = model[str]
        dict[str] = vec
    except:
        leg = jieba.lcut_for_search(str)
        vec = model.get_mean_vector(leg)
        dict[str] = vec


def calculate_similarity(vec1, vec2):
    num = float(np.dot(vec1, vec2))
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


if __name__ == '__main__':
    model = KeyedVectors.load('TC.bin')
    file = open('standardMeta.txt', 'r', encoding='utf-8')
    myDict = {}
    for line in file:
        get_dict(myDict, line.strip(), model)
    while True:
        target_str = input().strip().split()
        topk = int(target_str[1])
        target_str = target_str[0]
        try:
            match_vec = model[target_str]
        except:
            leg = jieba.lcut(target_str, use_paddle=True)
            match_vec = model.get_mean_vector(target_str)
        finally:
            array = []
            for k, v in myDict.items():
                array.append((k, calculate_similarity(v.tolist(), match_vec.tolist())))
            array = sorted(array, key=lambda x: x[1], reverse=True)[:topk]
            data = dict(array)
            output = {target_str: data}
            json.dump(output, open(target_str + '.json', 'w', encoding='utf-8'), indent=4,
                      ensure_ascii=False)
            rawWordVec = []
            word2ind = []
            for key, value in myDict.items():
                rawWordVec.append(value)
                word2ind.append(key)
            rawWordVec = np.array(rawWordVec)
            X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
            # 可视化
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            ax.set_facecolor('white')
            ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')
            # 设置中文字体 否则乱码
            words = []
            for i in range(0, topk):
                words.append(array[i][0])
            # 设置中文字体 否则乱码
            zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf', size=16)
            for w in words:
                if w in word2ind:
                    ind = word2ind.index(w)
                    xy = X_reduced[ind]
                    plt.plot(xy[0], xy[1], '.', alpha=1, color='orange', markersize=10)
                    plt.text(xy[0], xy[1], w, fontproperties=zhfont1, alpha=1, color='red')
            plt.show()