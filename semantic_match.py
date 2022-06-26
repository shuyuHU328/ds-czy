from gensim.models import KeyedVectors
import jieba


def get_dict(dict, target_str, model):
    leg = jieba.lcut(target_str)
    vec = model.get_mean_vector(leg)
    dict[target_str] = vec


if __name__ == '__main__':
    model = KeyedVectors.load('test.bin')
    file = open('standardMeta.txt', 'r', encoding='utf-8')
    myDict = {}
    for line in file:
        get_dict(myDict, line.strip(), model)
    print(len(myDict))
    # print(model.n_similarity(leg1, leg2))
