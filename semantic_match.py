from gensim.models import KeyedVectors

if __name__ == '__main__':
    wv_from_text=KeyedVectors.load_word2vec_format('TC_Model.bin',binary=True)
