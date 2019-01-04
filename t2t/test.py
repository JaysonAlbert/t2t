import numpy as np
import tqdm
from nltk.corpus import wordnet
from sklearn.metrics import f1_score

from t2t.embedding_server import EmbeddingServer
from t2t.new_semcor import new_semcor
from t2t.util import disambiguation


def get_sentence_str(input):
    return ' '.join(np.array(input)[:,0])


pos_map = {
    'ADJ': 'a',
    'VERB': 'v',
    'NOUN': 'n',
    'ADV': 'r',
}


def predict(instance, sentence):
    sent_str = get_sentence_str(sentence)
    lemma = np.array(instance)[:,1]
    pos = np.array(instance)[:, 2]

    preds = []
    for l,p in zip(lemma, pos):
        p = pos_map[p]
        pred = disambiguation(sent_str, l, embedding_server.embedding, p)
        preds.append(pred)

    return preds


def evaluate(instance, sentence, golds):
    preds = predict(instance, sentence)
    res = []
    for p, g in zip(preds, golds):
        g = [wordnet.lemma_from_key(i).synset() for i in g[1]]
        res.append(p in g)
    return res


def precision(res):
    return sum(res) * 1.0 / len(res)


def f1(res):
    return f1_score(np.ones(len(res)),res)


embedding_server = EmbeddingServer()


res = []
for index, ((instance, sentence), golds) in  tqdm.tqdm(enumerate(zip(new_semcor.both(name='senseval2'), new_semcor.golds('senseval2')))):
    res.extend(evaluate(instance, sentence, golds))
    print("precision for {}th examples: {}, f1 score: {}".format(index, precision(res), f1(res)))
