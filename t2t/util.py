import numpy as np
from nltk.corpus import wordnet


def similarity(source, target):
    return np.inner(source, target)


def disambiguation(context_sentence, ambiguous_word, embedding_fn, pos=None, synsets=None):
    if not synsets:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        pos = ['a', 's'] if pos == 'a' else [pos]
        synsets = [ss for ss in synsets if str(ss.pos()) in pos]

    if not synsets:
        return wordnet.synsets(ambiguous_word)[0]

    score = [(similarity(embedding_fn(context_sentence), embedding_fn(ss.definition())), ss) for ss in synsets]

    _, sense = max(
        score
    )

    return sense
