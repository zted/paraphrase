import numpy as np
from nltk import word_tokenize
import json

# max_examples = 10000000000
# maxlen = 20
# minlen = 5
# hidden_dim = 128
# embedding_dim = 100
# someFile = '/export/home1/NoCsBack/hci/ted/translate_datadir/giga-fren.release2.fixed.en'


def load_word_embeddings(fname):
    # loads word embeddings into a dictionary
    embDict = {}
    with open(fname, 'r') as f:
        for line in f:
            splits = line.strip().split(' ')
            word = splits[0]
            wordvec = np.array(map(float, splits[1:]))
            embDict[word] = wordvec
    return embDict


def load_vqa_questions(filename, minlen, maxlen, word2idx, max_examples):
    print('Begin loading from {}'.format(filename))
    raw_sentences = []
    vectorized_sentences = []
    with open(filename, 'r') as json_data:
        d = json.load(json_data)
        questions = []
        for q in d['questions']:
            questions.append(q['question'])
        for q in set(questions):
            # remove duplicate questions by converting list of strings to set
            skip = False
            stripped = q.strip()
            try:
                tokens = word_tokenize(stripped)
            except UnicodeDecodeError:
                continue
            if len(tokens) <= maxlen and len(tokens) >= minlen:
                # only take sentences that aren't too long nor too short
                lowered = [t.lower() for t in tokens]
                sentVec = []
                for w in lowered:
                    try:
                        # see if we have a word embedding for this word, if not we skip
                        idx = word2idx[w]
                        sentVec.append(idx)
                    except KeyError:
                        skip = True
                        break
                if not skip:
                    # all words in the sentence are in the embeddings, so we use this sentence
                    # vectorize it
                    raw_sentences.append(' '.join(lowered))
                    vectorized_sentences.append(pad_instance(sentVec, maxlen))
                    if len(vectorized_sentences) >= max_examples:
                        # we have enough
                        break
    print('Finished loading {} sentences from {}'.format(len(vectorized_sentences), filename))
    return raw_sentences, vectorized_sentences


def load_google_sentences(filename, minlen, maxlen, word2idx, max_examples):
    print('Begin loading from {}'.format(filename))
    raw_sentences = []
    vectorized_sentences = []
    with open(filename, 'r') as f:
        for q in f:
            skip = False
            stripped = q.strip()
            try:
                tokens = word_tokenize(stripped)
            except UnicodeDecodeError:
                continue
            if len(tokens) <= maxlen and len(tokens) >= minlen:
                # only take sentences that aren't too long nor too short
                lowered = [t.lower() for t in tokens]
                sentVec = []
                for w in lowered:
                    try:
                        # see if we have a word embedding for this word, if not we skip
                        idx = word2idx[w]
                        sentVec.append(idx)
                    except KeyError:
                        skip = True
                        break
                if not skip:
                    # all words in the sentence are in the embeddings, so we use this sentence
                    # vectorize it
                    raw_sentences.append(' '.join(lowered))
                    vectorized_sentences.append(pad_instance(sentVec, maxlen))
                    if len(vectorized_sentences) >= max_examples:
                        # we have enough
                        break
    print('Finished loading {} sentences from {}'.format(len(vectorized_sentences), filename))
    return raw_sentences, vectorized_sentences


def load_brown_questions(minlen, maxlen, word2idx, max_examples):
    from nltk.corpus import brown
    raw_sentences = []
    sentences = brown.sents(categories=['hobbies', 'news', 'fiction', 'adventure', 'reviews', 'editorials', 'romance'])
    vectorized_sentences = []
    for tokens in sentences:
        skip = False
        if len(tokens) <= maxlen and len(tokens) >= minlen:
            # only take sentences that aren't too long nor too short
            lowered = [t.lower() for t in tokens]
            sentVec = []
            for w in lowered:
                try:
                    # see if we have a word embedding for this word, if not we skip
                    idx = word2idx[w]
                    sentVec.append(idx)
                except KeyError:
                    skip = True
                    break
            if not skip:
                # all words in the sentence are in the embeddings, so we use this sentence
                # vectorize it
                raw_sentences.append(' '.join(lowered))
                vectorized_sentences.append(pad_instance(sentVec, maxlen))
                vectorized_sentences.append(pad_instance(sentVec, maxlen))
                if len(vectorized_sentences) >= max_examples:
                    # we have enough
                    break
    print('Finished loading {} sentences from brown corpus'.format(len(vectorized_sentences)))
    return raw_sentences, vectorized_sentences


def pad_instance(arr, maxlen, pad_symbol=0):
    someLen = len(arr)
    assert someLen <= maxlen
    diff = maxlen - someLen
    padded = arr + [pad_symbol] * diff
    return padded


def pad_sequences(arrays, maxlen, pad_symbol=0):
    newArray = []
    for a in arrays:
        newArray.append(pad_instance(a, maxlen, pad_symbol=pad_symbol))
    return newArray



j1 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_test2015_questions.json'
j2 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_test-dev2015_questions.json'
j3 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_train2014_questions.json'
j4 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_val2014_questions.json'