import numpy as np

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation, GRU, TimeDistributed, Embedding, RepeatVector
from keras.callbacks import Callback
from nltk.corpus import brown


maxlen = 20
minlen = 3
hidden_dim = 128
embedding_dim = 100
batch_size = 50
num_epochs = 1


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


def build_model(max_len, num_words, embedding_dim, hidden_dim, embeddings=None):
    model = Sequential()
    model.add(Embedding(input_dim=num_words+1,
                        input_length=max_len,
                        output_dim=embedding_dim,
                        mask_zero=True,
                        weights=embeddings))
    model.add(GRU(output_dim=hidden_dim,
                  input_length=max_len))
    model.add(RepeatVector(max_len))
    model.add(GRU(output_dim=hidden_dim,
                  input_length=max_len,
                  return_sequences=True))
    model.add(TimeDistributed(Dense(num_words)))
    model.add(Activation("softmax"))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_text(model, seed_words, temperature, word2idx, idx2word, N):
    # keep this here for reference
    model.reset_states()
    for seed_word in seed_words:
        seed_word_idx = word2idx[seed_word]
        x = np.zeros(shape=model.input_shape)
        x[0, 0] = seed_word_idx
        probs = model.predict(x, verbose=0)

    # now we start generating text
    probs = probs[0, 0, :]
    next_word_idx = sample(probs, temperature)
    generated_text_idx = [next_word_idx]
    generated_text = [idx2word[next_word_idx]]
    for i in xrange(N - 1):
        last_word_idx = generated_text_idx[-1]
        x = np.zeros(shape=model.input_shape)
        x[0, 0] = last_word_idx
        probs = model.predict(x, verbose=0)
        probs = probs[0, 0, :]
        next_word_idx = sample(probs, temperature)
        generated_text_idx.append(next_word_idx)
        generated_text.append(idx2word[next_word_idx])
    return generated_text


def generate_text2(model, inputs, idx2word):
    model.reset_states()
    # don't sample, just take the prediction with the highest probability
    probs = model.predict(np.matrix(inputs), verbose=0)
    generated_text = []
    print probs
    print probs.shape
    for p in probs[0]:
        idx = np.argmax(p) # -1 because we
        generated_text.append(idx2word[idx])
    return generated_text


class ResetStates(Callback):
    # is this callback necessary?
    def on_epoch_begin(self, epoch, logs={}):
        self.model.reset_states()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


brownSents = brown.sents(categories=['hobbies', 'news', 'fiction', 'adventure', 'reviews', 'editorials', 'romance'])
print('Finished loading sentences from brown corpus')
gloveDict = load_word_embeddings('glove.6B.100d.txt')
print('Finished loading word embeddings')
word2idx = {}
sentences = []
unique_words = set([])
count = 1
raw_sentences = []
print('Vectorizing sentences')
for s in brownSents:
    skip = False
    if len(s) <= maxlen and len(s) >= minlen:
        # only take sentences that aren't too long nor too short
        lowered = [t.lower() for t in s]
        words_to_add = []
        for w in lowered:
            if w not in unique_words:
                try:
                    # see if we have a word embedding for this word, if not we skip
                    gloveDict[w]
                    words_to_add.append(w)
                except KeyError as e:
                    skip = True
                    break
        if not skip:
            # all words in the sentence are in the embeddings, so we use this sentence
            # vectorize it
            raw_sentences.append(' '.join(lowered))
            for w in set(words_to_add):
                unique_words.add(w)
                word2idx[w] = count
                count += 1
            sentVec = [word2idx[t] for t in lowered]
            sentences.append(sentVec)

# we start at count = 1 because the input sequence is padded with 0's until a vector of 20 is reached
# cannot use the index 0 for any words in the vocab
num_words = len(word2idx)
assert count-1 == num_words
embeddingsMatrix = np.zeros([num_words+1, embedding_dim])
# create embeddings matrix corresponding to the word indices

print('Creating weights for embeddings layer')
for word, idx in word2idx.items():
    embeddingsMatrix[idx, :] = gloveDict[word]
gloveDict = None

print('Building model')
model = build_model(maxlen, num_words, embedding_dim, hidden_dim, embeddings=[embeddingsMatrix])

sentences_target = np.zeros(shape=(len(sentences), maxlen, num_words), dtype=bool)
print('Creating target data')
for ns, sent in enumerate(sentences):
    for nt, t in enumerate(sent):
        sentences_target[ns, nt, t-1] = 1
        # -1 because we don't use padding for output but index 0 is padding in input

# train on the first 10000 data points
X = pad_sequences(sentences, maxlen=maxlen, padding='post')[0:10000]
Y = sentences_target[0:10000]
model.fit(X, Y, batch_size=batch_size, nb_epoch=num_epochs)
model_name = 'paraphrase.brown.h5'
model.save_weights(model_name, overwrite=True)

seed_str = list(X[0])
model.load_weights(model_name)
generated_text = generate_text2(model, seed_str, list(unique_words))
print('Original sentence:\n{}'.format(raw_sentences[0]))
print('Paraphrased:\n{}'.format(' '.join(generated_text)))
