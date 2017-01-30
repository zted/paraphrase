import numpy as np

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Embedding, RepeatVector
from keras.callbacks import Callback
from nltk import word_tokenize
from nltk.corpus import brown


def load_word_embeddings(fname):
    idx = {}
    with open(fname, 'r') as f:
        for line in f:
            splits = line.strip().split(' ')
            word = splits[0]
            wordvec = np.array(map(float, splits[1:]))
            idx[word] = wordvec
    return idx


brownSents = brown.sents(categories=['hobbies', 'news', 'fiction', 'adventure', 'reviews', 'editorials', 'romance'])

gloveDict = load_word_embeddings('glove.6B.100d.txt')
maxlen = 20
sentences = []
words = set([])
word2idx = {}
hidden_dim = 128
embedding_dim = 100
batch_size = 50
num_epochs = 20
count = 0

for s in brownSents:
    skip = False
    if len(s) <= maxlen:
        lowered = [t.lower() for t in s]
        for t in lowered:
            if t not in words:
                try:
                    gloveDict[t]
                    words.add(t)
                    word2idx[t] = count
                    count += 1
                except KeyError as e:
                    skip = True
        if not skip:
            sentVec = [word2idx[t] for t in lowered]
            sentences.append(sentVec)


num_words = len(word2idx)
embeddingsMatrix = np.zeros((len(words), embedding_dim))


for k, idx in word2idx.items():
    embeddingsMatrix[idx, :] = word2idx[k]

gloveDict = None


def get_model(max_len, num_words, embedding_dim, hidden_dim, batch_size, embeddings=None):
    model = Sequential()
    model.add(Embedding(input_dim=num_words,
                        input_length=max_len,
                        batch_input_shape=[batch_size, max_len],
                        mask_zero=True,
                        output_dim=embedding_dim,
                        weights=embeddings))
    model.add(LSTM(output_dim=hidden_dim,
                   return_sequences=False,
                   stateful=False))
    model.add(RepeatVector(max_len))
    model.add(LSTM(output_dim=hidden_dim,
                   return_sequences=True,
                   stateful=True))
    model.add(TimeDistributed(Dense(num_words), input_shape=(max_len, hidden_dim)))
    model.add(Activation("softmax"))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_text(model, seed_words, temperature, word2idx, idx2word, N):
    model.reset_states()
    # first we initialize the state of the LSTM using the seed_str
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
    x = np.zeros(shape=model.input_shape)
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


class ResetStates(Callback):

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


model = get_model(maxlen, num_words, embedding_dim, hidden_dim, batch_size, embeddings=[embeddingsMatrix])

print len(sentences)
print num_words

sentences_target = np.zeros(shape=(len(sentences), maxlen, num_words), dtype=bool)
for ns, sent in enumerate(sentences):
    for nt, t in enumerate(sent):
        sentences_target[ns, nt, t] = 1

X = np.array(pad_sequences(sentences, maxlen=maxlen, padding='post'))[0:10000]
Y = np.array(sentences_target)[0:10000]
model_name = 'paraphrase.brown.h5'
model.fit(X, Y, batch_size=batch_size, nb_epoch=num_epochs, callbacks=[ResetStates()])
model.save_weights(model_name, overwrite=True)

seed_str = 'as the river flows'
seed_str = [w.lower() for w in word_tokenize(seed_str)]
model.load_weights(model_name)
generated_text = generate_text(model, seed_str, 1.1, word2idx, words, 3000)
print generated_text[0:10]
