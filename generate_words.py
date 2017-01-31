import numpy as np
import corpora
import preprocessing

import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential

from keras.layers import Dense, Activation, LSTM, TimeDistributed, Embedding

corpus = corpora.shakespeare_sonnets
words, word2idx = preprocessing.get_words(corpus)


def get_model(num_timesteps, num_words, embedding_dim, hidden_dim, batch_size):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, 
                        input_length=num_timesteps,
                        batch_input_shape=[batch_size, num_timesteps],
                        output_dim=embedding_dim))
    model.add(LSTM(output_dim=hidden_dim, 
                   batch_input_shape=[batch_size, num_timesteps, embedding_dim], 
                   return_sequences=True, 
                   stateful=True))
    model.add(TimeDistributed(Dense(num_words), input_shape=(num_timesteps, hidden_dim)))
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
    probs = probs[0,0,:]
    next_word_idx = sample(probs, temperature)
    generated_text_idx = [next_word_idx]
    generated_text = [idx2word[next_word_idx]]
    for i in xrange(N - 1):
        last_word_idx = generated_text_idx[-1]
        x = np.zeros(shape=model.input_shape)
        x[0, 0] = last_word_idx
        probs = model.predict(x, verbose=0)
        probs = probs[0,0,:]
        next_word_idx = sample(probs, temperature)
        generated_text_idx.append(next_word_idx)
        generated_text.append(idx2word[next_word_idx])
    return generated_text
    
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model_name = 'wordlevel.%s.h5' % os.path.basename(corpus)

hidden_dim = 512
embedding_dim = 100

from nltk import word_tokenize
seed_str = '''To be or not to be'''
seed_str = [w.lower() for w in word_tokenize(seed_str)]
trained_model_test = get_model(1, len(word2idx), embedding_dim, hidden_dim, 1)
trained_model_test.load_weights(model_name)
generated_text = generate_text(trained_model_test, seed_str, 1.1, word2idx, words, 30)
print generated_text[0:10]