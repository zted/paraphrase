import corpora
import preprocessing
corpus = corpora.shakespeare_sonnets
words, word2idx = preprocessing.get_words(corpus)

import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential

from keras.layers import Dense, Activation, LSTM, Reshape, TimeDistributed, Embedding
from keras.callbacks import Callback


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


class ResetStates(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        self.model.reset_states()

def train_model(num_timesteps, hidden_dim, embedding_dim, batch_size, num_epochs, corpus, word2idx, model=None):
    num_chars = len(word2idx)
    model = model if model else get_model(num_timesteps, num_chars, embedding_dim, hidden_dim, batch_size)
    examples = preprocessing.vectorized_example_stream(corpus, num_timesteps, batch_size, word2idx, word_level=True)
    total_num_chars = preprocessing.count_words(corpus)
    total_num_chars = total_num_chars - total_num_chars % (num_timesteps * batch_size)
    samples_per_epoch = total_num_chars//num_timesteps
    model.fit_generator(examples, samples_per_epoch, num_epochs, callbacks=[ResetStates()])
    return model

num_timesteps = 30
hidden_dim = 512
embedding_dim = 100
batch_size = 32
num_epochs = 3
trained_model = train_model(num_timesteps, hidden_dim, embedding_dim, batch_size, 
                            num_epochs, corpus,word2idx)

model_name = 'wordlevel.%s.h5' % os.path.basename(corpus)
trained_model.save_weights(model_name, overwrite=True)

