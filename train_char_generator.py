import corpora
import preprocessing

corpus = corpora.shakespeare_sonnets
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential

from keras.layers import Dense, Activation, LSTM, Reshape, TimeDistributed, Embedding
from keras.callbacks import Callback


chars, char2idx = preprocessing.get_chars(corpus)

def get_model(num_timesteps, num_chars, hidden_dim, batch_size):
    model = Sequential()
    model.add(LSTM(output_dim=hidden_dim, 
                   batch_input_shape=[batch_size, num_timesteps, num_chars], 
                   return_sequences=True, 
                   stateful=True))
    model.add(TimeDistributed(Dense(num_chars), input_shape=(num_timesteps, hidden_dim)))
    model.add(Activation("softmax"))
    #model.add(Reshape((num_timesteps, num_chars)))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class ResetStates(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        self.model.reset_states()

def train_model(num_timesteps, hidden_dim, batch_size, num_epochs, corpus, char2idx, model=None):
    num_chars = len(char2idx)
    model = model if model else get_model(num_timesteps, num_chars, hidden_dim, batch_size)
    weights = './charlevel.shakespeare_sonnets.txt.h5'
    model.load_weights(weights)
    examples = preprocessing.vectorized_example_stream(corpus, num_timesteps, batch_size, char2idx, word_level=False)
    total_num_chars = preprocessing.count_chars(corpus)
    total_num_chars = total_num_chars - total_num_chars % (num_timesteps * batch_size)
    samples_per_epoch = total_num_chars//num_timesteps
    model.fit_generator(examples, samples_per_epoch, num_epochs, callbacks=[ResetStates()])
    return model


num_timesteps = 30
hidden_dim = 512
batch_size = 32
num_epochs = 7
trained_model = train_model(num_timesteps, hidden_dim, batch_size, 
                            num_epochs, corpus,char2idx)

model_name = 'charlevel.%s.h5' % os.path.basename(corpus)
trained_model.save_weights(model_name, overwrite=True)