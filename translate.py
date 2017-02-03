import tensorflow as tf
import numpy as np
from nltk.corpus import brown

import os
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

max_examples = 10000
maxlen = 17
minlen = 3
hidden_dim = 128
embedding_dim = 100
batch_size = 50
num_epochs = 1000
validate_cycle = 100
validation_size = 5


def pad_sequences(arrays, maxlen, pad_symbol=0):
    newArray = []
    for a in arrays:
        someLen = len(a)
        assert someLen <= maxlen
        diff = maxlen - someLen
        padded = a + [pad_symbol] * diff
        newArray.append(padded)
    return newArray


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


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def embedding_rnn_seq2seq2(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None,
                          initializer=None):
  """
  Same as the one in tensorflow's seq2seq.py except with the option to pass in an initializer
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
    if dtype is not None:
      scope.set_dtype(dtype)
    else:
      dtype = scope.dtype

    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size, initializer=initializer)
    _, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return tf.nn.seq2seq.embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          num_decoder_symbols,
          embedding_size,
          output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse) as scope:
        outputs, state = tf.nn.seq2seq.embedding_rnn_decoder(
            decoder_inputs, encoder_state, cell, num_decoder_symbols,
            embedding_size, output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(structure=encoder_state,
                                    flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state


brownSents = brown.sents(categories=['hobbies', 'news', 'fiction', 'adventure', 'reviews', 'editorials', 'romance'])
print('Finished loading sentences from brown corpus')
gloveDict = load_word_embeddings('glove.6B.100d.txt')
print('Finished loading word embeddings')
word2idx = {}
sentences = []
unique_words = set(['NONE'])
all_words = ['NONE']
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
                all_words.append(w)
                word2idx[w] = count
                count += 1
            sentVec = [word2idx[t] for t in lowered]
            sentences.append(sentVec)
            if len(sentences) >= max_examples:
                # we have enough
                break

# we start at count = 1 because the input sequence is padded with 0's until a vector of 20 is reached
# cannot use the index 0 for any words in the vocab
num_words = len(unique_words)
assert count == num_words
assert count == len(all_words)
embeddingsMatrix = np.zeros([num_words, embedding_dim])
# create embeddings matrix corresponding to the word indices
print('Creating weights for embeddings layer')
for word, idx in word2idx.items():
    embeddingsMatrix[idx, :] = gloveDict[word]
word2idx['NONE'] = 0
gloveDict = None

print('Building model')

X = pad_sequences(sentences, maxlen=maxlen)

seq_length = maxlen
vocab_size = num_words
memory_dim = 100

enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))
cell = tf.nn.rnn_cell.GRUCell(memory_dim)
dec_outputs, dec_memory = embedding_rnn_seq2seq2(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim, pretrained_embeddings=tf.constant_initializer(embeddingsMatrix))
loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
tf.summary.scalar("loss", loss)
magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
tf.summary.scalar("magnitude at t=1", magnitude)
summary_op = tf.summary.merge_all()
learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())


def train_batch(X, Y):
    X = np.array(X).T
    Y = np.array(Y).T
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary


allWords = all_words
nb_batches = int(np.ceil(len(X) / float(batch_size)))

print('Number of training examples we have: {}\nNumber of unique words: {}'.format(len(sentences),num_words-1))
print('Begin training')
for t in range(num_epochs):

    for i in range(nb_batches):
        batch_start = i * batch_size
        batch_end = min(len(X), batch_start+batch_size)
        X_batch = X[batch_start:batch_end]
        loss_t, summary = train_batch(X_batch, X_batch)

    if (t+1) % validate_cycle == 0:
        val_indices = np.random.choice(range(max_examples), validation_size, replace=False)
        X_test = np.array([X[v] for v in val_indices])
        X_test = np.array(X_test).T
        feed_dict = {enc_inp[t]: X_test[t] for t in range(seq_length)}
        print ('Finished training epoch {}\nLoss: {}'.format(t+1, loss_t))
        dec_outputs_batch = sess.run(dec_outputs, feed_dict)
        answers = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
        answers = np.array(answers).T

        generated_text = []
        for sent in answers:
            sentString = []
            for token in sent:
                sentString.append(allWords[token])
            generated_text.append(' '.join(sentString))

        print('Original sentences:\n{}'.format([raw_sentences[v] for v in val_indices]))
        print(generated_text)


unseen_sent = 'defendants testifies after fears about election'
unseen_vectorized = [word2idx[w] for w in unseen_sent.split(' ')]
vector_padded = np.array(pad_sequences([unseen_vectorized], maxlen))
X_test = np.array(vector_padded)
X_test = np.array(X_test).T
feed_dict = {enc_inp[t]: X_test[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)
answers = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
unseen_prediction = []
for a in answers:
    unseen_prediction.append(allWords[a[0]])
print('\nUnseen Prediction:')
print ' '.join(unseen_prediction)