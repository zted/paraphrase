import tensorflow as tf
import numpy as np
import data_prep as D
import sys

import os
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

try:
    tfSessionFile = sys.argv[1]
except IndexError as e:
    tfSessionFile = None

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

max_examples = 100000000000000
maxlen = 30
minlen = 5
embedding_dim = 100
batch_size = 100
num_epochs = 5
validate_cycle = 1000
# validate every x batches
validation_size = 5
recurrent_dim = 100

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


def train_batch(X, Y):
    X = np.array(X).T
    Y = np.array(Y).T
    feed_dict = {enc_inp[t]: X[t] for t in range(maxlen)}
    feed_dict.update({labels[t]: Y[t] for t in range(maxlen)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary


def predict_sentence(X, enc, maxlen):
    feed_dict = {enc[t]: X[t] for t in range(maxlen)}
    raw_predictions = sess.run(dec_outputs, feed_dict)
    most_probable_ans = [seqs.argmax(axis=1) for seqs in raw_predictions]
    most_probable_ans = np.array(most_probable_ans).T
    sampled_ans = [sample_sequence(seqs) for seqs in raw_predictions]

    most_probable_text = []
    sampled_text = []
    for s1, s2 in zip(most_probable_ans, sampled_ans):
        s1Str = []
        s2Str = []
        for t1, t2 in zip(s1, s2):
            s1Str.append(all_words[t1])
            s2Str.append(all_words[t2])
        most_probable_text.append(' '.join(s1Str))
        sampled_text.append(' '.join(s2Str))

    print('Most probable predictions\n{}'.format(most_probable_text))
    print('Sampled precitions\n{}'.format(sampled_text))


def sample_sequence(seq):
    out = []
    for distribution in seq:
        out.append(sample(distribution))
    return out


def sample(preds):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


gloveDict = D.load_word_embeddings('glove.6B.100d.txt')
gloveNum = len(gloveDict)
embeddingsMatrix = np.zeros([gloveNum+1, embedding_dim])
# create embeddings matrix corresponding to the word indices
print('Creating weights for embeddings layer')

word2idx = {'PAD':0}
all_words = ['PAD']
for n, (word, vec) in enumerate(gloveDict.items()):
    word2idx[word] = n+1
    embeddingsMatrix[n+1, :] = vec
    all_words.append(word)
gloveDict = None

print('Finished loading word embeddings')
print('Loading Sentences')
j1 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_test2015_questions.json'
j2 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_test-dev2015_questions.json'
j3 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_train2014_questions.json'
j4 = '/export/home1/NoCsBack/hci/ted/data/OpenEnded_mscoco_val2014_questions.json'
gigaCorpus = '/export/home1/NoCsBack/hci/ted/translate_datadir/google_processed_minlen5_maxlen30.txt'

raw_sentences0, sentences0 = D.load_google_sentences_nodict(j4, minlen, maxlen, word2idx, max_examples)
raw_sentences1, sentences1 = D.load_vqa_questions(j1, minlen, maxlen, word2idx, max_examples)
raw_sentences2, sentences2 = D.load_vqa_questions(j2, minlen, maxlen, word2idx, max_examples)
raw_sentences3, sentences3 = D.load_vqa_questions(j3, minlen, maxlen, word2idx, max_examples)
raw_sentences4, sentences4 = D.load_vqa_questions(j4, minlen, maxlen, word2idx, max_examples)

raw_sentences = raw_sentences0 + raw_sentences1 + raw_sentences2 + raw_sentences3 + raw_sentences4
sentences = sentences0 + sentences1 + sentences2 + sentences3 + sentences4

X = sentences
sentences = None
vocab_size = len(word2idx)

unseen_sent = ['defendants testifies after fears about election', 'i like taking walks', 'the dog is a great and caring animal']
unseen_vectorized = [[word2idx[w] for w in sent.split(' ')] for sent in unseen_sent]
print unseen_vectorized
unseen_sent_padded = np.array(D.pad_sequences(unseen_vectorized, maxlen))
X_unseen = np.array(unseen_sent_padded)
X_unseen = np.array(X_unseen).T

print('Building model')
enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(maxlen)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(maxlen)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, recurrent_dim))
cell = tf.nn.rnn_cell.GRUCell(recurrent_dim)
dec_outputs, dec_memory = embedding_rnn_seq2seq2(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim, initializer=tf.constant_initializer(embeddingsMatrix))
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
saver = tf.train.Saver()

nb_batches = int(np.ceil(len(X) / float(batch_size)))
print('Number of training examples we have: {}\n'.format(len(X)))

if tfSessionFile is not None:
    loader = tf.train.import_meta_graph(tfSessionFile)
    loader.restore(sess, tf.train.latest_checkpoint('./'))

print('Begin training')
for t in range(num_epochs):
    for i in range(nb_batches):
        batch_start = i * batch_size
        batch_end = min(len(X), batch_start+batch_size)
        X_batch = X[batch_start:batch_end]
        loss_t, summary = train_batch(X_batch, X_batch)

        if (i+1) % validate_cycle == 0:
            print ('Training epoch {} batch {}/{}\nLoss: {}'.format(t+1, i+1, nb_batches, loss_t))
            saver.save(sess, 'my-model')
            val_indices = np.random.choice(range(max_examples), validation_size, replace=False)
            print('Randomly selected training sentences:\n{}'.format([raw_sentences[v] for v in val_indices]))
            X_test = np.array([X[v] for v in val_indices])
            X_test = np.array(X_test).T
            predict_sentence(X_test, enc_inp, maxlen)

            print('\nUnseen sentences:\n{}'.format(unseen_sent))
            predict_sentence(X_unseen, enc_inp, maxlen)

saver.save(sess, 'my-model')