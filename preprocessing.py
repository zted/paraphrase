import codecs
import operator

from nltk import word_tokenize
import numpy as np


def get_chars(corpus):
    with codecs.open(corpus, 'rb', 'utf-8') as f:
        characters = set()
        for l in f:
            for character in l:
                characters.add(character)
    characters = sorted(characters)
    chars2idx = {char:idx for idx, char in enumerate(characters)}
    return characters, chars2idx


def get_words(corpus, max_size=10000):
    wordcounts = {}
    for w in word_stream(corpus, 0, None):
        if w in wordcounts:
            wordcounts[w] += 1
        else:
            wordcounts[w] = 1
    wordcounts = sorted(wordcounts.items(), key=operator.itemgetter(1), reverse=True) # sort by key
    if len(wordcounts) > max_size:
        wordcounts = wordcounts[:max_size]
        print('Truncating vocab.')
    words = ['<unk>']
    for w, _ in wordcounts:
        words.append(w)
    word2idx = {w:i for i, w in enumerate(words)}
    return words, word2idx


def char_stream(filename, start_position, end_position):
    with codecs.open(filename, 'rb', 'utf-8') as f:
        f.seek(start_position)
        position = start_position
        while True:
            l = f.next()
            for char in l:
                yield char
                position += 1
                if position == end_position:
                    raise StopIteration


def word_stream(filename, start_position, end_position):
    position = 0
    with codecs.open(filename, 'rb', 'utf-8') as f:
        for line in f:
            for w in word_tokenize(line):
                w = w.lower()
                if end_position != None and position == end_position:
                    raise StopIteration
                if position >= start_position:
                   yield w
                position += 1
            # add new line token as a word
            if end_position != None and position == end_position:
                raise StopIteration
            if position >= start_position:
                yield w
            w = '\n'
            yield w
            position +=1


def batch_stream(filename, start_positions, end_positions, word_level):
    stream_fn = word_stream if word_level else char_stream
    streams = [stream_fn(filename, start_pos, end_pos) for start_pos, end_pos in zip(start_positions, end_positions)]
    while True:
        try:
            batch = [next(stream) for stream in streams]
            yield batch
        except StopIteration: # one epoch done
            streams = [stream_fn(filename, start_pos, end_pos) for start_pos, end_pos in zip(start_positions, end_positions)]


def example_stream(filename, num_timesteps, batch_size, word_level):
    total_num_symbols = count_words(filename) if word_level else count_chars(filename)
    print('total num symbols', total_num_symbols)
    num_symbols_per_batch = batch_size * num_timesteps
    total_num_symbols = total_num_symbols - total_num_symbols % num_symbols_per_batch
    start_positions = [i * total_num_symbols//batch_size for i in xrange(batch_size)]
    end_positions = [start_pos + total_num_symbols/batch_size for start_pos in start_positions]
    batched_symbols = batch_stream(filename, start_positions, end_positions, word_level)
    symbols = [next(batched_symbols)]
    while True:
        symbols = [symbols[-1]]
        for t in xrange(num_timesteps):
            symbols.append(next(batched_symbols))
        inputs = symbols[:-1] # all but the last symbols are the inputs
        targets = symbols[1:] # all but the first symbols are the targets
        yield inputs, targets


def count_chars(filename):
    total_num_chars = 0
    with codecs.open(filename, 'rb', 'utf-8') as f:
        for l in f: 
            total_num_chars += len(l)
    return total_num_chars


def count_words(filename):
    total_num_words = 0
    for w in word_stream(filename, 0, None):
        total_num_words += 1
    return total_num_words


def vectorize_chars(inputs, targets, char2idx):
    num_timesteps = len(inputs)
    batch_size = len(inputs[0])
    num_chars = len(char2idx)
    inputs_tensor = np.zeros(shape=(batch_size, num_timesteps, num_chars))
    for t, batch_t in enumerate(inputs):
        for i, char in enumerate(batch_t):
            charidx = char2idx[char]
            inputs_tensor[i, t, charidx] = 1
    targets_tensor = np.zeros(shape=(batch_size, num_timesteps, num_chars))
    for t, batch_t in enumerate(targets):
        for i, char in enumerate(batch_t):
            charidx = char2idx[char]
            targets_tensor[i, t, charidx] = 1
    return inputs_tensor, targets_tensor


def vectorize_words(inputs, targets, word2idx):
    num_timesteps = len(inputs)
    batch_size = len(inputs[0])
    inputs_tensor = np.zeros(shape=(batch_size, num_timesteps))
    for t, batch_t in enumerate(inputs):
        for i, word in enumerate(batch_t):
            wordidx = word2idx[word] if word in word2idx else word2idx['<unk>']
            inputs_tensor[i, t] = wordidx
    num_words = len(word2idx)
    targets_tensor = np.zeros(shape=(batch_size, num_timesteps, num_words))
    for t, batch_t in enumerate(targets):
        for i, word in enumerate(batch_t):
            wordidx = word2idx[word] if word in word2idx else word2idx['<unk>']
            targets_tensor[i, t, wordidx] = 1
    return inputs_tensor, targets_tensor


def vectorized_example_stream(filename, num_timesteps, batch_size, symbol2idx, word_level=True):
    stream = example_stream(filename, num_timesteps, batch_size, word_level)
    for inputs, targets in stream:
        if word_level:
            yield vectorize_words(inputs, targets, symbol2idx)
        else:
            yield vectorize_chars(inputs, targets, symbol2idx)
