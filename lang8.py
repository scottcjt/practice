import numpy as np

import collections
import random
import string
import sys


_PUNCTS = '.,!;:\'"`~!@#$%^&*()-_=+\|[]{}<>?'
_FACES = 'DPO'
_EYES = ';:'

_PEEPHOLES = [
    ('a . m .', 'a.m.'),
    ('. . . ', '...'),
    # some n't is mis-processed to n ' t
    ('n \' t ', ' n\'t '),
    ('N \' T ', ' N\'T '),
    ('n \' t\t', ' n\'t\t'),
    ('N \' T\t', ' N\'T\t'),
]

_MAPPINGS = [
    ('.', '<eos>'),
    ('\'"`', '<quote>'),
    ('([{<', '<Lbracket>'),
    (')]}>', '<Rbracket>'),
]


def _preprocess_line(line):
    line = line.strip()

    # Peepholes
    for old, new in _PEEPHOLES:
        line = line.replace(old, new)

    # TODO: Detect emoji here?

    # Grouping punctuations
    # for old, new in _MAPPINGS:
    #     line = line.replace(old, new)

    return line


def _preprocess_tokens(tokens):
    for i in range(len(tokens)):
        if tokens[i].isnumeric():
            tokens[i] = '<num>'
    for i in range(len(tokens)):
        if len(tokens[i]) == 1 and not tokens[i].isalpha():
            tokens[i] = '<pun>'

    # cont = True
    # while cont:
    #     cont = False
    #     for i in range(1, len(tokens)):
    #         if tokens[i-1] in _PUNCTS and tokens[i] in _PUNCTS:
    #             merged = tokens[i-1] + tokens[i]
    #             tokens[i-1:i+1] = [merged]
    #             print('merged: ' + merged)
    #             cont = True
    #             break
    return tokens


def _preprocess(line):
    line = _preprocess_line(line)
    x, y = line.split('\t')
    return x, y


def _tokenize(sentence, codebook):
    tokens = sentence.split()
    ids = [codebook(tok) for tok in tokens]
    return ids
    # keys_of_seq = []
    # for tok in tokens:
    #     keys = [codebook(k) for k in tok]
    #     keys_of_seq.append(keys)
    #
    # return keys_of_seq


def _form_sentence_nparray(seq, pad_to):
    npa = np.zeros([len(seq), pad_to], np.int32)
    lengths = np.zeros([len(seq)], np.int32)

    for i, elems in enumerate(seq):
        # truncate first
        elems_npa = np.array(elems[:pad_to])
        lengths[i] = len(elems_npa)

        # pad second
        if len(elems_npa) < pad_to:
            pad = pad_to - len(elems_npa)
            elems_npa = np.pad(elems_npa, (0, pad), mode='constant', constant_values=0)

        npa[i] = elems_npa

    return npa, lengths


# def _form_batch_nparray(batch, keys_pad):
#     sentences_npa = [_form_sentence_nparray(sen, keys_pad) for sen in batch]
#     sentences_npa, lengths_npa = zip(*sentences_npa)
#     words_pad = max(map(lambda x: x.shape[0], sentences_npa))
#
#     npa = np.zeros([len(batch), words_pad, keys_pad], np.int32)
#     for i, s in enumerate(sentences_npa):
#         npa[i, :s.shape[0], :s.shape[1]] = s
#
#     lengths = np.zeros([len(batch), words_pad], np.int32)
#     for i, l in enumerate(lengths_npa):
#         lengths[i, :len(l)] = l
#
#     return npa, lengths

def _build_batch_with_paddings(seqs, maxlen, init_val):
    npa = np.full([len(seqs), maxlen], init_val, dtype=np.int32)
    for i, s in enumerate(seqs):
        npa[i, :len(s)] = s

    return npa, np.array(list(map(len, seqs)), dtype=np.int32)


def _build_batch_nparray(*args, init_val):
    """Build ndarrays for all inputs, padding their inner dimension to init_val.
    """
    maxlen = 0
    for seqs in args:
        candidate = max([len(seq) for seq in seqs])
        maxlen = max(candidate, maxlen)

    pairs = [_build_batch_with_paddings(seqs, maxlen, init_val) for seqs in args]
    return pairs


# def _form_batch_nparrays(xseqs, yseqs, zseqs):
#     xs, xlens = _build_batch_nparray(xseqs)
#     ys, ylens = _build_batch_nparray(yseqs)
#     zs, zlens = _build_batch_nparray(zseqs)
#
#     assert all((y == z for y, z in zip(ylens, zlens)))
#
#     return xs, xlens, ys, ylens, zs, zlens


# x = _tokenize('this is not a good day to crash 12345678901234567890xxx .')
# y = _form_sentence_nparray(x, 20)
# x2 = _tokenize('one two three four .')
# y2 = _form_sentence_nparray(x2, 20)
#
# z = _form_batch_nparray([y, y2], 20)
# print(z)
# exit(1)

# def _read_file(filename):
#     with open(filename, mode='r') as fp:
#         xs, ys =


class Lang8Data(object):
    TRAIN = 'train'
    VALIDATE = 'validate'
    TEST = 'test'

    _word_maxlen = 20
    _punct_whitelist = '.,?!-\''

    Batch = collections.namedtuple('Batch', 'xs xlens ys ylens zs zlens')

    def _tidy(self):
        unwanted = list(self._punct_whitelist) + ['<pun>', '<num>']
        unwanted = [self.word2code(x) for x in unwanted]

        lines2 = []
        for line in self.lines:
            line = _preprocess_line(line)
            x, y = line.split('\t')
            xt = _tokenize(x, self.word2code)
            yt = _tokenize(y, self.word2code)

            if len(xt) < 3 or len(xt) > 96 or len(yt) > 96:
                continue

            c = collections.Counter(xt)
            unwanted_count = sum([c[x] for x in unwanted])
            if unwanted_count >= (len(xt) / 2):
                # print(x)
                continue

            lines2.append(line)

        print('Read in {} lines, keep {} lines'.format(len(self.lines), len(lines2)))
        random.shuffle(lines2)
        self.lines = lines2

    @classmethod
    def _force_unk(cls, word):
        return len(word) == 0 or len(word) > cls._word_maxlen

    @staticmethod
    def _is_digits(word):
        return all([c in string.digits for c in word])

    @classmethod
    def _is_punct(cls, word):
        return len(word) == 1 and word not in cls._punct_whitelist and word in string.punctuation

    def _init_codebook(self):
        # unk must be 0
        # symbols = ['<unk>', '<pun>', '.', '?', '!', ',', '-']
        # symbols.extend(string.ascii_letters)
        # symbols.extend(string.digits)
        #
        # self.char_codebook = {c: i for i, c in enumerate(symbols)}

        codes = ['<pad>', '<unk>',
                 '<start>', '<end>',
                 '<num>', '<pun>'] + [c for c in self._punct_whitelist]
        pick = self.vocab_size - len(codes)

        # Pick words from `correct' parts.
        counter = collections.Counter()
        for line in self.lines:
            line = line.split('\t')[1].strip()
            words = line.split()
            for w in words:
                # exclude those too short/long
                if self._force_unk(w):
                    continue
                # exclude numbers
                if self._is_digits(w):
                    continue
                # exclude punctuations
                if self._is_punct(w):
                    continue
                counter[w] += 1

        picked = counter.most_common(pick)
        picked = map(lambda x: x[0], picked)
        codes.extend(picked)
        assert len(codes) <= self.vocab_size

        self.word_codebook = {w:i for i, w in enumerate(codes)}
        self.word_codebook_rev = {i:w for i, w in enumerate(codes)}
        self._unk = self.word_codebook['<unk>']

    # def char_code(self, c):
    #     if c in self.char_codebook:
    #         return self.char_codebook[c]
    #     if c in string.punctuation:
    #         return self.char_codebook['<pun>']
    #     return self.char_codebook['<unk>']

    def word2code(self, w):
        if self._is_punct(w):
            return self.word_codebook['<pun>']
        elif self._is_digits(w):
            return self.word_codebook['<num>']
        else:
            return self.word_codebook.get(w, self._unk)

    def code2word(self, c):
        return self.word_codebook_rev.get(c, '<oov>')

    @property
    def pad_symbol(self):
        return self.word_codebook['<pad>']

    @property
    def start_symbol(self):
        return self.word_codebook['<start>']

    @property
    def end_symbol(self):
        return self.word_codebook['<end>']

    # @property
    # def char_max(self):
    #     return len(self.char_codebook)
    class Samples(object):
        def __init__(self, samples, bucketing=False):
            self._samples = samples
            self._cursor = 0
            self._sort = bucketing

            if self._sort:
                self._samples.sort(key=len)

        def get(self, n):
            take = self._samples[self._cursor:self._cursor + n]
            self._cursor += len(take)
            if len(take) < n:
                wrap = n - len(take)
                take += self._samples[:wrap]
                self._cursor = wrap

            return take

        @property
        def size(self):
            return len(self._samples)

    def __init__(self, filename, keys_per_word=20, vocab_size=20000, bucketing=True):
        with open(filename, mode='r') as fp:
            self.lines = fp.readlines()

        self.keys_per_word = keys_per_word
        self.vocab_size = vocab_size

        self._init_codebook()
        self._tidy()

        total = len(self.lines)
        train = int(total * 0.8)
        validate = int(total * 0.1)
        test = int(total * 0.1)
        train += total - (train + validate + test)

        self._data_cats = {self.TRAIN: self.Samples(self.lines[:train], bucketing),
                           self.VALIDATE: self.Samples(self.lines[train:train+validate]),
                           self.TEST: self.Samples(self.lines[train:train+validate:])}

    def next_batch(self, n=128, cat=TRAIN):
        data = self._data_cats[cat]
        picked = data.get(n)

        xstrs, ystrs = zip(*[_preprocess(line) for line in picked])

        xtoks = [_tokenize(x, self.word2code) for x in xstrs]
        ytoks = [[self.start_symbol] + _tokenize(y, self.word2code) for y in ystrs]
        ztoks = []
        for ytok in ytoks:
            ztoks.append(ytok[1:] + [self.end_symbol])

        xs, ys, zs = _build_batch_nparray(xtoks, ytoks, ztoks, init_val=self.pad_symbol)
        # xs = _build_batch_nparray(xtoks, self.pad_symbol)
        # ys = _build_batch_nparray(ytoks, self.pad_symbol)
        # zs = _build_batch_nparray(ztoks, self.pad_symbol)
        return self.Batch(*xs, *ys, *zs)

    @property
    def epoch_batches(self, batch_size=128):
        return self._data_cats[self.TRAIN].size // batch_size


# data = Lang8Data('lang8-10p')
#
# b = data.next_batch(128)
# for x, xl, y, yl, z, zl in zip(b.xs, b.xlens, b.ys, b.ylens, b.zs, b.zlens):
#     xw = [data.code2word(c) for c in x]
#     yw = [data.code2word(c) for c in y]
#     zw = [data.code2word(c) for c in z]
#     print(xw, end='')
#     print(' -> ', end='')
#     print(yw, end='')
#     print(' -> ', end='')
#     print(zw)
#
