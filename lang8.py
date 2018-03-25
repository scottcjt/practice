import numpy as np

import collections
import random
import re
import string
import tempfile


_PEEPHOLES = [
    ('a . m . ', 'a.m. '),
    ('. . . ', '... '),
    # some n't is mis-processed to n ' t
    ('n \' t ', ' n\'t '),
    ('N \' T ', ' N\'T '),
    ('n \' t\t', ' n\'t\t'),
    ('N \' T\t', ' N\'T\t'),
]


def _peephole(line):
    line = line.strip()

    # Peepholes
    for old, new in _PEEPHOLES:
        line = line.replace(old, new)

    # The first letter of correct answer should always be capital letter.
    def capitalize(m):
        return m.group(0).upper()

    line = re.sub('\t[a-z]', capitalize, line)
    return line


def _preprocessed(fp):
    fp.seek(0, 0)
    for line in fp:
        yield _peephole(line)


def _tokenize(sentence, codebook):
    tokens = sentence.split()
    ids = []
    for tok in tokens:
        x = codebook(tok)
        if isinstance(x, list):
            ids += x
        else:
            ids.append(x)
    return ids


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

    @staticmethod
    def _corpus_iter(fp):
        fp.seek(0, 0)
        while True:
            for line in fp:
                yield line.strip()
            fp.seek(0, 0)

    def _tidy_and_build_corpus(self, lines):
        unwanted = list(self._punct_whitelist) + ['<pun>', '<num>', '<unk>']
        unwanted = [self.word2code(x) for x in unwanted]

        valid_lines = []
        total_unks = 0
        for i, line in enumerate(lines):
            x, y = line.split('\t')
            xt = _tokenize(x, self.word2code)
            yt = _tokenize(y, self.word2code)

            if len(xt) < 3 or len(xt) > 96 or len(yt) > 96:
                continue

            c = collections.Counter(xt)
            total_unks += c[self._unk]
            unwanted_count = sum([c[x] for x in unwanted])
            if unwanted_count >= (len(xt) / 2):
                continue

            valid_lines.append(line + '\n')

        print('Read in {} lines, keep {} lines (total {} <unk>s)'.format(i, len(valid_lines), total_unks))

        total = len(valid_lines)
        train_n = int(total * 0.8)
        eval_n = int(total * 0.1)
        test_n = int(total * 0.1)
        train_n += total - (train_n + eval_n + test_n)

        random.shuffle(valid_lines)

        self._training_tmp = tempfile.TemporaryFile(mode='w+')
        self._training_tmp.writelines(valid_lines[:train_n])
        self._training_tmp.flush()
        self._train_n = train_n

        self._eval_tmp = tempfile.TemporaryFile(mode='w+')
        self._eval_tmp.writelines(valid_lines[train_n:train_n+eval_n])
        self._eval_tmp.flush()
        self._eval_n = eval_n

        self._test_tmp = tempfile.TemporaryFile(mode='w+')
        self._test_tmp.writelines(valid_lines[train_n+eval_n:])
        self._test_tmp.flush()
        self._test_n = test_n

        self._cats = {
            self.TRAIN: self._corpus_iter(self._training_tmp),
            self.VALIDATE: self._corpus_iter(self._eval_tmp),
            self.TEST: self._corpus_iter(self._test_tmp),
        }

    @classmethod
    def _force_unk(cls, word):
        return len(word) == 0 or len(word) > cls._word_maxlen

    @staticmethod
    def _is_digits(word):
        return all([c in string.digits for c in word])

    @classmethod
    def _is_punct(cls, word):
        return len(word) == 1 and word not in cls._punct_whitelist and word in string.punctuation

    def _build_codebook(self, lines):
        if self._statfile:
            def log(msg):
                print(msg, file=self._statfile)
        else:
            def log(msg):
                pass

        codes = ['<pad>', '<unk>',
                 '<start>', '<end>',
                 '<num>', '<pun>',
                 '<cap>', '<small>'] + [c for c in self._punct_whitelist]
        pick = self.vocab_size - len(codes)

        total = 0
        counter = collections.Counter()
        for line in lines:
            # Pick words from `correct' parts.
            y = line.split('\t')[1].strip()

            words = y.split()
            for w in words:
                # exclude those too short/long
                if self._force_unk(w):
                    continue
                # exclude numbers
                if self._is_digits(w):
                    continue
                # included punctuations
                if w in self._punct_whitelist:
                    continue
                # excluded punctuations
                if self._is_punct(w):
                    continue
                counter[w] += 1
                total += 1

        log('raw words={} total_count={}'.format(len(counter), total))
        log('')

        # Try to merge the same word of capital/small letters.
        WORD_MERGE_RATE = 3
        all_smalls = collections.Counter()
        non_smalls = collections.Counter()
        for w, n in counter.items():
            if w == w.lower():
                all_smalls[w] = n
            else:
                non_smalls[w] = n

        merged = 0
        for w, n in non_smalls.items():
            lower_w_count = all_smalls.get(w.lower(), 0)
            if lower_w_count == 0:
                continue
            # If the same word appears both as lower and upper cases, compare their probabilities:
            if lower_w_count > WORD_MERGE_RATE * n:
                # Most appearances are lower case means it should be a lower case word.
                counter[w.lower()] += n
                del counter[w]
                log('merge {} -> {} ({} vs {})'.format(w, w.lower(), n, lower_w_count))
                merged += 1
            elif lower_w_count * WORD_MERGE_RATE < n:
                # Most appearances are upper/mixed case means it should be a upper case word?
                counter[w] += lower_w_count
                del counter[w.lower()]
                log('merge {} -> {} ({} vs {})'.format(w.lower(), w, lower_w_count, n))
                merged += 1
            else:
                log('{} / {} ({} vs {}) are preserved'.format(w, w.lower(), n, lower_w_count))

        log('{} words are merged (coeff={})'.format(merged, WORD_MERGE_RATE))

        picked = counter.most_common(pick)
        log('')
        log('vocabs have appearance count < {} are discarded'.format(picked[-1][1]))

        n = 0
        for w, c in picked:
            n += c
        log('coverage rate: {}'.format(n / total))

        picked = map(lambda x: x[0], picked)
        codes.extend(picked)
        assert len(codes) <= self.vocab_size

        self.word_codebook = {w:i for i, w in enumerate(codes)}
        self.word_codebook_rev = {i:w for i, w in enumerate(codes)}
        self._unk = self.word_codebook['<unk>']

        # write vocab table
        log('')
        log('valid vocabs:')
        for w in codes:
            log(w)

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
            ret = self.word_codebook.get(w)
            if ret:
                return ret
            # If we have 'Apple' but there is only 'apple' in dict,
            # emit [<cap>, 'apple']
            ret = self.word_codebook.get(w[0].lower() + w[1:])
            if ret:
                return [self.word_codebook['<cap>'], ret]
            # Vise versa.
            ret = self.word_codebook.get(w.upper() + w[1:])
            if ret:
                return [self.word_codebook['<small>'], ret]
            return self._unk

    def code2word(self, c):
        return self.word_codebook_rev.get(c, '<oov>')

    def reconstruct(self, ids):
        ret = []
        i = 0
        while i < len(ids):
            w = self.code2word(ids[i])
            i += 1
            if w == 'cap':
                w = self.code2word(ids[i]).capitalize()
                i += 1
            elif w == 'small':
                w = self.code2word(ids[i]).lower()
                i += 1
            ret.append(w)

        return ' '.join(ret)

    @property
    def pad_symbol(self):
        return self.word_codebook['<pad>']

    @property
    def start_symbol(self):
        return self.word_codebook['<start>']

    @property
    def end_symbol(self):
        return self.word_codebook['<end>']

    def __init__(self, filename, statfile=None, vocab_size=20000):
        self.vocab_size = vocab_size

        if statfile:
            self._statfile = open(statfile, mode='w')

        with open(filename, mode='r') as fp:
            preprocessed = _preprocessed(fp)
            self._build_codebook(preprocessed)

            preprocessed = _preprocessed(fp)
            self._tidy_and_build_corpus(preprocessed)

        if statfile:
            self._statfile.close()
            del self._statfile

    def next_batch(self, n=128, cat=TRAIN):
        data = self._cats[cat]

        picked = [x.split('\t') for _, x in zip(range(n), data)]

        xstrs, ystrs = zip(*picked)

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
        return self._train_n // batch_size


# data = Lang8Data('lang8-1p', 'lang8_vocab')
#
# for _ in range(100):
#     b = data.next_batch(cat=data.TEST)
#     for i, (seq, l) in enumerate(zip(b.xs, b.xlens)):
#         text = data.reconstruct(seq[:l])
#         if text[0].islower():
#             answer = data.reconstruct(b.zs[i][:b.zlens[i]])
#             print('In: ' + text)
#             print('Expect: ' + answer)

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
