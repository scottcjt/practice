import tensorflow as tf
import lang8

from tensorflow.contrib import seq2seq


MODE_EVAL = 0
MODE_TRAIN = 1
KEEP = 0.7


class GrammarCorrectionModel(object):
    @property
    def _reuse_vars(self):
        return self._mode != 'train'

    def _variable(self, name, **kwargs):

        v = tf.get_variable(name, **kwargs)
        if self._mode == 'train':
            v = tf.nn.dropout(v, KEEP)

        return v

    def _rnn_cells(self, num_units, layers):
        cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units) for _ in range(layers)]
        if self._mode == 'train':
            cells = [tf.nn.rnn_cell.DropoutWrapper(c, KEEP, KEEP, KEEP)
                     for c in cells]

        if layers > 1:
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        else:
            cells = cells[0]

        return cells

    def _build_main_graph(self, xs, xlens, ys, ylens):
        with tf.variable_scope('word_model', reuse=self._reuse_vars):
            embeds = self._variable('embeddings',
                                    dtype=tf.float32, shape=[self._word_symbols, self._word_embedding_size])

            with tf.variable_scope('encoder', reuse=self._reuse_vars):
                fw_cells = self._rnn_cells(self._word_model_rnn_hidden_size, self._word_model_rnn_layers // 2)
                bw_cells = self._rnn_cells(self._word_model_rnn_hidden_size, self._word_model_rnn_layers // 2)

                batch_input_embeds = tf.nn.embedding_lookup(embeds, xs)

                rnn_out, rnn_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cells, bw_cells, batch_input_embeds, xlens, dtype=tf.float32)

            with tf.variable_scope('decoder', reuse=self._reuse_vars):
                # Attention only consumes encoder outputs.
                attention = seq2seq.LuongAttention(self._decoder_attention_size, tf.concat(rnn_out, -1), xlens)
                cells = self._rnn_cells(self._word_model_rnn_hidden_size, self._word_model_rnn_layers)
                cells = seq2seq.AttentionWrapper(cells, attention)
                decode_init_state = cells.zero_state(self._batch_size, tf.float32)

                # This layer sits just before softmax. It seems that if an activation is placed here,
                # the network will not converge well. Why?
                def apply_dropout(v):
                    if self._mode == 'train':
                        return tf.nn.dropout(v, KEEP)
                    else:
                        return v

                final_projection = tf.layers.Dense(
                    self._word_symbols, kernel_regularizer=apply_dropout, use_bias=False)

                if self._mode != 'infer':
                    batch_target_embeds = tf.nn.embedding_lookup(embeds, ys)
                    helper = seq2seq.TrainingHelper(batch_target_embeds, ylens)
                    decoder = seq2seq.BasicDecoder(cells, helper, decode_init_state, final_projection)
                    (logits, ids), state, lengths = seq2seq.dynamic_decode(decoder)
                    return logits, ids, lengths
                else:
                    helper = seq2seq.GreedyEmbeddingHelper(
                        embeds,
                        tf.tile([self._start_token], [self._batch_size]),
                        self._end_token)
                    decoder = seq2seq.BasicDecoder(cells, helper, decode_init_state, final_projection)
                    max_iters = tf.reduce_max(xlens) * 2
                    (logits, ids), state, lengths = seq2seq.dynamic_decode(
                        decoder, maximum_iterations=max_iters)
                    return logits, ids, lengths

    def _pad_batch(self, xs, maxlen):
        pad = tf.maximum(maxlen - tf.shape(xs)[1], 0)
        padded_xs = tf.pad(xs, [[0, 0], [0, pad]], constant_values=-1)
        return padded_xs

    def _compare_batch(self, xs, xlens, ys, ylens):
        assert xs.shape.is_compatible_with([self._batch_size, None])
        assert xlens.shape == [self._batch_size]
        assert ys.shape.is_compatible_with([self._batch_size, None])
        assert ylens.shape == [self._batch_size]

        dirty_comp = tf.equal(xs, ys)
        # mask all `unused' elements as true (use any of two length arrays)
        inv_mask = tf.logical_not(
            tf.sequence_mask(xlens, maxlen=tf.shape(xs)[1]))
        masked = tf.logical_or(dirty_comp, inv_mask)
        # the same size & the same values in range
        equal_samples = tf.logical_and(tf.equal(xlens, ylens), tf.reduce_all(masked, axis=1))

        return equal_samples

    def _build_eval(self, targets, target_lens):
        """Perform true evaluation and compute accuracy/recall."""
        assert targets.shape.is_compatible_with([self._batch_size, None])
        assert target_lens.shape.is_compatible_with([self._batch_size])

        with tf.variable_scope('eval'):
            # Input xs and zs always have the same size.
            # xs == zs[:, :-1] -> unrelated sample (should not be modified)
            target_lens_no_eos = target_lens - 1
            non_relevant = self._compare_batch(
                self._xs_ph, self._xlens_ph, targets, target_lens_no_eos)
            relevant = tf.logical_not(non_relevant)

            # output == zs -> correct prediction
            # pad two id arrays to the same shape
            maxlen = tf.maximum(tf.shape(self._infer_outs)[1], tf.shape(targets)[1])
            padded_out = self._pad_batch(self._infer_outs, maxlen)
            padded_targets = self._pad_batch(targets, maxlen)
            #
            # padded_out = tf.Print(padded_out, [maxlen, tf.shape(padded_out), tf.shape(padded_targets)])

            # TP+TN
            accurate_samples = self._compare_batch(
                padded_out, self._infer_lens, padded_targets, target_lens)

            # TP = relevant (TP+FN) & (TP+TN)
            tp_samples = tf.logical_and(relevant, accurate_samples)
            # FP = (FP+TN) & (FP+FN)
            fp_samples = tf.logical_and(non_relevant, tf.logical_not(accurate_samples))

            # accuracy = correctly predict (modified or not)
            accurate_count = tf.count_nonzero(accurate_samples, dtype=tf.float32)
            accuracy = accurate_count / self._batch_size
            tp_count = tf.count_nonzero(tp_samples, dtype=tf.float32)
            fp_count = tf.count_nonzero(fp_samples, dtype=tf.float32)
            # precision = TP / TP+FP
            precision = tp_count / (tp_count + fp_count)
            # recall = TP / TP+FN
            recall = tp_count / accurate_count
            recall = tf.where(tf.is_nan(recall), 0.0, recall)

            # Compute a `copy' rate (xs == output).
            # Returning a copy of input can get about 50% accuracy...
            maxlen = tf.maximum(tf.shape(self._infer_outs)[1], tf.shape(self._xs_ph)[1])
            padded_out = self._pad_batch(self._infer_outs, maxlen)
            padded_in = self._pad_batch(self._xs_ph, maxlen)
            out_lens_no_eos = self._infer_lens - 1
            copied = self._compare_batch(
                padded_in, self._xlens_ph, padded_out, out_lens_no_eos)
            copied = tf.count_nonzero(copied) / self._batch_size

            return accuracy, precision, recall, copied

    def _build_loss(self, logits, expected, lengths):
        with tf.variable_scope('loss'):
            # logits: [BATCH, N, EMBED]
            # expected: [BATCH, M]
            # The problem is N is not always equal to M...
            # M sometimes has padding caused by a large xs plus small ys/zs.
            # Based on this observation, N will always <= M.
            max_expected_len = tf.shape(logits)[1]
            expected = expected[:, :max_expected_len]

            weights = tf.sequence_mask(lengths, dtype=tf.float32)
            loss = seq2seq.sequence_loss(logits, expected, weights)
            return loss

    # def _build_learning_rate_op(self, loss):
    #     """Simply half the learning rate after loss < 2.0.
    #     TODO: lol...use a better curve?
    #     """
    #     rate = tf.cond(loss < tf.constant(3.0),
    #                    lambda: tf.constant(0.0005),
    #                    lambda: tf.constant(0.001))
    #
    #     return rate
    #
    def _build_train_op(self, rate, steps, loss):
        opt = tf.train.AdamOptimizer(rate)
        gradients, params = zip(*opt.compute_gradients(loss))
        gradients, norm = tf.clip_by_global_norm(gradients, 5.0)

        return norm, opt.apply_gradients(zip(gradients, params), global_step=steps)

    def __init__(self, start_symbol, end_symbol, pad_symbol):
        # self._char_model_rnn_hidden_size = 128
        # self._char_model_rnn_layers = 2
        # self._char_symbols = 72

        self._batch_size = 32
        self._word_symbols = 20000
        self._start_token = start_symbol
        self._end_token = end_symbol
        self._pad_token = pad_symbol

        self._word_embedding_size = 64
        self._word_model_rnn_hidden_size = 128
        self._word_model_rnn_layers = 2
        self._decoder_attention_size = 128

        self._xs_ph = tf.placeholder(tf.int32, [self._batch_size, None], 'input/xs')
        self._xlens_ph = tf.placeholder(tf.int32, [self._batch_size], 'input/xlens')
        self._ys_ph = tf.placeholder(tf.int32, [self._batch_size, None], 'input/ys')
        self._ylens_ph = tf.placeholder(tf.int32, [self._batch_size], 'input/ylens')
        self._zs_ph = tf.placeholder(tf.int32, [self._batch_size, None], 'input/zs')

        self._log_items = dict()
        self._build_model()

    def _build_model(self):
        self._global_step = tf.get_variable('global_step',
                                            dtype=tf.int32, initializer=0, trainable=False)

        # todo: adjust this!?
        self._learning_rate = tf.constant(0.001)

        # training
        self._mode = 'train'

        logits, _, x = self._build_main_graph(self._xs_ph, self._xlens_ph,
                                              self._ys_ph, self._ylens_ph)

        self._loss = self._build_loss(logits, self._zs_ph, self._ylens_ph)

        self._grad_norm, self._train_op = self._build_train_op(
            self._learning_rate, self._global_step, self._loss)

        self._summaries = tf.summary.merge([
            tf.summary.scalar('train/loss', self._loss),
            tf.summary.scalar('train/gradient_norm', self._grad_norm),
            ])

        # no dropout
        self._mode = 'validate'

        logits, outs, lens = self._build_main_graph(self._xs_ph, self._xlens_ph,
                                                    self._ys_ph, self._ylens_ph)

        self._v_loss = self._build_loss(logits, self._zs_ph, self._ylens_ph)
        self._v_outs = outs
        self._v_lens = lens

        # decoder has no hint, no dropout (cannot compute loss)
        self._mode = 'infer'

        _, infer_outs, infer_lens = self._build_main_graph(self._xs_ph, self._xlens_ph,
                                                           None, None)

        self._infer_outs = tf.identity(infer_outs, 'output/xs')
        self._infer_lens = tf.identity(infer_lens, 'output/xlens')

        self._accuracy, self._precision, self._recall, self._copied = self._build_eval(self._zs_ph, self._ylens_ph)

        self._v_summaries = tf.summary.merge([
            tf.summary.scalar('eval/loss', self._v_loss),
            tf.summary.scalar('eval/accuracy', self._accuracy),
            tf.summary.scalar('eval/precision', self._precision),
            tf.summary.scalar('eval/recall', self._recall),
            tf.summary.scalar('eval/copy', self._copied),
            ])

    def run(self, data_feeder):
        saver = tf.train.Saver()
        with tf.summary.FileWriter('train/logs', tf.get_default_graph()) as summary_writer:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                epochs = 10
                n = data_feeder.epoch_batches
                interval = 10
                save_interval = 100

                print('training {} epochs'.format(epochs))
                print('Start training, 1 epoch has {} batches (batch_size={})...'.format(n, self._batch_size))

                for i in range(n*epochs):
                    batch = data_feeder.next_batch(self._batch_size)
                    feeds = {
                        self._xs_ph: batch.xs,
                        self._xlens_ph: batch.xlens,
                        self._ys_ph: batch.ys,
                        self._ylens_ph: batch.ylens,
                        self._zs_ph: batch.zs,
                    }
                    fetches = [self._loss, self._grad_norm, self._train_op]
                    if i % interval == 0:
                        fetches += [self._summaries]

                    ret = sess.run(fetches, feeds)
                    ret = dict(zip(fetches, ret))
                    loss, grad_norm = ret[self._loss], ret[self._grad_norm]

                    if i % interval == 0:
                        batch = data_feeder.next_batch(self._batch_size, cat=data_feeder.VALIDATE)
                        feeds = {
                            self._xs_ph: batch.xs,
                            self._xlens_ph: batch.xlens,
                            self._ys_ph: batch.ys,
                            self._ylens_ph: batch.ylens,
                            self._zs_ph: batch.zs,
                        }
                        fetches = [self._v_loss, self._v_summaries]

                        v_loss, v_summaries = sess.run(fetches, feeds)

                        # Write summaries.
                        step = self._global_step.eval(sess)
                        summary_writer.add_summary(ret[self._summaries], global_step=step)
                        summary_writer.add_summary(v_summaries, global_step=step)

                        print('batch {:5d}: loss={:.4f} v_loss={:.4f} norm={:.4f}'.format(i, loss, v_loss, grad_norm))

                    if i % save_interval == 0:
                        saver.save(sess, 'train/checkpoints/step-{}'.format(i))


data = lang8.Lang8Data('lang8-1p', 'lang8-1p_vocab')

model = GrammarCorrectionModel(data.start_symbol, data.end_symbol, data.pad_symbol)
model.run(data)


#
# def main():
#
#
# if __name__ == '__main__':
#     exit(main())
