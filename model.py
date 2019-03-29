import tensorflow as tf

import sys
import utils
import enc_and_dec


class q_generation:
    PAD = 0
    GO = 1
    EOS = 2
    UNK = 3
    
    def __init__(self, params):
        self.dtype = params['dtype']
	self.voca_size = params['voca_size']
	self.embedding_size = params['embedding_size']
	self.hidden_size = params['hidden_size']
        self.cell_type = params['cell_type']
	self.pre_embedding = params['pre_embedding']
	self.embedding_trainable = params['embedding_trainable']
	self.enc_type = params['enc_type']
	self.enc_layer = params['encoder_layer']
	self.dec_layer = params['decoder_layer']
	self.maxlen_dec_train = params['maxlen_dec_train'] # for loss calculation
	self.maxlen_dec_dev = params['maxlen_dec_dev'] # for loss calculation
	self.rnn_dropout = params['dropout']
	self.attn = params['attn']
        self.beam_width = params['beam_width']
        self.length_penalty_weight = params['length_penalty_weight']
	self.sample_prob = params['sample_prob']
	self.learning_rate = params['learning_rate']
	self.decay_step = params['decay_step'] # learning rate decay
	self.decay_rate = params['decay_rate'] # learning rate decay step

    def run(self, features, labels, mode, params):
        
        self.enc_inputs = tf.to_int32(features['enc_inputs'])
        if mode != tf.estimator.ModeKeys.PREDICT:
	    self.dec_inputs = tf.to_int32(features['dec_inputs'])

        else:
            self.dec_inputs = None

	self._build_embedding(self.enc_inputs, self.dec_inputs)

	self.enc_input_length = self._calculate_length(self.enc_inputs)
	self.batch_size = tf.shape(self.enc_inputs)[0]

	if self.dec_inputs is not None:
            self.dec_input_length = self._calculate_length(self.dec_inputs)
	else:
	    self.dec_input_length = None

	with tf.variable_scope('EncoderScope'):
            encoder = enc_and_dec.Encoder(self.enc_type, 
                    self.enc_layer, self.hidden_size,
                    self.cell_type, self.rnn_dropout,
                    self.dtype, mode)

	    encoder_outputs, encoder_state = encoder.run(self.embd_enc_inputs, self.enc_input_length)

	with tf.variable_scope('DecoderScope'):
	    decoder = enc_and_dec.Decoder(self.enc_type, 
                    self.attn, self.voca_size,
                    self.beam_width, self.length_penalty_weight,
		    self.dec_layer, self.hidden_size * 2 * (self.enc_type == 'bi'),
                    self.cell_type, self.rnn_dropout,
		    self.dtype, mode, self.sample_prob)

	    # Add attention wrapper to decoder cell
	    decoder.set_attention_cell(encoder_outputs, self.enc_input_length, encoder_state, self.enc_layer)
        
        if not (mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0):
	    self.logits = decoder.run(self.embd_dec_inputs, self.dec_input_length, self.dec_embedding, self.GO, self.EOS)
	    self.predictions = tf.argmax(self.logits, axis = -1)
        
        else: # Beam decoding
            self.predictions = decoder.run(self.embd_dec_inputs, self.dec_input_length, self.dec_embedding, self.GO, self.EOS)

	self._calculate_loss(mode)
	return self._update_or_output(mode)


    def _calculate_length(self, inputs):
        input_length = tf.reduce_sum(
                tf.to_int32(tf.not_equal(inputs, self.PAD)), -1)
        return input_length

    def _build_embedding(self, enc_inputs, dec_inputs):
	# Make embedded inputs
	# Same tensor name  == embedding sharing
	self.embd_enc_inputs, self.enc_embedding = utils.embed_op(enc_inputs, self.pre_embedding, 
                self.voca_size, self.embedding_size, 
		self.embedding_trainable, self.dtype, 
		name = 'embedding')

	if dec_inputs is not None:
            self.embd_dec_inputs, self.dec_embedding = utils.embed_op(dec_inputs, self.pre_embedding, 
                    self.voca_size, self.embedding_size, 
		    self.embedding_trainable, self.dtype,
		    name = 'embedding')

        else:
            self.embd_dec_inputs = None
            self.dec_embedding = self.enc_embedding # Enc and Dec share embedding

    def _calculate_loss(self, mode):
	if mode == tf.estimator.ModeKeys.PREDICT:
            return 
		
	self.labels = tf.concat([self.dec_inputs[:, 1:], tf.zeros([self.batch_size, 1], dtype = tf.int32)], axis = 1, name = 'labels')
	maxlen_label = self.maxlen_dec_train if mode == tf.estimator.ModeKeys.TRAIN else self.maxlen_dec_dev
	current_length = tf.shape(self.logits)[1]

	def concat_padding():
            num_pad = maxlen_label - current_length
	    padding = tf.zeros([self.batch_size, num_pad, self.voca_size], dtype = self.dtype)
            return tf.concat([self.logits, padding], axis = 1)

	def slice_to_maxlen():
	    return tf.slice(self.logits, [0,0,0], [self.batch_size, maxlen_label, self.voca_size])

	self.logits = tf.cond(current_length < maxlen_label,
                concat_padding,
		slice_to_maxlen)

	weight_pad = tf.sequence_mask(self.dec_input_length, maxlen_label, self.dtype)
	self.loss = tf.contrib.seq2seq.sequence_loss(
                self.logits, 
		self.labels,
		weight_pad,
		average_across_timesteps = True,
		average_across_batch = True,
		softmax_loss_function = None # default : sparse_softmax_cross_entropy
                )

    def _update_or_output(self, mode):
        eval_metric_ops = {
                'bleu' : utils.bleu_score(self.labels, self.predictions)
                }

        if mode == tf.estimator.ModeKeys.PREDICT:
	    return tf.estimator.EstimatorSpec(
                    mode = mode,
		    predictions = {
		    'question' : self.predictions
                    })

	# Optimizer
        if self.decay_step is not None:
            self.learning_rate = tf.train.exponential_decay(
                    self.learning_rate, 
		    tf.train.get_global_step(), 
		    self.decay_step, 
		    self.decay_rate, 
		    staircase = True)

	optimizer = tf.train.AdamOptimizer(self.learning_rate)

	grad_and_var = optimizer.compute_gradients(self.loss, tf.trainable_variables())
	grad, var = zip(*grad_and_var)
        # grad, norm = tf.clip_by_global_norm(grad, 5)
	train_op = optimizer.apply_gradients(zip(grad, var), global_step = tf.train.get_global_step())

	return tf.estimator.EstimatorSpec(
                mode = mode,
		loss = self.loss,
		train_op = train_op,
		eval_metric_ops = eval_metric_ops)








