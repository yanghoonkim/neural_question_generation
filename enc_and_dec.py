import tensorflow as tf 

class _BaseClass(object):
    def __init__(self, 
            num_layer = 1, hidden_size = 512,
	    cell_type = 'gru', dropout = 0.1,
	    dtype = tf.float32, mode = tf.estimator.ModeKeys.TRAIN):

	    self.num_layer = num_layer
	    self.hidden_size = hidden_size
	    self.cell_type = cell_type
	    self.dropout = dropout
	    self.dtype = dtype
	    self.mode = mode

    # Build cell for encoder and decoder
    def _create_cell(self):
        def rnn_cell():
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size) if self.cell_type == 'gru' else tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            return tf.contrib.rnn.DropoutWrapper(cell, 
                    input_keep_prob = 1 - self.dropout if self.mode == tf.estimator.ModeKeys.TRAIN else 1)
        return rnn_cell() if self.num_layer == 1 else tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(self.num_layer)])


class Encoder(_BaseClass):
    def __init__(self, enc_type ='bi', 
            num_layer = 1, hidden_size = 512,
            cell_type = 'lstm', dropout = 0.1,
	    dtype = tf.float32, mode = tf.estimator.ModeKeys.TRAIN):
        
        super(Encoder, self).__init__(
                num_layer = num_layer,
    	        hidden_size = hidden_size,
                cell_type = cell_type,
		dropout = dropout,
		dtype = dtype,
		mode = mode
                )
	self.enc_type = enc_type

    def run(self, embd_input, sequence_length):
        if self.enc_type == 'mono':
            #encoder_output: [batch_size, max_time, hidden_size]
	    #encoder_state: last hidden_state of encoder, [batch_size, hidden_size]
	    enc_cell = self._create_cell()

	    self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(
                    enc_cell, 
		    inputs = embd_input,
        	    sequence_length = sequence_length,
		    dtype = self.dtype)


	elif self.enc_type == 'bi':
	    enc_cell_fw = self._create_cell()
	    enc_cell_bw = self._create_cell()

	    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    enc_cell_fw, enc_cell_bw,
		    inputs = embd_input,
		    sequence_length = sequence_length,
		    dtype = self.dtype)

	    self.encoder_output = tf.concat(encoder_output, -1)
	    if type(encoder_state[0]) is not tuple:
                # if num_layer == 1
                self.encoder_state = tf.concat(encoder_state, -1)
	    else:
                self.encoder_state = tuple(tf.concat([state_fw, state_bw], -1) for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]))
		
	else:
	    raise ValueError('Invalid input %s' %self.enc_type)

        return self.encoder_output, self.encoder_state 


class Decoder(_BaseClass):
    def __init__(self, enc_type ='bi', 
            attn_type = 'bahdanau', voca_size = None,
	    num_layer = 1, hidden_size = 512, 
            cell_type = 'lstm', dropout = 0.1, 
            dtype = tf.float32, mode = tf.estimator.ModeKeys.TRAIN,
	    sample_prob = 0.25):
        
        super(Decoder, self).__init__(
                num_layer = num_layer,
		hidden_size = hidden_size,
                cell_type = cell_type,
		dropout = dropout,
		dtype = dtype,
		mode = mode
		)
	self.enc_type = enc_type
	self.attn_type = attn_type
	self.voca_size = voca_size
	self.sample_prob = sample_prob


    def run(self, embd_input, sequence_length, embedding, start_token = 1, end_token = 2):
        # batch_size should not be specified
	# if fixed, then the redundant evaluation data will make error
	# it may related to bug of tensorflow api

	# Helper for decoder
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs = embd_input,
                    sequence_length = sequence_length,
                    embedding = embedding,
                    sampling_probability = self.sample_prob)
        else: # EVAL & TEST
            start_token = start_token * tf.ones([self.batch_size], dtype = tf.int32)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding, start_token, end_token
                    )

        # Start decoding
        initial_state = self.out_dec_cell.zero_state(dtype = self.dtype, batch_size = self.batch_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(
        	self.out_dec_cell, helper, initial_state,
        	output_layer = None)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = None)
        else: # Test & Eval
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = self. max_iter)

        self.logits = outputs.rnn_output

       	return self.logits


    def set_attentional_cell(self, memory, memory_length, encoder_state, enc_num_layer):
        self.batch_size = tf.shape(memory)[0]

    	dec_cell = self._create_cell()
	attention_mechanism = self._attention(memory, memory_length)

	initial_cell_state = encoder_state if self.num_layer == enc_num_layer else None
	attn_dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_cell, attention_mechanism,
		attention_layer_size = self.hidden_size,
		initial_cell_state = initial_cell_state)

	self.out_dec_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_dec_cell, self.voca_size)

        # Set maximum iteration for GreedyHelper(Eval and Test)
	self.max_iter = None if self.mode == tf.estimator.ModeKeys.TRAIN else tf.round(tf.reduce_max(memory_length) * 2)


    def _attention(self, memory, memory_length):
        if self.attn_type == 'bahdanau':
	    return tf.contrib.seq2seq.BahdanauAttention(
                    self.hidden_size,
	            memory,
	            memory_length)
	elif self.attn_type == 'normed_bahdanau':
	    return tf.contrib.seq2seq.BahdanauAttention(
                    self.hidden_size,
	            memory,
	            memory_length,
	            normalize = True)
	    
	elif self.attn_type == 'luong':
	    return tf.contrib.seq2seq.LuongAttention(
                    self.hidden_size,
	            memory,
	            memory_length)

	elif self.attn_type == 'scaled_luong':
            return tf.contrib.seq2seq.LuongAttention(
                    self.hidden_size,
	            memory,
	            memory_length,
                    scale = True)
	else:
	    raise ValueError('Unknown attention mechanism : %s' %attn_type)
