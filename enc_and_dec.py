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
	    if self.cell_type is 'gru':
                if self.num_layer == 1:
                    self.encoder_state = tf.concat(encoder_state, -1)
	        else: # multi layer
                    self.encoder_state = tuple(tf.concat([state_fw, state_bw], -1) for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]))

            else: # lstm
                if self.num_layer == 1:
                    encoder_state_c = tf.concat([encoder_state[0].c, encoder_state[1].c], axis = 1)
                    encoder_state_h = tf.concat([encoder_state[0].h, encoder_state[1].h], axis = 1)
                    self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_state_c, h = encoder_state_h)
                else: # multi layer
                    _encoder_state = list()
                    for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]):
                        partial_state_c = tf.concat([state_fw.c, state_bw.c], axis = 1)
                        partial_state_h = tf.concat([state_fw.h, state_bw.h], axis = 1)
                        partial_state = tf.contrib.rnn.LSTMStateTuple(c = partial_state_c, h = partial_state_h)
                        _encoder_state.append(partial_state)
                    self.encoder_state = tuple(_encoder_state)

		
	else:
	    raise ValueError('Invalid input %s' %self.enc_type)

        return self.encoder_output, self.encoder_state 


class Decoder(_BaseClass):
    def __init__(self, enc_type ='bi', 
            attn_type = 'bahdanau', voca_size = None,
            beam_width = 0, length_penalty_weight = 1,
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
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
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

        # Decoder initial state setting
        if (self.mode != tf.estimator.ModeKeys.PREDICT or self.beam_width == 0):
            initial_state = self.out_dec_cell.zero_state(dtype = self.dtype, batch_size = self.batch_size)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.out_dec_cell, helper, initial_state,
        	    output_layer = None)
        else:
            initial_state = self.out_dec_cell.zero_state(dtype = self.dtype, batch_size = self.batch_size * self.beam_width)
            print type(self.length_penalty_weight)
            print '----------------------------------'
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell = self.out_dec_cell,
                    embedding = embedding,
                    start_tokens = start_token,
                    end_token = end_token,
                    initial_state = initial_state,
                    beam_width = self.beam_width,
                    length_penalty_weight = self.length_penalty_weight)


        if self.mode == tf.estimator.ModeKeys.TRAIN:
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished = True, maximum_iterations = None)
            return outputs.rnn_output
            
        # Test with Beam decoding
        elif (self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0):
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished = False, maximum_iterations = self.max_iter)
            predictions = outputs.predicted_ids # [batch, length, beam_width]
            predictions = tf.transpose(predictions, [0, 2, 1]) # [batch, beam_width, length]
            predictions = predictions = predictions[:, 0, :] # [batch, length]
            return predictions


        else: # Greedy decoder (Test & Eval)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = self. max_iter)
            return outputs.rnn_output




    def set_attention_cell(self, memory, memory_length, encoder_state, enc_num_layer):
        self.batch_size = tf.shape(memory)[0]

    	dec_cell = self._create_cell()
            
        if (self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0):
            memory = tf.contrib.seq2seq.tile_batch(memory, self.beam_width)
            memory_length = tf.contrib.seq2seq.tile_batch(memory_length, self.beam_width)

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
	    raise ValueError('Unknown attention mechanism : %s' %self.attn_type)
