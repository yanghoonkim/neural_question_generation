import numpy as np
import tensorflow as tf

import sys
sys.path.append('submodule/')
from mytools import *


def q_generation(features, labels, mode, params):

    dtype = params['dtype']
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']   
    
    sentence = features['s'] # [batch, length]
    len_s = features['len_s']
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        question = features['q'] # label
        len_q = features['len_q']
    else:
        question = None
        len_q = None
    
    # Embedding for sentence, question and rnn encoding of sentence
    with tf.variable_scope('SharedScope'):
        # Embedded inputs
        # Same name == embedding sharing
        embd_s = embed_op(sentence, params, name = 'embedding')
        if question is not None:
            embd_q = embed_op(question[:, :-1], params, name = 'embedding')

        # Build encoder cell
        def gru_cell_enc():
            cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            return tf.contrib.rnn.DropoutWrapper(cell, 
                    output_keep_prob = 1 - params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1)
        def gru_cell_dec():
            cell = tf.nn.rnn_cell.GRUCell(hidden_size * 2)
            return tf.contrib.rnn.DropoutWrapper(cell,
                    output_keep_prob = 1 - params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1)

        encoder_cell_fw = gru_cell_enc() if params['encoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([gru_cell_enc() for _ in range(params['encoder_layer'])])
        encoder_cell_bw = gru_cell_enc() if params['encoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([gru_cell_enc() for _ in range(params['encoder_layer'])])


        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: last hidden state of encoder, [batch_size, num_units]
        #encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        #    encoder_cell, embd_s,
        #    sequence_length=len_s,
        #    dtype = tf.float32    
        #    )

        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw,
                encoder_cell_bw,
                inputs = embd_s,
                sequence_length = len_s,
                dtype = dtype)

        encoder_outputs = tf.concat(encoder_outputs, -1)
        encoder_state = tf.concat(encoder_state, -1) if type(encoder_state) is not tuple else tuple(tf.concat([state_fw, state_bw], -1)for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]))
        
    # This part should be moved into QuestionGeneration scope    
    with tf.variable_scope('SharedScope/EmbeddingScope', reuse = True):
        embedding_q = tf.get_variable('embedding')
 
    # Rnn decoding of sentence with attention 
    with tf.variable_scope('QuestionGeneration'):
        # Memory for attention
        attention_states = encoder_outputs

        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                hidden_size * 2, attention_states,
                memory_sequence_length=len_s)

        # batch_size should not be specified
        # if fixed, then the redundant eval_data will make error
        # it may related to bug of tensorflow api
        batch_size = attention_mechanism._batch_size

        # Build decoder cell
        decoder_cell = gru_cell_dec() if params['decoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([gru_cell_dec() for _ in range(params['decoder_layer'])])

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=hidden_size,
                initial_cell_state = encoder_state)

        decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, voca_size)

        # Helper for decoder cell
        if mode == tf.estimator.ModeKeys.TRAIN:
            len_q = tf.cast(len_q, tf.int32)
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs = embd_q,
                    sequence_length = len_q,
                    embedding = embedding_q,
                    sampling_probability = 0.25)
        else: # EVAL & TEST
            start_token = params['start_token'] * tf.ones([batch_size], dtype = tf.int32)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_q, start_token, params['end_token']
                    )

        # Decoder
        initial_state = decoder_cell.zero_state(dtype = dtype, batch_size = batch_size)
        projection_q = tf.layers.Dense(voca_size, use_bias = True)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state,
            output_layer=None)

        # Dynamic decoding
        if mode == tf.estimator.ModeKeys.TRAIN:
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = None)
        else: # Test & Eval
            max_iter = params['maxlen_q_dev']
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = max_iter)
        
        logits_q = outputs.rnn_output
        #logits_q = projection_q(outputs.rnn_output)

    
    # Predictions
    softmax_q = tf.nn.softmax(logits_q)
    predictions_q = tf.argmax(softmax_q, axis = -1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {
                    'question' : predictions_q
                    })
    # Loss
    label_q = tf.cast(question[:,1:], tf.int32, name = 'label_q')
    maxlen_q = params['maxlen_q_train'] if mode == tf.estimator.ModeKeys.TRAIN else params['maxlen_q_dev']
    current_length = tf.shape(logits_q)[1]
    def concat_padding():
        num_pad = maxlen_q - current_length
        padding = tf.zeros([batch_size, num_pad, params['voca_size']], dtype = dtype)

        return tf.concat([logits_q, padding], axis = 1)

    def slice_to_maxlen():
        return tf.slice(logits_q, [0,0,0], [batch_size, maxlen_q, params['voca_size']])

    logits_q = tf.cond(current_length < maxlen_q,
            concat_padding,
            slice_to_maxlen)
    
    weight_q = tf.sequence_mask(len_q, maxlen_q, dtype)

    loss_q = tf.contrib.seq2seq.sequence_loss(
            logits_q, 
            label_q,
            weight_q, # [batch, length]
            average_across_timesteps = True,
            average_across_batch = True,
            softmax_loss_function = None # default : sparse_softmax_cross_entropy
            )
    
    loss_reg = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss = loss_q + params['regularization'] * loss_reg

    # eval_metric for estimator
    eval_metric_ops = None

    # Summary
    tf.summary.scalar('loss_reg', loss_reg)
    tf.summary.scalar('loss_question', loss_q)
    tf.summary.scalar('total_loss', loss)


    # Optimizer
    learning_rate = params['learning_rate']
    if params['decay_step'] is not None:
        learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), params['decay_step'], params['decay_rate'], staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grad_and_var = optimizer.compute_gradients(loss, tf.trainable_variables())
    grad, var = zip(*grad_and_var)
    clipped_grad, norm = tf.clip_by_global_norm(grad, 5)
    train_op = optimizer.apply_gradients(zip(clipped_grad, var), global_step = tf.train.get_global_step())
        
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
