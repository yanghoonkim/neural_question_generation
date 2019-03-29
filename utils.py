import numpy as np
import tensorflow as tf
import nltk

def embed_op(inputs, pre_embedding, voca_size, embedding_size = None, embedding_trainable = False, dtype  = tf.float32, name = 'embedding'):
    if pre_embedding == None:
        with tf.variable_scope('EmbeddingScope', reuse = tf.AUTO_REUSE):
            embedding = tf.get_variable(
                    name, 
                    [voca_size, embedding_size], 
                    dtype = dtype,

                    )
    else:
        embedding = np.load(pre_embedding)
        with tf.variable_scope('EmbeddingScope', reuse = tf.AUTO_REUSE):
            init = tf.constant_initializer(embedding)
            embedding_size = embedding.shape[-1]
            embedding = tf.get_variable(
                    name,
                    [voca_size, embedding_size],
                    initializer = init,
                    dtype = dtype,
                    trainable = embedding_trainable 
                    )

    tf.summary.histogram(embedding.name + '/value', embedding)
    return tf.nn.embedding_lookup(embedding, inputs), embedding


def bleu_score(labels, predictions,
               weights=None, metrics_collections=None,
               updates_collections=None, name=None):

    def _nltk_blue_score(labels, predictions):

        # slice after <eos>
        predictions = predictions.tolist()
        for i in range(len(predictions)):
            prediction = predictions[i]
            if 2 in prediction: # 2: EOS
                predictions[i] = prediction[:prediction.index(2)+1]

        labels = [
            [[w_id for w_id in label if w_id != 0]] # 0: PAD
            for label in labels.tolist()]
        predictions = [
            [w_id for w_id in prediction]
            for prediction in predictions]

        return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

    score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
    return tf.metrics.mean(score * 100)