import numpy as np
import tensorflow as tf

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
