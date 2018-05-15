import argparse
import pickle as pkl
import numpy as np
import tensorflow as tf

import params
import model

FLAGS = None

def remove_eos(sentence, eos = '<EOS>', pad = '<PAD>'):
    if eos in sentence:
        return sentence[:sentence.index(eos)] + '\n'
    elif pad in sentence:
        return sentence[:sentence.index(pad)] + '\n'
    else:
        return sentence + '\n'

def write_result(predict_results, dic_dir):
    print 'Load dic file...'
    with open(dic_dir) as dic:
        dic_file = pkl.load(dic)
    reversed_dic = dict((y,x) for x,y in dic_file.iteritems())
    
    print 'Writing into file...'
    with open(FLAGS.pred_dir, 'w') as f:
        while True:
            try : 
                output = predict_results.next()
                indices = [reversed_dic[index] for index in output['question']]
                sentence = ' '.join(indices)
                sentence = remove_eos(sentence)
                f.write(sentence.encode('utf-8'))

            except StopIteration:
                break
    
def main(unused):
    
    # Enable logging for tf.estimator
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Config
    config = tf.contrib.learn.RunConfig(
            model_dir = FLAGS.model_dir, 
            keep_checkpoint_max = 10, 
            save_checkpoints_steps = 100)
    
    # Load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # Define estimator
    q_generation = model.q_generation(model_params)
    nn = tf.estimator.Estimator(model_fn=q_generation.run, config = config, params=model_params)
    
    # Load training data
    train_sentence = np.load(FLAGS.train_sentence) # train_data
    train_question = np.load(FLAGS.train_question) # train_label

    # Data shuffling for training data
    permutation = np.random.permutation(len(train_sentence))
    train_sentence = train_sentence[permutation]
    train_question = train_question[permutation]
    
    # Training input function for estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'enc_inputs': train_sentence, 'dec_inputs': train_question},
        y=None, # useless value
        batch_size = model_params['batch_size'],
        num_epochs=FLAGS.num_epochs,
        shuffle=True)
    

    # Load evaluation data
    eval_sentence = np.load(FLAGS.eval_sentence)
    eval_question = np.load(FLAGS.eval_question)

    # Evaluation input function for estimator
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'enc_inputs': eval_sentence, 'dec_inputs': eval_question},
        y = None,
        batch_size = model_params['batch_size'],
        num_epochs=1,
        shuffle=False)

    # define experiment
    exp_nn = tf.contrib.learn.Experiment(
            estimator = nn, 
            train_input_fn = train_input_fn, 
            eval_input_fn = eval_input_fn,
            train_steps = None,
            min_eval_frequency = 100)

    # train and evaluate
    if FLAGS.mode == 'train':
        exp_nn.train_and_evaluate()
    
    elif FLAGS.mode == 'eval':
        exp_nn.evaluate(delay_secs = 0)

    else: # 'pred'
        # Load test data
        test_sentence = np.load(FLAGS.test_sentence)

        # prediction input function for estimator
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {'enc_inputs' : test_sentence},
                y = None,
                batch_size = model_params['batch_size'],
                num_epochs = 1,
                shuffle = False)

        # prediction
        predict_results = nn.predict(input_fn = pred_input_fn)

        # write result(question) into file
        write_result(predict_results, FLAGS.dic_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'train, eval')
    parser.add_argument('--train_sentence', type = str, default= '', help = 'path to the training sentence.')
    parser.add_argument('--train_question', type = str, default = '', help = 'path to the training question.')

    parser.add_argument('--eval_sentence', type = str, default = '', help = 'path to the evaluation sentence. ')
    parser.add_argument('--eval_question', type = str, default = '', help = 'path to the evaluation question.')

    parser.add_argument('--test_sentence', type = str, default = '', help = 'path to the test sentence.')
    
    parser.add_argument('--dic_dir', type = str, help = 'path to the dictionary')
    parser.add_argument('--model_dir', type = str, help = 'path to save the model')
    parser.add_argument('--pred_dir', type = str, help = 'path to save the predictions')
    parser.add_argument('--params', type = str, help = 'parameter setting')
    parser.add_argument('--num_epochs', type = int, default = 10, help = 'training epoch size')
    FLAGS = parser.parse_args()

    tf.app.run(main)
