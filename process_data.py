import pickle as pkl
import numpy as np
from collections import defaultdict

# Source file
TRAIN_SRC = 'data/xinyadu_data/src-train.txt'
TRAIN_TGT = 'data/xinyadu_data/tgt-train.txt'
DEV_SRC = 'data/xinyadu_data/src-dev.txt'
DEV_TGT = 'data/xinyadu_data/tgt-dev.txt'
TEST_SRC = 'data/xinyadu_data/src-test.txt'
TEST_TGT = 'data/xinyadu_data/tgt-test.txt'

# Target file
sentence_outfile_train = 'data/processed/train_sentence.npy'
question_outfile_train = 'data/processed/train_question.npy'

sentence_outfile_dev = 'data/processed/dev_sentence.npy'
question_outfile_dev = 'data/processed/dev_question.npy'

sentence_outfile_test = 'data/processed/test_sentence.npy'
question_outfile_test = 'data/processed/test_question.npy'


maxlen_s_train = 60
maxlen_q_train = 30

maxlen_s_dev = 60
maxlen_q_dev = 25

maxlen_s_test = 60
maxlen_q_test = 25

dic_size = 34000
vocab_dir = 'data/processed/vocab_xinyadu.dic'

# Read data and Check max-length >>>

# Train data
with open(TRAIN_SRC) as f:
    sentence_train = [line.split() for line in f.readlines()]

with open(TRAIN_TGT) as f:
    question_train = [line.split() for line in f.readlines()]

maxlen = max([len(sentence) for sentence in sentence_train])
print 'train sentences max length : %d '%maxlen

maxlen = max([len(sentence) for sentence in question_train])
print 'train questions max length : %d \n'%maxlen


# Dev data
with open(DEV_SRC) as f:
    sentence_dev = [line.split() for line in f.readlines()]

with open(DEV_TGT) as f:
    question_dev = [line.split() for line in f.readlines()]

maxlen = max([len(sentence) for sentence in sentence_dev])
print 'dev sentences max length : %d '%maxlen

maxlen = max([len(sentence) for sentence in question_dev])
print 'dev questions max length : %d \n'%maxlen

# Test data
with open(TEST_SRC) as f:
    sentence_test = [line.split() for line in f.readlines()]

with open(TEST_TGT) as f:
    question_test = [line.split() for line in f.readlines()] 
    
maxlen = max([len(sentence) for sentence in sentence_test])
print 'dev sentences max length : %d '%maxlen

maxlen = max([len(sentence) for sentence in question_test])
print 'dev questions max length : %d \n'%maxlen

# <<< Read data and Check max-length

# >>> Filtering data with max-length

print 'restrict max length of train_sentence as %d'%maxlen_s_train
print 'restrict max length of train_question as %d'%maxlen_q_train
print 'restrict max length of dev_sentence as %d'%maxlen_s_dev
print 'restrict max length of dev_question as %d'%maxlen_q_dev
print 'restrict max length of test_sentence as %d'%maxlen_s_test
print 'restrict max length of test_question as %d\n'%maxlen_q_test

def filter_with_maxlen(maxlen_s, maxlen_q, sentence, question):
    # Filtering with maxlen(sentence)
    temp_sentence = list()
    temp_question = list()
    
    for i, line in enumerate(sentence):
        if (len(line) <= maxlen_s):
            temp_sentence.append(line)
            temp_question.append(question[i])
            

    # Filtering with maxlen(question)
    filtered_sentence = list()
    filtered_question = list()

    for i, line in enumerate(temp_question):
        if len(line) <= maxlen_q:
            filtered_sentence.append(temp_sentence[i])
            filtered_question.append(line)
            
    return filtered_sentence, filtered_question

# Train data
filtered_sentence_train, filtered_question_train = filter_with_maxlen(
    maxlen_s_train, maxlen_q_train, sentence_train, question_train)        

# Dev data
filtered_sentence_dev, filtered_question_dev = filter_with_maxlen(
    maxlen_s_dev, maxlen_q_dev, sentence_dev, question_dev)

# Test data
filtered_sentence_test, filtered_question_test = filter_with_maxlen(
    maxlen_s_test, maxlen_q_test, sentence_test, question_test)

# Save filtered data
with open('data/processed/sentence_train.txt', 'w') as f:
    for line in filtered_sentence_train:
        f.write(' '.join(line) + '\n')

with open('data/processed/question_train.txt', 'w') as f:
    for line in filtered_question_train:
        f.write(' '.join(line) + '\n')
        
with open('data/processed/sentence_dev.txt', 'w') as f:
    for line in filtered_sentence_dev:
        f.write(' '.join(line) + '\n')
        
with open('data/processed/question_dev.txt', 'w') as f:
    for line in filtered_question_dev:
        f.write(' '.join(line) + '\n')

with open('data/processed/sentence_test.txt', 'w') as f:
    for line in filtered_sentence_test:
        f.write(' '.join(line) + '\n')
        
with open('data/processed/question_test.txt', 'w') as f:
    for line in filtered_question_test:
        f.write(' '.join(line) + '\n')
# <<< Filtering data with max-length

# Make vocab with filtered sentences and questions(train) >>>

all_sentence  = filtered_sentence_train + filtered_question_train

# Make vocab with word frequency
wordcount = defaultdict(int)
for sentence in all_sentence:
    for word in sentence:
        wordcount[word]+=1
sorted_wordlist = [(k, wordcount[k]) for k in sorted(wordcount, key=wordcount.get, reverse=True)]

print 'resize dictionary with %d most frequent words...'%dic_size
resized_dic = dict(sorted_wordlist[:dic_size])

word2idx = dict()
word2idx['<PAD>'] = 0
word2idx['<GO>'] = 1
word2idx['<EOS>'] = 2
word2idx['<UNK>'] = 3
idx = 4
for word in resized_dic:
    word2idx[word] = idx
    idx += 1

# Save dic
print 'save Dic File...'
with open(vocab_dir, 'w') as f:
    pkl.dump(word2idx, f)

# <<< Make vocab with filtered sentences and questions(train)

# Process data with vocab >>>
def process(data, vocab, maxlen, if_go = False):
    if if_go:
        maxlen = maxlen + 2 # include <GO> and <EOS>
    processed_data = list()
    length_data = list()
    for line in data:
        processed_data.append([])
        if if_go:
            processed_data[-1].append(word2idx['<GO>'])
        for word in line:
            if word in word2idx:
                processed_data[-1].append(word2idx[word])
            else:
                processed_data[-1].append(word2idx['<UNK>'])
        if if_go:
            processed_data[-1].append(word2idx['<EOS>'])
        length_data.append(len(processed_data[-1])) 
        processed_data[-1] = processed_data[-1] + [word2idx['<PAD>']] * (maxlen - len(processed_data[-1]))
    return processed_data, length_data

# Train data
processed_sentence_train, length_sentence_train = process(filtered_sentence_train, word2idx, maxlen_s_train, if_go = False)
processed_question_train, length_question_train = process(filtered_question_train, word2idx, maxlen_q_train, if_go = True)

# Eval data
processed_sentence_dev, length_sentence_dev = process(filtered_sentence_dev, word2idx, maxlen_s_dev, if_go = False)
processed_question_dev, length_question_dev = process(filtered_question_dev, word2idx, maxlen_q_dev, if_go = True)

# Test data
processed_sentence_test, length_sentence_test = process(filtered_sentence_test, word2idx, maxlen_s_test, if_go = False)
processed_question_test, length_question_test = process(filtered_question_test, word2idx, maxlen_q_test, if_go = True)

print 'Processing Complete'
# <<< Process data with vocab

np.save(sentence_outfile_train, processed_sentence_train)
np.save(question_outfile_train, processed_question_train)

np.save(sentence_outfile_dev, processed_sentence_dev)
np.save(question_outfile_dev, processed_question_dev)

np.save(sentence_outfile_test, processed_sentence_test)
np.save(question_outfile_test, processed_question_test)
print 'Saving Complete'


