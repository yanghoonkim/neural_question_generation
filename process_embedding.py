import numpy as np
import pickle as pkl

# Load & process GloVe
data_dir = 'data/'
glove = 'glove.840B.300d'

print 'Reading Glove...'
f = open(data_dir + glove + '.txt')
lines = f.readlines()
f.close()

print 'Processing Glove...\n'
embedding = dict()
for line in lines:
    splited = line.split()
    embedding[splited[0]] = map(float, splited[1:])

# Save glove as dic file
np.save(data_dir + 'processed/' + glove + '.dic', embedding)


# Make pre-trianed embedding with GloVe
with open('data/processed/vocab_xinyadu.dic') as f:
    vocab = pkl.load(f)
    
embedding_vocab = np.tile(embedding['UNKNOWN'],[len(vocab),1])
'''
vocab['<PAD>'] = 0
vocab['<GO>'] = 1
vocab['<EOS>'] = 2
vocab['<UNK>'] = 3
'''
embedding_vocab[0] = 0.0 # vocab['<PAD>'] = 1
embedding_vocab[1] = embedding['<s>']
embedding_vocab[2] = embedding['EOS']
embedding_vocab[3] = embedding['UNKNOWN']

unk_num = 0
for word, idx in vocab.items():
    if word in embedding:
        embedding_vocab[idx] = embedding[word]
    else:
        unk_num += 1


np.save('data/processed/glove_embedding.npy', embedding_vocab)

# check how many unknown words
print 'vocab size : %d' %len(embedding_vocab)
print 'unknown word size : %d' %unk_num
