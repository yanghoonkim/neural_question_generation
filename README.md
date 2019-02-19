# neural question_generation
Implemenration of &lt;Learning to Ask: Neural Question Generation for Reading Comprehension> by Xinya Du et al.

**The source code still needs to be modified**

1. **Model**

  - Embedding
    - Pretrained GloVe embeddings
    - Randomly initialized embeddings
  
  - RNN-based seq2seq
    - GRU/LSTM
  
  - To be updated
    - Beam decoder
    - Post-processing code for unknown words
    
2. **Dataset**

processed data provided by [Xinya Du et al.](https://arxiv.org/pdf/1705.00106.pdf)

## Requirements

- python 2.7
- numpy
- Tensorflow 1.4

## Usage

1. Data preprocessing

```
mkdir data/processed
python process_data.py
```

2. Download & process GloVe

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/
unzip data/glove.840B.300d.zip -d data/
python process_embedding.py # This will take a couple of minutes
```

3. Train model

```
# data_name : dataset name in run.sh
# model_name : the name of weight set, you can train multiple models with the same parameters by taking different model name 
bash run.sh [data_name] train [model_name]
# example : bash run.sh squad train model_0
```

4. Test model

```
mkdir result # only for the first time
bash run.sh [data_name] pred [model_name]
# example : bash run.sh squad pred model_0
# the result will be saved in the result folder
```

