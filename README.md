# neural_question_generation
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
# data_name : dataset name which is defined in run.sh
# hyperparameters : hyperparameters setting which is defined in params.py
# epochs: training epochs

bash run.sh train [data_name] [hyperparameters] [epochs]
# example : bash run.sh trian squad basic_params 10
```

4. Test model

```
mkdir result # only for the first time, predicted result will be saved here
bash run.sh pred [data_name] [hyperparameters] 0 
# example : bash run.sh pred squad basic_params 0 
```

