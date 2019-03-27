#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash
train(){
	MODE='train'
}

eval(){
	MODE='eval'
}

pred(){
	MODE='pred'
}


squad(){
	TRAIN_SENTENCE='data/processed/train_sentence.npy'
	TRAIN_QUESTION='data/processed/train_question.npy'
	DEV_SENTENCE='data/processed/dev_sentence.npy'
	DEV_QUESTION='data/processed/dev_question.npy'
	TEST_SENTENCE='data/processed/test_sentence.npy'
	PRED_DIR='result/pred.txt'
}


# Pass the first argument as mode
# Pass the second argument as the name of dataset
# Pass the third argument as the hyperparameter setting
# Pass the fourth argument as training epochs
$1
$2

PARAMS=$3
MODEL_DIR=./store_model/$PARAMS
NUM_EPOCHS=$4

python main.py \
	--mode=$MODE \
	--train_sentence=$TRAIN_SENTENCE \
	--train_question=$TRAIN_QUESTION \
	--eval_sentence=$DEV_SENTENCE \
	--eval_question=$DEV_QUESTION \
	--test_sentence=$TEST_SENTENCE \
	--dic_dir=$DIC_DIR \
	--model_dir=$MODEL_DIR \
	--pred_dir=$PRED_DIR \
	--params=$PARAMS \
	--num_epochs=$NUM_EPOCHS
