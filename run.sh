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
	PARAMS=basic_params
}


# Pass the first argument as the name of dataset
# Pass the second argument as mode
# Pass the third argument to control GPU usage
$1
$2

NUM_EPOCHS=5
MODEL_DIR=./store_model/$3

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
