#!/bin/bash

# our model BBN dataset
for ((i=1; i<=5; i++)); do
  time python main_our.py --dataset=BBN --data_directory=~/EACL-2017/fnet/data/processed/f3/ --char_embedding_size=200 --rnn_hidden_neurons=100 --char_rnn_hidden_neurons=200 --keep_prob=0.5 --learning_rate=0.0005 --joint_embedding_size=500 --epochs=10 --batch_size=1000 --use_clean --use_mention --finetune --finetune_directory=../ckpt/Wiki_1.1 --uid=T_BBN_model.$i
done

# our model OntoNotes dataset
for ((i=1; i<=5; i++)); do
  time python main_our.py --dataset=OntoNotes --data_directory=~/EACL-2017/fnet/data/processed/f3/ --char_embedding_size=200 --rnn_hidden_neurons=100 --char_rnn_hidden_neurons=200 --keep_prob=0.5 --learning_rate=0.0005 --joint_embedding_size=500 --epochs=5 --batch_size=800 --use_clean --use_mention --finetune --finetune_directory=../ckpt/Wiki_1.1 --uid=T_OntoNotes_model.$i
done

