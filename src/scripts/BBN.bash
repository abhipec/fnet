#!/bin/bash

# our model 
for ((i=1; i<=5; i++)); do
  time python main_our.py --dataset=BBN --data_directory=~/EACL-2017/fnet/data/processed/f1/ --char_embedding_size=200 --rnn_hidden_neurons=100 --char_rnn_hidden_neurons=200 --keep_prob=0.5 --learning_rate=0.0005 --joint_embedding_size=500 --epochs=50 --batch_size=1000 --use_mention --use_clean --uid=BBN_1.$i
done

# our no mention
#for ((i=1; i<=5; i++)); do
#  time python main_our.py --dataset=BBN --data_directory=~/EACL-2017/fnet/data/processed/f1/ --char_embedding_size=200 --rnn_hidden_neurons=100 --char_rnn_hidden_neurons=200 --keep_prob=0.5 --learning_rate=0.0005 --joint_embedding_size=500 --epochs=50 --batch_size=1000 --use_clean --uid=BBN_2.$i
#done

# our model all clean
#for ((i=1; i<=5; i++)); do
#  time python main_our.py --dataset=BBN --data_directory=~/EACL-2017/fnet/data/processed/f1/ --char_embedding_size=200 --rnn_hidden_neurons=100 --char_rnn_hidden_neurons=200 --keep_prob=0.5 --learning_rate=0.0005 --joint_embedding_size=500 --epochs=50 --batch_size=1000 --use_mention --uid=BBN_3.$i
#done
