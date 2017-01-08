#!/bin/bash

# BBN
for ((i=1; i<=5; i++)); do
  time python main_shimaoka.py --dataset=BBN --data_directory=~/EACL-2017/fnet/data/processed/f2/ --attention_size=50 --batch_size=1000 --epochs=50 --keep_prob=0.5 --learning_rate=0.0005 --rnn_hidden_neurons=100 --uid=Shimaoka_BBN.$i
done

# OntoNotes
for ((i=1; i<=5; i++)); do
  time python main_shimaoka.py --dataset=OntoNotes --data_directory=~/EACL-2017/fnet/data/processed/f2/ --attention_size=50 --batch_size=1000 --epochs=10 --keep_prob=0.5 --learning_rate=0.0005 --rnn_hidden_neurons=100 --uid=Shimaoka_OntoNotes.$i
done

# Wiki
for ((i=1; i<=5; i++)); do
  time python main_shimaoka.py --dataset=Wiki --data_directory=~/EACL-2017/fnet/data/processed/f2/ --attention_size=50 --batch_size=1000 --epochs=15 --keep_prob=0.5 --learning_rate=0.001 --rnn_hidden_neurons=100 --uid=Shimaoka_Wiki.$i
done
