#!/bin/bash
# our model 
for ((i=1; i<=5; i++)); do
  python main_tl_feature.py --dataset=BBN --data_directory=~/EACL-2017/fnet/data/processed/f4/ --learning_rate=0.0005 --joint_embedding_size=500 --epochs=10 --batch_size=1000 --use_clean --threshold=0.5 --uid=tf_unnorm.$i
done


for ((i=1; i<=5; i++)); do
  python main_tl_feature.py --dataset=OntoNotes --data_directory=~/EACL-2017/fnet/data/processed/f4/ --learning_rate=0.0005 --joint_embedding_size=500 --epochs=5 --batch_size=1000 --use_clean --threshold=1.9 --uid=tf_unnorm_OntoNotes_19.$i
done

