# FNET

## Publication
Fine-Grained Entity Type Classification by Jointly Learning Representations and Label Embeddings. Abhishek, Ashish Anand and Amit Awekar. EACL 2017.

## data

Download the necessary data as per instructions mentioned in data/processed/f1/README.md file. 

Directory structure:

- /home/
  - EACL-2017
    - fnet
    - glove.840B.300d

## dependencies
Python3 version of TensorFlow (0.10.0rc0) framework is used in this experiment.

``` bash
pip install numpy docopt pandas plotly matplotlib scipy sklearn 
```
Compile Cpp libraries.
```bash
cd src/lib
bash compile_gcc_5.bash
```
## run

```bash
cd src
bash scripts/BBN.bash
bash scripts/OntoNotes.bash
bash scripts/Wiki.bash
```
This will create model checkpoints in the ckpt directory.

Please have a look at the scripts and modify necessary variables.

Report result:

```bash
python report_results.py ~/EACL-2017/fnet/ckpt/
```

## Feature level transfer learning experiment
Download the necessary data as per instructions mentioned in data/processed/f4/README.md file.

```bash
bash scripts/tl.bash
python report_results.py ~/EACL-2017/fnet/ckpt/
```

## Preprocessing steps (Optional)
These steps will convert the original data https://github.com/shanzhenren/AFET to tfrecord format used in this code.

Download the necessary data as per instructions mentioned in data/AFET/dataset/README.md file.

Also download and extract GloVe vectors (http://nlp.stanford.edu/data/glove.840B.300d.zip) in glove.840B.300d directory.

Dataset names used: BBN, Wiki and OntoNotes.

Preprocess data and generate train, development and test set. 

```bash
cd data_processing/
python sanitizer.py BBN ~/EACL-2017/fnet/data/AFET/ 10 ~/EACL-2017/fnet/data/sanitized/

```
Convert json to Tfrecord format
```bash
python data_processing/json_to_tfrecord.py BBN ~/EACL-2017/fnet/data/sanitized/ ~/EACL-2017/glove.840B.300d/glove.840B.300d.txt f1 ~/EACL-2017/fnet/data/processed/
python data_processing/json_to_tfrecord.py BBN ~/EACL-2017/fnet/data/sanitized/ ~/EACL-2017/glove.840B.300d/glove.840B.300d.txt f2 ~/EACL-2017/fnet/data/processed/
python data_processing/json_to_tfrecord.py BBN ~/EACL-2017/fnet/data/sanitized/ ~/EACL-2017/glove.840B.300d/glove.840B.300d.txt f3 ~/EACL-2017/fnet/data/processed/
```

| data_format | alias | remarks |
|---|---|---|
| our  | f1  | Used in our, our-NoM, our-AllC| 
| Attentive  |  f2 | Used in Attentive| 
| transfer-learning-model  | f3   | Used in model level transfer learning| 


### Transfer learning experiments
1. Train our model on Wiki dataset.
2. Note down its uid. 
3. Modify ../ckpt/uid/checkpint file such that it points to the best performing checkpoint. 
4. Change the fintune_directory parameter in the following scripts to include uid noted in step 2. 

#### Model level transfer learning
```bash
bash scripts/transfer_learning_model.bash
```
#### Feature level transfer learning
```bash
bash scripts/transfer_learning_feature_dumping.bash
bash scripts/tl.bash
```
Report result
```bash
python report_results.py ~/EACL-2017/fnet/ckpt/
```

### type-wise analysis
Please change the dataset and the path of result file that need to be analysed type wise.
```bash
python class_wise_analysis.py --all_labels_file=../data/sanitized/BBN/sanitized_labels.txt  --json_file=../data/sanitized/BBN/sanitized_test.json --result_file=../ckpt/Wiki_1.2/result_7.txt --dataset=Wiki
```
