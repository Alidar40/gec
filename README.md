# Grammatical Error Correction - Home Assignment
Home assignment to conduct multiple experiments for Grammatical Error Correction task

Reference:

https://github.com/michaelchen110/Grammar-Correction/

https://github.com/grammarly/gector

## Overview
Были рассмотрены три различные нейросетевые архитектуры применительно к задаче исправления грамматических ошибок:
1. Sequence to sequence
2. Transformer
3. GECToR

## Requirements

Для seq2seq и transformer:
```
Python 3.9
Cuda 11.4
```

Для gector:
```angular2html
Python 3.7 обязательно
Используйте отдельный venv, файл requirements.txt лежит в gector/
```


## Dataset
Для seq2seq и transformer использовался синтетический датасет грамматических ошибок.
Тот же, что использовали авторы GECToR.

https://github.com/awasthiabhijeet/PIE/tree/master/errorify

Для ускорения обучения эксперименты были проведены только на первых 1000 датасета.

Чтобы разбить датасет на train/test/val, воспользуйтесь скриптом train_test_split.sh

<hr/>
Для GECToR использовалась русскоязычная часть датасета cLang-8

https://github.com/google-research-datasets/clang8

Датасет также необходимо предобработать. См. инструкцию в ```gector/README.md```

## Embeddings
Эксперименты были проведены с использованием двух раличных эмбеддингов - Elmo и Glove.

### Elmo
Скачайте эмбединги Elmo и поместите их в ```embeddings/elmo```
```
wget -P data/embs/ -O options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget -P data/embs/ -O weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
```

### Glove

```
python -m spacy download en_core_web_lg 
```

## Transformer Quickstart

### 1. Обучение

Скриптом запуска обучения является ```transformer/transformer_train.sh```:

```
python transformer_train.py \
  -exp_name local_elmo \
  -src ../data/synthetic_1k/ \
  -model experiments/ \
  -corpus synthetic \
  -epochs 50 \
  -batch_size 256 \
  -en elmo \
  -de elmo \
  -resume_checkpoint experiments/local_elmo/synthetic.glove.glove.transformer.pt
```

Здесь можно задать различные параметры обучения: тип эмбеддингов (```elmo```,```glove```), размер батча и пр.

### 2. Инференс
Скриптом запуска инференса является ```transformer/transformer_pred.sh```:
```
python transformer_pred.py \
  -exp_name local_elmo \
  -model experiments/ \
  -corpus synthetic \
  -checkpoint synthetic.glove.glove.transformer.pt \
  -en elmo \
  -de elmo
```

Тестовое предложение: ```She see Tom is catched by policeman in park at last night. ``` (находится в ```./test.src```)

Результат работы после обучения 15 эпох (~1.5 часа на ноутбучной GTX-3060) с эмбеддингами Elmo.
```
Predicting synthetic elmo elmo ...
Source: She see <unk> is <unk> by <unk> in <unk> at last night . 
Target: She saw <unk> caught by a <unk> in the <unk> last night .
Translation: She of 
```
Как видим, качества нет вообще как такового. В исходном предложении много ```<unk>``` - надо либо файнтюнить эмбединги, либо брать новые

```
Load synthetic vocabuary; vocab size = 1848
Predicting synthetic glove glove ...
Source: She see <unk> is <unk> by <unk> in <unk> at last night .
Target: She saw <unk> caught by a <unk> in the <unk> last night .
Translation: She <unk> is <unk> is a <unk> <unk> in at Mr. Gross 's month . 
```

Аналогичная ситуация

### 3. Эксперименты
1. Эмбеддинги elmo
2. Эмбеддинги glove
3. Эмбеддинги glove, с увеличеным количеством слоёв с 6 до 8 и увеличеным dropout с 0.1 до 0.3, а так же в 10 раз больший датасет


## Seq2seq Quickstart

### 1. Обучение

Скриптом запуска обучения является ```seq2seq/seq2seq_train.sh```:

```
python seq2seq_train.py \
  -exp_name local_elmo_fixed \
  -src ../data/synthetic_1k/ \
  -model experiments/ \
  -corpus synthetic \
  -epochs 100 \
  -batch_size 32 \
  -emb_type elmo \
  -resume_checkpoint experiments/local_elmo/checkpoints/acc_96.31_loss_80.25_step_2500.pt
```

### 2. Инференс
Скриптом запуска инференса является ```transformer/transformer_pred.sh```:
```
python seq2seq_pred.py \
  -exp_name local_elmo \
  -model experiments/ \
  -test_src ./test.src
```

```
['She see Tom is catched by policeman in park at last night.']
['some how is being how how by them how ; last 1 how some how by how how how at last last 1 how how being been how how by them at last being how how ; might some how by them how how ; last been how their some how how how at last being how how by them how ; last been how their some how how how at last being how how by them how ; last been how their some how how how at last being how how by them how ; last been how their some how how how at last being how how by them how ; last been how their some how how how at last being how how by them how ; last been how their some how how how at last being how how by them how ; last been how']
```

Ничего не обучилось :(

### 3. Эксперименты
1. Эмбеддинги elmo
2. Эмбеддинги elmo и всего один LSTM-слой
3. Эмбеддинги elmo и fix_embeddings=False
4. Эмбеддинги elmo и hidsize=1024
5. Эмбеддинги elmo и в 10 раз больший датасет
6. Эмбеддинги glove и в 10 раз больший датасет


## GECToR Quickstart

### 1. Обучение

```
python train.py --train_set ../data/gector_lang8_russian/train.gec --dev_set ../data/gector_lang8_russian/test.gec --model_dir ./models/rubert-tiny2  --transformer_model cointegrated/rubert-tiny2 --n_epoch 1000
```
Доп. параметры см. в ```gector/README.md```


### 2. Инференс
```
python predict.py --model_path ./models/rubert-tiny2/best.th --vocab_path ./models/rubert-tiny2/vocabulary --input_file ../test_rus.src  --output_file pred.txt --transformer_model cointegrated/rubert-tiny2
```

Что получилось:
```
Input: She see Tom is catched by policeman in park at last night.
Pred:  She said Tom was catched by policeman in the park last night.
```

Изменил смысл предложения, но лучше, учитывая сколько и на каком датасете это обучали.

### 3. Эксперименты
1. Предтренированный ruRoberta-large с датасетом cLang8 Russian
2. Предтренированный rubert-tiny2 с датасетом cLang8 Russian
3. Предтренированный roberta-base на синтетическом датасете размером 10k

