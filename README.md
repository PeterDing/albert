# An Implementation for ALBERT with Tensorflow 2.0

An Implementation of [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) with Tensorflow 2.0.

## Create Training Dataset

```
python3 create_pretraining_data.py \
    --file_patterns="data/*txt" \
    --max_seq_length=512 \
    --max_mask_length=20 \
    --tfrecord_file_num=8 \
    --whole_word_mask=False \
    --vocab_file='data/vocab.txt'
```

## Pre Training

```
python3 run_pretraining.py \
    --strategy=mirror \
    --model_type=base \
    --train_files='data-training/*' \
    --eval_files='data-training/*'
```
