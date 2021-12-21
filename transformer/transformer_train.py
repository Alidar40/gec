"""
Reference: 
    codebase: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    torchtext load pretrained embeddings: http://anie.me/On-Torchtext/

Prelims:
    pip install torch numpy matplotlib spacy torchtext seaborn 
    python -m spacy download en 

Usage:
python transformer/transformer_train.py \
    -src data/test/ \
    -model data/models/ \
    -corpus lang8_small \
    -en basic -de basic

Other options for embeddings:
    -en basic -de basic
    -en glove -de basic
    -en glove -de glove
    -en elmo -de basic
    -en elmo -de elmo

TODO
    -en basic -de glove
    -en basic -de elmo
    -en glove -de elmo
    -en elmo -de glove
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from torchtext.legacy import data, datasets
import spacy

import wandb

import os
import sys
import random
import argparse

from model import MyIterator, LabelSmoothing, NoamOpt, MultiGPULossCompute, SimpleLossCompute
from model import make_model, rebatch, run_epoch, batch_size_fn, get_emb, greedy_decode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_name', '--EXP_NAME')
    parser.add_argument('-src', '--SRC_DIR')
    parser.add_argument('-model', '--MODEL_DIR')
    parser.add_argument('-corpus', '--DATA')
    parser.add_argument('-en', '--EN_EMB')
    parser.add_argument('-de', '--DE_EMB')
    parser.add_argument('-batch_size', '--BATCH_SIZE', type=int)
    parser.add_argument('-epochs', '--EPOCHS', type=int)
    parser.add_argument('-resume_checkpoint', '--RESUME_CHECKPOINT')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    EXP_NAME = args.EXP_NAME
    SRC_DIR = args.SRC_DIR
    MODEL_DIR = os.path.join(args.MODEL_DIR, EXP_NAME)
    DATA = args.DATA
    EN_EMB = args.EN_EMB
    DE_EMB = args.DE_EMB
    SEQ_TRAIN = True if DE_EMB == 'elmo' else False
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHES = args.EPOCHS
    resume_checkpoint = args.RESUME_CHECKPOINT

    # TODO currently hidden size is fixed, should be able to adjust 
    #      based on src and trg embeddings respectively
    # EMB_DIM should be multiple of h (default 8), look at MultiHeadedAttention
    if 'glove' in EN_EMB:
        EMB_DIM = 200
    elif 'elmo' in EN_EMB:
        EMB_DIM = 1024
    else:
        EMB_DIM = 512


    options_file = "embeddings/elmo/options.json"
    weight_file = "embeddings/elmo/weights.hdf5"

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # GPU to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = ("cpu")
    # devices = [0, 1, 2, 3]

    #####################
    #   Data Loading    #
    #####################
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    MIN_FREQ = 2

    spacy_en = spacy.load('en_core_web_lg')
    # spacy_en = spacy.load('ru_core_news_lg')
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    train = datasets.TranslationDataset(path=os.path.join(SRC_DIR, DATA),
            exts=('.train.src', '.train.trg'), fields=(TEXT, TEXT))
    val = datasets.TranslationDataset(path=os.path.join(SRC_DIR, DATA), 
            exts=('.val.src', '.val.trg'), fields=(TEXT, TEXT))

    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    random_idx = random.randint(0, len(train) - 1)
    print(train[random_idx].src)
    print(train[random_idx].trg)

    ###############
    #  Vocabuary  #
    ###############
    vocab_file = os.path.join(MODEL_DIR, '%s.vocab' % (DATA))
    if os.path.exists(vocab_file):
        TEXT.vocab = torch.load(vocab_file)
    else:
        print("Save %s vocabuary..." % (DATA), end='\t') 
        TEXT.build_vocab(train.src, min_freq=MIN_FREQ, vectors='glove.6B.200d')
        print("vocab size = %d" % (len(TEXT.vocab)))
        torch.save(TEXT.vocab, vocab_file)

    pad_idx = TEXT.vocab.stoi["<blank>"]

    #####################
    #   Word Embedding  #
    #####################
    encoder_emb, decoder_emb = get_emb(EN_EMB, DE_EMB, TEXT.vocab, device, 
                                       d_model=EMB_DIM,
                                       elmo_options=options_file, 
                                       elmo_weights=weight_file)

    ##########################
    #   Training the System  #
    ##########################
    model = make_model(len(TEXT.vocab), encoder_emb, decoder_emb, 
                       d_model=EMB_DIM).to(device)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print("Restart from last checkpoint...")
        model.load_state_dict(torch.load(resume_checkpoint))

    criterion = LabelSmoothing(size=len(TEXT.vocab), padding_idx=pad_idx, smoothing=0.1).to(device)
    model_opt = NoamOpt(EMB_DIM, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, 
                        betas=(0.9, 0.98), eps=1e-9))

    # calculate parameters
    total_params = sum(p.numel() for p in model.parameters()) // 1000000
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000
    rate = trainable_params / total_params
    print("Model parameters trainable (%d M) / total (%d M) = %f" % (trainable_params, total_params, rate))

    print("Training %s %s %s..." % (DATA, EN_EMB, DE_EMB))

    ### SINGLE GPU
    wandb.init(project="gec_transformer", name=EXP_NAME, mode="online")
    global_step = 0
    for epoch in range(1, EPOCHES+1):
        model.train()
        loss_compute = SimpleLossCompute(model.generator, criterion, opt=model_opt)
        global_step = run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model, loss_compute, TEXT.vocab, wandb, epoch, global_step,  seq_train=SEQ_TRAIN)

        model.eval()
        total_loss, total_tokens = 0, 0
        losses = list()
        for batch in (rebatch(pad_idx, b) for b in valid_iter):
            out = greedy_decode(model, TEXT.vocab, batch.src, batch.src_mask, trg=batch.trg)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            loss = loss / batch.ntokens
            losses.append(loss)
            wandb.log({
                'epoch': epoch,
                'global_step': global_step,
                'valid_loss': loss,
            })
            total_tokens += batch.ntokens

        wandb.log({
            'epoch': epoch,
            'global_step': global_step,
            'valid_epoch_loss': np.mean(losses),
        })

        print("Save model...")
        model_file = os.path.join(MODEL_DIR, 'loss_%s_step_%s.transformer.pt' % (np.mean(losses), global_step))
        torch.save(model.state_dict(), model_file)

        print("Epoch %d/%d - Loss: %f" % (epoch, EPOCHES, total_loss / total_tokens))


if __name__ == "__main__":
    main()



