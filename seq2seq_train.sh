python seq2seq/seq2seq_train.py \
  -exp_name glove_synthetic10k \
  -src data/synthetic_10k/ \
  -model seq2seq/experiments/ \
  -corpus synthetic \
  -epochs 100 \
  -batch_size 24 \
  -emb_type glove \
#  -resume_checkpoint experiments/local_elmo/checkpoints/acc_96.31_loss_80.25_step_2500.pt

