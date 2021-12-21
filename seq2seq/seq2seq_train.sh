python seq2seq_train.py \
  -exp_name elmo_synthetic10k \
  -src ../data/synthetic_10k/ \
  -model experiments/ \
  -corpus synthetic \
  -epochs 100 \
  -batch_size 24 \
  -emb_type elmo \
#  -resume_checkpoint experiments/local_elmo/checkpoints/acc_96.31_loss_80.25_step_2500.pt

