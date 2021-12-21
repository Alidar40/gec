cd data/synthetic_1k

python ../../preprocessing/prepare_csv.py \
    -i synthetic.src \
    -train synthetic.train.src \
    -train_r 0.7 \
    -test synthetic.test.src \
    -test_r 0.15 \
    -val synthetic.val.src \
    -val_r 0.15

python ../../preprocessing/prepare_csv.py \
    -i synthetic.trg \
    -train synthetic.train.trg \
    -train_r 0.7 \
    -test synthetic.test.trg \
    -test_r 0.15 \
    -val synthetic.val.trg \
    -val_r 0.15

cd -
