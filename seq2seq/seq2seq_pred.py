# test_src as argument
# pickle: train_dataset, encoder, decoder, opts
# python seq2seq_pred.py -test_src ./data/lang8_english_src_test_100k.txt

from model import *
import argparse
import pickle
from allennlp.modules.elmo import Elmo, batch_to_ids
import h5py

def parse_args():
    parser = argparse.ArgumentParser()        
    parser.add_argument('-test_src')
    parser.add_argument('-exp_name', '--EXP_NAME')
    parser.add_argument('-model', '--MODEL_DIR')
    args = parser.parse_args()
    return args

def load_model(prefix):
    train_dataset = pickle.load(open(os.path.join(prefix,'train_dataset.pkl'),'rb'))
    encoder = pickle.load(open(os.path.join(prefix,'encoder.pkl'),'rb'))
    decoder = pickle.load(open(os.path.join(prefix,'decoder.pkl'),'rb'))
    opts = pickle.load(open(os.path.join(prefix,'opts.pkl'),'rb'))
    return train_dataset, encoder, decoder, opts

def main():
    args = parse_args()
    test_src = args.test_src
    exp_name = args.EXP_NAME
    model_dir = os.path.join(args.MODEL_DIR, exp_name)
    train_dataset, encoder, decoder, opts = load_model(model_dir)
    test_src_texts = []
    with codecs.open(test_src, 'r', 'utf-8') as f:
        test_src_texts = f.readlines()
    print(test_src_texts[:5])
    out_texts = []

    # Pretrained embedding: elmo 
    elmo = None
    if opts.pretrained_embeddings == 'elmo_input' or opts.pretrained_embeddings == 'elmo_both':
        options_file = "embeddings/elmo/options.json"
        weight_file = "embeddings/elmo/weights.hdf5"
        elmo = Elmo(options_file, weight_file, 1, dropout=0)

    for src_text in test_src_texts:
        _, out_text, _ = translate(src_text.strip(), train_dataset, encoder, decoder, opts, elmo, max_seq_len=opts.max_seq_len)
        out_texts.append(out_text)
    print(out_texts[:5])
    count = 0
    with codecs.open('seq2seq_pred.txt', 'w', 'utf-8') as f:
        for text in out_texts:
            count += 1
            f.write(text + '\n')
    print('Done testing')
    
if __name__ == '__main__':
    main()
