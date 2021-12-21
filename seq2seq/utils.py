import os
import pickle

import torch

def save_model(prefix, train_dataset, encoder, decoder, opts):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    pickle.dump(train_dataset, open(os.path.join(prefix, 'train_dataset.pkl'), 'wb'))
    pickle.dump(encoder, open(os.path.join(prefix, 'encoder.pkl'), 'wb'))
    pickle.dump(decoder, open(os.path.join(prefix, 'decoder.pkl'), 'wb'))
    pickle.dump(opts, open(os.path.join(prefix, 'opts.pkl'), 'wb'))


def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    total_accuracy, total_loss, global_step):
    checkpoint = {
        'opts': opts,
        'global_step': global_step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }

    checkpoint_path = 'seq2seq/experiments/%s/checkpoints/acc_%.2f_loss_%.2f_step_%d.pt' % (
        experiment_name, total_accuracy, total_loss, global_step)

    directory, filename = os.path.split(os.path.abspath(checkpoint_path))

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path
