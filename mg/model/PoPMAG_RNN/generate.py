import torch
import numpy as np
import os
import optparse
import sys
import torch.utils.data as Data

sys.path.append('/data2/qt/MusicGeneration/mg/model/')
import PoPMAG_RNN.config as config
from utils.data import Melody_Arrangement_Dataset
from PoPMAG_RNN.config import device
from PoPMAG_RNN.network import PoPMAG_RNN
from mg.model.utils.MuMIDI import MuMIDI_EventSeq


# pylint: disable=E1101,E1102

# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=1)

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/PoPMAG_RNN/save_model/256_256_2_epoch_4_3399.pth',
                      #default='/data2/qt/MusicGeneration/mg/model/Event_MelodyRNN/save_model/epoch_128.pth',
                      help = 'pth file containing the trained model')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/egs/dataset/lmd_matched_split/valid.pth')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/PoPMAG_RNN/output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=5)

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=1.0)

    parser.add_option('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=3)

    parser.add_option('-S', '--stochastic-beam-search',
                      dest='stochastic_beam_search',
                      action='store_true',
                      default=False)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.0)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


opt = getopt()

# ------------------------------------------------------------------------

output_dir = opt.output_dir
save_path = opt.save_path
data_path = opt.data_path
batch_size = opt.batch_size
max_len = opt.max_len
greedy_ratio = opt.greedy_ratio
use_beam_search = opt.beam_size > 0
stochastic_beam_search = opt.stochastic_beam_search
beam_size = opt.beam_size
temperature = opt.temperature
init_zero = opt.init_zero

use_beam_search = True
if use_beam_search:
    greedy_ratio = 'DISABLED'
else:
    beam_size = 'DISABLED'

assert os.path.isfile(save_path), f'"{save_path}" is not a file'

# ------------------------------------------------------------------------

print('-' * 70)
print('Saved model path:', save_path)
print('Batch size:', batch_size)
print('Max length:', max_len)
print('Greedy ratio:', greedy_ratio)
print('Beam size:', beam_size)
print('Beam search stochastic:', stochastic_beam_search)
print('Output directory:', output_dir)
print('Temperature:', temperature)
print('Init zero:', init_zero)
print('-' * 70)


# ========================================================================
# Generating
# ========================================================================
model_config = config.model
print(model_config)#{'init_dim': 32, 'event_dim': 308, 'hidden_dim': 256, 'rnn_layers': 2, 'dropout': 0.5}
model = PoPMAG_RNN(**model_config)
model.load_state_dict(torch.load(save_path))
#device = torch.device('cpu')
model.to(device)
model.eval()
print(model)
print('-' * 70)

def load_dataset():
    global data_path
    dataset = Melody_Arrangement_Dataset.load_file(data_path)
    dataset_size = len(dataset.melody_seqs)
    assert dataset_size > 0
    return dataset

print('Loading dataset')
# print(os.path.isdir(data_path))
dataset = load_dataset()

if init_zero:
    init = torch.zeros(batch_size, model.init_dim).to(device)
else:
    init = torch.randn(batch_size, model.init_dim).to(device)

num_workers = 0 if sys.platform.startswith('win32') else 10
batch_gen = Data.DataLoader(dataset,
                            batch_size,
                            collate_fn=dataset.FastBatchify,
                            shuffle=True,
                            drop_last=True,
                            num_workers=num_workers)

os.makedirs(output_dir, exist_ok=True)

def write_midi(path, output):
    event_len = len(output)
    event = MuMIDI_EventSeq.to_event(output)
    MuMIDI_EventSeq.write_midi(event, path)
    return event_len

with torch.no_grad():
    for iteration, (src, src_mask, tar, tar_mask, label, label_mask) in enumerate(batch_gen):
        # print(device)
        src = src.to(device)
        src_mask = src_mask.to(device)
        # tar = tar.to(device)
        # tar_mask = tar_mask.to(device)
        # print(f'src={src.shape}')
        # print(f'tar={tar.shape}')
        # print(f'label={label.shape}')

        comp_src = model.compression(src)

        outputs = model.generate_arrangement(init, comp_src, src_mask, max_len)
        # ========================================================================
        # Saving
        # ========================================================================

        for i, output in enumerate(outputs):
            name = f'iter-{iteration}-pred-{i:03d}.mid'
            path = os.path.join(output_dir, name)
            output = output.cpu().numpy().T
            event_len = write_midi(path, output)
            print(f'===> {path} ({event_len} events)')

            name = f'iter-{iteration}-gt-{i:03d}.mid'
            path = os.path.join(output_dir, name)
            event_len = write_midi(path)
            print(f'===> {path} ({event_len} events)')
        break

print('Done')
# outputs = outputs.cpu().numpy().T  # [batch, steps]



# espnet/egs2/jsut/tts1 -> 3d (ddl:4.7 12:00pm)