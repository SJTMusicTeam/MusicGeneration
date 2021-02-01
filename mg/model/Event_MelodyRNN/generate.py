import torch
import numpy as np
import os
import optparse
import sys
sys.path.append('/data2/qt/MusicGeneration/mg/model/')

import Event_MelodyRNN.config as config
from Event_MelodyRNN.config import device
from Event_MelodyRNN.network import Event_Melody_RNN
from utils.sequence import EventSeq, Control, ControlSeq
import utils.shared as utils

# pylint: disable=E1101,E1102

# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    # parser.add_option('-c', '--control',
    #                   dest='control',
    #                   type='string',
    #                   default=None,
    #                   help=('control or a processed data file path, '
    #                         'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
    #                         '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
    #                         '";3" (which gives all pitches the same probability), '
    #                         'or "/path/to/processed/midi/file.data" '
    #                         '(uses control sequence from the given processed data)'))

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=8)

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/Event_MelodyRNN/save_model/epoch_271.pth',
                      #default='/data2/qt/MusicGeneration/mg/model/Event_MelodyRNN/save_model/epoch_128.pth',
                      help = 'pth file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/Event_MelodyRNN/output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=10000)

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=1.0)

    parser.add_option('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=0)

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
batch_size = opt.batch_size
max_len = opt.max_len
greedy_ratio = opt.greedy_ratio
use_beam_search = opt.beam_size > 0
stochastic_beam_search = opt.stochastic_beam_search
beam_size = opt.beam_size
temperature = opt.temperature
init_zero = opt.init_zero

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
model = Event_Melody_RNN(**model_config)
model.load_state_dict(torch.load(save_path))
#device = torch.device('cpu')
model.to(device)
model.eval()
print(model)
print('-' * 70)

if init_zero:
    init = torch.zeros(batch_size, model.init_dim).to(device)
else:
    init = torch.randn(batch_size, model.init_dim).to(device)

with torch.no_grad():
    if use_beam_search:
        outputs = model.beam_search(init, max_len, beam_size,
                                    temperature=temperature,
                                    stochastic=stochastic_beam_search,
                                    verbose=True)
    else:
        outputs = model.generate(init, max_len,
                                 greedy=greedy_ratio,
                                 temperature=temperature,
                                 verbose=True)

outputs = outputs.cpu().numpy().T  # [batch, steps]


# ========================================================================
# Saving
# ========================================================================

os.makedirs(output_dir, exist_ok=True)

for i, output in enumerate(outputs):
    name = f'output-{i:03d}.mid'
    path = os.path.join(output_dir, name)
    n_notes = utils.event_indeces_to_midi_file(output, path)
    print(f'===> {path} ({n_notes} notes)')
