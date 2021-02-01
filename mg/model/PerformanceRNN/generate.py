import torch
import numpy as np
import os
import sys
import optparse
import utils
from PerformanceRNN.config import device, model as model_config
from PerformanceRNN.network import PerformanceRNN
from utils.sequence import EventSeq, Control, ControlSeq

# pylint: disable=E1101,E1102

# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--control',
                      dest='control',
                      type='string',
                      default=None,
                      help=('control or a processed data file path, '
                            'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
                            '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
                            '";3" (which gives all pitches the same probability), '
                            'or "/path/to/processed/midi/file.data" '
                            '(uses control sequence from the given processed data)'))

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=8)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=0)

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
sess_path = opt.sess_path
batch_size = opt.batch_size
max_len = opt.max_len
greedy_ratio = opt.greedy_ratio
control = opt.control
use_beam_search = opt.beam_size > 0
stochastic_beam_search = opt.stochastic_beam_search
beam_size = opt.beam_size
temperature = opt.temperature
init_zero = opt.init_zero

if use_beam_search:
    greedy_ratio = 'DISABLED'
else:
    beam_size = 'DISABLED'

assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

if control is not None:
    if os.path.isfile(control) or os.path.isdir(control):
        if os.path.isdir(control):
            files = list(utils.find_files_by_extensions(control))
            assert len(files) > 0, f'no file in "{control}"'
            control = np.random.choice(files)
        _, compressed_controls = torch.load(control)
        controls = ControlSeq.recover_compressed_array(compressed_controls)
        if max_len == 0:
            max_len = controls.shape[0]
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)
        control = f'control sequence from "{control}"'

    else:
        pitch_histogram, note_density = control.split(';')
        pitch_histogram = list(filter(len, pitch_histogram.split(',')))
        if len(pitch_histogram) == 0:
            pitch_histogram = np.ones(12) / 12
        else:
            pitch_histogram = np.array(list(map(float, pitch_histogram)))
            assert pitch_histogram.size == 12
            assert np.all(pitch_histogram >= 0)
            pitch_histogram = pitch_histogram / pitch_histogram.sum() \
                if pitch_histogram.sum() else np.ones(12) / 12
        note_density = int(note_density)
        assert note_density in range(len(ControlSeq.note_density_bins))
        control = Control(pitch_histogram, note_density)
        controls = torch.tensor(control.to_array(), dtype=torch.float32)
        controls = controls.repeat(1, batch_size, 1).to(device)
        control = repr(control)

else:
    controls = None
    control = 'NONE'

assert max_len > 0, 'either max length or control sequence length should be given'

# ------------------------------------------------------------------------

print('-' * 70)
print('Session:', sess_path)
print('Batch size:', batch_size)
print('Max length:', max_len)
print('Greedy ratio:', greedy_ratio)
print('Beam size:', beam_size)
print('Beam search stochastic:', stochastic_beam_search)
print('Output directory:', output_dir)
print('Controls:', control)
print('Temperature:', temperature)
print('Init zero:', init_zero)
print('-' * 70)


# ========================================================================
# Generating
# ========================================================================

state = torch.load(sess_path, map_location=device)
model = PerformanceRNN(**state['model_config']).to(device)
model.load_state_dict(state['model_state'])
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
                                    controls=controls,
                                    temperature=temperature,
                                    stochastic=stochastic_beam_search,
                                    verbose=True)
    else:
        outputs = model.generate(init, max_len,
                                 controls=controls,
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
