import criterion
from layers import *
import config
from network import MusicTransformer
from data import Data
import utils
from processor import decode_midi, encode_midi

import datetime
import optparse
import os

from tensorboardX import SummaryWriter


def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=8)

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      #default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/save_model/.pth',
                      default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/save_model/train-1226-0.0.pth',
                      help = 'pth file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/MuiscTransformer/output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=1500)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.0)

    return parser.parse_args()[0]


args = get_options()

# config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
# gen_summary_writer = SummaryWriter(gen_log_dir)

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
mt.load_state_dict(torch.load(args.save_path))
mt.test()

print(config.condition_file)
if config.condition_file is not None:
    inputs = np.array([encode_midi(config.condition_file)[:100]])
else:
    inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs)
result = mt(inputs, config.length)

for i in result:
    print(i)

# decode_midi(result, file_path=config.save_path)
path = config.save_path+'example.mid'
# decode_midi(result, file_path=config.save_path)

n_notes = utils.event_indeces_to_midi_file(result, path)
print(f'===> {path} ({n_notes} notes)')
# gen_summary_writer.close()
