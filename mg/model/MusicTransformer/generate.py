import criterion
from layers import *
import config
from network import MusicTransformer
from data import Data
import utils
from processor import decode_midi, encode_midi

import datetime
import argparse

from tensorboardX import SummaryWriter


def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/save_model/')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/egs/dataset/maestro/train_processed/')

    parser.add_option('-e', '--epochs',
                      dest='epochs',
                      type='int',
                      default=20000)

    parser.add_option('-i', '--saving-interval',
                      dest='saving_interval',
                      type='int',
                      default=50)

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=config.train['batch_size'])

    parser.add_option('-l', '--learning-rate',
                      dest='learning_rate',
                      type='float',
                      default=config.train['learning_rate'])

    parser.add_option('-w', '--window-size',
                      dest='window_size',
                      type='int',
                      default=config.train['window_size'])

    parser.add_option('-S', '--stride-size',
                      dest='stride_size',
                      type='int',
                      default=config.train['stride_size'])

    parser.add_option('-g', '--multi_gpu',
                      dest='multi_gpu',
                      type='string',
                      default='False')

    parser.add_option('-m', '--load_path',
                      dest='load_path',
                      type='string',
                      default=None)

    parser.add_option('-M', '--max_seq',
                      dest='max_seq',
                      type='int',
                      default=2048)

    return parser.parse_args()[0]

args = get_options()

config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
mt.load_state_dict(torch.load(args.model_dir+'/final.pth'))
mt.test()

print(config.condition_file)
if config.condition_file is not None:
    inputs = np.array([encode_midi('dataset/midi/BENABD10.mid')[:500]])
else:
    inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs)
result = mt(inputs, config.length, gen_summary_writer)

for i in result:
    print(i)

decode_midi(result, file_path=config.save_path)

gen_summary_writer.close()
