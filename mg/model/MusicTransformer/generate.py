import torch

from layers import *
import config
from network import MusicTransformer
from criterion import SmoothCrossEntropyLoss, CustomSchedule
import utils
import sequence
from data import Data
from metrics import *
import datetime
import optparse


def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=4)

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      #default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/save_model/.pth',
                      default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/309model_save/train-3049-0.6044921875.pth',
                      help = 'pth file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/MuiscTransformer/output/')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/egs/dataset/classic_piano/')

    parser.add_option('-l', '--max-length',
                      dest='max_seq',
                      type='int',
                      default=1500)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.0)


    return parser.parse_args()[0]


args = get_options()
data_path = args.data_path
pickle_dir = args.data_path
max_seq = args.max_seq
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
checkpoint = torch.load(args.save_path)
mt.load_state_dict(checkpoint['net'])
mt.to(config.device)
mt.eval()

print(pickle_dir)
dataset = Data(pickle_dir, max_seq)
print(dataset)
# # init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size)
})
eval_x, eval_y = dataset.slide_seq2seq_batch(2, config.max_seq, 'test')
eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

eval_preiction, _ = mt.forward(eval_x)

eval_metrics = metric_set(eval_preiction, eval_y)
print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))
# print([[24, 28, 31]] * args.batch_size )
mt.test()
print(config.condition_file)
if config.condition_file is not None:
    _note_seq = sequence.NoteSeq.from_midi_file(config.condition_file)
    _event_seq = sequence.EventSeq.from_note_seq(_note_seq)
    # print(_event_seq)
    inputs = np.array([_event_seq.to_array()[:500] ] * args.batch_size, dtype=np.int16)
else:
    inputs = np.array([ [24, 28, 31] ] * args.batch_size, dtype=np.int16)
inputs = torch.LongTensor(inputs).to(config.device)
result = mt(inputs, config.length)

# for i in result:
#     print(i)
print(result)
# decode_midi(result, file_path=config.save_path)

# decode_midi(result, file_path=config.save_path)
for i in range(len(inputs)):
    path = config.save_path+'example_'+str(i)+'_3049sample_uncond2000.mid'
    res = result[i]
    n_notes = utils.event_indeces_to_midi_file(res, path)
    print(f'===> {path} ({n_notes} notes)')
# gen_summary_writer.close()
