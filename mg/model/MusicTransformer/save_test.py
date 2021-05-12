from network import MusicTransformer
from metrics import *
from criterion import SmoothCrossEntropyLoss, CustomSchedule
import config
from data import Data

import utils
import datetime
import time
import optparse
import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter

# nn.Parameter(
def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/save_model/')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/egs/dataset/classic_piano/')

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
                      default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/309model_save/train-3049-0.6044921875.pth')

    parser.add_option('-M', '--max_seq',
                      dest='max_seq',
                      type='int',
                      default=2048)

    return parser.parse_args()[0]

options = get_options()
# ------------------------------------------------------------------------

saving_interval = options.saving_interval
window_size = options.window_size
stride_size = options.stride_size

data_path = options.data_path
l_r = options.learning_rate
batch_size = options.batch_size
pickle_dir = options.data_path
max_seq = options.max_seq
epochs = options.epochs
load_path = options.load_path
save_path = options.save_path

if options.multi_gpu == 'True' :
    multi_gpu = True
else:
    multi_gpu = False


event_dim = config.event_dim #EventSeq.dim()
model_config = config.model

device = config.device
limlen = config.window_size

print('-' * 70)

print('Save path:', save_path)
print('Dataset path:', data_path)
print('Saving interval:', saving_interval)
print('-' * 70)

print('Hyperparameters:', utils.dict2params(model_config))
print('Learning rate:', l_r)
print('Batch size:', batch_size)
print('Window size:', window_size)
print('Stride size:', stride_size)
print('Device:', device)
print('-' * 70)


# ========================================================================
# Load model and dataset
# ========================================================================
# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size)
})

start_epoch = 0
def load_model():
    global model_config, device, start_epoch
    model = MusicTransformer(**model_config)
    opt = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)
    if load_path is not None:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

        start_epoch = checkpoint['epoch'] + 1

        # model.load_state_dict(torch.load(load_path))
        model.to(device)
        print(f'Success load {load_path}')
        for item in model.state_dict().keys():
            print( item )
        model.eval()
        eval_x, eval_y = dataset.slide_seq2seq_batch(2, config.max_seq, 'valid')
        eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
        eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

        eval_preiction, _ = model.forward(eval_x)

        eval_metrics = metric_set(eval_preiction, eval_y)
        print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))

    model.to(device)
    return model, scheduler

"""
def load_dataset(limlen):
    global data_path
    dataset = Event_Dataset(data_path, limlen, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset
"""
# load data
print(pickle_dir)
dataset = Data(pickle_dir, max_seq)
print(dataset)



print('Loading model')
mt, scheduler = load_model()
