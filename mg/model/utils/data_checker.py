import torch
from torch import nn
from torch import optim
import time
import optparse
import numpy as np
import os
import sys
import torch.utils.data as Data
sys.path.append('/data2/qt/MusicGeneration/mg/model/')

import utils.shared as utils
import Event_MelodyRNN.config as config
from utils.data import Event_Dataset, MyDataset, SeqBatchify
from Event_MelodyRNN.network import Event_Melody_RNN
from utils.sequence import EventSeq
import Event_MelodyRNN.config
# ========================================================================
# Settings
# ========================================================================
# python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -i 10
def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/Event_MelodyRNN/save_model/')

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

    parser.add_option('-T', '--teacher-forcing-ratio',
                      dest='teacher_forcing_ratio',
                      type='float',
                      default=config.train['teacher_forcing_ratio'])

    parser.add_option('-n', '--clip_norm',
                      dest='clip_norm',
                      type='float',
                      default=config.train['clip_norm'])

    parser.add_option('-t', '--use-transposition',
                      dest='use_transposition',
                      action='store_true',
                      default=config.train['use_transposition'])

    parser.add_option('-p', '--model-params',
                      dest='model_params',
                      type='string',
                      default='')

    parser.add_option('-r', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False)

    parser.add_option('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]

options = get_options()
# ------------------------------------------------------------------------

save_path = options.save_path
data_path = options.data_path
epochs = options.epochs
saving_interval = options.saving_interval

learning_rate = options.learning_rate
batch_size = options.batch_size
window_size = options.window_size
stride_size = options.stride_size
use_transposition = options.use_transposition
teacher_forcing_ratio = options.teacher_forcing_ratio
clip_norm = options.clip_norm

reset_optimizer = options.reset_optimizer
enable_logging = options.enable_logging

event_dim = EventSeq.dim()
model_config = config.model
# print('model_config_before: ', model_config)
model_params = utils.params2dict(options.model_params)
# print('model_params: ', model_params)
model_config.update(model_params)
# print('model_config_after: ', model_config)
device = config.device

print('-' * 70)

print('Save path:', save_path)
print('Dataset path:', data_path)
print('Saving interval:', saving_interval)
print('-' * 70)

print('Hyperparameters:', utils.dict2params(model_config))
print('Learning rate:', learning_rate)
print('Batch size:', batch_size)
print('Window size:', window_size)
print('Stride size:', stride_size)
print('Teacher forcing ratio:', teacher_forcing_ratio)
print('Random transposition:', use_transposition)
print('Reset optimizer:', reset_optimizer)
print('Enabling logging:', enable_logging)
print('Device:', device)
print('-' * 70)


# ========================================================================
# Load model and dataset
# ========================================================================


def load_dataset():
    global data_path
    dataset = Event_Dataset(data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset


print('Loading dataset')
# print(os.path.isdir(data_path))
dataset = load_dataset()
print(dataset)
dataset.count(5000)
"""
5000
16/962
the ratio of length of events less than 5000 is 1.6632016632016633%
"""
print('-' * 70)

