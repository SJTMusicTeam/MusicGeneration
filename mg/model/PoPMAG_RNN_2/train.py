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
import PoPMAG_RNN_2.config as config
from utils.data import Melody_Arrangement_Dataset
from PoPMAG_RNN_2.network import PoPMAG_RNN
from utils.MuMIDI import MuMIDI_EventSeq
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
                      default='/data2/qt/MusicGeneration/mg/model/PoPMAG_RNN/save_model/')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/egs/dataset/lmd_matched_MuMIDI/train.pth')

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

event_dim = MuMIDI_EventSeq.dim()
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
#device = torch.device('cpu')
def load_model():
    global model_config, device, learning_rate
    model = PoPMAG_RNN(**model_config)
    # model.load_state_dict(torch.load('/data2/qt/MusicGeneration/mg/model/PoPMAG_RNN/save_model/256_256_2_epoch_0_599.pth'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def load_dataset():
    global data_path
    dataset = Melody_Arrangement_Dataset.load_file(data_path)
    dataset_size = len(dataset.melody_seqs)
    assert dataset_size > 0
    return dataset


print('Loading model')
model, optimizer = load_model()
print(model)

print('-' * 70)

print('Loading dataset')
# print(os.path.isdir(data_path))
dataset = load_dataset()
print(dataset)

print('-' * 70)


# ------------------------------------------------------------------------

def save_model(epoch):
    global model, optimizer, model_config, save_path
    print('Saving to', save_path+'256_256_2_epoch_'+epoch+'.pth')
    torch.save(model.state_dict(),  save_path+'256_256_2_epoch_'+epoch+'.pth')
    # torch.save({'model_config': model_config,
    #             'model_state': model.state_dict(),
    #             'model_optimizer_state': optimizer.state_dict()}, save_path)
    print('Done saving')


# ========================================================================
# Training
# ========================================================================

# if enable_logging:
#     from torch.utils.tensorboard import SummaryWriter
#     writer = SummaryWriter()

last_saving_time = time.time()
loss_function = nn.CrossEntropyLoss(reduction='none')

num_workers = 0 if sys.platform.startswith('win32') else 10
batch_gen = Data.DataLoader(dataset,
                            batch_size,
                            collate_fn=dataset.FastBatchify,
                            shuffle=True,
                            drop_last=True,
                            num_workers=num_workers)


for epoch in range(epochs):
    try:
        l_sum, n = 0, 0
        # for iteration, (src, tar) in enumerate(batch_gen):
        for iteration, (src, src_mask, tar, tar_mask, label, label_mask) in enumerate(batch_gen):
            # print(device)
            src = src.to(device)
            src_mask = src_mask.to(device)
            tar = tar.to(device)
            tar_mask = tar_mask.to(device)
            label = label.to(device)
            label_mask = label_mask.to(device)
            # print(f'src={src.shape}')
            # print(f'tar={tar.shape}')
            # print(f'label={label.shape}')
            # print(f'tar={tar[0, 0, 0]}')
            # print(f'label={label[0, 0, 0]}')
            # print(f'label_mask={label_mask[0, 0, 0]}')

            comp_src = model.compression(src)
            comp_tar = model.compression(tar)
            # print(f'comp_src={comp_src.shape}')
            # print(f'comp_tar={comp_tar.shape}')
            
            # comp_src, src_mask = model.sequence_compression(s)# (batch * bar_num * bar_len * embedding)
            # comp_tar, tar_mask = model.sequence_compression(t)# (batch * bar_num' * bar_len' * embedding)
            # comp_src.to(device)
            # comp_tar.to(device)
            # print(f'comp_src={comp_src.shape}')

            # print(events.shape)
            init = torch.randn(batch_size, model.init_dim).to(device)
            # # print(events.shape)
            # # print(label.shape)
            outputs = model.Train(init, src=comp_src, src_mask=src_mask, tar=comp_tar, tar_mask=tar_mask)
            init.detach_()
            # print(f'output_shape={outputs.shape}')
            # print(f'outputs.dev={outputs.device}, label.dev={label.device}')
            # output = [batch, mx_bar_nums, mx_bar_events, 3, mx_event_dim]
            # label = [batch, mx_bar_nums, mx_bar_events, 3]
            loss = loss_function(outputs.view(-1, model.mx_dim), label.view(-1))
            # loss = [loss_function(outputs[i], label[:, :, :, i]) * label_mask[:, :, :, i] for i in range(3)]

            # print(f'loss.shape={loss.shape}')
            loss = torch.mean(loss * label_mask.view(-1))
            if torch.isnan(loss):
                loss = 0
                continue
            # loss = torch.mean(loss)
            # print(f'loss={loss}')
            model.zero_grad()
            #
            loss.backward()
            #
            l_sum += loss.item()
            n += 1
            # # norm = utils.compute_gradient_norm(model.parameters())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            # # nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=1)
            #
            optimizer.step()
            #
            if (iteration+1) % 100 == 0:
                print(f'epoch {epoch}, iter {iteration}, loss: {loss.item()}')
                save_model(str(epoch)+'_'+str(iteration))

        print(f'epoch {epoch}, ave-loss: {l_sum/n}, epoch time: {time.time()-last_saving_time}')
        last_saving_time = time.time()
        save_model(str(epoch)+'_')

    except KeyboardInterrupt:
        save_model(str(epoch)+'_')
        break


