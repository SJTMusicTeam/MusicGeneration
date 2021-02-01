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

    parser.add_option('-q', '--limit-length',
                      dest='limlen',
                      type='int',
                      default=config.limlen)

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
limlen = config.limlen

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
    model = Event_Melody_RNN(**model_config)
    # model.load_state_dict(torch.load('/data2/qt/MusicGeneration/mg/model/Event_MelodyRNN/save_model/segment_epoch_27.pth'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def load_dataset(limlen):
    global data_path
    dataset = Event_Dataset(data_path, limlen, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset


print('Loading model')
model, optimizer = load_model()
print(model)

print('-' * 70)

print('Loading dataset')
# print(os.path.isdir(data_path))
dataset = load_dataset(limlen)
print(dataset)

print('-' * 70)


# ------------------------------------------------------------------------

def save_model(epoch):
    global model, optimizer, model_config, save_path
    print('Saving to', save_path+config.train_mode+'_epoch_'+str(epoch)+'.pth')
    torch.save(model.state_dict(),  save_path+config.train_mode+'_epoch_'+str(epoch)+'.pth')
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
loss_function = nn.CrossEntropyLoss()

if config.train_mode == 'window':

    data = dataset.batches(batch_size, window_size, stride_size)
    mydataset = MyDataset(data)
    num_workers = 0 if sys.platform.startswith('win32') else 8
    batch_gen = Data.DataLoader(mydataset,
                                batch_size,
                                collate_fn=dataset.Batchify,
                                shuffle=True,
                                drop_last=True,
                                num_workers=num_workers)

    for epoch in range(epochs):
        try:
            l_sum = 0
            for iteration, events in enumerate(batch_gen):
                # print(events.shape)
                events.dtype = np.int16
                events = torch.LongTensor(events).to(device)
                assert events.shape[0] == window_size

                init = torch.randn(batch_size, model.init_dim).to(device)
                outputs = model.generate(init, window_size, events=events[:-1],
                                         teacher_forcing_ratio=teacher_forcing_ratio, output_type='logit')
                assert outputs.shape[:2] == events.shape[:2]

                loss = loss_function(outputs.view(-1, event_dim), events.view(-1))
                model.zero_grad()
                loss.backward()

                l_sum += loss.item()

                norm = utils.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)
                #nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # if enable_logging:
                #     writer.add_scalar('model/loss', loss.item(), iteration)
                #     writer.add_scalar('model/norm', norm.item(), iteration)

                if (iteration+1)%100 == 0:
                    print(f'epoch {epoch}, iter {iteration}, loss: {loss.item()}')

            # if (epoch+1) % saving_interval == 0:
            print(f'epoch {epoch}, ave-loss: {l_sum}, epoch time: {time.time()-last_saving_time}')
            last_saving_time = time.time()
            save_model(epoch)

        except KeyboardInterrupt:
            save_model(epoch)
            break

elif config.train_mode=='sequence':
    #data = dataset.batches(batch_size, window_size, stride_size)
    mydataset = MyDataset(dataset.samples)
    num_workers = 0 if sys.platform.startswith('win32') else 8
    batch_gen = Data.DataLoader(mydataset,
                                batch_size,
                                collate_fn=SeqBatchify,
                                shuffle=True,
                                drop_last=True,
                                num_workers=num_workers)

    for epoch in range(epochs):
        try:
            l_sum = 0
            for iteration, (events, label, lengths) in enumerate(batch_gen):
                # print(events.shape)
                # events.dtype = np.int16
                events = torch.LongTensor(events).to(device)
                label = torch.LongTensor(label).to(device)

                init = torch.randn(batch_size, model.init_dim).to(device)
                # print(events.shape)
                # print(events)
                # print(label.shape)
                outputs = model.train(init, events=events, lengths=lengths)
                init.detach_()
                # print(outputs.shape)
                loss = loss_function(outputs.view(-1, event_dim), label)
                model.zero_grad()
                loss.backward()

                l_sum += loss.item()

                norm = utils.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)

                optimizer.step()

                if (iteration+1)% 30 == 0:
                    print(f'epoch {epoch}, iter {iteration}, loss: {loss.item()}')

            print(f'epoch {epoch}, ave-loss: {l_sum}, epoch time: {time.time()-last_saving_time}')
            last_saving_time = time.time()
            save_model(epoch)

        except KeyboardInterrupt:
            save_model(epoch)
            break
elif config.train_mode=='segment':
    window_size = np.min(dataset.seqlens)
    stride_size = window_size//5
    data = dataset.batches(batch_size, window_size, stride_size)
    print(f'Window Size = {window_size}')
    print(f'Stride = {stride_size}')
    print(f'Iteration={len(data)//batch_size}')
    mydataset = MyDataset(data)
    num_workers = 0 if sys.platform.startswith('win32') else 8
    batch_gen = Data.DataLoader(mydataset,
                                batch_size,
                                collate_fn=dataset.SegBatchify,
                                shuffle=True,
                                drop_last=True,
                                num_workers=num_workers)

    for epoch in range(epochs):
        try:
            l_sum, n = 0, 0
            for iteration, (events, label) in enumerate(batch_gen):
                # print(events.shape)
                events.dtype = np.int16
                label.dtype = np.int16
                events = torch.LongTensor(events).to(device)
                label = torch.LongTensor(label).to(device)
                init = torch.randn(batch_size, model.init_dim).to(device)
                # print(events.shape)
                # print(label.shape)
                outputs = model.train(init, events=events)
                # init.detach_()
                # print(outputs.shape)
                loss = loss_function(outputs.view(-1, event_dim), label.view(-1))
                model.zero_grad()

                loss.backward()

                l_sum += loss.item()
                n += 1
                norm = utils.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)

                optimizer.step()

                if (iteration+1)% 30 == 0:
                    print(f'epoch {epoch}, iter {iteration}, loss: {loss.item()}')

            print(f'epoch {epoch}, ave-loss: {l_sum/n}, epoch time: {time.time()-last_saving_time}')
            last_saving_time = time.time()
            save_model(epoch)

        except KeyboardInterrupt:
            save_model(epoch)
            break


