import torch
# import sys, os
# print(os.path.dirname(os.path.abspath('__file__')))
#
# sys.path.append('/data2/qt/MusicGeneration/mg/model')
# print(os.path.dirname(os.path.abspath('__file__')))

from sequence import EventSeq
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

event_dim = EventSeq.dim()
pad_token = EventSeq.dim()
# token_sos = event_dim + 1
# token_eos = event_dim + 2
# vocab_size = event_dim + 3
vocab_size = EventSeq.dim()

save_path = '/data2/qt/MusicGeneration/mg/model/MusicTransformer/output/'
condition_file = '/data2/qt/MusicGeneration/egs/dataset/maestro/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--1.midi'
length = 2500
threshold_len = 500

pickle_dir = '/data2/qt/MusicGeneration/egs/dataset/maestro/'

load_path = None
dropout = 0.2
debug = False

num_layers = 8
max_seq = 2048
embedding_dim = 256

l_r = 0.0001
batch_size = 4
window_size = 2048
stride_size = 10
accum_grad = 32
label_smooth = 0.1
epochs = 50000

model = {
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'max_seq': max_seq,
    'num_layer': num_layers,
    'dropout': dropout,
}

train = {
    'learning_rate': l_r,
    'batch_size': batch_size,
    'window_size': 2048,
    'stride_size': 10,
    'accum_grad' : 25,
}
