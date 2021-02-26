import torch
from utils.sequence import EventSeq, ControlSeq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
train_mode = "segment"#segment, window, sequence
limlen = 1200
model = {
    'init_dim': 32,
    'event_dim': EventSeq.dim(),
    'hidden_dim': 512,
    'rnn_layers': 3,
    'dropout': 0.3,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 100,
    'window_size': 200,
    'stride_size': 10,
    'use_transposition': False,
    'teacher_forcing_ratio': 1.0,
    'clip_norm' : 1.0
}
