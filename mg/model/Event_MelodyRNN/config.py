import torch
from utils.sequence import EventSeq, ControlSeq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
train_mode = "sequence"#segment

model = {
    'init_dim': 32,
    'event_dim': EventSeq.dim(),
    'hidden_dim': 256,
    'rnn_layers': 2,
    'dropout': 0.5,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 8,
    'window_size': 200,
    'stride_size': 10,
    'use_transposition': False,
    'teacher_forcing_ratio': 1.0,
    'clip_norm' : 5.0
}
