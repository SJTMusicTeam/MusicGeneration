import torch
from utils.MuMIDI import MuMIDI_EventSeq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# train_mode = "segment"#segment, window, sequence
train_mode = "bar"
model = {
    'init_dim': 32,
    'event_dim': MuMIDI_EventSeq.dim(),
    'bar_dim': 200,#187
    'embed_dim' : 512,
    'hidden_dim': 512,
    'rnn_layers': 3,
    'dropout': 0.3,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 4,
    'window_size': 200,
    'stride_size': 10,
    'use_transposition': False,
    'teacher_forcing_ratio': 1.0,
    'clip_norm' : 1.0
}
