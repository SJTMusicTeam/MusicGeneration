import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
event_dim = 388
pad_token = 388
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3
condition_file = None

pickle_dir = '/data2/qt/MusicTransformer-pytorch/dataset/processed/ecomp'

load_path = None
dropout = 0.2
debug = False

num_layers = 8
max_seq = 2048
embedding_dim = 256

l_r = 0.0001
batch_size = 2
window_size = 2048
stride_size = 10
accum_grad = 1#25
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
