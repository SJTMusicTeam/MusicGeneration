import torch
import time
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import random
import sys
import pandas as pd


# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128 # (stop playing all previous notes)
MELODY_NO_EVENT = 129 # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.
option = "maestro"
data_train_url = "../../egs/dataset/"+option

with np.load(data_train_url+"/train.npz", allow_pickle=True) as data:
    train_set = data['train']

print("Training melodies:", len(train_set))
train_set=sorted(train_set,key = lambda i:len(i),reverse=True)

vocab_size = 130 # known 0-127 notes + 128 note_off + 129 no_event
num_steps = 30
embed_size = 64
batch_size = 64
drop_out = 0.5
num_hiddens = 64
num_layers = 2
epochs = 20000
clip_norm = 5
learning_rate = 0.001
pred_period = 50
num_gen = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

# Do some stats on the corpus.
all_notes = np.concatenate(train_set)
print("Number of notes:")
print(all_notes.shape)
all_notes_df = pd.DataFrame(all_notes)
print("Notes that do appear:")
unique, counts = np.unique(all_notes, return_counts=True)
print(unique)
print("Notes that don't appear:")
print(np.setdiff1d(np.arange(0,129),unique))

def slice_sequence(sequence, num_steps):
    """Slice a sequence into redundant sequences of lenght num_steps."""
    seq_slices = []
    labels = []
    for i in range(len(sequence) - num_steps - 1):#[0...sequence)
        seq_slices.append(sequence[i: i + num_steps])
        labels.append(sequence[i+num_steps])
    return seq_slices, labels

def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype,device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch,    n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

"""
def data_iter_consecutive(sequence,batch_size):
    data_len = len(sequence)
    batch_len = data_len // batch_size
    sequence = torch.tensor(sequence[0:batch_size*batch_len],dtype=torch.long, device=device).view(batch_size, batch_len)
    epoch_size = (batch_len - 1)//num_steps
    for i in range(epoch_size):  # [0...sequence)
        k = i * num_steps
        X = sequence[:, k: k + num_steps]
        Y = sequence[:, k + 1: k + num_steps + 1]
        yield  torch.tensor(np.array(X), dtype=torch.long, device=device), \
               torch.tensor(np.array(Y), dtype=torch.long, device=device)
"""
"""
def batchify(sequence):
    #用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, list中的每个元素都是__getitem__得到的结果
    #Slice a sequence into redundant sequences of lenght num_steps.
    seq_slices, labels = [], []
    data_len = len(sequence)
    batch_len = data_len // batch_size
    sequence = sequence[0:batch_len*batch_size].view(batch_size,batch_len)
    epoch_size = batch_len-1
    for i in range(epoch_size):  # [0...sequence)
        k = i * num_steps
        X = sequence[:, k: k + num_steps]
        Y = sequence[:, k + 1: k + num_steps + 1]
        seq_slices.extend( sequence[i: i + num_steps] )
        labels.extend( sequence[i+1:i + num_steps+1] )
    #print(seq_slices)
    #print(labels)
    return (torch.tensor(np.array(seq_slices).reshape(-1, num_steps), dtype=torch.long, device=device)
            ,torch.tensor(np.array(labels).reshape(-1, num_steps), dtype=torch.long, device=device))

"""
"""
def data_iter_batch(sequence,num_steps):
    data_len = len(sequence[0])
    # sequence = torch.tensor(sequence[0:batch_size*batch_len],dtype=torch.long, device=device).view(batch_size, batch_len)
    epoch_size = data_len - 1 - num_steps
    for i in range(epoch_size):  # [0...sequence)
        X = sequence[:, i: i + num_steps]
        Y = sequence[:, i + 1: i + num_steps + 1]
        yield  torch.tensor(np.array(X), dtype=torch.long, device=device), \
               torch.tensor(np.array(Y), dtype=torch.long, device=device)
"""

def batchify(data):
    return data
"""
    #用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, list中的每个元素都是__getitem__得到的结果
    #Slice a sequence into redundant sequences of lenght num_steps.
    lengths = [len(item) for item in data]
    #print(lengths)
    mx_length = max(lengths)
    padding_res = np.zeros((len(data),mx_length))
    for i in range(len(data)):
        mx = len(data[i])
        padding_res[i,:mx]=np.array(data[i])
    #print(data)
    #padding_res = torch.autograd.Variable([torch.tensor(item, dtype=torch.long, device=device) for item in data])
    #padding_res = nn.utils.rnn.pack_padded_sequence(input=padding_res, lengths=lengths, batch_first=True)
    padding_res = torch.tensor(padding_res , dtype = torch.long, device = device)

    #padding_res = nn.utils.rnn.pad_packed_sequence(input=padding_res, batch_first=True)
    print(padding_res)
    return padding_res,lengths#torch.tensor( padding_res, dtype=torch.long, device=device)
"""
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)

"""
    'attention_rnn':
        MelodyRnnConfig(
            generator_pb2.GeneratorDetails(
                id='attention_rnn',
                description='Melody RNN with lookback encoding and attention.'),
            note_seq.KeyMelodyEncoderDecoder(
                min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE),
            contrib_training.HParams(
                batch_size=128,
                rnn_layer_sizes=[128, 128],
                dropout_keep_prob=0.5,
                attn_length=40,
                clip_norm=3,
                learning_rate=0.001))
"""

def flatten_padded_sequences(outs, lengths):
    batch, mx_length, vocab_size = outs.shape
    res = outs[0, :lengths[0] - 1, :]
    for i in range(batch - 1):
        res = torch.cat([res, outs[i + 1, :lengths[i + 1] - 1, :]], dim=0)
    return res.view(-1, vocab_size)

def flatten_padded_label_sequences(X, lengths):
    batch = len(X)
    Y = torch.tensor(np.array(X[0])[1:lengths[0]],dtype=torch.long,device=device)
    for i in range(batch-1):
        Y = torch.cat([Y,torch.tensor(np.array(X[i+1])[1:lengths[i+1]],dtype=torch.long,device=device)],dim=0)
    return Y

class Melody_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super(Melody_RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.state = None
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False,
                               dropout=0.5)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输⼊
        """
        self.attantion =  nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
        """
        self.decoder = nn.Linear(num_hiddens, vocab_size)

    def forward(self, inputs, state=None):
        lengths = np.array([len(item) for item in inputs])
        mx_length = np.max(lengths)
        padding_res = np.zeros((len(inputs), mx_length))
        for i in range(len(inputs)):
            mx = len(inputs[i])
            padding_res[i, :mx] = np.array(inputs[i])
        padding_res = torch.tensor(padding_res, dtype=torch.long, device=device)
        #padding_res = torch.autograd.Variable(padding_res)
        # inputs的形状是(批量⼤⼩, 词数)，因为LSTM需要将序列⻓度(seq_len)作为第⼀维，所以将输⼊转置后
        # 再提取词特征，输出形状为(词数, 批量⼤⼩, 词向量维度)

        # print(inputs.shape)
        embeddings = self.embedding(padding_res)
        # embeddings = self.embedding(inputs.permute(1, 0))
        # embeddings = (batch_size X mx_length X embed_size)
        # print(embeddings.shape)
        # rnn.LSTM只传⼊输⼊embeddings，因此只返回最后⼀层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量⼤⼩, 隐藏单元个数)
        embedding_packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        packed_output, self.state = self.encoder(embedding_packed, state)  # output, (h, c)
        outputs, inputs_size = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        #print('outputs')
        #print(outputs.shape)

        #out形状为(批量⼤⼩,词数,字典大小)
        outs = self.decoder(outputs)
        #print('outs')
        #print(outs.shape)
        # res = outs[0,:lengths[0]-1,:]
        # for i in range(batch_size-1):
        #     res = torch.cat([res,outs[i+1,:lengths[i+1]-1,:]],dim = 0)
        res = flatten_padded_sequences(outs, lengths)
        return res.view(-1,self.vocab_size), self.state
"""
def attention_forward(model, enc_states, dec_state):
    #enc_states: (时间步数, 批量大小, 隐藏单元个数)
    #dec_state: (批量大小, 隐藏单元个数)
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
    alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
    return (alpha * enc_states).sum(dim=0)  # 返回背景变量
"""

def train_melody_rnn(model, data, num_epochs, optimizer, loss, clip_norm, batch_size, pred_period, device):
    mydataset = MyDataset(data)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    model.to(device)

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = Data.DataLoader(mydataset,
                                    batch_size,
                                    #shuffle=True,
                                    collate_fn=batchify,
                                    num_workers=num_workers)

        # for batch,batch_lengths in data_iter:
        for X in data_iter:
            (output, state) = model(X)
            lengths = np.array([len(item) for item in X])
            Y = flatten_padded_label_sequences(X,lengths)
            y_len = Y.shape[0]
            l = loss(output, Y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)
            #print('loss')
            #print(l.item())
            optimizer.step()
            l_sum += l.item() * y_len
            n += y_len

        if (epoch + 1) % 1 == 0:
            if (epoch + 1) % 10 == 0:
                torch.save(net.state_dict(), 'basic_rnn_' + str(epoch) + '.pth')
            print('epoch %d, ave-loss %f, time %.2f sec' % (
                    epoch + 1, l_sum/n, time.time() - start))


net = Melody_RNN(vocab_size, embed_size, num_hiddens, num_layers)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
loss = nn.CrossEntropyLoss()
#mini_data = [[2,3,5],[4,6]]
#mini_data = train_set[-2:]
#print(mini_data)
train_melody_rnn(net, train_set, epochs, optimizer, loss, clip_norm, batch_size, pred_period, device)
#torch.save(net.state_dict(), 'basic_rnn.pth')