import torch
import time
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import random
import sys


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

vocab_size = 130 # known 0-127 notes + 128 note_off + 129 no_event
num_steps = 30
embed_size = 64
batch_size = 12
drop_out = 0.5
num_hiddens = 64
num_layers = 2
epochs = 10000
clip_norm = 3
learning_rate = 0.001
pred_period = 50
num_gen = 100
att_length = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

def flatten_padded_sequences(outs, lengths):
    batch, mx_length, vocab_size = outs.shape
    res = outs[0, :lengths[0] - 1, :]
    for i in range(batch - 1):
        res = torch.cat([res, outs[i + 1, :lengths[i + 1] - 1, :]], dim=0)
    return res.view(-1, vocab_size)

def flatten_padded_label_sequences(X, lengths):
    batch = len(X)
    Y = np.array(X[0])[1:lengths[0]]
    for i in range(batch-1):
        Y = np.concatenate([Y,np.array(X[i+1])[1:lengths[i+1]]], axis=0)
    return Y

def batchify(data):
    lengths = np.array([len(item) for item in data])
    mx_length = np.max(lengths)
    X = np.zeros((len(data), mx_length))
    for i in range(len(data)):
        mx = len(data[i])
        X[i, :mx] = np.array(data[i])
    Y = flatten_padded_label_sequences(data, lengths)
    return X, Y, lengths

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
class Melody_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, att_length=40):
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

        self.decoder = nn.Linear(2*num_hiddens, vocab_size)


        self.W_h = nn.Linear(num_layers*num_hiddens,num_hiddens)  # self.attention_size
        self.W_c = nn.Linear(num_layers*num_hiddens,num_hiddens)
        self.v = nn.Linear(num_hiddens,1)
        self.att_length = att_length
        # nn.init.uniform_(self.W_h, -0.1, 0.1)
        # nn.init.uniform_(self.W_c, -0.1, 0.1)
        # nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, inputs, lengths,state=None):
        #padding_res = torch.autograd.Variable(padding_res)
        # inputs的形状是(批量⼤⼩, 词数)，因为LSTM需要将序列⻓度(seq_len)作为第⼀维，所以将输⼊转置后
        # 再提取词特征，输出形状为(词数, 批量⼤⼩, 词向量维度)
        batch = inputs.shape[0]
        embeddings = self.embedding(inputs.permute(1, 0))
        #embeddings = (mx_length X batch_size X embed_size)
        # print(embeddings.shape)
        # rnn.LSTM只传⼊输⼊embeddings，因此只返回最后⼀层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数*layers, 批量⼤⼩, 隐藏单元个数)
        output, (h, c) = self.encoder(embeddings[0,:,:].unsqueeze(0), state)
        outputs, h_steps, c_steps = output, h, c

        steps = np.max(lengths)
        for i in range(1,steps):
            output, (h,c) = self.encoder(embeddings[1,:,:].unsqueeze(0), state)
            outputs = torch.cat([outputs, output], dim=0)
            h_steps = torch.cat([h_steps, h], dim=0)
            c_steps = torch.cat([c_steps, c], dim=0)

        outputs = outputs.permute(0, 1, 2).view(batch, steps, -1)
        h_steps = h_steps.permute(0, 1, 2).view(batch, steps, -1)# (batch, max_sequence, layers*hidden)
        c_steps = c_steps.permute(0, 1, 2).view(batch, steps, -1)

        Wh = self.W_h(h_steps)# (batch, max_sequence, hidden)
        Wc = self.W_c(c_steps)
        Att = torch.tensor(np.zeros((batch, steps, self.att_length, self.num_hiddens)),dtype=torch.float32,device=device)
        for i in range(1,steps):
            att_pre = max(self.att_length-i,0)
            seq_pre = max(i-self.att_length, 0)
            Att[:, i, att_pre:, :] = Wh[:, seq_pre:i, :]
        Wh_att = Att
        for i in range(steps):
            Att[:, i, :, :] += Wc[:, i, :].unsqueeze(1)
        Att_weight = self.v(torch.tan(Att))# (batch, max_sequence, att_length, 1)
        # Att_weight = Att_weight.squeeze(3)# (batch, max_sequence, att_length)
        #print(Att_weight.shape)
        for i in range(self.att_length):# mask
            att_pre = max(self.att_length - i, 0)
            Att_weight[:, :, 0:att_pre] = -1e9
        #print(Att_weight)
        softmax_weight = F.softmax(Att_weight, dim=2)# (batch, max_sequence, att_length)
        attention_output = torch.sum(softmax_weight * Wh_att,dim=2).squeeze(2)
        # (batch, max_sequence, att_length, hidden) = (batch, max_sequence, hidden)
        concat_h = torch.cat([attention_output, outputs],dim=2)
        #out形状为(批量⼤⼩,词数,字典大小)
        outs = self.decoder(concat_h)
        res = flatten_padded_sequences(outs, lengths)
        return res.view(-1,self.vocab_size), self.state

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

        for X, Y, lengths in data_iter:
            X = torch.tensor(X,dtype=torch.long,device=device)
            Y = torch.tensor(Y,dtype=torch.long,device=device)
            (output, state) = model(X, lengths)
            y_len = Y.shape[0]
            l = loss(output, Y.long())

            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)
            optimizer.step()
            l_sum += l.item() * y_len
            n += y_len

        if (epoch + 1) % 1 == 0:
            if (epoch + 1) % pred_period == 0:
                torch.save(net.state_dict(), 'attention_rnn_' + str(epoch) + '.pth')
            print('epoch %d, ave-loss %f, time %.2f sec' % (
                    epoch + 1, l_sum/n, time.time() - start))


net = Melody_RNN(vocab_size, embed_size, num_hiddens, num_layers, att_length)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
loss = nn.CrossEntropyLoss()
#mini_data = [[2,3,5],[4,6]]
#mini_data = train_set[-2:]
#print(mini_data)
train_melody_rnn(net, train_set, epochs, optimizer, loss, clip_norm, batch_size, pred_period, device)
#torch.save(net.state_dict(), 'basic_rnn.pth')