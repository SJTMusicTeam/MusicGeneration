import torch
from music21 import converter, instrument, note, chord, stream, midi#多乐器是否可行
import time
import numpy as np
import pandas as pd
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

data_train_url = "../../egs/dataset/maestro/train"
data_vaild_url = "../../egs/dataset/maestro/vaild"
data_test_url = "../../egs/dataset/maestro/test"

def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(np.round(stream.flat.highestTime / 0.25)) # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame({'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    df = df.sort_values(['pos','pitch'], ascending=[True, False]) # sort the dataframe properly
    df = df.drop_duplicates(subset=['pos']) # drop duplicate values
    # part 2, convert into a sequence of note events
    #output = np.zeros(df.off.max() + 1, dtype=np.int16) + np.int16(MELODY_NO_EVENT)
    output = np.zeros(total_length+2, dtype=np.int16) + np.int16(MELODY_NO_EVENT)  # set array full of no events by default.
    # Fill in the output list
    """
    for row in df.iterrows():
        output[row[1].on] = row[1].pitch  # set note on
        output[row[1].off] = MELODY_NOTE_OFF
    """
    for i in range(total_length):
        if not df[df.pos==i].empty:
            n = df[df.pos==i].iloc[0] # pick the highest pitch at each semiquaver
            output[i] = n.pitch # set note on
            output[i+n.dur] = MELODY_NOTE_OFF

    return output

def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = df.duration.diff(-1) * -1 * 0.25  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[['code','duration']]

def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = note.Rest() # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream

"""
## Play a melody stream
#sp = midi.realtime.StreamPlayer(melody_stream)
#sp.play()
wm_mid = converter.parse("../../egs/dataset/Giant-MIDI/test.mid")
#wm_mid.show()
wm_mel_rnn = streamToNoteArray(wm_mid)
print(wm_mel_rnn)

#noteArrayToStream(wm_mel_rnn).show()
noteArrayToStream(wm_mel_rnn).write("midi", "../../egs/dataset/tmp_res/test.mid")
"""
def creat_dataset(url):
    path = url+"/"
    #midi_files = glob.glob(url+"/*.mid")  # this won't work, no files there.
    #print(midi_files)
    print(os.listdir(path))
    training_arrays = []
    start = time.perf_counter()
    for f in os.listdir(path):#midi_files:
        try:
            s = converter.parse(url+"/"+f)
        except:
            continue
        # for p in s.parts: # extract all voices
        #     arr = streamToNoteArray(p)
        #     training_arrays.append(arr)
        arr = streamToNoteArray(s.parts[0])  # just extract first voice
        training_arrays.append(arr)

    print("Converted: ", url, ", it took", time.perf_counter() - start)
    print(training_arrays)
    training_dataset = np.array(training_arrays)
    return training_dataset
"""
dataset = creat_dataset(data_vaild_url)
np.savez('melody_vaild_dataset.npz', vaild=dataset)
dataset = creat_dataset(data_test_url)
np.savez('melody_testing_dataset.npz', test=dataset)
dataset = creat_dataset(data_train_url)
np.savez('melody_training_dataset.npz', train=dataset)
"""

with np.load('melody_vaild_dataset.npz', allow_pickle=True) as data:
    train_set = data['vaild']

print("Training melodies:", len(train_set))
#print(train_set[:5])

vocab_size = 130 # known 0-127 notes + 128 note_off + 129 no_event
num_steps = 30
embed_size = 64
batch_size = 2
drop_out = 0.5
num_hiddens = 128
num_layers = 2
epochs = 30
clip_norm = 5
learning_rate = 0.001
pred_period = 2
num_gen = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def slice_sequence(sequence, num_steps):
    """Slice a sequence into redundant sequences of lenght num_steps."""
    seq_slices = []
    labels = []
    for i in range(len(sequence) - num_steps - 1):#[0...sequence)
        seq_slices.append(sequence[i: i + num_steps])
        labels.append(sequence[i+num_steps])
    return seq_slices, labels
"""
slices = []
labels = []
for seq in train_set:
    slice, label = slice_sequence(seq, num_steps)
    slices.extend( slice )
    labels.extend( label )

print(len(labels))
#torch.long torch.float32
slices=torch.tensor(np.array(slices).reshape(-1, num_steps), dtype=torch.float32, device=device)
labels=torch.tensor(labels, dtype=torch.float32, device=device)
"""
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype,device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

#x = torch.tensor([0, 2])
#print(one_hot(x, vocab_size))

def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch,    n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

#print(slices[:3])
#print(to_onehot(slices[:3],vocab_size))


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
def batchify(data):
    #用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, list中的每个元素都是__getitem__得到的结果
    #Slice a sequence into redundant sequences of lenght num_steps.
    seq_slices, labels = [], []
    for sequence in data:
        for i in range(len(sequence)-num_steps-1):  # [0...sequence)
            seq_slices.extend( sequence[i: i + num_steps] )
            labels.extend( sequence[i+1:i + num_steps+1] )
    #print(seq_slices)
    #print(labels)
    return (torch.tensor(np.array(seq_slices).reshape(-1, num_steps), dtype=torch.long, device=device)
            ,torch.tensor(np.array(labels).reshape(-1, num_steps), dtype=torch.long, device=device))
"""

"""
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)
"""
"""
mydataset = MyDataset(train_set)
num_workers = 0 if sys.platform.startswith('win32') else 4
data_iter = Data.DataLoader(mydataset, batch_size, shuffle=True,
                            collate_fn=batchify,
                            num_workers=num_workers)
cnt  = 0
for seq, labels in data_iter:
    #print(seq.shape)
    print(labels.shape)
    cnt += 1
    if cnt ==5 :
        break
"""

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
        #inputs = to_onehot(inputs,vocab_size)
        # inputs的形状是(批量⼤⼩, 词数)，因为LSTM需要将序列⻓度(seq_len)作为第⼀维，所以将输⼊转置后
        # 再提取词特征，输出形状为(词数, 批量⼤⼩, 词向量维度)

        # print(inputs.shape)
        embeddings = self.embedding(inputs.permute(1, 0))
        # print(embeddings.shape)
        # rnn.LSTM只传⼊输⼊embeddings，因此只返回最后⼀层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量⼤⼩, 隐藏单元个数)
        outputs, self.state = self.encoder(embeddings, state)  # output, (h, c)
        # print(outputs.shape)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输⼊。它的形状为
        # (批量⼤⼩, 2 * 隐藏单元个数)。
        #encoding = torch.cat((outputs[0], outputs[-1]), -1)
        #print(encoding.shape)
        #out形状为(词数,批量⼤⼩,字典大小)
        outs = F.softmax(self.decoder(outputs),dim=2)
        # print(outs.shape)

        return outs.view(-1,self.vocab_size), self.state
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
    # mydataset = MyDataset(data)
    # num_workers = 0 if sys.platform.startswith('win32') else 4

    model.to(device)

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        # data_iter = Data.DataLoader(mydataset, batch_size, shuffle=True,
        #                             collate_fn=batchify,
        #                             num_workers=num_workers)
        for corpus in data:
            # mydataset = MyDataset(corpus)
            # data_iter = Data.DataLoader(mydataset, batch_size, shuffle=True,
            #                             collate_fn=batchify,
            #                             num_workers=num_workers)
            state = None
            for X, Y in data_iter_consecutive(corpus,batch_size):
            # for X, Y in data_iter:
                if state is not None:
                    # 使⽤detach函数从计算图分离隐藏状态, 这是为了
                    # 使模型参数的梯度计算只依赖⼀次迭代读取的⼩批量序列(防⽌梯度计算开销太⼤)
                    if isinstance(state, tuple):  # LSTM, state:(h, c)
                        state = (state[0].detach(), state[1].detach())
                    else:
                        state = state.detach()
                (output, state) = model(X,state)
                # Y的形状是(batch_size, num_steps)，转置后再变成⻓度为
                # (batch * num_steps) 的向量，这样跟输出的⾏⼀⼀对应
                # print("------------")
                # print(Y.shape)
                # print(torch.transpose(Y, 0, 1).contiguous().shape)
                y = torch.transpose(Y, 0, 1).contiguous().view(-1)
                l = loss(output, y.long())

                optimizer.zero_grad()
                l.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)

                optimizer.step()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, ave-loss %f, time %.2f sec' % (
                    epoch + 1, l_sum/n, time.time() - start))


net = Melody_RNN(vocab_size, embed_size, num_hiddens, num_layers)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()
#train_melody_rnn(net, train_set[:2],epochs, optimizer, loss, clip_norm, batch_size, pred_period, device)

def predict_melody_rnn(model, num_gen, prefix, device):
    state = None
    output = prefix
    for t in range(num_gen):
        if t<len(prefix):
            X = torch.tensor(prefix[t], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            pass
        else:
            if t >= len(prefix):
                X = torch.tensor(int(Y.argmax(dim=1).item()), device=device).view(1, 1)
            output.append( int(Y.argmax(dim=1).item()) )
    return output

res = predict_melody_rnn(net, num_gen , [60], device)
print(res)
res_melody = noteArrayToStream(res)
res_melody.write("midi", "test.mid")



"""
batch_size = self._batch_size()
num_seqs = len(event_sequences)
num_batches = int(np.ceil(num_seqs / float(batch_size)))
"""
