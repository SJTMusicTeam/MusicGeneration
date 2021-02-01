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
# print(len(train_set[0]))
# print(len(train_set[1]))
# print(len(train_set[2]))
train_set=sorted(train_set,key = lambda i:len(i),reverse=True)
# print(len(train_set[0]))
# print(len(train_set[1]))
# print(len(train_set[2]))

vocab_size = 130 # known 0-127 notes + 128 note_off + 129 no_event
num_steps = 30
embed_size = 64
batch_size = 3
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

def batchify(data):
    return data

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

"""
class AttentionCellWrapper(rnn_cell.RNNCell):
  # Basic attention cell wrapper.
  # 
  # Implementation based on https://arxiv.org/abs/1601.06733.
  # 

  def __init__(self,
               cell,
               attn_length,
               attn_size=None,
               attn_vec_size=None,
               input_size=None,
               state_is_tuple=True,
               reuse=None):
    # Create a cell with attention.
    # 
    # Args:
    #   cell: an RNNCell, an attention is added to it.
    #   attn_length: integer, the size of an attention window.
    #   attn_size: integer, the size of an attention vector. Equal to
    #       cell.output_size by default.
    #   attn_vec_size: integer, the number of convolutional features calculated
    #       on attention state and a size of the hidden layer built from
    #       base cell state. Equal attn_size to by default.
    #   input_size: integer, the size of a hidden linear layer,
    #       built from inputs and attention. Derived from the input tensor
    #       by default.
    #   state_is_tuple: If True, accepted and returned states are n-tuples, where
    #     `n = len(cells)`.  By default (False), the states are all
    #     concatenated along the column axis.
    #   reuse: (optional) Python boolean describing whether to reuse variables
    #     in an existing scope.  If not `True`, and the existing scope already has
    #     the given variables, an error is raised.
    # 
    # Raises:
    #   TypeError: if cell is not an RNNCell.
    #   ValueError: if cell returns a state tuple but the flag
    #       `state_is_tuple` is `False` or if attn_length is zero or less.
    # 
    super(AttentionCellWrapper, self).__init__(_reuse=reuse)
    assert_like_rnncell("cell", cell)
    if _is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError(
          "Cell returns tuple of states, but the flag "
          "state_is_tuple is not set. State size is: %s" % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError(
          "attn_length should be greater than zero, got %s" % str(attn_length))
    if not state_is_tuple:
      tf.logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

  @property
  def state_size(self):
    size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length)
    if self._state_is_tuple:
      return size
    else:
      return sum(list(size))

  @property
  def output_size(self):
    return self._attn_size

  def call(self, inputs, state):
    #Long short-term memory cell with attention (LSTMA).
    if self._state_is_tuple:
      state, attns, attn_states = state
    else:
      states = state
      state = tf.slice(states, [0, 0], [-1, self._cell.state_size])
      attns = tf.slice(states, [0, self._cell.state_size],
                       [-1, self._attn_size])
      attn_states = tf.slice(
          states, [0, self._cell.state_size + self._attn_size],
          [-1, self._attn_size * self._attn_length])
    attn_states = tf.reshape(attn_states,
                             [-1, self._attn_length, self._attn_size])
    input_size = self._input_size
    if input_size is None:
      input_size = inputs.get_shape().as_list()[1]
    if self._linear1 is None:
      self._linear1 = _Linear([inputs, attns], input_size, True)
    inputs = self._linear1([inputs, attns])
    cell_output, new_state = self._cell(inputs, state)
    if self._state_is_tuple:
      new_state_cat = tf.concat(tf.nest.flatten(new_state), 1)
    else:
      new_state_cat = new_state
    new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
    with tf.variable_scope("attn_output_projection"):
      if self._linear2 is None:
        self._linear2 = _Linear([cell_output, new_attns], self._attn_size, True)
      output = self._linear2([cell_output, new_attns])
    new_attn_states = tf.concat(
        [new_attn_states, tf.expand_dims(output, 1)], 1)
    new_attn_states = tf.reshape(
        new_attn_states, [-1, self._attn_length * self._attn_size])
    new_state = (new_state, new_attns, new_attn_states)
    if not self._state_is_tuple:
      new_state = tf.concat(list(new_state), 1)
    return output, new_state

  def _attention(self, query, attn_states):
    conv2d = tf.nn.conv2d
    reduce_sum = tf.math.reduce_sum
    softmax = tf.nn.softmax
    tanh = tf.math.tanh

    with tf.variable_scope("attention"):
      k = tf.get_variable("attn_w",
                          [1, 1, self._attn_size, self._attn_vec_size])
      v = tf.get_variable("attn_v", [self._attn_vec_size])
      hidden = tf.reshape(attn_states,
                          [-1, self._attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      if self._linear3 is None:
        self._linear3 = _Linear(query, self._attn_vec_size, True)
      y = self._linear3(query)
      y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          tf.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = tf.reshape(d, [-1, self._attn_size])
      new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
      return new_attns, new_attn_states

"""


class RNN_Attention(nn.Module):
    def __init__(self):
        super(RNN_Attention, self).__init__()

        self.W_h = nn.Parameter(torch.Tensor(num_hiddens * num_layers, num_hiddens * num_layers))  # self.attention_size
        self.W_c = nn.Parameter(torch.Tensor(num_hiddens * num_layers, num_hiddens * num_layers))
        self.v = nn.Parameter(torch.Tensor(1, num_hiddens * num_layers))
        nn.init.uniform_(self.W_h, -0.1, 0.1)
        nn.init.uniform_(self.W_c, -0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)
    """
    with tf.variable_scope("attention"):
        k = tf.get_variable("attn_w",
                            [1, 1, self._attn_size, self._attn_vec_size])
        v = tf.get_variable("attn_v", [self._attn_vec_size])
        hidden = tf.reshape(attn_states,
                            [-1, self._attn_length, 1, self._attn_size])
        hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        if self._linear3 is None:
            self._linear3 = _Linear(query, self._attn_vec_size, True)
        y = self._linear3(query)
        y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
        s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
        a = softmax(s)
        d = reduce_sum(
            tf.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
        new_attns = tf.reshape(d, [-1, self._attn_size])
        new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
        return new_attns, new_attn_states
    """
    def forward(self):
        pass

def Make_RNN(rnn_layer_sizes, num_layers, bidirectional=False, dropout=0.5, attn_length=50):
    cells = nn.Sequential()
    for i in range(len(num_layers)):
        cell = nn.Sequential()
        cell.add_module('layer-'+str(i)+'-LSTM',nn.LSTMCell(input_size=rnn_layer_sizes[i], hidden_size=rnn_layer_sizes[i+1]))
        # cell = contrib_rnn.AttentionCellWrapper(cell, attn_length, state_is_tuple=True)
        # cell.add_module('layer-'+str(i)+'-attention',RNN_Attention())
        cell.add_module('layer-'+str(i)+'-dropout',drop_out)
        cells.add_module(str(i)+'th layer:',cell)

    return cells

class Melody_Attention_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super(Melody_Attention_RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.state = None
        self.embedding = nn.Embedding(vocab_size, embed_size)
        """
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False,
                               dropout=0.5)#eval = 1.0
        """
        self.encoder = Make_RNN(rnn_layer_sizes=[embed_size,[num_hiddens]*num_layers],
                               num_layers=num_layers,
                               bidirectional=False,
                               dropout=0.5,
                                attn_length=40)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输⼊
        #self.att_h = nn.Linear()

        """
        self.attantion =  nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
        """
        self.decoder = nn.Linear(num_hiddens, vocab_size)

    """
    def attention(self, h, c):
        u = torch.tanh(torch.matmul(h, self.W_h)+torch.matmul(c, self.W_c))  # [batch, seq_len, hidden_dim*2]
        att = torch.matmul(self.v, u)
        #att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_h = h * att_score  # [batch, seq_len, hidden_dim*2]
        context = torch.sum(scored_h, dim=1)  # [batch, hidden_dim*2]

        return context
    """

    def forward(self, inputs, state=None):
        lengths = np.array([len(item) for item in inputs])
        mx_length = np.max(lengths)
        padding_res = np.zeros((len(inputs), mx_length))
        for i in range(len(inputs)):
            mx = len(inputs[i])
            padding_res[i, :mx] = np.array(inputs[i])
        padding_res = torch.tensor(padding_res, dtype=torch.long, device=device)
        padding_res = torch.autograd.Variable(padding_res)
        # inputs的形状是(批量⼤⼩, 词数)，输出形状为(批量⼤⼩, 词数, 词向量维度)

        # print(padding_res.shape)
        embeddings = self.embedding(padding_res)
        # embeddings = (batch_size X mx_length X embed_size)
        embedding_packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        packed_output, self.state = self.encoder(embedding_packed, state)  # output, (h, c)
        print('h')
        print(self.state[0].shape)#h:(lstm层数 X batch_size X hidden_nums)

        outputs, inputs_size = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # outputs形状是(词数, 批量⼤⼩, 隐藏单元个数)
        #print('outputs')
        #print(outputs.shape)

        #out形状为(批量⼤⼩,词数,字典大小)
        outs = F.softmax(self.decoder(outputs),dim=2)
        #print('outs')
        #print(outs.shape)
        res = outs[0,:lengths[0]-1,:]
        for i in range(batch_size-1):
            res = torch.cat([res,outs[i+1,:lengths[i+1]-1,:]],dim = 0)

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
                                    collate_fn=batchify,
                                    num_workers=num_workers)

        for X in data_iter:
            (output, state) = model(X)
            """
            def flatten_padded_sequences():
                indices = tf.where(tf.sequence_mask(lengths))
                return tf.gather_nd(maybe_padded_sequences, indices)
            """
            lengths = np.array([len(item) for item in X])
            Y = torch.tensor(np.array(X[0])[:lengths[0]-1],dtype=torch.long,device=device)
            for i in range(batch_size-1):
                Y = torch.cat([Y,torch.tensor(np.array(X[i+1])[:lengths[i+1]-1],dtype=torch.long,device=device)],dim=0)
            y_len = Y.shape[0]
            l = loss(output, Y.long())

            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)
            optimizer.step()
            l_sum += l.item() * y_len
            n += y_len

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, ave-loss %f, time %.2f sec' % (
                    epoch + 1, l_sum/n, time.time() - start))


net = Melody_Attention_RNN(vocab_size, embed_size, num_hiddens, num_layers)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()
mini_data = [[2,3,5,7],[9,10,11],[4,6]]
#mini_data = train_set[-2:]
print(mini_data)
train_melody_rnn(net, mini_data,epochs, optimizer, loss, clip_norm, batch_size, pred_period, device)
#torch.save(net.state_dict(), 'attention_rnn.pth')


"""
class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1*2, len(X), n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]
"""