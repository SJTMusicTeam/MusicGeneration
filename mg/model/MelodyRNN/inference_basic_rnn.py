import torch
from music21 import converter, instrument, note, chord, stream, midi#多乐器是否可行
import time
import numpy as np
import pandas as pd
import os
from torch import nn
import torch.nn.functional as F
#from utils import midi2note
import torch.utils.data as Data
import random
import sys


# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128 # (stop playing all previous notes)
MELODY_NO_EVENT = 129 # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.

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


vocab_size = 130 # known 0-127 notes + 128 note_off + 129 no_event
embed_size = 64
drop_out = 0.5
num_hiddens = 128
num_layers = 2
num_gen = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, self.state = self.encoder(embeddings, state)
        outs = F.softmax(self.decoder(outputs),dim=2)
        return outs.view(-1,self.vocab_size), self.state

net = Melody_RNN(vocab_size, embed_size, num_hiddens, num_layers)
net.load_state_dict(torch.load('baisc_rnn.pth'))
def predict_melody_rnn(model, num_gen, prefix, device):
    net.eval()
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

res = predict_melody_rnn(net, num_gen , [60,50], device)
print(res)
res_melody = noteArrayToStream(res)
res_melody.write("midi", "test.mid")



"""
batch_size = self._batch_size()
num_seqs = len(event_sequences)
num_batches = int(np.ceil(num_seqs / float(batch_size)))
"""
