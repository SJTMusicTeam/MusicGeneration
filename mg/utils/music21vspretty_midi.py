from music21 import converter, instrument, note, chord, stream, midi#多乐器是否可行
import time
import numpy as np
import pandas as pd
import os
import random
import sys
import pretty_midi
from pretty_midi import PrettyMIDI, Note, Instrument
import copy
import itertools
import collections

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
    # cnt = 0
    for element in stream.flat:
        # if cnt<50:
        #     print('offset=  %f, quarterLength= %f '%(element.offset,element.quarterLength))
        # cnt += 1
        if isinstance(element, note.Note):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=np.int)
    print(np_stream_list)
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


## Play a melody stream
#sp = midi.realtime.StreamPlayer(melody_stream)
#sp.play()
fpath = "../../egs/dataset/maestro/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--3.midi"
wm_mid = converter.parse(fpath)
# print(wm_mid.flat)
#wm_mid.show()
wm_mel_rnn = streamToNoteArray(wm_mid)[:50]
print(wm_mel_rnn)

#noteArrayToStream(wm_mel_rnn).show()
#noteArrayToStream(wm_mel_rnn).write("midi", "../../egs/dataset/tmp_res/music_test.mid")

music = pretty_midi.PrettyMIDI(fpath)
notes = itertools.chain(*[
            inst.notes for inst in music.instruments
            if inst.program in range(128) and not inst.is_drum])
print(notes)
print([inst.notes[:50] for inst in music.instruments
       if inst.program in range(128) and not inst.is_drum])