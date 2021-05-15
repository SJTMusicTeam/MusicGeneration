'''Music extraction

Uses miditoolkit for interacting with MIDI files
'''
import copy
import miditoolkit
import numpy as np

from miditoolkit.midi.containers import Instrument


def skyline(mido_obj, instr_idx=0):
    '''Melody extraction based on Skyline algorithm

    Based on https://dl.acm.org/doi/10.1145/319463.319470

    Args:
        mido_obj: MidiFile
        instr_idx: index of instrument to extract melody from
    
    Return:
        new_midi_obj: MidiFile
    '''
    start2note = {}
    for note in mido_obj.instruments[instr_idx].notes:
        start = note.start
        if start in start2note:
            start2note[start].append(note)
        else:
            start2note[start] = [note]
    starts = sorted(list(start2note.keys()))
    skyline_notes = []
    for si, start in enumerate(starts):
        notes = start2note[start]
        pitches = [n.pitch for n in notes]
        note = copy.deepcopy(notes[np.argmax(pitches)])
        if si < len(starts)-1:
            note.end = min(note.end, starts[si+1])
        skyline_notes.append(note)
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.markers = mido_obj.markers
    new_midi_obj.tempo_changes = mido_obj.tempo_changes
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = skyline_notes
    new_midi_obj.instruments = [piano_track]
    return new_midi_obj


def top(mido_obj, instr_idx=0, top_thres=0.5):
    '''Melody extraction using time overlap parameter (TOP)

    Based on http://alumni.media.mit.edu/~chaiwei/papers/msthesis.pdf

    Args:
        mido_obj: MidiFile
        instr_idx: index of instrument to extract melody from
        top_thres: TOP threshold
    
    Return:
        new_midi_obj: MidiFile
    '''
    notes = mido_obj.instruments[instr_idx].notes.copy()
    notes = sorted(notes, key=lambda x: x.pitch, reverse=True)
    top_notes = []
    for n in notes:
        overlap = 0.0
        for tn in top_notes:
            o = max(0.0, min(n.end, tn.end)-max(n.start, tn.start))
            overlap += o
        ctop = overlap / (n.end-n.start)
        if ctop <= top_thres:
            top_notes.append(n)
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.markers = mido_obj.markers
    new_midi_obj.tempo_changes = mido_obj.tempo_changes
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = top_notes
    new_midi_obj.instruments = [piano_track]
    return new_midi_obj
