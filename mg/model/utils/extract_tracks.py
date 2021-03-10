import os
import os.path
import random
from pathlib import Path
import numpy as np
import pypianoroll
import pretty_midi
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from collections import defaultdict

tracks_name = ['melody', 'piano', 'bass', 'guitar', 'drum', 'string']
instrument_numbers = {}
instrument_numbers['piano'] = [1,2,3,4,5,6,7,8] #Piano
instrument_numbers['bass'] = [33,34,35,36,37,38,39,40] #Bass
instrument_numbers['guitar'] = [25,26,27,28,29,30,31,32] #Guitar
instrument_numbers['drum'] = [114,115,116,117,118,119] #drum
# instrument_numbers['string']

def remove_drum_empty_track(midi_file, drop_drum=True):
    '''
    Data Filtration:
    1. read pretty midi data, (dorp_drum: remove the drum track)
    2. remove emtpy track,
    also remove track with fewer than 20 notes of the track
    #also remove track with fewer than 10% notes of the track
    with most notes
    ********
    Return: pretty_midi object, pypianoroll object
    '''

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)

    except Exception as e:
        print(f'exceptions {e} when read the file {midi_file}')
        return None, None

    ### remove drum track
    if drop_drum:
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                midi_data.instruments.remove(instrument)

    ### remove empty track
    pypiano_data = pypianoroll.from_pretty_midi(midi_data)
    note_count = [np.count_nonzero(np.any(track.pianoroll, axis=1)) for track in pypiano_data.tracks]
    empty_indices = np.array(note_count) < 20  # note_count / np.max(note_count) < 0.1
    remove_indices = np.arange(len(pypiano_data.tracks))[empty_indices]

    for index in sorted(remove_indices, reverse=True):
        del pypiano_data.tracks[index]
        del midi_data.instruments[index]

    return midi_data, pypiano_data

def check_melody(name):
    names = [item.lower() for item in name.split()]
    if 'melody' in names:
        return True
    if 'flute' in names:
        return True
    return False

def get_merged(collection):
    #will hold all merged instrument tracks
    merge_piano_roll_list = defaultdict(list)
    for instrument in collection.keys():
        try:
            #merged_pianorolls = np.empty(shape=(0,32,128))
            #concatenate/stack all tracks for a single instrument
            if len(collection[instrument]) > 1:
                if collection[instrument]:
                    merged_pianorolls = np.stack([track.pianoroll \
                        for track in collection[instrument]], -1)
                    # print(collection[instrument][0].pianoroll.shape)
                    # print(merged_pianorolls)
                    # print(merged_pianorolls.shape)
                    # merged_pianorolls = merged_pianorolls[:, :, :, 0]
                    merged_piano_rolls = np.any(merged_pianorolls, axis=-1)
                    # print(merged_piano_rolls)
                    # print(merged_piano_rolls.shape)
                    merge_piano_roll_list[instrument] = merged_piano_rolls
            else:
                merge_piano_roll_list[instrument] = np.squeeze(collection[instrument])
        except Exception as e:
            print("ERROR!!!!!----> Cannot concatenate/merge track for instrument:", \
                  instrument, " with error ", e)
    # merge_piano_roll_list = np.stack([track for track in merge_piano_roll_list], -1)
    # return merge_piano_roll_list.reshape(-1,32,128,4)
    return merge_piano_roll_list

# def merged_tracks(collection):
#     multitrack[category_list[key]].get_merged_pianoroll()

#def pianoroll_to_tracks(merged_pianoroll):


def extract_merge(midi_path, instrument_numbers):
    pretty_midi_data, pypiano_data = remove_drum_empty_track(midi_path, drop_drum=False)
    # pretty_midi_data = pretty_midi.PrettyMIDI(midi_path)
    music_tracks = pypianoroll.from_pretty_midi(pretty_midi_data)
    print(music_tracks)
    collection = defaultdict(list)
    program_id = defaultdict(list)
    program = {}  # json.load()
    program['melody'] = program['bass'] = program['drum'] = -1
    # dict.fromkeys(['piano', 'bass','guitar','drum','string'], [])
    # print(collection)

    for idx, track in enumerate(music_tracks.tracks):
        # print(track)
        # print(pretty_midi_data.instruments[idx].name)

        if track.program == program['melody'] or check_melody(track.name) \
                or check_melody(pretty_midi_data.instruments[idx].name):
            collection['melody'].append(track)
            program_id['melody'].append(idx)
        elif track.program in instrument_numbers['piano']:
            collection['piano'].append(track)
            program_id['piano'].append(idx)
        elif track.program in instrument_numbers['bass'] or track.program == program['bass']:
            collection['bass'].append(track)
            program_id['bass'].append(idx)
        elif track.program in instrument_numbers['guitar']:
            collection['guitar'].append(track)
            program_id['guitar'].append(idx)
        elif track.program in instrument_numbers['drum'] or track.program == program['drum']:
            collection['drum'].append(track)
            program_id['drum'].append(idx)
        else:
            collection['string'].append(track)
            program_id['string'].append(idx)
        # print(collection)

    # print(collection['melody'])
    # print(collection['piano'])
    # print(collection['bass'])
    # print(collection['drum'])
    # print(collection['guitar'])
    # print(collection['string'])
    # for key in tracks_name:
    #     collection[key] = music_tracks[collection[key]].get_merged_pianoroll()
    merged_pianoroll = get_merged(collection)
    #merged_tracks = pianoroll_to_tracks(merged_pianoroll)
    pypiano_mult = pypianoroll.Multitrack(name=music_tracks.name, resolution=music_tracks.resolution, \
                                          tempo=music_tracks.tempo, downbeat=music_tracks.downbeat)
    for key in tracks_name:
        if len(merged_pianoroll[key]) != 0:
            IS_DRUM = False
            if key == 'drum':
                IS_DRUM = True
            track = pypianoroll.StandardTrack(name=key, program=program_id[key][0], \
                              is_drum=IS_DRUM, pianoroll=merged_pianoroll[key])
            pypiano_mult.append(track)

    return pypiano_mult

if __name__ == '__main__':
    pp = '../../../egs/dataset/multi_tracks/3a3a1f1c159b128c0715c6dbb56dd612.mid'
    Mult = extract_merge(pp, instrument_numbers)
    print(Mult)
    pf = '../../../egs/dataset/multi_tracks/six_tracks_test.mid'
    music = Mult.to_pretty_midi()
    music.write(pf)
