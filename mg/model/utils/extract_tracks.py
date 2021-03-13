import os
import os.path
import torch
from pathlib import Path
import numpy as np
import pypianoroll
import pretty_midi
from collections import defaultdict
import json
import sys
sys.path.append('/data2/qt/MusicGeneration/mg/model/')
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor
from utils.shared import find_files_by_extensions

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

def check_name(name, name2,target):
    names = [item.lower() for item in name.split()]
    if target in names:
        return True
    if target == 'melody' and 'flute' in names:
        return True
    names = [item.lower() for item in name2.split()]
    if target in names:
        return True
    if target == 'melody' and 'flute' in names:
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

def count_tracks(collection,program_id):
    cnt = 0
    for key in tracks_name:
        if len(program_id[key])!=0:
            cnt += 1
        # print(f'key = {key}, tracks = {collection[key]}')
    return cnt

def extract_merge(midi_path, instrument_numbers):
    pretty_midi_data, pypiano_data = remove_drum_empty_track(midi_path, drop_drum=False)
    # pretty_midi_data = pretty_midi.PrettyMIDI(midi_path)
    music_tracks = pypianoroll.from_pretty_midi(pretty_midi_data)
    # print(music_tracks)
    collection = defaultdict(list)
    program_id = defaultdict(list)
    program = {}  # json.load()
    program['melody'] = program['bass'] = program['drum'] = -1
    # dict.fromkeys(['piano', 'bass','guitar','drum','string'], [])
    # print(collection)

    for idx, track in enumerate(music_tracks.tracks):
        # print(track)
        # print(pretty_midi_data.instruments[idx].name)

        if track.program == program['melody'] or \
                check_name(track.name, pretty_midi_data.instruments[idx].name, 'melody'):
            collection['melody'].append(track)
            program_id['melody'].append(idx)
        elif track.program in instrument_numbers['drum'] or track.program == program['drum'] or \
                check_name(track.name, pretty_midi_data.instruments[idx].name, 'drum') :
            collection['drum'].append(track)
            program_id['drum'].append(idx)
        elif track.program in instrument_numbers['piano'] or \
                check_name(track.name, pretty_midi_data.instruments[idx].name, 'piano'):
            collection['piano'].append(track)
            program_id['piano'].append(idx)
        elif track.program in instrument_numbers['bass'] or track.program == program['bass'] or \
                check_name(track.name, pretty_midi_data.instruments[idx].name, 'bass'):
            collection['bass'].append(track)
            program_id['bass'].append(idx)
        elif track.program in instrument_numbers['guitar'] or \
                check_name(track.name, pretty_midi_data.instruments[idx].name, 'guitar'):
            collection['guitar'].append(track)
            program_id['guitar'].append(idx)
        else:
            collection['string'].append(track)
            program_id['string'].append(idx)
        # print(collection)

    cnt = count_tracks(collection, program_id)
    if cnt < 3 or (cnt == 2 and len(program_id['melody'])==0 ):
        return None
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
            if key == 'melody' or key == 'string':
                Pro = program_id[key][0]
            else:
                Pro = instrument_numbers[key][0]
            track = pypianoroll.StandardTrack(name=key, program=Pro, \
                              is_drum=IS_DRUM, pianoroll=merged_pianoroll[key])
            pypiano_mult.append(track)

    return pypiano_mult


# multi-threading
def preprocess_merge_midi(path, output_dir):
    Mult = extract_merge(path, instrument_numbers)
    if Mult is None:
        return
    music = Mult.to_pretty_midi()
    # print(path.split('/'))
    name = path.split('/')[-1]
    # print(name)
    # output_name = os.path.join(output_dir, name)
    code = hashlib.md5(path.encode()).hexdigest()
    save_path = os.path.join(output_dir, '{}_{}'.format(code,name))
    # print(os.path.join(output_dir, output_name))
    music.write(save_path)
    print(f'success for file:{save_path}')
    return

def preprocess_merge_midi_files_under(input_dir, output_dir, num_workers, dict_path):
    global program_dict
    program_dict = get_program_dict(dict_path=dict_path)
    if input_dir[-1] != '/':
        input_dir += '/'
    if output_dir[-1] != '/':
        output_dir += '/'

    midi_paths = list(find_files_by_extensions(input_dir, ['.mid', '.midi']))
    os.makedirs(output_dir, exist_ok=True)

    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            # name_with_sub_folder = path.replace(midi_root, "")
            # output_name = os.path.join(save_dir, name_with_sub_folder)
            executor.submit(preprocess_merge_midi, path, output_dir)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue

    # for path, future in Bar('Processing').iter(results):
    #     print(' ', end='[{}]'.format(path), flush=True)
    #     name = os.path.basename(path)
    #     code = hashlib.md5(path.encode()).hexdigest()
    #     save_path = os.path.join(output_dir, out_fmt.format(name, code))
    #     torch.save(future.result(), save_path)

    print('Done')

def get_program_dict(dict_path):
    with open(dict_path, 'r') as f:
        program_dict = json.load(f)
    return program_dict

# dict_path = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_output/program_result.json'
# dict_path = '/data2/qt/midi-miner/example/output/program_result.json'
# program_dict = get_program_dict(dict_path=dict_path)


if __name__ == '__main__':
    # print(program_dict)
    # pp = '../../../egs/dataset/multi_tracks/3a3a1f1c159b128c0715c6dbb56dd612.mid'
    # Mult = extract_merge(pp, instrument_numbers)
    # # print(Mult)
    # pf = '../../../egs/dataset/multi_tracks/six_tracks_test.mid'
    # music = Mult.to_pretty_midi()
    # music.write(pf)

    preprocess_merge_midi_files_under(
        input_dir=sys.argv[1],
        output_dir=sys.argv[2],
        num_workers=int(sys.argv[3]),
        dict_path=sys.argv[4])
# python extract_tracks.py /data2/qt/MusicGeneration/egs/dataset/lmd_matched /data2/qt/MusicGeneration/egs/dataset/lmd_matched_merged 10 /data2/qt/MusicGeneration/egs/dataset/lmd_matched_output/program_result.json
