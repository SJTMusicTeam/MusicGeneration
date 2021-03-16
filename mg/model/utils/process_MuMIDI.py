import os
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor
sys.path.append('/data2/qt/MusicGeneration/mg/model/')
from utils.MuMIDI import MuMIDI_EventSeq
from utils.shared import find_files_by_extensions


def preprocess_MuMIDI_event(path, output_dir):
    melody_events, arrange_events = MuMIDI_EventSeq.extract_split_events(path)
    melody_words = MuMIDI_EventSeq.to_array(melody_events)
    arrange_words = MuMIDI_EventSeq.to_array(arrange_events)

    name = path.split('/')[-1].split('.')[0] + '.data'
    state = {'melody' : melody_words, 'arrangement' : arrange_words}
    save_path = os.path.join(output_dir, name)
    torch.save(state, save_path)
    print(f'success for file:{save_path}')
    return

def preprocess_midi_files_under(input_dir, output_dir, num_workers):
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
            executor.submit(preprocess_MuMIDI_event, path, output_dir)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(f' Error for {path}')
            continue

    print('Done')


if __name__ == '__main__':
    # pp = '/data2/qt/MusicGeneration/egs/dataset/multi_tracks/six_tracks_test.mid'
    # pa = '/data2/qt/MusicGeneration/egs/dataset/tmp_res/test_mumidi_bef.midi'
    # pb = '/data2/qt/MusicGeneration/egs/dataset/tmp_res/test_mumidi_aft.midi'
    # #events = preprocess_REMI_event(pp)
    # DEFAULT_TRACKS = ['melody', 'piano', 'bass', 'guitar', 'string', 'drum']
    # melody_events, arrange_events = MuMIDI_EventSeq.extract_split_events(pp)
    # print(f'melody_events={melody_events}')
    # print(f'arrange_events={arrange_events}')
    # melody_words = MuMIDI_EventSeq.to_array(melody_events)
    # print(f'melody_words={melody_words}')
    # arrange_words = MuMIDI_EventSeq.to_array(arrange_events)
    # print(f'arrange_words={arrange_words}')



    preprocess_midi_files_under(
        input_dir=sys.argv[1],
        output_dir=sys.argv[2],
        num_workers=int(sys.argv[3],
        ))

# python process_MuMIDI.py /data2/qt/MusicGeneration/egs/dataset/lmd_matched_merged /data2/qt/MusicGeneration/egs/dataset/lmd_matched_split 10
