import os
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor
sys.path.append('/data2/qt/MusicGeneration/mg/model/')
from utils.MuMIDI import MuMIDI_EventSeq
from utils.shared import find_files_by_extensions
from utils.data import Melody_Arrangement_Dataset
import logging
import coloredlogs


def preprocess_MuMIDI_event(path, output_dir):
    try:
        name = path.split('/')[-1].split('.')[0] + '.data'
        save_path = os.path.join(output_dir, name)
        if os.path.exists(save_path):
            return
        melody_events, arrange_events = MuMIDI_EventSeq.extract_split_events(path)
        if melody_events is None:
            logger.info(f'Missing melody or tracks for {path}')
            return
        melody_words = MuMIDI_EventSeq.to_array(melody_events)
        arrange_words = MuMIDI_EventSeq.to_array(arrange_events)

        melody_seq_bar = MuMIDI_EventSeq.segmentation(melody_words)
        arrange_seq_bar = MuMIDI_EventSeq.segmentation(arrange_words)

        melody_bar, melody_bar_mask = Melody_Arrangement_Dataset.get_mask(melody_seq_bar, 0)
        arrange_bar, arrange_bar_mask = Melody_Arrangement_Dataset.get_mask(arrange_seq_bar, -1)
        label, label_mask = Melody_Arrangement_Dataset.label_mask(arrange_words)

        state = {'melody' : melody_words, 'arrangement' : arrange_words, 'melody_bar' : melody_bar, 'melody_bar_mask' : melody_bar_mask\
            , 'arrangement_bar' : arrange_bar, 'arrange_bar_mask':arrange_bar_mask, 'label':label, 'label_mask': label_mask }

        torch.save(state, save_path)
        logger.info(f'success for file:{save_path}')
    except Exception as e:
        logger.warning(e)
        logger.info(f' Error for {path}')
    return

def preprocess_midi_files_under(input_dir, output_dir, num_workers):
    if input_dir[-1] != '/':
        input_dir += '/'
    if output_dir[-1] != '/':
        output_dir += '/'

    midi_paths = list(find_files_by_extensions(input_dir, ['.mid', '.midi']))
    print(len(midi_paths))
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


    logger = logging.getLogger(__name__)
    logger.handlers = []
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)

    # pp = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_merged/0a8b29fbd2d0aa0b2a0078464ea093aa_896e8cff892d1293590bce1cae38109c.mid'
    # out_dir = '/data2/qt/MusicGeneration/egs/dataset/lmd_matched_split'
    # preprocess_MuMIDI_event(pp, out_dir)

    preprocess_midi_files_under(
        input_dir=sys.argv[1],
        output_dir=sys.argv[2],
        num_workers=int(sys.argv[3],
        ))

# python process_MuMIDI.py /data2/qt/MusicGeneration/egs/dataset/lmd_matched_merged /data2/qt/MusicGeneration/egs/dataset/lmd_matched_split 10
