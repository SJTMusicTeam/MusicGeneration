import os
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor
from utils.sequence import NoteSeq, EventSeq
import utils.shared


def preprocess_midi_event(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    return event_seq.to_array()

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.shared.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi_event, path)))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue

    for path, future in Bar('Processing').iter(results):
        print(' ', end='[{}]'.format(path), flush=True)
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        torch.save(future.result(), save_path)

    print('Done')


if __name__ == '__main__':
    # pp = '../../../egs/dataset/maestro/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--3.midi'
    # preprocess_midi_event(pp)
    preprocess_midi_files_under(
        midi_root=sys.argv[1],
        save_dir=sys.argv[2],
        num_workers=int(sys.argv[3],
        type='event'))
