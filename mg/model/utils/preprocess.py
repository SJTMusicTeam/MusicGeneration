import os
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor
from utils.sequence import NoteSeq, EventSeq, ControlSeq
import utils.shared


def preprocess_midi_event(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    return event_seq.to_array()

def preprocess_midi_event_control(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()

def preprocess_midi_files_under(midi_root, save_dir, num_workers, type='event'):
    midi_paths = list(utils.shared.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            if type=='event':
                results.append((path, executor.submit(preprocess_midi_event, path)))
            else:
                if type=='event_control':
                    results.append((path, executor.submit(preprocess_midi_event_control, path)))

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
    preprocess_midi_files_under(
        midi_root=sys.argv[1],
        save_dir=sys.argv[2],
        num_workers=int(sys.argv[3],
        type='event'))
