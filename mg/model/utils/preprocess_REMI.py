import os
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor
from utils.REMI import REMI_EventSeq
import utils.shared


def preprocess_REMI_event(path):
    remi_event_seq = REMI_EventSeq.extract_events(path)
    #print(remi_event_seq[:15])
    return REMI_EventSeq.to_array(remi_event_seq)

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.shared.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_REMI_event, path)))
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
    pp = '../../../egs/dataset/tmp_res/test_seq_bef.midi'
    pb = '../../../egs/dataset/tmp_res/test_remi_bef.midi'
    pa = '../../../egs/dataset/tmp_res/test_remi_aft.midi'
    #events = preprocess_REMI_event(pp)
    events = REMI_EventSeq.extract_events(pp)
    words  = REMI_EventSeq.to_array(events)
    event = REMI_EventSeq.to_event(words)
    print(events[:10])
    print(event[:10])
    REMI_EventSeq.write_midi(events,pa)
    REMI_EventSeq.write_midi(event, pb)

    # preprocess_midi_files_under(
    #     midi_root=sys.argv[1],
    #     save_dir=sys.argv[2],
    #     num_workers=int(sys.argv[3],
    #     ))
