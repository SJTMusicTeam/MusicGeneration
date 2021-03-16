import numpy as np
import collections
import miditoolkit
import copy
import utils.chord_inference as chord_inference
from collections import defaultdict

# parameters for input
# DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 32 # 16 for REMI

DEFAULT_DURATION_STEP = 60 #120,60 3840/120 = 32 or 1920/60 = 32
DEFAULT_DURATION_RANGE = range(DEFAULT_DURATION_STEP, 1921)
DEFAULT_DURATION_BINS = np.arange(DEFAULT_DURATION_RANGE.start, DEFAULT_DURATION_RANGE.stop,
                                  DEFAULT_DURATION_STEP, dtype=int)

DEFAULT_tempo_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

DEFAULT_VELOCITY = 100
DEFAULT_PITCH_RANGE = range(0, 127)

DEFAULT_VELOCITY_STEPS = 4 #32
DEFAULT_VELOCITY_RANGE = range(0, 129)
DEFAULT_VELOCITY_BINS = np.arange(DEFAULT_VELOCITY_RANGE.start, DEFAULT_VELOCITY_RANGE.stop,
                                  DEFAULT_VELOCITY_STEPS)

# parameters for output
DEFAULT_RESOLUTION = 480

DEFAULT_TRACKS = ['melody', 'piano', 'bass', 'guitar', 'string', 'drum']
tracks_idx = {}
for idx, track in enumerate(DEFAULT_TRACKS):
    tracks_idx[track] = idx

chord_quality = ['maj', 'min', 'dim', 'aug', 'dom']  # 5
chord_root = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']  # 12
chord_map = {}
new_idx = 0
for i in range(len(chord_quality)):
    for j in range(len(chord_root)):
        chord_map[chord_root[j] + ':' + chord_quality[i]] = new_idx
        new_idx += 1
chord_map['N:N'] = new_idx
new_idx += 1
inv_chord_map = {v: k for k, v in zip(chord_map.keys(), chord_map.values())}

instrument_numbers = {}
instrument_numbers['melody'] = [73]
instrument_numbers['piano'] = [1,2,3,4,5,6,7,8] #Piano
instrument_numbers['bass'] = [33,34,35,36,37,38,39,40] #Bass
instrument_numbers['guitar'] = [25,26,27,28,29,30,31,32] #Guitar
instrument_numbers['drum'] = [114,115,116,117,118,119] #drum
instrument_numbers['string'] = [66]


# inv_chord_map = {}
# for k,v in zip(chord_map.keys(),chord_map.values()):
#     inv_chord_map[v]=k
# print(inv_chord_map)

def get_velocity_bins():
    n = MuMIDI_EventSeq.velocity_range.stop - MuMIDI_EventSeq.velocity_range.start
    return np.arange(MuMIDI_EventSeq.velocity_range.start,
                     MuMIDI_EventSeq.velocity_range.stop,
                     n / (MuMIDI_EventSeq.velocity_steps - 1))


# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, track=''):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.track = track

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, track={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.track)


# read notes and tempo changes from midi (multi-tracks)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # print(midi_obj.instruments)

    # note
    note_items = []
    for instr in range(len(midi_obj.instruments)):
        notes = midi_obj.instruments[instr].notes
        notes.sort(key=lambda x: (x.start, x.pitch))
        for note in notes:
            note_items.append(Item(
                name='note',
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
                track=midi_obj.instruments[instr].name))
    note_items.sort(key=lambda x: x.start)
    # print(note_items)

    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # print(tempo_items)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    # print(wanted_ticks)

    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    # print(tempo_items)

    return note_items, tempo_items


# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items


# extract chord
def extract_chords(items):
    method = chord_inference.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0],
            track=''))
    return output


# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    # items.sort(key=lambda x: x.start)
    items.sort(key=lambda x: (x.start, x.track))
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    l = 0
    r = 0
    mx = len(items)
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):

        # left:  a[i - 1] < v <= a[i]
        # right: a[i - 1] <= v < a[i]
        # l = np.searchsorted(items, db1,'left')
        # r = np.searchsorted(items, db2,'right')
        while l < mx and items[l].start < db1:
            l += 1
        while r < mx and items[r].start <= db2:
            r += 1
        if l < r:
            insiders = items[l:r]
        else:
            insiders = []
        # for item in items:
        #     if (item.start >= db1) and (item.start < db2):
        #         insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups


# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)


# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='bar',
            time=None,
            value=0,
            # value=None,
            text='{}'.format(n_downbeat)))
        last_position = -1
        last_track = ''
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            if index != last_position:
                last_position = index
                events.append(Event(
                    name='position',
                    time=item.start,
                    value=index,
                    # value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))
            if item.track != last_track and item.track != '':
                last_track = item.track
                events.append(Event(
                    name=f'track_{item.track}',
                    time=item.start,
                    value=tracks_idx[item.track],
                    # value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))

            if item.name == 'note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS,
                    item.velocity,
                    side='right') - 1
                events.append(Event(
                    name='note_velocity',
                    time=item.start,
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='note_on',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                events.append(Event(
                    name='note_duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'chord':
                events.append(Event(
                    name='chord',
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'tempo':
                tempo = item.pitch
                if tempo in DEFAULT_tempo_INTERVALS[0]:
                    tempo_style = Event('tempo_class', item.start, 0, None)  # slow
                    tempo_value = Event('tempo_value', item.start,
                                        tempo - DEFAULT_tempo_INTERVALS[0].start, None)
                elif tempo in DEFAULT_tempo_INTERVALS[1]:
                    tempo_style = Event('tempo_class', item.start, 1, None)  # mid
                    tempo_value = Event('tempo_value', item.start,
                                        tempo - DEFAULT_tempo_INTERVALS[1].start, None)
                elif tempo in DEFAULT_tempo_INTERVALS[2]:
                    tempo_style = Event('tempo_class', item.start, 2, None)  # fast
                    tempo_value = Event('tempo_value', item.start,
                                        tempo - DEFAULT_tempo_INTERVALS[2].start, None)
                elif tempo < DEFAULT_tempo_INTERVALS[0].start:
                    tempo_style = Event('tempo_class', item.start, 0, None)  # slow
                    tempo_value = Event('tempo_value', item.start, 0, None)
                elif tempo > DEFAULT_tempo_INTERVALS[2].stop:
                    tempo_style = Event('tempo_class', item.start, 2, None)  # fast
                    tempo_value = Event('tempo_value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)

    return events


#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events


def write_midi(events, output_path, prompt_path=None):
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events) - 3):
        if events[i].name == 'bar' and i > 0:
            temp_notes.append('bar')
            temp_chords.append('bar')
            temp_tempos.append('bar')
        elif events[i].name == 'position' and \
                events[i + 1].name == 'note_velocity' and \
                events[i + 2].name == 'note_on' and \
                events[i + 3].name == 'note_duration':
            # start time and end time from position
            position = int(events[i].value)
            # velocity
            index = int(events[i + 1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i + 2].value)
            # duration
            index = int(events[i + 3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == 'position' and events[i + 1].name == 'chord':
            position = int(events[i].value)
            temp_chords.append([position, events[i + 1].value])
        elif events[i].name == 'position' and \
                events[i + 1].name == 'tempo_class' and \
                events[i + 2].name == 'tempo_value':
            position = int(events[i].value)
            if events[i + 1].value == 'slow':
                tempo = DEFAULT_tempo_INTERVALS[0].start + int(events[i + 2].value)
            elif events[i + 1].value == 'mid':
                tempo = DEFAULT_tempo_INTERVALS[1].start + int(events[i + 2].value)
            elif events[i + 1].value == 'fast':
                tempo = DEFAULT_tempo_INTERVALS[2].start + int(events[i + 2].value)
            temp_tempos.append([position, tempo])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.midi.containers.Note(start=st, end=et, pitch=pitch,
                                                          velocity=velocity))  # velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION * 4 * 4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0] + last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)


class MuMIDI_EventSeq:
    pitch_range = DEFAULT_PITCH_RANGE
    velocity_range = DEFAULT_VELOCITY_RANGE
    velocity_steps = DEFAULT_VELOCITY_STEPS
    duration_bins = DEFAULT_DURATION_BINS

    def __init__(self, events=[]):
        pass

    @staticmethod
    def dim():
        return sum(MuMIDI_EventSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        """
        * note_on(pitch) : 128 : 0-127
        * note_duration : 32: 0-31
        * note_velocity: 32: 0-31
        * bar: 1: 0
        * position: 32: 1-32
        * track: 6: melody, piano, bass, guitar, string, drum
        * tempo: 3 + 60
            tempo_style: 3:low,mid,fast  [range(30, 90), range(90, 150), range(150, 210)]
            tempo_value: 180: 30-90(60),90-150(60),150-210(60)
        * chord: 84? 60?
        12 chord roots
        ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        7 chord qualities
        (major, minor, diminished, augmented, major7, minor7, half_diminished)
        'maj','min','dim','aug','dom' # 5
        """
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(MuMIDI_EventSeq.pitch_range)
        feat_dims['note_duration'] = len(MuMIDI_EventSeq.duration_bins)
        feat_dims['note_velocity'] = len(DEFAULT_VELOCITY_BINS)
        feat_dims['bar'] = 1
        feat_dims['position'] = DEFAULT_FRACTION
        feat_dims['track'] = len(DEFAULT_TRACKS)
        feat_dims['tempo_class'] = len(DEFAULT_tempo_INTERVALS)
        feat_dims['tempo_value'] = len(DEFAULT_tempo_INTERVALS[0])
        feat_dims['chord'] = len(chord_map)
        return feat_dims

    @staticmethod
    def dims_feat():
        offset = 0
        feat_ranges = collections.OrderedDict()
        idxs_feat = collections.OrderedDict()
        for feat_name, feat_dim in MuMIDI_EventSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            for i in range(0, feat_dim):
                idxs_feat[offset + i] = (feat_name, i)
            offset += feat_dim
        # print(idxs_feat)
        return idxs_feat

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in MuMIDI_EventSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        # print('feat_range:')
        # print(feat_ranges)
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
        n = MuMIDI_EventSeq.velocity_range.stop - MuMIDI_EventSeq.velocity_range.start
        return np.arange(MuMIDI_EventSeq.velocity_range.start,
                         MuMIDI_EventSeq.velocity_range.stop,
                         n / (MuMIDI_EventSeq.velocity_steps - 1))

    @staticmethod
    def extract_events(input_path):  # detail
        note_items, tempo_items = read_items(input_path)
        # note_item:   Item(name=Note, start=956, end=1530, velocity=55, pitch=59)
        # tempo_items: Item(name=tempo, start=0, end=None, velocity=None, pitch=120)
        note_items = quantize_items(note_items)
        # Item(name=Note, start=960, end=1534, velocity=55, pitch=59)

        max_time = note_items[-1].end

        chord_items = extract_chords(note_items)
        # Item(name=Chord, start=0, end=960, velocity=None, pitch=N:N)
        # Item(name=Chord, start=960, end=2880, velocity=None, pitch=G:maj)

        items = chord_items + tempo_items + note_items
        groups = group_items(items, max_time)
        events = item2event(groups)
        return events

    @staticmethod
    def to_array(events):
        feat_idxs = MuMIDI_EventSeq.feat_ranges()
        idxs = []
        for event in events:
            # print(f'event_name = {event.name}, event_val = {event.value}')
            if event.name == 'chord':
                idxs.append(feat_idxs[event.name][chord_map[event.value]])
            elif event.name.startswith('track'):
                idxs.append(feat_idxs[event.name[:5]][event.value])
            else:
                idxs.append(feat_idxs[event.name][event.value])
        dtype = np.uint8 if MuMIDI_EventSeq.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)

    @staticmethod
    def to_event(words):
        idxs_feat = MuMIDI_EventSeq.dims_feat()
        events = []
        for word in words:
            event_name, event_value = idxs_feat[word]
            if event_name == 'chord':
                event_value = inv_chord_map[event_value]
            if event_name == 'track':
                event_name = event_name + '_' + DEFAULT_TRACKS[event_value]
            events.append(Event(event_name, None, event_value, None))
        return events

    @staticmethod
    def from_array(words):
        events = MuMIDI_EventSeq.to_event(words)
        return events

    @staticmethod
    def write_midi(events, output_path):
        # get downbeat and note (no time)
        temp_notes = []
        temp_chords = []
        temp_tempos = []
        position = -1
        track = ''
        for i in range(len(events) - 3):
            if events[i].name == 'bar' and i > 0:
                temp_notes.append('bar')
                temp_chords.append('bar')
                temp_tempos.append('bar')
                track = ''
            else:
                if events[i].name == 'position':
                    position = int(events[i].value)
                if events[i].name.startswith('track'):
                    track = events[i].name.split('_')[-1]
                if  events[i].name == 'note_velocity' and \
                    events[i + 1].name == 'note_on' and \
                    events[i + 2].name == 'note_duration':
                    # start time and end time from position
                    # velocity
                    index = int(events[i].value)
                    velocity = int(DEFAULT_VELOCITY_BINS[index])
                    # pitch
                    pitch = int(events[i + 1].value)
                    # duration
                    index = int(events[i + 2].value)
                    duration = DEFAULT_DURATION_BINS[index]
                    # adding
                    temp_notes.append([position, velocity, pitch, duration, track])
                elif events[i].name == 'chord':
                    temp_chords.append([position, events[i].value])
                elif    events[i].name == 'tempo_class' and \
                        events[i + 1].name == 'tempo_value':
                    position = int(events[i].value)
                    """
                    if events[i + 1].value == 0:#'slow':
                        tempo = DEFAULT_tempo_INTERVALS[0].start + int(events[i + 2].value)
                    elif events[i + 1].value == 1:#'mid':
                        tempo = DEFAULT_tempo_INTERVALS[1].start + int(events[i + 2].value)
                    elif events[i + 1].value == 2:#'fast':
                        tempo = DEFAULT_tempo_INTERVALS[2].start + int(events[i + 2].value)
                    """
                    tempo = DEFAULT_tempo_INTERVALS[events[i].value].start + int(events[i + 1].value)
                    temp_tempos.append([position, tempo])
        # get specific time for notes
        ticks_per_beat = DEFAULT_RESOLUTION
        ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4
        notes = defaultdict(list)
        current_bar = 0
        for note in temp_notes:
            if note == 'bar':
                current_bar += 1
            else:
                position, velocity, pitch, duration, track = note
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                # duration (end time)
                et = st + duration
                notes[track].append(miditoolkit.midi.containers.Note(start=st, end=et, pitch=pitch,
                                                              velocity=velocity))  # Note(velocity, pitch, st, et))
        # get specific time for chords
        if len(temp_chords) > 0:
            chords = []
            current_bar = 0
            for chord in temp_chords:
                if chord == 'bar':
                    current_bar += 1
                else:
                    position, value = chord
                    # position (start time)
                    current_bar_st = current_bar * ticks_per_bar
                    current_bar_et = (current_bar + 1) * ticks_per_bar
                    flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                    st = flags[position]
                    chords.append([st, value])
        
        # get specific time for tempos
        tempos = []
        current_bar = 0
        for tempo in temp_tempos:
            if tempo == 'bar':
                current_bar += 1
            else:
                position, value = tempo
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                tempos.append([int(st), value])
        # write
        """
        Doing        
        """
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        # inst = {}
        for track in DEFAULT_TRACKS:
            IS_DRUM = False
            if track == 'drum':
                IS_DRUM = True

            # print(f'track = {track}, pro_id={instrument_numbers[track]}')
            Pro_id = instrument_numbers[track][0]
            inst = miditoolkit.midi.containers.Instrument(program=Pro_id, is_drum=IS_DRUM,name=track)
            inst.notes = notes[track]
            midi.instruments.append(inst)
        print(midi.instruments)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
        # write
        print(output_path)
        midi.dump(output_path)
        return midi


if __name__ == '__main__':
    #print(DEFAULT_DURATION_BINS)
    #print(len(DEFAULT_DURATION_BINS))
    # pp = '../../../egs/dataset/lmd_matched_merged/six_tracks_test.mid'
    pp = '/data2/qt/MusicGeneration/egs/dataset/multi_tracks/six_tracks_test.mid'
    pa = '/data2/qt/MusicGeneration/egs/dataset/tmp_res/test_mumidi_bef.midi'
    pb = '/data2/qt/MusicGeneration/egs/dataset/tmp_res/test_mumidi_aft.midi'
    #events = preprocess_REMI_event(pp)
    events = MuMIDI_EventSeq.extract_events(pp)
    words  = MuMIDI_EventSeq.to_array(events)
    event = MuMIDI_EventSeq.to_event(words)
    print(*events[:10],sep='\n')
    print('*'*10)
    print(*event[:10],sep='\n')
    MuMIDI_EventSeq.write_midi(events, pa)
    MuMIDI_EventSeq.write_midi(event, pb)

    # preprocess_midi_files_under(
    #     midi_root=sys.argv[1],
    #     save_dir=sys.argv[2],
    #     num_workers=int(sys.argv[3],
    #     ))
    # note_items, tempo_items = read_items(pp)
    # # note_item:   Item(name=Note, start=956, end=1530, velocity=55, pitch=59)
    # # tempo_items: Item(name=tempo, start=0, end=None, velocity=None, pitch=120)
    # note_items = quantize_items(note_items)
    # # Item(name=Note, start=960, end=1534, velocity=55, pitch=59)
    #
    # max_time = note_items[-1].end
    #
    # chord_items = extract_chords(note_items)
    # # print(chord_items)
    # # Item(name=Chord, start=0, end=960, velocity=None, pitch=N:N)
    # # Item(name=Chord, start=960, end=2880, velocity=None, pitch=G:maj)
    #
    # items = chord_items + tempo_items + note_items
    # groups = group_items(items, max_time)
    # # for group in groups:
    # #     print(group)
    # events = item2event(groups)
    # print(*events[:100], sep='\n')


